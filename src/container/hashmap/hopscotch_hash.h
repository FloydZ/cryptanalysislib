#ifndef CRYPTANALYSISLIB_HOPSCOTCH_HASH_H
#define CRYPTANALYSISLIB_HOPSCOTCH_HASH_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "growth_policy.h"

#if (defined(__GNUC__) && (__GNUC__ == 4) && (__GNUC_MINOR__ < 9))
#define TSL_HH_NO_RANGE_ERASE_WITH_CONST_ITERATOR
#endif

namespace tsl {
	namespace detail_hopscotch_hash {

		///
		/// \tparam T
		template <typename T>
		struct make_void {
			using type = void;
		};

		template <typename T, typename = void>
		struct has_is_transparent : std::false_type {};

		template <typename T>
		struct has_is_transparent<T,
		                          typename make_void<typename T::is_transparent>::type>
		    : std::true_type {};

		template <typename T, typename = void>
		struct has_key_compare : std::false_type {};

		template <typename T>
		struct has_key_compare<T, typename make_void<typename T::key_compare>::type>
		    : std::true_type {};

		template <typename U>
		struct is_power_of_two_policy : std::false_type {};

		template <std::size_t GrowthFactor>
		struct is_power_of_two_policy<cryptanalysislib::hh::power_of_two_growth_policy<GrowthFactor>>
		    : std::true_type {};


		/**
		* Each bucket may store up to three elements:
		* - An aligned storage to store a value_type object with placement-new.
		* - An (optional) hash of the value in the bucket.
		* - An unsigned integer of type neighborhood_bitmap used to tell us which
		* buckets in the neighborhood of the current bucket contain a value with a hash
		* belonging to the current bucket.
		*
		* For a bucket 'bct', a bit 'i' (counting from 0 and from the least significant
		* bit to the most significant) set to 1 means that the bucket 'bct + i'
		* contains a value with a hash belonging to bucket 'bct'. The bits used for
		* that, start from the third least significant bit. The two least significant
		* bits are reserved:
		* - The least significant bit is set to 1 if there is a value in the bucket
		* storage.
		* - The second least significant bit is set to 1 if there is an overflow. More
		* than NeighborhoodSize values give the same hash, all overflow values are
		* stored in the m_overflow_elements list of the map.
		*
		* Details regarding hopscotch hashing an its implementation can be found here:
		*  https://tessil.github.io/2016/08/29/hopscotch-hashing.html
		*/
		static const std::size_t NB_RESERVED_BITS_IN_NEIGHBORHOOD = 2;

		using truncated_hash_type = std::uint_least32_t;

		/**
		* Helper class that stores a truncated hash if StoreHash is true and nothing
		* otherwise.
		*/
		template <bool StoreHash>
		class hopscotch_bucket_hash {
		public:
			[[nodiscard]] constexpr inline bool bucket_hash_equal(std::size_t /*hash*/) const noexcept {
				return true;
			}

			[[nodiscard]] constexpr inline truncated_hash_type truncated_bucket_hash() const noexcept {
				return 0;
			}

		protected:
			constexpr inline void copy_hash(const hopscotch_bucket_hash&) noexcept {}
			constexpr inline void set_hash(truncated_hash_type /*hash*/) noexcept {}
		};

		template <>
		class hopscotch_bucket_hash<true> {
		public:
			[[nodiscard]] constexpr inline bool bucket_hash_equal(const std::size_t hash) const noexcept {
				return m_hash == truncated_hash_type(hash);
			}

			[[nodiscard]] constexpr inline truncated_hash_type truncated_bucket_hash() const noexcept {
				return m_hash;
			}

		protected:
			constexpr inline void copy_hash(const hopscotch_bucket_hash& bucket) noexcept {
				m_hash = bucket.m_hash;
			}

			constexpr inline void set_hash(const truncated_hash_type hash) noexcept { m_hash = hash; }

		private:
			truncated_hash_type m_hash;
		};

		template <typename ValueType, unsigned int NeighborhoodSize, bool StoreHash>
		class hopscotch_bucket : public hopscotch_bucket_hash<StoreHash> {
		private:
			static const std::size_t MIN_NEIGHBORHOOD_SIZE = 4;
			static const std::size_t MAX_NEIGHBORHOOD_SIZE = 64u - NB_RESERVED_BITS_IN_NEIGHBORHOOD;

			static_assert(NeighborhoodSize >= 4, "NeighborhoodSize should be >= 4.");
			// We can't put a variable in the message, ensure coherence
			static_assert(MIN_NEIGHBORHOOD_SIZE == 4);

			static_assert(NeighborhoodSize <= 62, "NeighborhoodSize should be <= 62.");
			// We can't put a variable in the message, ensure coherence
			static_assert(MAX_NEIGHBORHOOD_SIZE == 62);

			static_assert(!StoreHash || NeighborhoodSize <= 30,
			              "NeighborhoodSize should be <= 30 if StoreHash is true.");
			// We can't put a variable in the message, ensure coherence
			static_assert(MAX_NEIGHBORHOOD_SIZE - 32 == 30);

			using bucket_hash = hopscotch_bucket_hash<StoreHash>;

		public:
			using value_type = ValueType;
			using neighborhood_bitmap = LogTypeTemplate<NeighborhoodSize + NB_RESERVED_BITS_IN_NEIGHBORHOOD>;

			constexpr hopscotch_bucket() noexcept : bucket_hash(), m_neighborhood_infos(0) {
				ASSERT(empty());
			}

			constexpr hopscotch_bucket(const hopscotch_bucket& bucket) noexcept (
			        std::is_nothrow_copy_constructible<value_type>::value)
			    : bucket_hash(bucket), m_neighborhood_infos(0) {
				if (!bucket.empty()) {
					::new (static_cast<void*>(std::addressof(m_value))) value_type(bucket.value());
				}

				m_neighborhood_infos = bucket.m_neighborhood_infos;
			}

			constexpr hopscotch_bucket(hopscotch_bucket&& bucket) noexcept (
			        std::is_nothrow_move_constructible<value_type>::value)
			    : bucket_hash(std::move(bucket)), m_neighborhood_infos(0) {
				if (!bucket.empty()) {
					::new (static_cast<void*>(std::addressof(m_value))) value_type(std::move(bucket.value()));
				}

				m_neighborhood_infos = bucket.m_neighborhood_infos;
			}

			constexpr hopscotch_bucket& operator=(const hopscotch_bucket& bucket) noexcept (
			        std::is_nothrow_copy_constructible<value_type>::value) {
				if (this != &bucket) {
					remove_value();

					bucket_hash::operator=(bucket);
					if (!bucket.empty()) {
						::new (static_cast<void*>(std::addressof(m_value)))
						        value_type(bucket.value());
					}

					m_neighborhood_infos = bucket.m_neighborhood_infos;
				}

				return *this;
			}

			constexpr hopscotch_bucket& operator=(hopscotch_bucket&&) = delete;

			constexpr ~hopscotch_bucket() noexcept {
				if (!empty()) {
					destroy_value();
				}
			}

			constexpr inline neighborhood_bitmap neighborhood_infos() const noexcept {
				return neighborhood_bitmap(m_neighborhood_infos >> NB_RESERVED_BITS_IN_NEIGHBORHOOD);
			}

			constexpr inline void set_overflow(bool has_overflow) noexcept {
				if (has_overflow) {
					m_neighborhood_infos = neighborhood_bitmap(m_neighborhood_infos | 2);
				} else {
					m_neighborhood_infos = neighborhood_bitmap(m_neighborhood_infos & ~2);
				}
			}

			[[nodiscard]] constexpr inline bool has_overflow() const noexcept {
				return (m_neighborhood_infos & 2) != 0;
			}

			[[nodiscard]] constexpr inline bool empty() const noexcept {
				return (m_neighborhood_infos & 1) == 0;
			}

			constexpr inline void toggle_neighbor_presence(const std::size_t ineighbor) noexcept {
				ASSERT(ineighbor <= NeighborhoodSize);
				m_neighborhood_infos = neighborhood_bitmap(
				        m_neighborhood_infos ^
				        (1ull << (ineighbor + NB_RESERVED_BITS_IN_NEIGHBORHOOD)));
			}

			[[nodiscard]] constexpr inline bool check_neighbor_presence(const std::size_t ineighbor) const noexcept {
				ASSERT(ineighbor <= NeighborhoodSize);
				if (((m_neighborhood_infos >>
				      (ineighbor + NB_RESERVED_BITS_IN_NEIGHBORHOOD)) &
				     1) == 1) {
					return true;
				}

				return false;
			}

			[[nodiscard]] constexpr inline value_type& value() noexcept {
				ASSERT(!empty());
				return *std::launder(reinterpret_cast<value_type*>(std::addressof(m_value)));

			}

			[[nodiscard]] constexpr inline const value_type& value() const noexcept {
				ASSERT(!empty());
				return *std::launder(reinterpret_cast<const value_type*>(std::addressof(m_value)));
			}

			template <typename... Args>
			constexpr void set_value_of_empty_bucket(truncated_hash_type hash,
			                               Args&&... value_type_args) noexcept {
				ASSERT(empty());

				::new (static_cast<void*>(std::addressof(m_value)))
				        value_type(std::forward<Args>(value_type_args)...);
				set_empty(false);
				this->set_hash(hash);
			}

			constexpr void swap_value_into_empty_bucket(hopscotch_bucket& empty_bucket) noexcept {
				ASSERT(empty_bucket.empty());
				if (!empty()) {
					::new (static_cast<void*>(std::addressof(empty_bucket.m_value)))
					        value_type(std::move(value()));
					empty_bucket.copy_hash(*this);
					empty_bucket.set_empty(false);

					destroy_value();
					set_empty(true);
				}
			}

			constexpr inline void remove_value() noexcept {
				if (!empty()) {
					destroy_value();
					set_empty(true);
				}
			}

			constexpr inline void clear() noexcept {
				if (!empty()) {
					destroy_value();
				}

				m_neighborhood_infos = 0;
				empty();
			}

			constexpr inline static truncated_hash_type truncate_hash(std::size_t hash) noexcept {
				return truncated_hash_type(hash);
			}

		private:
			constexpr void set_empty(bool is_empty) noexcept {
				if (is_empty) {
					m_neighborhood_infos = neighborhood_bitmap(m_neighborhood_infos & ~1);
				} else {
					m_neighborhood_infos = neighborhood_bitmap(m_neighborhood_infos | 1);
				}
			}

			constexpr void destroy_value() noexcept {
				ASSERT(!empty());
				value().~value_type();
			}

		private:
			neighborhood_bitmap m_neighborhood_infos;
			alignas(value_type) unsigned char m_value[sizeof(value_type)];
		};

		/**
		 * Internal common class used by (b)hopscotch_map and (b)hopscotch_set.
		 *
		 * ValueType is what will be stored by hopscotch_hash (usually std::pair<Key, T>
		 * for a map and Key for a set).
		 *
		 * KeySelect should be a FunctionObject which takes a ValueType in parameter and
		 * returns a reference to the key.
		 *
		 * ValueSelect should be a FunctionObject which takes a ValueType in parameter
		 * and returns a reference to the value. ValueSelect should be void if there is
		 * no value (in a set for example).
		 *
		 * OverflowContainer will be used as containers for overflown elements. Usually
		 * it should be a list<ValueType> or a set<Key>/map<Key, T>.
		 */
		template <class ValueType,
		          class KeySelect,
		          class ValueSelect,
		          class Hash,
		          class KeyEqual,
		          class Allocator,
		          unsigned int NeighborhoodSize,
		          bool StoreHash,
		          class GrowthPolicy,
		          class OverflowContainer>
		class hopscotch_hash : private Hash, private KeyEqual, private GrowthPolicy {
		private:
			template <typename U>
			using has_mapped_type = typename std::integral_constant<bool, !std::is_same<U, void>::value>;

			static_assert(noexcept(std::declval<GrowthPolicy>().bucket_for_hash(std::size_t(0))),
			        "GrowthPolicy::bucket_for_hash must be noexcept.");
			static_assert(noexcept(std::declval<GrowthPolicy>().clear()),
			              "GrowthPolicy::clear must be noexcept.");

		public:
			template <bool IsConst>
			class hopscotch_iterator;

			using key_type = typename KeySelect::key_type;
			using value_type = ValueType;
			using size_type = std::size_t;
			using difference_type = std::ptrdiff_t;
			using hasher = Hash;
			using key_equal = KeyEqual;
			using allocator_type = Allocator;
			using reference = value_type&;
			using const_reference = const value_type&;
			using pointer = value_type*;
			using const_pointer = const value_type*;
			using iterator = hopscotch_iterator<false>;
			using const_iterator = hopscotch_iterator<true>;

		private:
			using hopscotch_bucket =
			        tsl::detail_hopscotch_hash::hopscotch_bucket<ValueType, NeighborhoodSize,
			                                                     StoreHash>;
			using neighborhood_bitmap = typename hopscotch_bucket::neighborhood_bitmap;

			using buckets_allocator = typename std::allocator_traits<
			        allocator_type>::template rebind_alloc<hopscotch_bucket>;
			using buckets_container_type =
			        std::vector<hopscotch_bucket, buckets_allocator>;

			using overflow_container_type = OverflowContainer;

			static_assert(std::is_same<typename overflow_container_type::value_type,
			                           ValueType>::value,
			              "OverflowContainer should have ValueType as type.");

			static_assert(std::is_same<typename overflow_container_type::allocator_type,
			                           Allocator>::value,
			              "Invalid allocator, not the same type as the value_type.");

			using iterator_buckets = typename buckets_container_type::iterator;
			using const_iterator_buckets =
			        typename buckets_container_type::const_iterator;

			using iterator_overflow = typename overflow_container_type::iterator;
			using const_iterator_overflow =
			        typename overflow_container_type::const_iterator;

		public:
			/**
       		 * The `operator*()` and `operator->()` methods return a const reference and
       		 * const pointer respectively to the stored value type.
       		 *
       		 * In case of a map, to get a modifiable reference to the value associated to
       		 * a key (the `.second` in the stored pair), you have to call `value()`.
       		 */
			template <bool IsConst>
			class hopscotch_iterator {
				friend class hopscotch_hash;

			private:
				using iterator_bucket = typename std::conditional<
				        IsConst, typename hopscotch_hash::const_iterator_buckets,
				        typename hopscotch_hash::iterator_buckets>::type;
				using iterator_overflow = typename std::conditional<
				        IsConst, typename hopscotch_hash::const_iterator_overflow,
				        typename hopscotch_hash::iterator_overflow>::type;

				hopscotch_iterator(iterator_bucket buckets_iterator,
				                   iterator_bucket buckets_end_iterator,
				                   iterator_overflow overflow_iterator) noexcept
				    : m_buckets_iterator(buckets_iterator),
				      m_buckets_end_iterator(buckets_end_iterator),
				      m_overflow_iterator(overflow_iterator) {}

			public:
				using iterator_category = std::forward_iterator_tag;
				using value_type = const typename hopscotch_hash::value_type;
				using difference_type = std::ptrdiff_t;
				using reference = value_type&;
				using pointer = value_type*;

				constexpr hopscotch_iterator() noexcept {}

				// Copy constructor from iterator to const_iterator.
				template <bool TIsConst = IsConst,
				         typename std::enable_if<TIsConst>::type* = nullptr>
				constexpr hopscotch_iterator(const hopscotch_iterator<!TIsConst>& other) noexcept
				    : m_buckets_iterator(other.m_buckets_iterator),
				      m_buckets_end_iterator(other.m_buckets_end_iterator),
				      m_overflow_iterator(other.m_overflow_iterator) {}

				constexpr hopscotch_iterator(const hopscotch_iterator& other) = default;
				constexpr hopscotch_iterator(hopscotch_iterator&& other) = default;
				constexpr hopscotch_iterator& operator=(const hopscotch_iterator& other) = default;
				constexpr hopscotch_iterator& operator=(hopscotch_iterator&& other) = default;

				constexpr const typename hopscotch_hash::key_type& key() const noexcept {
					if (m_buckets_iterator != m_buckets_end_iterator) {
						return KeySelect()(m_buckets_iterator->value());
					}

					return KeySelect()(*m_overflow_iterator);
				}

				template <
				        class U = ValueSelect,
				        typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
				typename std::conditional<IsConst, const typename U::value_type&,
				                          typename U::value_type&>::type
				constexpr value() const noexcept {
					if (m_buckets_iterator != m_buckets_end_iterator) {
						return U()(m_buckets_iterator->value());
					}

					return U()(*m_overflow_iterator);
				}

				constexpr inline reference operator*() const noexcept {
					if (m_buckets_iterator != m_buckets_end_iterator) {
						return m_buckets_iterator->value();
					}

					return *m_overflow_iterator;
				}

				constexpr inline pointer operator->() const noexcept {
					if (m_buckets_iterator != m_buckets_end_iterator) {
						return std::addressof(m_buckets_iterator->value());
					}

					return std::addressof(*m_overflow_iterator);
				}

				constexpr inline hopscotch_iterator& operator++() noexcept {
					if (m_buckets_iterator == m_buckets_end_iterator) {
						++m_overflow_iterator;
						return *this;
					}

					do {
						++m_buckets_iterator;
					} while (m_buckets_iterator != m_buckets_end_iterator &&
					         m_buckets_iterator->empty());

					return *this;
				}

				constexpr inline hopscotch_iterator operator++(int) noexcept {
					hopscotch_iterator tmp(*this);
					++*this;

					return tmp;
				}

				friend bool operator==(const hopscotch_iterator& lhs,
				                       const hopscotch_iterator& rhs) {
					return lhs.m_buckets_iterator == rhs.m_buckets_iterator &&
					       lhs.m_overflow_iterator == rhs.m_overflow_iterator;
				}

				friend bool operator!=(const hopscotch_iterator& lhs,
				                       const hopscotch_iterator& rhs) {
					return !(lhs == rhs);
				}

			private:
				iterator_bucket m_buckets_iterator;
				iterator_bucket m_buckets_end_iterator;
				iterator_overflow m_overflow_iterator;
			};

		public:
			template <
			        class OC = OverflowContainer,
			        typename std::enable_if<!has_key_compare<OC>::value>::type* = nullptr>
			hopscotch_hash(size_type bucket_count, const Hash& hash,
			               const KeyEqual& equal, const Allocator& alloc,
			               float max_load_factor)
			    : Hash(hash),
			      KeyEqual(equal),
			      GrowthPolicy(bucket_count),
			      m_buckets_data(alloc),
			      m_overflow_elements(alloc),
			      m_buckets(static_empty_bucket_ptr()),
			      m_nb_elements(0) {
				if (bucket_count > max_bucket_count()) {
					ASSERT(false);
					// "The map exceeds its maximum size.");

				}

				if (bucket_count > 0) {
					static_assert(NeighborhoodSize - 1 > 0, "");

					// Can't directly construct with the appropriate size in the initializer
					// as m_buckets_data(bucket_count, alloc) is not supported by GCC 4.8
					m_buckets_data.resize(bucket_count + NeighborhoodSize - 1);
					m_buckets = m_buckets_data.data();
				}

				this->max_load_factor(max_load_factor);

				// Check in the constructor instead of outside a function to avoid
				// compilation issues when value_type is not complete.
				static_assert(std::is_nothrow_move_constructible<value_type>::value ||
				                      std::is_copy_constructible<value_type>::value,
				              "value_type must be either copy constructible or nothrow "
				              "move constructible.");
			}

			template <
			        class OC = OverflowContainer,
			        typename std::enable_if<has_key_compare<OC>::value>::type* = nullptr>
			constexpr hopscotch_hash(size_type bucket_count, const Hash& hash,
			               const KeyEqual& equal, const Allocator& alloc,
			               float max_load_factor, const typename OC::key_compare& comp) noexcept
			    : Hash(hash),
			      KeyEqual(equal),
			      GrowthPolicy(bucket_count),
			      m_buckets_data(alloc),
			      m_overflow_elements(comp, alloc),
			      m_buckets(static_empty_bucket_ptr()),
			      m_nb_elements(0) {
				if (bucket_count > max_bucket_count()) {
					ASSERT(false);
					// "The map exceeds its maximum size.");
				}

				if (bucket_count > 0) {
					static_assert(NeighborhoodSize - 1 > 0, "");

					// Can't directly construct with the appropriate size in the initializer
					// as m_buckets_data(bucket_count, alloc) is not supported by GCC 4.8
					m_buckets_data.resize(bucket_count + NeighborhoodSize - 1);
					m_buckets = m_buckets_data.data();
				}

				this->max_load_factor(max_load_factor);

				// Check in the constructor instead of outside of a function to avoid
				// compilation issues when value_type is not complete.
				static_assert(std::is_nothrow_move_constructible<value_type>::value ||
				                      std::is_copy_constructible<value_type>::value,
				              "value_type must be either copy constructible or nothrow "
				              "move constructible.");
			}

			constexpr hopscotch_hash(const hopscotch_hash& other) noexcept
			    : hopscotch_hash(other, other.get_allocator()) {}

			constexpr hopscotch_hash(const hopscotch_hash& other,
			                         const Allocator& alloc) noexcept
			    : Hash(other),
			      KeyEqual(other),
			      GrowthPolicy(other),
			      m_buckets_data(other.m_buckets_data, alloc),
			      m_overflow_elements(other.m_overflow_elements),
			      m_buckets(m_buckets_data.empty() ? static_empty_bucket_ptr()
			                                       : m_buckets_data.data()),
			      m_nb_elements(other.m_nb_elements),
			      m_min_load_threshold_rehash(other.m_min_load_threshold_rehash),
			      m_max_load_threshold_rehash(other.m_max_load_threshold_rehash),
			      m_max_load_factor(other.m_max_load_factor) {}

			constexpr hopscotch_hash(hopscotch_hash&& other) noexcept(
			        std::is_nothrow_move_constructible<Hash>::value&&
			        std::is_nothrow_move_constructible<KeyEqual>::value&&
			        std::is_nothrow_move_constructible<GrowthPolicy>::value&& std::
			        is_nothrow_move_constructible<buckets_container_type>::value&&
			        std::is_nothrow_move_constructible<
			                overflow_container_type>::value)
			    : Hash(std::move(static_cast<Hash&>(other))),
			      KeyEqual(std::move(static_cast<KeyEqual&>(other))),
			      GrowthPolicy(std::move(static_cast<GrowthPolicy&>(other))),
			      m_buckets_data(std::move(other.m_buckets_data)),
			      m_overflow_elements(std::move(other.m_overflow_elements)),
			      m_buckets(m_buckets_data.empty() ? static_empty_bucket_ptr()
			                                       : m_buckets_data.data()),
			      m_nb_elements(other.m_nb_elements),
			      m_min_load_threshold_rehash(other.m_min_load_threshold_rehash),
			      m_max_load_threshold_rehash(other.m_max_load_threshold_rehash),
			      m_max_load_factor(other.m_max_load_factor) {
				other.GrowthPolicy::clear();
				other.m_buckets_data.clear();
				other.m_overflow_elements.clear();
				other.m_buckets = static_empty_bucket_ptr();
				other.m_nb_elements = 0;
				other.m_min_load_threshold_rehash = 0;
				other.m_max_load_threshold_rehash = 0;
			}

			constexpr hopscotch_hash& operator=(const hopscotch_hash& other) noexcept {
				if (&other != this) {
					Hash::operator=(other);
					KeyEqual::operator=(other);
					GrowthPolicy::operator=(other);

					m_buckets_data = other.m_buckets_data;
					m_overflow_elements = other.m_overflow_elements;
					m_buckets = m_buckets_data.empty() ? static_empty_bucket_ptr()
					                                   : m_buckets_data.data();
					m_nb_elements = other.m_nb_elements;

					m_min_load_threshold_rehash = other.m_min_load_threshold_rehash;
					m_max_load_threshold_rehash = other.m_max_load_threshold_rehash;
					m_max_load_factor = other.m_max_load_factor;
				}

				return *this;
			}

			constexpr hopscotch_hash& operator=(hopscotch_hash&& other) noexcept {
				other.swap(*this);
				other.clear();

				return *this;
			}

			constexpr inline allocator_type get_allocator() const noexcept {
				return m_buckets_data.get_allocator();
			}

			/**
   			 * Iterators
   			 */
			constexpr inline iterator begin() noexcept {
				auto begin = m_buckets_data.begin();
				while (begin != m_buckets_data.end() && begin->empty()) {
					++begin;
				}

				return iterator(begin, m_buckets_data.end(), m_overflow_elements.begin());
			}

			constexpr inline const_iterator begin() const noexcept { return cbegin(); }

			constexpr inline const_iterator cbegin() const noexcept {
				auto begin = m_buckets_data.cbegin();
				while (begin != m_buckets_data.cend() && begin->empty()) {
					++begin;
				}

				return const_iterator(begin, m_buckets_data.cend(),
				                      m_overflow_elements.cbegin());
			}

			constexpr inline iterator end() noexcept {
				return iterator(m_buckets_data.end(), m_buckets_data.end(),
				                m_overflow_elements.end());
			}

			constexpr inline const_iterator end() const noexcept { return cend(); }

			constexpr inline const_iterator cend() const noexcept {
				return const_iterator(m_buckets_data.cend(), m_buckets_data.cend(),
				                      m_overflow_elements.cend());
			}

			/*
   			 * Capacity
   			 */
			[[nodiscard]] constexpr inline bool empty() const noexcept { return m_nb_elements == 0; }
			[[nodiscard]] constexpr inline size_type size() const noexcept { return m_nb_elements; }
			[[nodiscard]] constexpr inline size_type max_size() const noexcept { return m_buckets_data.max_size(); }

			/*
   			 * Modifiers
   			 */
			constexpr inline void clear() noexcept {
				for (auto& bucket : m_buckets_data) {
					bucket.clear();
				}

				m_overflow_elements.clear();
				m_nb_elements = 0;
			}

			constexpr inline std::pair<iterator, bool> insert(const value_type& value) {
				return insert_impl(value);
			}

			template <class P, typename std::enable_if<std::is_constructible<
			                          value_type, P&&>::value>::type* = nullptr>
			constexpr inline std::pair<iterator, bool> insert(P&& value) noexcept {
				return insert_impl(value_type(std::forward<P>(value)));
			}

			constexpr inline std::pair<iterator, bool> insert(value_type&& value) noexcept {
				return insert_impl(std::move(value));
			}

			constexpr inline iterator insert(const_iterator hint,
			                                 const value_type& value) noexcept {
				if (hint != cend() &&
				    compare_keys(KeySelect()(*hint), KeySelect()(value))) {
					return mutable_iterator(hint);
				}

				return insert(value).first;
			}

			template <class P, typename std::enable_if<std::is_constructible<
			                          value_type, P&&>::value>::type* = nullptr>
			constexpr inline iterator insert(const_iterator hint, P&& value) noexcept {
				return emplace_hint(hint, std::forward<P>(value));
			}

			constexpr inline iterator insert(const_iterator hint,
			                                 const value_type&& value) noexcept {
				if (hint != cend() &&
				    compare_keys(KeySelect()(*hint), KeySelect()(value))) {
					return mutable_iterator(hint);
				}

				return insert(std::move(value)).first;
			}

			template <class InputIt>
			constexpr inline void insert(InputIt first, InputIt last) noexcept {
				if constexpr (std::is_base_of<std::forward_iterator_tag,
				            typename std::iterator_traits<InputIt>::iterator_category>::value) {
					const auto nb_elements_insert = std::distance(first, last);
					const std::size_t nb_elements_in_buckets =
					        m_nb_elements - m_overflow_elements.size();
					const std::size_t nb_free_buckets =
					        m_max_load_threshold_rehash - nb_elements_in_buckets;
					ASSERT(m_nb_elements >= m_overflow_elements.size());
					ASSERT(m_max_load_threshold_rehash >= nb_elements_in_buckets);

					if (nb_elements_insert > 0 &&
					    nb_free_buckets < std::size_t(nb_elements_insert)) {
						reserve(nb_elements_in_buckets + std::size_t(nb_elements_insert));
					}
				}

				for (; first != last; ++first) {
					insert(*first);
				}
			}

			template <class M>
			constexpr inline std::pair<iterator, bool> insert_or_assign(const key_type& k,
			                                                            M&& obj) noexcept {
				return insert_or_assign_impl(k, std::forward<M>(obj));
			}

			template <class M>
			constexpr inline std::pair<iterator, bool> insert_or_assign(key_type&& k, M&& obj) noexcept {
				return insert_or_assign_impl(std::move(k), std::forward<M>(obj));
			}

			template <class M>
			constexpr inline iterator insert_or_assign(const_iterator hint, const key_type& k, M&& obj) noexcept {
				if (hint != cend() && compare_keys(KeySelect()(*hint), k)) {
					auto it = mutable_iterator(hint);
					it.value() = std::forward<M>(obj);

					return it;
				}

				return insert_or_assign(k, std::forward<M>(obj)).first;
			}

			template <class M>
			constexpr inline iterator insert_or_assign(const_iterator hint, key_type&& k, M&& obj) noexcept {
				if (hint != cend() && compare_keys(KeySelect()(*hint), k)) {
					auto it = mutable_iterator(hint);
					it.value() = std::forward<M>(obj);

					return it;
				}

				return insert_or_assign(std::move(k), std::forward<M>(obj)).first;
			}

			template <class... Args>
			constexpr inline std::pair<iterator, bool> emplace(Args&&... args) noexcept {
				return insert(value_type(std::forward<Args>(args)...));
			}

			template <class... Args>
			constexpr inline iterator emplace_hint(const_iterator hint, Args&&... args) noexcept {
				return insert(hint, value_type(std::forward<Args>(args)...));
			}

			template <class... Args>
			constexpr inline std::pair<iterator, bool> try_emplace(const key_type& k, Args&&... args) noexcept {
				return try_emplace_impl(k, std::forward<Args>(args)...);
			}

			template <class... Args>
			constexpr inline std::pair<iterator, bool> try_emplace(key_type&& k, Args&&... args) noexcept {
				return try_emplace_impl(std::move(k), std::forward<Args>(args)...);
			}

			template <class... Args>
			constexpr inline iterator try_emplace(const_iterator hint, const key_type& k, Args&&... args) noexcept {
				if (hint != cend() && compare_keys(KeySelect()(*hint), k)) {
					return mutable_iterator(hint);
				}

				return try_emplace(k, std::forward<Args>(args)...).first;
			}

			template <class... Args>
			constexpr inline iterator try_emplace(const_iterator hint, key_type&& k, Args&&... args) noexcept {
				if (hint != cend() && compare_keys(KeySelect()(*hint), k)) {
					return mutable_iterator(hint);
				}

				return try_emplace(std::move(k), std::forward<Args>(args)...).first;
			}

			/**
             * Here to avoid `template<class K> size_type erase(const K& key)` being used
             * when we use an iterator instead of a const_iterator.
             */
			constexpr inline iterator erase(iterator pos) noexcept { return erase(const_iterator(pos)); }

			constexpr iterator erase(const_iterator pos) noexcept {
				const std::size_t ibucket_for_hash = bucket_for_hash(hash_key(pos.key()));

				if (pos.m_buckets_iterator != pos.m_buckets_end_iterator) {
					auto it_bucket =
					        m_buckets_data.begin() +
					        std::distance(m_buckets_data.cbegin(), pos.m_buckets_iterator);
					erase_from_bucket(*it_bucket, ibucket_for_hash);

					return ++iterator(it_bucket, m_buckets_data.end(),
					                  m_overflow_elements.begin());
				} else {
					auto it_next_overflow =
					        erase_from_overflow(pos.m_overflow_iterator, ibucket_for_hash);
					return iterator(m_buckets_data.end(), m_buckets_data.end(),
					                it_next_overflow);
				}
			}

			constexpr iterator erase(const_iterator first, const_iterator last) noexcept {
				if (first == last) {
					return mutable_iterator(first);
				}

				auto to_delete = erase(first);
				while (to_delete != last) {
					to_delete = erase(to_delete);
				}

				return to_delete;
			}

			template <class K>
			constexpr inline size_type erase(const K& key) noexcept {
				return erase(key, hash_key(key));
			}

			template <class K>
			constexpr size_type erase(const K& key,
			                                 const std::size_t hash) noexcept{
				const std::size_t ibucket_for_hash = bucket_for_hash(hash);

				hopscotch_bucket* bucket_found =
				        find_in_buckets(key, hash, m_buckets + ibucket_for_hash);
				if (bucket_found != nullptr) {
					erase_from_bucket(*bucket_found, ibucket_for_hash);

					return 1;
				}

				if (m_buckets[ibucket_for_hash].has_overflow()) {
					auto it_overflow = find_in_overflow(key);
					if (it_overflow != m_overflow_elements.end()) {
						erase_from_overflow(it_overflow, ibucket_for_hash);

						return 1;
					}
				}

				return 0;
			}

			constexpr void swap(hopscotch_hash& other) noexcept {
				using std::swap;

				swap(static_cast<Hash&>(*this), static_cast<Hash&>(other));
				swap(static_cast<KeyEqual&>(*this), static_cast<KeyEqual&>(other));
				swap(static_cast<GrowthPolicy&>(*this), static_cast<GrowthPolicy&>(other));
				swap(m_buckets_data, other.m_buckets_data);
				swap(m_overflow_elements, other.m_overflow_elements);
				swap(m_buckets, other.m_buckets);
				swap(m_nb_elements, other.m_nb_elements);
				swap(m_min_load_threshold_rehash, other.m_min_load_threshold_rehash);
				swap(m_max_load_threshold_rehash, other.m_max_load_threshold_rehash);
				swap(m_max_load_factor, other.m_max_load_factor);
			}

			/*
   			 * Lookup
   			 */
			template <class K, class U = ValueSelect,
			         typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
			constexpr inline typename U::value_type& at(const K& key) noexcept {
				return at(key, hash_key(key));
			}

			template <class K, class U = ValueSelect,
			         typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
			constexpr inline typename U::value_type& at(const K& key,
			                                            const std::size_t hash) noexcept{
				return const_cast<typename U::value_type&>(
				        static_cast<const hopscotch_hash*>(this)->at(key, hash));
			}

			template <class K, class U = ValueSelect,
			         typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
			constexpr inline const typename U::value_type& at(const K& key) const noexcept {
				return at(key, hash_key(key));
			}

			template <class K, class U = ValueSelect,
			         typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
			constexpr inline const typename U::value_type& at(const K& key, std::size_t hash) const noexcept {
				using T = typename U::value_type;

				const T* value =
				        find_value_impl(key, hash, m_buckets + bucket_for_hash(hash));
				if (value == nullptr) {
					// Couldnt fin key;
					ASSERT(false);
				} else {
					return *value;
				}
			}

			template <class K, class U = ValueSelect,
			         typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
			constexpr typename U::value_type& operator[](K&& key) noexcept {
				using T = typename U::value_type;

				const std::size_t hash = hash_key(key);
				const std::size_t ibucket_for_hash = bucket_for_hash(hash);

				T* value = find_value_impl(key, hash, m_buckets + ibucket_for_hash);
				if (value != nullptr) {
					return *value;
				} else {
					return insert_value(ibucket_for_hash, hash, std::piecewise_construct,
					                    std::forward_as_tuple(std::forward<K>(key)),
					                    std::forward_as_tuple())
					        .first.value();
				}
			}

			template <class K>
			constexpr inline size_type count(const K& key) const noexcept {
				return count(key, hash_key(key));
			}

			template <class K>
			constexpr inline size_type count(const K& key, std::size_t hash) const noexcept {
				return count_impl(key, hash, m_buckets + bucket_for_hash(hash));
			}

			template <class K>
			constexpr inline iterator find(const K& key) noexcept {
				return find(key, hash_key(key));
			}

			template <class K>
			constexpr inline iterator find(const K& key,
			                               const std::size_t hash) noexcept {
				return find_impl(key, hash, m_buckets + bucket_for_hash(hash));
			}

			template <class K>
			constexpr inline const_iterator find(const K& key) const noexcept {
				return find(key, hash_key(key));
			}

			template <class K>
			constexpr inline const_iterator find(const K& key, std::size_t hash) const noexcept {
				return find_impl(key, hash, m_buckets + bucket_for_hash(hash));
			}

			template <class K>
			constexpr inline bool contains(const K& key) const noexcept {
				return contains(key, hash_key(key));
			}

			template <class K>
			constexpr inline bool contains(const K& key,
			                               const std::size_t hash) const noexcept {
				return count(key, hash) != 0;
			}

			template <class K>
			constexpr inline std::pair<iterator, iterator> equal_range(const K& key) noexcept {
				return equal_range(key, hash_key(key));
			}

			template <class K>
			constexpr inline std::pair<iterator, iterator> equal_range(const K& key,
			                                                           const std::size_t hash) noexcept {
				iterator it = find(key, hash);
				return std::make_pair(it, (it == end()) ? it : std::next(it));
			}

			template <class K>
			constexpr inline std::pair<const_iterator, const_iterator> equal_range(const K& key) const noexcept {
				return equal_range(key, hash_key(key));
			}

			template <class K>
			constexpr inline std::pair<const_iterator, const_iterator> equal_range(
			        const K& key, std::size_t hash) const noexcept {
				const_iterator it = find(key, hash);
				return std::make_pair(it, (it == cend()) ? it : std::next(it));
			}

			/*
   			 * Bucket interface
   			 */
			[[nodiscard]] constexpr inline size_type bucket_count() const noexcept {
				/*
     			 * So that the last bucket can have NeighborhoodSize neighbors, the size of
     			 * the bucket array is a little bigger than the real number of buckets when
     			 * not empty. We could use some of the buckets at the beginning, but it is
     			 * faster this way as we avoid extra checks.
     			 */
				if (m_buckets_data.empty()) {
					return 0;
				}

				return m_buckets_data.size() - NeighborhoodSize + 1;
			}

			[[nodiscard]] constexpr inline size_type max_bucket_count() const noexcept {
				const std::size_t max_bucket_count =
				        std::min(GrowthPolicy::max_bucket_count(), m_buckets_data.max_size());
				return max_bucket_count - NeighborhoodSize + 1;
			}

			/*
   			 *  Hash policy
   			 */
			[[nodiscard]] constexpr inline float load_factor() const noexcept {
				if (bucket_count() == 0) {
					return 0;
				}

				return float(m_nb_elements) / float(bucket_count());
			}

			[[nodiscard]] constexpr inline float max_load_factor() const { return m_max_load_factor; }

			constexpr inline void max_load_factor(const float ml) noexcept {
				m_max_load_factor = std::max(0.1f, std::min(ml, 0.95f));
				m_min_load_threshold_rehash =
				        size_type(float(bucket_count()) * MIN_LOAD_FACTOR_FOR_REHASH);
				m_max_load_threshold_rehash =
				        size_type(float(bucket_count()) * m_max_load_factor);
			}

			constexpr inline void rehash(size_type count_) noexcept {
				count_ = std::max(count_, size_type(std::ceil(float(size()) / max_load_factor())));
				rehash_impl(count_);
			}

			constexpr inline void reserve(size_type count_) noexcept {
				rehash(size_type(std::ceil(float(count_) / max_load_factor())));
			}

			/*
   			 * Observers
   			 */
			constexpr inline hasher hash_function() const noexcept { return static_cast<const Hash&>(*this); }
			constexpr inline key_equal key_eq() const noexcept { return static_cast<const KeyEqual&>(*this); }

			/*
   			 * Other
   			 */
			constexpr inline iterator mutable_iterator(const_iterator pos) noexcept {
				if (pos.m_buckets_iterator != pos.m_buckets_end_iterator) {
					// Get a non-const iterator
					auto it = m_buckets_data.begin() +
					          std::distance(m_buckets_data.cbegin(), pos.m_buckets_iterator);
					return iterator(it, m_buckets_data.end(), m_overflow_elements.begin());
				} else {
					// Get a non-const iterator
					auto it = mutable_overflow_iterator(pos.m_overflow_iterator);
					return iterator(m_buckets_data.end(), m_buckets_data.end(), it);
				}
			}

			size_type overflow_size() const noexcept {
				return m_overflow_elements.size();
			}

			template <class U = OverflowContainer,
			         typename std::enable_if<has_key_compare<U>::value>::type* = nullptr>
			typename U::key_compare key_comp() const {
				return m_overflow_elements.key_comp();
			}

		private:
			template <class K>
			constexpr inline std::size_t hash_key(const K& key) const noexcept {
				return Hash::operator()(key);
			}

			template <class K1, class K2>
			constexpr inline bool compare_keys(const K1& key1, const K2& key2) const noexcept {
				return KeyEqual::operator()(key1, key2);
			}

			[[nodiscard]] constexpr inline std::size_t bucket_for_hash(std::size_t hash) const noexcept {
				const std::size_t bucket = GrowthPolicy::bucket_for_hash(hash);
				ASSERT(bucket < m_buckets_data.size() ||
				              (bucket == 0 && m_buckets_data.empty()));

				return bucket;
			}

			template <typename U = value_type,
			         typename std::enable_if<
			                 std::is_nothrow_move_constructible<U>::value>::type* = nullptr>
			constexpr void rehash_impl(size_type count_) noexcept {
				hopscotch_hash new_map = new_hopscotch_hash(count_);

				if (!m_overflow_elements.empty()) {
					new_map.m_overflow_elements.swap(m_overflow_elements);
					new_map.m_nb_elements += new_map.m_overflow_elements.size();

					for (const value_type& value : new_map.m_overflow_elements) {
						const std::size_t ibucket_for_hash =
						        new_map.bucket_for_hash(new_map.hash_key(KeySelect()(value)));
						new_map.m_buckets[ibucket_for_hash].set_overflow(true);
					}
				}

//#ifndef TSL_HH_NO_EXCEPTIONS
//				try {
//#endif
					const bool use_stored_hash =
					        USE_STORED_HASH_ON_REHASH(new_map.bucket_count());
					for (auto it_bucket = m_buckets_data.begin();
					     it_bucket != m_buckets_data.end(); ++it_bucket) {
						if (it_bucket->empty()) {
							continue;
						}

						const std::size_t hash =
						        use_stored_hash ? it_bucket->truncated_bucket_hash()
						                        : new_map.hash_key(KeySelect()(it_bucket->value()));
						const std::size_t ibucket_for_hash = new_map.bucket_for_hash(hash);

						new_map.insert_value(ibucket_for_hash, hash,
						                     std::move(it_bucket->value()));

						erase_from_bucket(*it_bucket, bucket_for_hash(hash));
					}
//#ifndef TSL_HH_NO_EXCEPTIONS
//				}
//				/*
//     			 * The call to insert_value may throw an exception if an element is added to
//     			 * the overflow list and the memory allocation fails. Rollback the elements
//     			 * in this case.
//     			 */
//				catch (...) {
//					m_overflow_elements.swap(new_map.m_overflow_elements);
//
//					const bool use_stored_hash =
//					        USE_STORED_HASH_ON_REHASH(new_map.bucket_count());
//					for (auto it_bucket = new_map.m_buckets_data.begin();
//					     it_bucket != new_map.m_buckets_data.end(); ++it_bucket) {
//						if (it_bucket->empty()) {
//							continue;
//						}
//
//						const std::size_t hash =
//						        use_stored_hash ? it_bucket->truncated_bucket_hash()
//						                        : hash_key(KeySelect()(it_bucket->value()));
//						const std::size_t ibucket_for_hash = bucket_for_hash(hash);
//
//						// The elements we insert were not in the overflow list before the
//						// switch. They will not be go in the overflow list if we rollback the
//						// switch.
//						insert_value(ibucket_for_hash, hash, std::move(it_bucket->value()));
//					}
//
//					throw;
//				}
//#endif

				new_map.swap(*this);
			}

			template <typename U = value_type,
			         typename std::enable_if<
			                 std::is_copy_constructible<U>::value &&
			                 !std::is_nothrow_move_constructible<U>::value>::type* = nullptr>
			constexpr void rehash_impl(size_type count_) noexcept {
				hopscotch_hash new_map = new_hopscotch_hash(count_);

				const bool use_stored_hash =
				        USE_STORED_HASH_ON_REHASH(new_map.bucket_count());
				for (const hopscotch_bucket& bucket : m_buckets_data) {
					if (bucket.empty()) {
						continue;
					}

					const std::size_t hash =
					        use_stored_hash ? bucket.truncated_bucket_hash()
					                        : new_map.hash_key(KeySelect()(bucket.value()));
					const std::size_t ibucket_for_hash = new_map.bucket_for_hash(hash);

					new_map.insert_value(ibucket_for_hash, hash, bucket.value());
				}

				for (const value_type& value : m_overflow_elements) {
					const std::size_t hash = new_map.hash_key(KeySelect()(value));
					const std::size_t ibucket_for_hash = new_map.bucket_for_hash(hash);

					new_map.insert_value(ibucket_for_hash, hash, value);
				}

				new_map.swap(*this);
			}

#ifdef TSL_HH_NO_RANGE_ERASE_WITH_CONST_ITERATOR
			constexpr iterator_overflow mutable_overflow_iterator(const_iterator_overflow it) noexcept {
				return std::next(m_overflow_elements.begin(),
				                 std::distance(m_overflow_elements.cbegin(), it));
			}
#else
			iterator_overflow mutable_overflow_iterator(const_iterator_overflow it) {
				return m_overflow_elements.erase(it, it);
			}
#endif

			// iterator is in overflow list
			iterator_overflow erase_from_overflow(const_iterator_overflow pos,
			                                      std::size_t ibucket_for_hash) {
#ifdef TSL_HH_NO_RANGE_ERASE_WITH_CONST_ITERATOR
				auto it_next = m_overflow_elements.erase(mutable_overflow_iterator(pos));
#else
				auto it_next = m_overflow_elements.erase(pos);
#endif
				m_nb_elements--;

				// Check if we can remove the overflow flag
				ASSERT(m_buckets[ibucket_for_hash].has_overflow());
				for (const value_type& value : m_overflow_elements) {
					const std::size_t bucket_for_value =
					        bucket_for_hash(hash_key(KeySelect()(value)));
					if (bucket_for_value == ibucket_for_hash) {
						return it_next;
					}
				}

				m_buckets[ibucket_for_hash].set_overflow(false);
				return it_next;
			}

			/**
   			 * bucket_for_value is the bucket in which the value is.
   			 * ibucket_for_hash is the bucket where the value belongs.
   			 */
			constexpr void erase_from_bucket(hopscotch_bucket& bucket_for_value,
			                       std::size_t ibucket_for_hash) noexcept {
				const std::size_t ibucket_for_value =
				        std::distance(m_buckets_data.data(), &bucket_for_value);
				ASSERT(ibucket_for_value >= ibucket_for_hash);

				bucket_for_value.remove_value();
				m_buckets[ibucket_for_hash].toggle_neighbor_presence(ibucket_for_value -
				                                                     ibucket_for_hash);
				m_nb_elements--;
			}

			template <class K, class M>
			constexpr inline std::pair<iterator, bool> insert_or_assign_impl(K&& key, M&& obj) noexcept {
				auto it = try_emplace_impl(std::forward<K>(key), std::forward<M>(obj));
				if (!it.second) {
					it.first.value() = std::forward<M>(obj);
				}

				return it;
			}

			template <typename P, class... Args>
			constexpr std::pair<iterator, bool> try_emplace_impl(P&& key, Args&&... args_value) noexcept {
				const std::size_t hash = hash_key(key);
				const std::size_t ibucket_for_hash = bucket_for_hash(hash);

				// Check if already presents
				auto it_find = find_impl(key, hash, m_buckets + ibucket_for_hash);
				if (it_find != end()) {
					return std::make_pair(it_find, false);
				}

				return insert_value(
				        ibucket_for_hash, hash, std::piecewise_construct,
				        std::forward_as_tuple(std::forward<P>(key)),
				        std::forward_as_tuple(std::forward<Args>(args_value)...));
			}

			template <typename P>
			constexpr inline std::pair<iterator, bool> insert_impl(P&& value) noexcept {
				const std::size_t hash = hash_key(KeySelect()(value));
				const std::size_t ibucket_for_hash = bucket_for_hash(hash);

				// Check if already presents
				auto it_find = find_impl(KeySelect()(value), hash, m_buckets + ibucket_for_hash);
				if (it_find != end()) {
					return std::make_pair(it_find, false);
				}

				return insert_value(ibucket_for_hash, hash, std::forward<P>(value));
			}

			template <typename... Args>
			constexpr inline std::pair<iterator, bool> insert_value(std::size_t ibucket_for_hash,
			                                       std::size_t hash,
			                                       Args&&... value_type_args) {
				if ((m_nb_elements - m_overflow_elements.size()) >=
				    m_max_load_threshold_rehash) {
					rehash(GrowthPolicy::next_bucket_count());
					ibucket_for_hash = bucket_for_hash(hash);
				}

				std::size_t ibucket_empty = find_empty_bucket(ibucket_for_hash);
				if (ibucket_empty < m_buckets_data.size()) {
					do {
						ASSERT(ibucket_empty >= ibucket_for_hash);

						// Empty bucket is in range of NeighborhoodSize, use it
						if (ibucket_empty - ibucket_for_hash < NeighborhoodSize) {
							auto it = insert_in_bucket(ibucket_empty, ibucket_for_hash, hash,
							                           std::forward<Args>(value_type_args)...);
							return std::make_pair(
							        iterator(it, m_buckets_data.end(), m_overflow_elements.begin()),
							        true);
						}
					}
					// else, try to swap values to get a closer empty bucket
					while (swap_empty_bucket_closer(ibucket_empty));
				}

				// Load factor is too low or a rehash will not change the neighborhood, put
				// the value in overflow list
				if (size() < m_min_load_threshold_rehash ||
				    !will_neighborhood_change_on_rehash(ibucket_for_hash)) {
					auto it = insert_in_overflow(ibucket_for_hash,
					                             std::forward<Args>(value_type_args)...);
					return std::make_pair(
					        iterator(m_buckets_data.end(), m_buckets_data.end(), it), true);
				}

				rehash(GrowthPolicy::next_bucket_count());
				ibucket_for_hash = bucket_for_hash(hash);

				return insert_value(ibucket_for_hash, hash,
				                    std::forward<Args>(value_type_args)...);
			}

			/**
             * Return true if a rehash will change the position of a key-value in the
             * neighborhood of ibucket_neighborhood_check. In this case a rehash is needed
             * instead of puting the value in overflow list.
             */
			[[nodiscard]] constexpr inline bool will_neighborhood_change_on_rehash(
			        size_t ibucket_neighborhood_check) const noexcept {
				std::size_t expand_bucket_count = GrowthPolicy::next_bucket_count();
				GrowthPolicy expand_growth_policy(expand_bucket_count);

				const bool use_stored_hash = USE_STORED_HASH_ON_REHASH(expand_bucket_count);
				for (size_t ibucket = ibucket_neighborhood_check;
				     ibucket < m_buckets_data.size() &&
				     (ibucket - ibucket_neighborhood_check) < NeighborhoodSize;
				     ++ibucket) {
					ASSERT(!m_buckets[ibucket].empty());

					const size_t hash =
					        use_stored_hash ? m_buckets[ibucket].truncated_bucket_hash()
					                        : hash_key(KeySelect()(m_buckets[ibucket].value()));
					if (bucket_for_hash(hash) != expand_growth_policy.bucket_for_hash(hash)) {
						return true;
					}
				}

				return false;
			}

			/*
   			 * Return the index of an empty bucket in m_buckets_data.
   			 * If none, the returned index equals m_buckets_data.size()
   			 */
			[[nodiscard]] constexpr inline std::size_t find_empty_bucket(std::size_t ibucket_start) const {
				const std::size_t limit = std::min(
				        ibucket_start + MAX_PROBES_FOR_EMPTY_BUCKET, m_buckets_data.size());
				for (; ibucket_start < limit; ibucket_start++) {
					if (m_buckets[ibucket_start].empty()) {
						return ibucket_start;
					}
				}

				return m_buckets_data.size();
			}

			/*
             * Insert value in ibucket_empty where value originally belongs to
             * ibucket_for_hash
             *
             * Return bucket iterator to ibucket_empty
             */
			template <typename... Args>
			constexpr inline iterator_buckets insert_in_bucket(std::size_t ibucket_empty,
			                                  std::size_t ibucket_for_hash,
			                                  std::size_t hash,
			                                  Args&&... value_type_args) noexcept {
				ASSERT(ibucket_empty >= ibucket_for_hash);
				ASSERT(m_buckets[ibucket_empty].empty());
				m_buckets[ibucket_empty].set_value_of_empty_bucket(
				        hopscotch_bucket::truncate_hash(hash),
				        std::forward<Args>(value_type_args)...);

				ASSERT(!m_buckets[ibucket_for_hash].empty());
				m_buckets[ibucket_for_hash].toggle_neighbor_presence(ibucket_empty -
				                                                     ibucket_for_hash);
				m_nb_elements++;

				return m_buckets_data.begin() + ibucket_empty;
			}

			template <
			        class... Args, class U = OverflowContainer,
			        typename std::enable_if<!has_key_compare<U>::value>::type* = nullptr>
			constexpr iterator_overflow insert_in_overflow(std::size_t ibucket_for_hash,
			                                     Args&&... value_type_args) noexcept {
				auto it = m_overflow_elements.emplace(
				        m_overflow_elements.end(), std::forward<Args>(value_type_args)...);

				m_buckets[ibucket_for_hash].set_overflow(true);
				m_nb_elements++;

				return it;
			}

			template <class... Args, class U = OverflowContainer,
			         typename std::enable_if<has_key_compare<U>::value>::type* = nullptr>
			constexpr iterator_overflow insert_in_overflow(std::size_t ibucket_for_hash,
			                                     Args&&... value_type_args) noexcept {
				auto it =
				        m_overflow_elements.emplace(std::forward<Args>(value_type_args)...)
				                .first;

				m_buckets[ibucket_for_hash].set_overflow(true);
				m_nb_elements++;

				return it;
			}

			/*
			 * Try to swap the bucket ibucket_empty_in_out with a bucket preceding it
			 * while keeping the neighborhood conditions correct.
			 *
			 * If a swap was possible, the position of ibucket_empty_in_out will be closer
			 * to 0 and true will re returned.
			 */
			constexpr bool swap_empty_bucket_closer(std::size_t& ibucket_empty_in_out) noexcept {
				ASSERT(ibucket_empty_in_out >= NeighborhoodSize);
				const std::size_t neighborhood_start =
				        ibucket_empty_in_out - NeighborhoodSize + 1;

				for (std::size_t to_check = neighborhood_start;
				     to_check < ibucket_empty_in_out; to_check++) {
					neighborhood_bitmap neighborhood_infos =
					        m_buckets[to_check].neighborhood_infos();
					std::size_t to_swap = to_check;

					while (neighborhood_infos != 0 && to_swap < ibucket_empty_in_out) {
						if ((neighborhood_infos & 1) == 1) {
							ASSERT(m_buckets[ibucket_empty_in_out].empty());
							ASSERT(!m_buckets[to_swap].empty());

							m_buckets[to_swap].swap_value_into_empty_bucket(
							        m_buckets[ibucket_empty_in_out]);

							ASSERT(!m_buckets[to_check].check_neighbor_presence(
							        ibucket_empty_in_out - to_check));
							ASSERT(
							        m_buckets[to_check].check_neighbor_presence(to_swap - to_check));

							m_buckets[to_check].toggle_neighbor_presence(ibucket_empty_in_out -
							                                             to_check);
							m_buckets[to_check].toggle_neighbor_presence(to_swap - to_check);

							ibucket_empty_in_out = to_swap;

							return true;
						}

						to_swap++;
						neighborhood_infos = neighborhood_bitmap(neighborhood_infos >> 1);
					}
				}

				return false;
			}

			template <class K, class U = ValueSelect,
			         typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
			constexpr typename U::value_type* find_value_impl(const K& key, std::size_t hash,
			                                        hopscotch_bucket* bucket_for_hash) noexcept {
				return const_cast<typename U::value_type*>(
				        static_cast<const hopscotch_hash*>(this)->find_value_impl(
				                key, hash, bucket_for_hash));
			}

			/*
   			 * Avoid the creation of an iterator to just get the value for operator[] and
   			 * at() in maps. Faster this way.
   			 *
   			 * Return null if no value for the key (TODO use std::optional when
   			 * available).
   			 */
			template <class K, class U = ValueSelect,
			         typename std::enable_if<has_mapped_type<U>::value>::type* = nullptr>
			constexpr const typename U::value_type* find_value_impl(
			        const K& key, std::size_t hash,
			        const hopscotch_bucket* bucket_for_hash) const noexcept {
				const hopscotch_bucket* bucket_found =
				        find_in_buckets(key, hash, bucket_for_hash);
				if (bucket_found != nullptr) {
					return std::addressof(ValueSelect()(bucket_found->value()));
				}

				if (bucket_for_hash->has_overflow()) {
					auto it_overflow = find_in_overflow(key);
					if (it_overflow != m_overflow_elements.end()) {
						return std::addressof(ValueSelect()(*it_overflow));
					}
				}

				return nullptr;
			}

			template <class K>
			constexpr size_type count_impl(const K& key, std::size_t hash,
			                     const hopscotch_bucket* bucket_for_hash) const noexcept {
				if (find_in_buckets(key, hash, bucket_for_hash) != nullptr) {
					return 1;
				} else if (bucket_for_hash->has_overflow() &&
				           find_in_overflow(key) != m_overflow_elements.cend()) {
					return 1;
				} else {
					return 0;
				}
			}

			template <class K>
			constexpr iterator find_impl(const K& key, std::size_t hash,
			                   hopscotch_bucket* bucket_for_hash) noexcept {
				hopscotch_bucket* bucket_found =
				        find_in_buckets(key, hash, bucket_for_hash);
				if (bucket_found != nullptr) {
					return iterator(m_buckets_data.begin() +
					                        std::distance(m_buckets_data.data(), bucket_found),
					                m_buckets_data.end(), m_overflow_elements.begin());
				}

				if (!bucket_for_hash->has_overflow()) {
					return end();
				}

				return iterator(m_buckets_data.end(), m_buckets_data.end(),
				                find_in_overflow(key));
			}

			template <class K>
			constexpr const_iterator find_impl(const K& key,
			                                   const std::size_t hash,
			                         const hopscotch_bucket* bucket_for_hash) const noexcept {
				const hopscotch_bucket* bucket_found =
				        find_in_buckets(key, hash, bucket_for_hash);
				if (bucket_found != nullptr) {
					return const_iterator(
					        m_buckets_data.cbegin() +
					                std::distance(m_buckets_data.data(), bucket_found),
					        m_buckets_data.cend(), m_overflow_elements.cbegin());
				}

				if (!bucket_for_hash->has_overflow()) {
					return cend();
				}

				return const_iterator(m_buckets_data.cend(), m_buckets_data.cend(),
				                      find_in_overflow(key));
			}

			template <class K>
			constexpr hopscotch_bucket* find_in_buckets(const K& key,
			                                            const std::size_t hash,
			                                  hopscotch_bucket* bucket_for_hash) noexcept {
				const hopscotch_bucket* bucket_found =
				        static_cast<const hopscotch_hash*>(this)->find_in_buckets(
				                key, hash, bucket_for_hash);
				return const_cast<hopscotch_bucket*>(bucket_found);
			}

			/**
   * Return a pointer to the bucket which has the value, nullptr otherwise.
   */
			template <class K>
			constexpr inline const hopscotch_bucket* find_in_buckets(
			        const K& key,
			        const std::size_t hash,
			        const hopscotch_bucket* bucket_for_hash) const noexcept {
				(void)hash;  // Avoid warning of unused variable when StoreHash is false;

				// TODO Try to optimize the function.
				// I tried to use ffs and  __builtin_ffs functions but I could not reduce
				// the time the function takes with -march=native

				neighborhood_bitmap neighborhood_infos =
				        bucket_for_hash->neighborhood_infos();
				while (neighborhood_infos != 0) {
					if ((neighborhood_infos & 1) == 1) {
						// Check StoreHash before calling bucket_hash_equal. Functionally it
						// doesn't change anything. If StoreHash is false, bucket_hash_equal is a
						// no-op. Avoiding the call is there to help GCC optimizes `hash`
						// parameter away, it seems to not be able to do without this hint.
						if ((!StoreHash || bucket_for_hash->bucket_hash_equal(hash)) &&
						    compare_keys(KeySelect()(bucket_for_hash->value()), key)) {
							return bucket_for_hash;
						}
					}

					++bucket_for_hash;
					neighborhood_infos = neighborhood_bitmap(neighborhood_infos >> 1);
				}

				return nullptr;
			}

			template <
			        class K, class U = OverflowContainer,
			        typename std::enable_if<!has_key_compare<U>::value>::type* = nullptr>
			iterator_overflow find_in_overflow(const K& key) {
				return std::find_if(m_overflow_elements.begin(), m_overflow_elements.end(),
				                    [&](const value_type& value) {
					                    return compare_keys(key, KeySelect()(value));
				                    });
			}

			template <
			        class K, class U = OverflowContainer,
			        typename std::enable_if<!has_key_compare<U>::value>::type* = nullptr>
			const_iterator_overflow find_in_overflow(const K& key) const {
				return std::find_if(m_overflow_elements.cbegin(),
				                    m_overflow_elements.cend(),
				                    [&](const value_type& value) {
					                    return compare_keys(key, KeySelect()(value));
				                    });
			}

			template <class K, class U = OverflowContainer,
			         typename std::enable_if<has_key_compare<U>::value>::type* = nullptr>
			iterator_overflow find_in_overflow(const K& key) {
				return m_overflow_elements.find(key);
			}

			template <class K, class U = OverflowContainer,
			         typename std::enable_if<has_key_compare<U>::value>::type* = nullptr>
			const_iterator_overflow find_in_overflow(const K& key) const {
				return m_overflow_elements.find(key);
			}

			template <
			        class U = OverflowContainer,
			        typename std::enable_if<!has_key_compare<U>::value>::type* = nullptr>
			constexpr inline hopscotch_hash new_hopscotch_hash(size_type bucket_count) {
				return hopscotch_hash(bucket_count, static_cast<Hash&>(*this),
				                      static_cast<KeyEqual&>(*this), get_allocator(),
				                      m_max_load_factor);
			}

			template <class U = OverflowContainer,
			         typename std::enable_if<has_key_compare<U>::value>::type* = nullptr>
			constexpr inline hopscotch_hash new_hopscotch_hash(size_type bucket_count) {
				return hopscotch_hash(bucket_count, static_cast<Hash&>(*this),
				                      static_cast<KeyEqual&>(*this), get_allocator(),
				                      m_max_load_factor, m_overflow_elements.key_comp());
			}

		public:
			static constexpr size_type DEFAULT_INIT_BUCKETS_SIZE = 0;
			static constexpr float DEFAULT_MAX_LOAD_FACTOR =
			        (NeighborhoodSize <= 30) ? 0.8f : 0.9f;

		private:
			static const std::size_t MAX_PROBES_FOR_EMPTY_BUCKET = 12 * NeighborhoodSize;
			static constexpr float MIN_LOAD_FACTOR_FOR_REHASH = 0.1f;

			/**
   * We can only use the hash on rehash if the size of the hash type is the same
   * as the stored one or if we use a power of two modulo. In the case of the
   * power of two modulo, we just mask the least significant bytes, we just have
   * to check that the truncated_hash_type didn't truncated too much bytes.
   */
			template <class T = size_type,
			         typename std::enable_if<
			                 std::is_same<T, truncated_hash_type>::value>::type* = nullptr>
			constexpr inline static bool USE_STORED_HASH_ON_REHASH(size_type /*bucket_count*/) noexcept {
				return StoreHash;
			}

			template <class T = size_type,
			         typename std::enable_if<
			                 !std::is_same<T, truncated_hash_type>::value>::type* = nullptr>
			constexpr inline static bool USE_STORED_HASH_ON_REHASH(size_type bucket_count) noexcept {
				(void)bucket_count;
				if (StoreHash && is_power_of_two_policy<GrowthPolicy>::value) {
					ASSERT(bucket_count > 0);
					return (bucket_count - 1) <=
					       std::numeric_limits<truncated_hash_type>::max();
				} else {
					return false;
				}
			}

			/**
     		 * Return an always valid pointer to an static empty hopscotch_bucket.
     		 */
			inline hopscotch_bucket* static_empty_bucket_ptr() const noexcept {
				static hopscotch_bucket empty_bucket;
				return &empty_bucket;
			}

		private:
			buckets_container_type m_buckets_data;
			overflow_container_type m_overflow_elements;

			/**
   			 * Points to m_buckets_data.data() if !m_buckets_data.empty() otherwise points
   			 * to static_empty_bucket_ptr. This variable is useful to avoid the cost of
   			 * checking if m_buckets_data is empty when trying to find an element.
   			 *
   			 * TODO Remove m_buckets_data and only use a pointer+size instead of a
   			 * pointer+vector to save some space in the hopscotch_hash object.
   			 */
			hopscotch_bucket* m_buckets;

			size_type m_nb_elements;

			/**
   			 * Min size of the hash table before a rehash can occurs automatically (except
   			 * if m_max_load_threshold_rehash os reached). If the neighborhood of a bucket
   			 * is full before the min is reacher, the elements are put into
   			 * m_overflow_elements.
   			 */
			size_type m_min_load_threshold_rehash;

			/**
   			 * Max size of the hash table before a rehash occurs automatically to grow the
   			 * table.
   			 */
			size_type m_max_load_threshold_rehash;

			float m_max_load_factor;
		};

	}  // end namespace detail_hopscotch_hash

}  // end namespace tsl
#endif//CRYPTANALYSISLIB_HOPSCOTCH_HASH_H
