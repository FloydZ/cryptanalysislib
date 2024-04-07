#ifndef CRYPTANALYSISLIB_HOPSCOTCH_H
#define CRYPTANALYSISLIB_HOPSCOTCH_H
// SOURCE:https://github.com/Tessil/hopscotch-map/blob/master/include/tsl/hopscotch_map.h

#include <algorithm>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <list>
#include <memory>
#include <type_traits>
#include <utility>

#include "hopscotch_hash.h"
// TODO #include "lock_free_hopscotch.h"
#include "growth_policy.h"
#include "ska_flat.h"

// TODO ranges constructor
// INTERNAL::TODO:
//		- new buckets in cache?
// 		- avx search?

template<typename HM>
concept InternalHashMapAble =
	requires(HM hm) {
	    HM::DEFAULT_INIT_BUCKET_SIZE;
	    HM::key_type;
	    HM::size_type;
	    HM::difference_type;
	    HM::hasher;
	    HM::key_equal;
	    HM::allocator_type;
	    HM::reference;
	    HM::const_reference;
	    HM::pointer;
	    HM::const_pointer;
	    HM::iterator;
	    HM::const_iterator;

	    /// memory stuff
	    hm.clear();
	    hm.reserve();

	    /// iterators
	    hm.begin();
		hm.cbegin();
	    hm.end();
	    hm.cend();
	    hm.erase();
	    hm.mutate_iterator();

	    /// insert
	    hm.insert();
	    hm.insert_or_assign();
	    hm.emplace();

	    /// capacity stuff
	    hm.empty();
	    hm.size();
	    hm.max_size();

	    /// lookup
	    hm.at();
	    hm[0];
	    hm.find();
	    hm.contains();

	    /// bucket stuff
	    hm.bucket_count();
	    hm.max_bucket_count();
	    hm.load_factor();
	    hm.max_load_factor();
	    hm.rehash();
	    hm.hash_function();
	    hm.key_eq();
	    hm.overflow_size();



	    /// kp
	    hm.swap();
	    hm.count();
	    hm.equal_range();
};

/**
 * Implementation of a hash map using the hopscotch hashing algorithm.
 *
 * The Key and the value T must be either nothrow move-constructible,
 * copy-constructible or both.
 *
 * The size of the neighborhood (NeighborhoodSize) must be > 0 and <= 62 if
 * StoreHash is false. When StoreHash is true, 32-bits of the hash will be
 * stored alongside the neighborhood limiting the NeighborhoodSize to <= 30.
 * There is no memory usage difference between 'NeighborhoodSize 62; StoreHash
 * false' and 'NeighborhoodSize 30; StoreHash true'.
 *
 * Storing the hash may improve performance on insert during the rehash process
 * if the hash takes time to compute. It may also improve read performance if
 * the KeyEqual function takes time (or incurs a cache-miss). If used with
 * simple Hash and KeyEqual it may slow things down.
 *
 * StoreHash can only be set if the GrowthPolicy is set to
 * tsl::power_of_two_growth_policy.
 *
 * GrowthPolicy defines how the map grows and consequently how a hash value is
 * mapped to a bucket. By default the map uses tsl::power_of_two_growth_policy.
 * This policy keeps the number of buckets to a power of two and uses a mask to
 * map the hash to a bucket instead of the slow modulo. You may define your own
 * growth policy, check tsl::power_of_two_growth_policy for the interface.
 *
 * If the destructors of Key or T throw an exception, behaviour of the class is
 * undefined.
 *
 * Iterators invalidation:
 *  - clear, operator=, reserve, rehash: always invalidate the iterators.
 *  - insert, emplace, emplace_hint, operator[]: if there is an effective
 * insert, invalidate the iterators if a displacement is needed to resolve a
 * collision (which mean that most of the time, insert will invalidate the
 * iterators). Or if there is a rehash.
 *  - erase: iterator on the erased element is the only one which become
 * invalid.
 */
template<class Key,
         class T,
         class Hash = std::hash<Key>,
         class KeyEqual = std::equal_to<Key>,
         class Allocator = std::allocator<std::pair<Key, T>>,
         unsigned int NeighborhoodSize = 62,
         bool StoreHash = false,
         class GrowthPolicy = cryptanalysislib::hh::power_of_two_growth_policy<2>>
class hopscotch_map {
private:
	template<typename U>
	using has_is_transparent = tsl::detail_hopscotch_hash::has_is_transparent<U>;

	class KeySelect {
	public:
		using key_type = Key;

		[[nodiscard]] constexpr const key_type &operator()(const std::pair<Key, T> &key_value) const noexcept {
			return key_value.first;
		}

		[[nodiscard]] constexpr inline key_type &operator()(std::pair<Key, T> &key_value) noexcept {
			return key_value.first;
		}
	};

	class ValueSelect {
	public:
		using value_type = T;

		[[nodiscard]] constexpr inline const value_type &operator()(const std::pair<Key, T> &key_value) const noexcept{
			return key_value.second;
		}

		[[nodiscard]] constexpr inline value_type &operator()(std::pair<Key, T> &key_value) noexcept {
			return key_value.second;
		}
	};

	using overflow_container_type = std::list<std::pair<Key, T>, Allocator>;

	// TODO generic over this

	using ht2 = sherwood_v3_table <
	                std::pair<Key, T>,
					KeySelect,
					ValueSelect,
	                Hash,
	                KeyOrValueHasher<Key, std::pair<Key, T>, Hash>,
	        		KeyEqual ,
	                KeyOrValueEquality<Key, std::pair<Key, T>, KeyEqual>,
	                Allocator,
	                typename std::allocator_traits<Allocator>::template rebind_alloc<sherwood_v3_entry<std::pair<Key, T>>>
	>;
	using ht = tsl::detail_hopscotch_hash::hopscotch_hash<
	        		std::pair<Key, T>,
	                KeySelect,
	        		ValueSelect,
	        		Hash,
	        		KeyEqual,
	        		Allocator,
	        		NeighborhoodSize,
	        		StoreHash,
	        		GrowthPolicy,
	        		overflow_container_type
	>;

public:
	using key_type = typename ht::key_type;
	using mapped_type = T;
	using value_type = typename ht::value_type;
	using size_type = typename ht::size_type;
	using difference_type = typename ht::difference_type;
	using hasher = typename ht::hasher;
	using key_equal = typename ht::key_equal;
	using allocator_type = typename ht::allocator_type;
	using reference = typename ht::reference;
	using const_reference = typename ht::const_reference;
	using pointer = typename ht::pointer;
	using const_pointer = typename ht::const_pointer;
	using iterator = typename ht::iterator;
	using const_iterator = typename ht::const_iterator;

	/*
    * Constructors
    */
	hopscotch_map() : hopscotch_map(ht::DEFAULT_INIT_BUCKETS_SIZE) {}

	constexpr explicit hopscotch_map(const size_type bucket_count,
	                       const Hash &hash = Hash(),
	                       const KeyEqual &equal = KeyEqual(),
	                       const Allocator &alloc = Allocator()) noexcept
	    : m_ht(bucket_count, hash, equal, alloc, ht::DEFAULT_MAX_LOAD_FACTOR) {}

	constexpr hopscotch_map(const size_type bucket_count,
	                        const Allocator &alloc) noexcept
	    : hopscotch_map(bucket_count, Hash(), KeyEqual(), alloc) {}

	constexpr hopscotch_map(const size_type bucket_count,
	                        const Hash &hash,
	              			const Allocator &alloc) noexcept
	    : hopscotch_map(bucket_count, hash, KeyEqual(), alloc) {}

	constexpr explicit hopscotch_map(const Allocator &alloc) noexcept
	    : hopscotch_map(ht::DEFAULT_INIT_BUCKETS_SIZE, alloc) {}

	constexpr hopscotch_map(const hopscotch_map &other,
	                        const Allocator &alloc) noexcept
	    : m_ht(other.m_ht, alloc) {}

	template<class InputIt>
	constexpr hopscotch_map(InputIt first, InputIt last,
	              const size_type bucket_count = ht::DEFAULT_INIT_BUCKETS_SIZE,
	              const Hash &hash = Hash(), const KeyEqual &equal = KeyEqual(),
	              const Allocator &alloc = Allocator()) noexcept
	    : hopscotch_map(bucket_count, hash, equal, alloc) {
		insert(first, last);
	}

	template<class InputIt>
	constexpr hopscotch_map(InputIt first,
	                        InputIt last,
	                        const size_type bucket_count,
	              			const Allocator &alloc)noexcept
	    : hopscotch_map(first, last, bucket_count, Hash(), KeyEqual(), alloc) {}

	template<class InputIt>
	constexpr hopscotch_map(InputIt first, InputIt last,
	                        const size_type bucket_count,
	              			const Hash &hash, const Allocator &alloc) noexcept
	    : hopscotch_map(first, last, bucket_count, hash, KeyEqual(), alloc) {}

	constexpr hopscotch_map(std::initializer_list<value_type> init,
	              size_type bucket_count = ht::DEFAULT_INIT_BUCKETS_SIZE,
	              const Hash &hash = Hash(), const KeyEqual &equal = KeyEqual(),
	              const Allocator &alloc = Allocator()) noexcept
	    : hopscotch_map(init.begin(), init.end(), bucket_count, hash, equal,
	                    alloc) {}

	constexpr hopscotch_map(std::initializer_list<value_type> init,
	                        size_type bucket_count,
	              			const Allocator &alloc) noexcept
	    : hopscotch_map(init.begin(), init.end(), bucket_count, Hash(),
	                    KeyEqual(), alloc) {}

	constexpr hopscotch_map(std::initializer_list<value_type> init,
	                        size_type bucket_count,
	              			const Hash &hash,
	                        const Allocator &alloc) noexcept
	    : hopscotch_map(init.begin(), init.end(), bucket_count, hash, KeyEqual(),
	                    alloc) {}

	constexpr hopscotch_map &operator=(std::initializer_list<value_type> ilist) noexcept {
		m_ht.clear();

		m_ht.reserve(ilist.size());
		m_ht.insert(ilist.begin(), ilist.end());

		return *this;
	}

	constexpr inline allocator_type get_allocator() const { return m_ht.get_allocator(); }

	/*
   	 * Iterators
   	 */
	constexpr inline iterator begin() noexcept { return m_ht.begin(); }
	constexpr inline const_iterator begin() const noexcept { return m_ht.begin(); }
	constexpr inline const_iterator cbegin() const noexcept { return m_ht.cbegin(); }

	constexpr inline iterator end() noexcept { return m_ht.end(); }
	constexpr inline const_iterator end() const noexcept { return m_ht.end(); }
	constexpr inline const_iterator cend() const noexcept { return m_ht.cend(); }

	/*
     * Capacity
     */
	[[nodiscard]] constexpr inline bool empty() const noexcept { return m_ht.empty(); }
	[[nodiscard]] constexpr inline size_type size() const noexcept { return m_ht.size(); }
	[[nodiscard]] constexpr inline size_type max_size() const noexcept { return m_ht.max_size(); }

	/*
     * Modifiers
     */
	constexpr inline void clear() noexcept { m_ht.clear(); }

	constexpr inline std::pair<iterator, bool> insert(const value_type &value) noexcept {
		return m_ht.insert(value);
	}

	template<class P,
	         typename std::enable_if<std::is_constructible<value_type, P &&>::value>::type * = nullptr>
	constexpr std::pair<iterator, bool> insert(P &&value) noexcept {
		return m_ht.insert(std::forward<P>(value));
	}

	constexpr std::pair<iterator, bool> insert(value_type &&value) noexcept {
		return m_ht.insert(std::move(value));
	}

	constexpr iterator insert(const_iterator hint, const value_type &value) noexcept {
		return m_ht.insert(hint, value);
	}

	template<class P, typename std::enable_if<std::is_constructible<
	                          value_type, P &&>::value>::type * = nullptr>
	constexpr inline iterator insert(const_iterator hint, P &&value) noexcept {
		return m_ht.insert(hint, std::forward<P>(value));
	}

	constexpr inline iterator insert(const_iterator hint, value_type &&value) noexcept {
		return m_ht.insert(hint, std::move(value));
	}

	template<class InputIt>
	constexpr void insert(InputIt first, InputIt last) noexcept {
		m_ht.insert(first, last);
	}

	constexpr void insert(std::initializer_list<value_type> ilist) noexcept {
		m_ht.insert(ilist.begin(), ilist.end());
	}

	template<class M>
	constexpr inline std::pair<iterator, bool> insert_or_assign(const key_type &k,
	                                                            M &&obj) noexcept {
		return m_ht.insert_or_assign(k, std::forward<M>(obj));
	}

	template<class M>
	constexpr std::pair<iterator, bool> insert_or_assign(key_type &&k, M &&obj) noexcept {
		return m_ht.insert_or_assign(std::move(k), std::forward<M>(obj));
	}

	template<class M>
	constexpr iterator insert_or_assign(const_iterator hint,
	                                    const key_type &k,
	                                    M &&obj) noexcept {
		return m_ht.insert_or_assign(hint, k, std::forward<M>(obj));
	}

	template<class M>
	constexpr iterator insert_or_assign(const_iterator hint,
	                                    key_type &&k,
	                                    M &&obj) noexcept {
		return m_ht.insert_or_assign(hint, std::move(k), std::forward<M>(obj));
	}

	/**
     * Due to the way elements are stored, emplace will need to move or copy the
     * key-value once. The method is equivalent to
     * insert(value_type(std::forward<Args>(args)...));
     *
     * Mainly here for compatibility with the std::unordered_map interface.
     */
	template<class... Args>
	constexpr std::pair<iterator, bool> emplace(Args &&...args) noexcept {
		return m_ht.emplace(std::forward<Args>(args)...);
	}

	/**
     * Due to the way elements are stored, emplace_hint will need to move or copy
     * the key-value once. The method is equivalent to insert(hint,
     * value_type(std::forward<Args>(args)...));
     *
     * Mainly here for compatibility with the std::unordered_map interface.
     */
	template<class... Args>
	constexpr iterator emplace_hint(const_iterator hint, Args &&...args) noexcept {
		return m_ht.emplace_hint(hint, std::forward<Args>(args)...);
	}

	template<class... Args>
	constexpr std::pair<iterator, bool> try_emplace(const key_type &k, Args &&...args) noexcept {
		return m_ht.try_emplace(k, std::forward<Args>(args)...);
	}

	template<class... Args>
	constexpr std::pair<iterator, bool> try_emplace(key_type &&k, Args &&...args) noexcept {
		return m_ht.try_emplace(std::move(k), std::forward<Args>(args)...);
	}

	template<class... Args>
	constexpr iterator try_emplace(const_iterator hint, const key_type &k, Args &&...args) noexcept {
		return m_ht.try_emplace(hint, k, std::forward<Args>(args)...);
	}

	template<class... Args>
	constexpr iterator try_emplace(const_iterator hint, key_type &&k, Args &&...args) noexcept {
		return m_ht.try_emplace(hint, std::move(k), std::forward<Args>(args)...);
	}

	constexpr inline iterator erase(iterator pos) noexcept { return m_ht.erase(pos); }
	constexpr inline iterator erase(const_iterator pos) noexcept { return m_ht.erase(pos); }
	constexpr inline iterator erase(const_iterator first, const_iterator last) noexcept {
		return m_ht.erase(first, last);
	}
	constexpr inline size_type erase(const key_type &key) noexcept { return m_ht.erase(key); }

	/**
     * Use the hash value 'precalculated_hash' instead of hashing the key. The
     * hash value should be the same as hash_function()(key). Useful to speed-up
     * the lookup to the value if you already have the hash.
     */
	constexpr inline size_type erase(const key_type &key,
	                                 const std::size_t precalculated_hash) noexcept {
		return m_ht.erase(key, precalculated_hash);
	}

	/**
     * This overload only participates in the overload resolution if the typedef
     * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
     * to Key.
     */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline size_type erase(const K &key) noexcept {
		return m_ht.erase(key);
	}

	/**
   * @copydoc erase(const K& key)
   *
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup to the value if you already have the hash.
   */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline size_type erase(const K &key,
	                                 const std::size_t precalculated_hash) noexcept {
		return m_ht.erase(key, precalculated_hash);
	}

	constexpr inline void swap(hopscotch_map &other) noexcept { other.m_ht.swap(m_ht); }

	/**
     * Lookup
     */
	constexpr inline T &at(const Key &key) noexcept { return m_ht.at(key); }

	/**
     * Use the hash value 'precalculated_hash' instead of hashing the key. The
     * hash value should be the same as hash_function()(key). Useful to speed-up
     * the lookup if you already have the hash.
     */
	constexpr inline T &at(const Key &key,
	                       const std::size_t precalculated_hash) noexcept {
		return m_ht.at(key, precalculated_hash);
	}

	constexpr inline const T &at(const Key &key) const noexcept { return m_ht.at(key); }

	/**
     * @copydoc at(const Key& key, std::size_t precalculated_hash)
     */
	constexpr inline const T &at(const Key &key, const std::size_t precalculated_hash) const noexcept{
		return m_ht.at(key, precalculated_hash);
	}

	/**
   * This overload only participates in the overload resolution if the typedef
   * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
   * to Key.
   */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline T &at(const K &key) noexcept {
		return m_ht.at(key);
	}

	/**
     * @copydoc at(const K& key)
     *
     * Use the hash value 'precalculated_hash' instead of hashing the key. The
     * hash value should be the same as hash_function()(key). Useful to speed-up
     * the lookup if you already have the hash.
     */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline T &at(const K &key, const std::size_t precalculated_hash) noexcept {
		return m_ht.at(key, precalculated_hash);
	}

	/**
     * @copydoc at(const K& key)
     */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline const T &at(const K &key) const noexcept {
		return m_ht.at(key);
	}

	/**
     * @copydoc at(const K& key, std::size_t precalculated_hash)
     */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline const T &at(const K &key,
	                             const std::size_t precalculated_hash) const noexcept {
		return m_ht.at(key, precalculated_hash);
	}

	constexpr inline T &operator[](const Key &key) noexcept { return m_ht[key]; }
	constexpr inline T &operator[](Key &&key) noexcept { return m_ht[std::move(key)]; }

	constexpr inline size_type count(const Key &key) const noexcept { return m_ht.count(key); }

	/**
   * Use the hash value 'precalculated_hash' instead of hashing the key. The
   * hash value should be the same as hash_function()(key). Useful to speed-up
   * the lookup if you already have the hash.
   */
	constexpr inline size_type count(const Key &key,
	                                 const std::size_t precalculated_hash) const noexcept {
		return m_ht.count(key, precalculated_hash);
	}

	/**
   * This overload only participates in the overload resolution if the typedef
   * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
   * to Key.
   */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline size_type count(const K &key) const {
		return m_ht.count(key);
	}

	/**
     * @copydoc count(const K& key) const
     *
     * Use the hash value 'precalculated_hash' instead of hashing the key. The
     * hash value should be the same as hash_function()(key). Useful to speed-up
     * the lookup if you already have the hash.
     */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline size_type count(const K &key,
	                          const std::size_t precalculated_hash) const noexcept {
		return m_ht.count(key, precalculated_hash);
	}

	constexpr inline iterator find(const Key &key) noexcept { return m_ht.find(key); }

	/**
     * Use the hash value 'precalculated_hash' instead of hashing the key. The
     * hash value should be the same as hash_function()(key). Useful to speed-up
     * the lookup if you already have the hash.
     */
	constexpr inline iterator find(const Key &key,
	                               const std::size_t precalculated_hash) noexcept {
		return m_ht.find(key, precalculated_hash);
	}

	constexpr inline const_iterator find(const Key &key) const noexcept { return m_ht.find(key); }

	/**
     * @copydoc find(const Key& key, std::size_t precalculated_hash)
     */
	const_iterator find(const Key &key, std::size_t precalculated_hash) const {
		return m_ht.find(key, precalculated_hash);
	}

	/**
     * This overload only participates in the overload resolution if the typedef
     * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
     * to Key.
     */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline iterator find(const K &key) noexcept {
		return m_ht.find(key);
	}

	/**
     * @copydoc find(const K& key)
     *
     * Use the hash value 'precalculated_hash' instead of hashing the key. The
     * hash value should be the same as hash_function()(key). Useful to speed-up
     * the lookup if you already have the hash.
     */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline iterator find(const K &key,
	                               const std::size_t precalculated_hash) noexcept {
		return m_ht.find(key, precalculated_hash);
	}

	/**
     * @copydoc find(const K& key)
     */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline const_iterator find(const K &key) const noexcept {
		return m_ht.find(key);
	}

	/**
     * @copydoc find(const K& key)
     *
     * Use the hash value 'precalculated_hash' instead of hashing the key. The
     * hash value should be the same as hash_function()(key). Useful to speed-up
     * the lookup if you already have the hash.
     */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline const_iterator find(const K &key,
	                                     const std::size_t precalculated_hash) const noexcept {
		return m_ht.find(key, precalculated_hash);
	}

	constexpr inline bool contains(const Key &key) const noexcept { return m_ht.contains(key); }

	/**
     * Use the hash value 'precalculated_hash' instead of hashing the key. The
     * hash value should be the same as hash_function()(key). Useful to speed-up
     * the lookup if you already have the hash.
     */
	constexpr inline bool contains(const Key &key,
	                        const std::size_t precalculated_hash) const noexcept {
		return m_ht.contains(key, precalculated_hash);
	}

	/**
   * This overload only participates in the overload resolution if the typedef
   * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
   * to Key.
   */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline bool contains(const K &key) const noexcept {
		return m_ht.contains(key);
	}

	/**
     * @copydoc contains(const K& key) const
     *
     * Use the hash value 'precalculated_hash' instead of hashing the key. The
     * hash value should be the same as hash_function()(key). Useful to speed-up
     * the lookup if you already have the hash.
     */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline bool contains(const K &key,
	                               const std::size_t precalculated_hash) const noexcept {
		return m_ht.contains(key, precalculated_hash);
	}

	constexpr inline std::pair<iterator, iterator> equal_range(const Key &key) noexcept {
		return m_ht.equal_range(key);
	}

	/**
     * Use the hash value 'precalculated_hash' instead of hashing the key. The
     * hash value should be the same as hash_function()(key). Useful to speed-up
     * the lookup if you already have the hash.
     */
	constexpr inline std::pair<iterator, iterator> equal_range(const Key &key,
	                                          const std::size_t precalculated_hash) noexcept {
		return m_ht.equal_range(key, precalculated_hash);
	}

	std::pair<const_iterator, const_iterator> equal_range(const Key &key) const {
		return m_ht.equal_range(key);
	}

	/**
    * @copydoc equal_range(const Key& key, std::size_t precalculated_hash)
    */
	constexpr inline std::pair<const_iterator, const_iterator> equal_range(
	        const Key &key,
	        const std::size_t precalculated_hash) const noexcept {
		return m_ht.equal_range(key, precalculated_hash);
	}

	/**
     * This overload only participates in the overload resolution if the typedef
     * KeyEqual::is_transparent exists. If so, K must be hashable and comparable
     * to Key.
     */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline std::pair<iterator, iterator> equal_range(const K &key) noexcept {
		return m_ht.equal_range(key);
	}

	/**
     * @copydoc equal_range(const K& key)
     *
     * Use the hash value 'precalculated_hash' instead of hashing the key. The
     * hash value should be the same as hash_function()(key). Useful to speed-up
     * the lookup if you already have the hash.
     */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline std::pair<iterator, iterator> equal_range(const K &key,
	                                          const std::size_t precalculated_hash) noexcept {
		return m_ht.equal_range(key, precalculated_hash);
	}

	/**
     * @copydoc equal_range(const K& key)
     */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline std::pair<const_iterator, const_iterator> equal_range(const K &key) const noexcept {
		return m_ht.equal_range(key);
	}

	/**
   * @copydoc equal_range(const K& key, std::size_t precalculated_hash)
   */
	template<
	        class K, class KE = KeyEqual,
	        typename std::enable_if<has_is_transparent<KE>::value>::type * = nullptr>
	constexpr inline std::pair<const_iterator, const_iterator> equal_range(
	        const K &key,
	        const std::size_t precalculated_hash) const noexcept {
		return m_ht.equal_range(key, precalculated_hash);
	}

	/**
     * Bucket interface
     */
	[[nodiscard]] constexpr inline size_type bucket_count() const noexcept { return m_ht.bucket_count(); }
	[[nodiscard]] constexpr inline size_type max_bucket_count() const noexcept { return m_ht.max_bucket_count(); }

	/**
     *  Hash policy
     */
	[[nodiscard]] constexpr inline float load_factor() const noexcept { return m_ht.load_factor(); }
	[[nodiscard]] constexpr inline float max_load_factor() const noexcept { return m_ht.max_load_factor(); }
	constexpr inline void max_load_factor(float ml) noexcept { m_ht.max_load_factor(ml); }

	constexpr inline void rehash(size_type count_) { m_ht.rehash(count_); }
	constexpr inline void reserve(size_type count_) { m_ht.reserve(count_); }

	/**
     * Observers
     */
	[[nodiscard]] constexpr inline hasher hash_function() const noexcept { return m_ht.hash_function(); }
	[[nodiscard]] constexpr inline key_equal key_eq() const noexcept { return m_ht.key_eq(); }

	/**
     * Convert a const_iterator to an iterator.
     */
	constexpr inline iterator mutable_iterator(const_iterator pos) noexcept {
		return m_ht.mutable_iterator(pos);
	}

	constexpr inline size_type overflow_size() const noexcept { return m_ht.overflow_size(); }

	friend bool operator==(const hopscotch_map &lhs, const hopscotch_map &rhs) {
		if (lhs.size() != rhs.size()) {
			return false;
		}

		for (const auto &element_lhs: lhs) {
			const auto it_element_rhs = rhs.find(element_lhs.first);
			if (it_element_rhs == rhs.cend() ||
			    element_lhs.second != it_element_rhs->second) {
				return false;
			}
		}

		return true;
	}

	friend bool operator!=(const hopscotch_map &lhs, const hopscotch_map &rhs) {
		return !operator==(lhs, rhs);
	}

	friend void swap(hopscotch_map &lhs, hopscotch_map &rhs) { lhs.swap(rhs); }

private:
	ht m_ht;
};

/**
	 * Same as `tsl::hopscotch_map<Key, T, Hash, KeyEqual, Allocator,
	 * NeighborhoodSize, StoreHash, tsl::hh::prime_growth_policy>`.
	 */
template<class Key, class T, class Hash = std::hash<Key>,
         class KeyEqual = std::equal_to<Key>,
         class Allocator = std::allocator<std::pair<Key, T>>,
         unsigned int NeighborhoodSize = 62, bool StoreHash = false>
using hopscotch_pg_map =
        hopscotch_map<Key, T, Hash, KeyEqual, Allocator, NeighborhoodSize,
                      StoreHash, cryptanalysislib::hh::prime_growth_policy>;

#endif//CRYPTANALYSISLIB_HOPSCOTCH_H
