#ifndef CRYPTANALYSISLIB_SKA_FLAT_H
#define CRYPTANALYSISLIB_SKA_FLAT_H

#include "growth_policy.h"
#include <cstdint>

// TODO remove
template<typename T, bool>
struct AssignIfTrue{
	void operator()(T & lhs, const T & rhs){
		lhs = rhs;
	}
	void operator()(T & lhs, T && rhs) {
		lhs = std::move(rhs);
	}
};
template<typename T>
struct AssignIfTrue<T, false>{
	void operator()(T &, const T &)
	{}
	void operator()(T &, T &&)
	{}
};


template<typename Result, typename Functor>
struct functor_storage : Functor {
	functor_storage() = default;
	functor_storage(const Functor &functor)
	    : Functor(functor) {
	}
	template<typename... Args>
	Result operator()(Args &&...args) {
		return static_cast<Functor &>(*this)(std::forward<Args>(args)...);
	}
	template<typename... Args>
	Result operator()(Args &&...args) const {
		return static_cast<const Functor &>(*this)(std::forward<Args>(args)...);
	}
};
template<typename Result, typename... Args>
struct functor_storage<Result, Result (*)(Args...)> {
	typedef Result (*function_ptr)(Args...);
	function_ptr function;
	functor_storage(function_ptr function)
	    : function(function) {
	}
	Result operator()(Args... args) const {
		return function(std::forward<Args>(args)...);
	}
	operator function_ptr &() {
		return function;
	}
	operator const function_ptr &() {
		return function;
	}
};
template<typename key_type, typename value_type, typename hasher>
struct KeyOrValueHasher : functor_storage<size_t, hasher> {
	typedef functor_storage<size_t, hasher> hasher_storage;
	KeyOrValueHasher() = default;
	KeyOrValueHasher(const hasher &hash)
	    : hasher_storage(hash) {
	}
	size_t operator()(const key_type &key) {
		return static_cast<hasher_storage &>(*this)(key);
	}
	size_t operator()(const key_type &key) const {
		return static_cast<const hasher_storage &>(*this)(key);
	}
	size_t operator()(const value_type &value) {
		return static_cast<hasher_storage &>(*this)(value.first);
	}
	size_t operator()(const value_type &value) const {
		return static_cast<const hasher_storage &>(*this)(value.first);
	}
	template<typename F, typename S>
	size_t operator()(const std::pair<F, S> &value) {
		return static_cast<hasher_storage &>(*this)(value.first);
	}
	template<typename F, typename S>
	size_t operator()(const std::pair<F, S> &value) const {
		return static_cast<const hasher_storage &>(*this)(value.first);
	}
};

template<typename key_type, typename value_type, typename key_equal>
struct KeyOrValueEquality : functor_storage<bool, key_equal> {
	typedef functor_storage<bool, key_equal> equality_storage;
	KeyOrValueEquality() = default;
	KeyOrValueEquality(const key_equal &equality)
	    : equality_storage(equality) {
	}
	bool operator()(const key_type &lhs, const key_type &rhs) {
		return static_cast<equality_storage &>(*this)(lhs, rhs);
	}
	bool operator()(const key_type &lhs, const value_type &rhs) {
		return static_cast<equality_storage &>(*this)(lhs, rhs.first);
	}
	bool operator()(const value_type &lhs, const key_type &rhs) {
		return static_cast<equality_storage &>(*this)(lhs.first, rhs);
	}
	bool operator()(const value_type &lhs, const value_type &rhs) {
		return static_cast<equality_storage &>(*this)(lhs.first, rhs.first);
	}
	template<typename F, typename S>
	bool operator()(const key_type &lhs, const std::pair<F, S> &rhs) {
		return static_cast<equality_storage &>(*this)(lhs, rhs.first);
	}
	template<typename F, typename S>
	bool operator()(const std::pair<F, S> &lhs, const key_type &rhs) {
		return static_cast<equality_storage &>(*this)(lhs.first, rhs);
	}
	template<typename F, typename S>
	bool operator()(const value_type &lhs, const std::pair<F, S> &rhs) {
		return static_cast<equality_storage &>(*this)(lhs.first, rhs.first);
	}
	template<typename F, typename S>
	bool operator()(const std::pair<F, S> &lhs, const value_type &rhs) {
		return static_cast<equality_storage &>(*this)(lhs.first, rhs.first);
	}
	template<typename FL, typename SL, typename FR, typename SR>
	bool operator()(const std::pair<FL, SL> &lhs, const std::pair<FR, SR> &rhs) {
		return static_cast<equality_storage &>(*this)(lhs.first, rhs.first);
	}
};


static constexpr int8_t min_lookups = 4;
template<typename T>
struct sherwood_v3_entry {
	constexpr sherwood_v3_entry() noexcept {}
	constexpr sherwood_v3_entry(const int8_t distance_from_desired) noexcept
	    : distance_from_desired(distance_from_desired) {}
	constexpr ~sherwood_v3_entry() {}
	static sherwood_v3_entry *empty_default_table() {
		static sherwood_v3_entry result[min_lookups] = {{}, {}, {}, {special_end_value}};
		return result;
	}

	[[nodiscard]] constexpr inline bool has_value() const noexcept {
		return distance_from_desired >= 0;
	}

	[[nodiscard]] constexpr inline bool is_empty() const noexcept {
		return distance_from_desired < 0;
	}

	[[nodiscard]] constexpr inline bool is_at_desired_position() const noexcept {
		return distance_from_desired <= 0;
	}

	template<typename... Args>
	constexpr inline void emplace(const int8_t distance, Args &&...args) noexcept {
		new (std::addressof(value)) T(std::forward<Args>(args)...);
		distance_from_desired = distance;
	}

	constexpr inline void destroy_value() noexcept {
		value.~T();
		distance_from_desired = -1;
	}

	int8_t distance_from_desired = -1;
	static constexpr int8_t special_end_value = 0;
	union {
		T value;
	};
};

/// TODO simplify
/// \tparam T
/// \tparam KeySelect
/// \tparam ValueSelect
/// \tparam ArgumentHash
/// \tparam Hasher
/// \tparam ArgumentEqual
/// \tparam Equal
/// \tparam ArgumentAlloc
/// \tparam EntryAlloc
template<typename T,
         typename KeySelect,
         typename ValueSelect,
         typename ArgumentHash,
         typename Hasher,
         typename ArgumentEqual,
         typename Equal,
         typename ArgumentAlloc,
         typename EntryAlloc>
class sherwood_v3_table : private EntryAlloc, private Hasher, private Equal {
	using Entry = sherwood_v3_entry<T>;
	using AllocatorTraits = std::allocator_traits<EntryAlloc>;
	using EntryPointer = typename AllocatorTraits::pointer;
	struct convertible_to_iterator;

	template<typename U>
	using has_mapped_type = typename std::integral_constant<bool, !std::is_same<U, void>::value>;

public:
	using FindKey = typename KeySelect::key_type;
	using value_type = T;
	using key_type = FindKey;
	using size_type = size_t;
	using difference_type = std::ptrdiff_t;
	using hasher = ArgumentHash;
	using key_equal = ArgumentEqual;
	using allocator_type = EntryAlloc;
	using reference = value_type &;
	using const_reference = const value_type &;
	using pointer = value_type *;
	using const_pointer = const value_type *;

	constexpr static size_t DEFAULT_INIT_BUCKETS_SIZE = 1024;
	static constexpr float DEFAULT_MAX_LOAD_FACTOR = 0.8f;

	constexpr sherwood_v3_table() noexcept {}

	constexpr explicit sherwood_v3_table(const size_type bucket_count,
	                                     const ArgumentHash &hash = ArgumentHash(),
	                                     const ArgumentEqual &equal = ArgumentEqual(),
	                                     const ArgumentAlloc &alloc = ArgumentAlloc(),
	                                     const double max_load_factor = 1) noexcept
	    : EntryAlloc(alloc), Hasher(hash), Equal(equal) {
		(void)max_load_factor;
		rehash(bucket_count);
	}

	constexpr sherwood_v3_table(const size_type bucket_count,
	                            const ArgumentAlloc &alloc) noexcept
	    : sherwood_v3_table(bucket_count, ArgumentHash(), ArgumentEqual(), alloc) {}

	constexpr sherwood_v3_table(const size_type bucket_count,
	                            const ArgumentHash &hash,
	                            const ArgumentAlloc &alloc) noexcept
	    : sherwood_v3_table(bucket_count, hash, ArgumentEqual(), alloc) {}

	constexpr explicit sherwood_v3_table(const ArgumentAlloc &alloc) noexcept
	    : EntryAlloc(alloc) {}

	template<typename It>
	constexpr sherwood_v3_table(It first, It last,
	                            const size_type bucket_count = 0,
	                            const ArgumentHash &hash = ArgumentHash(),
	                            const ArgumentEqual &equal = ArgumentEqual(),
	                            const ArgumentAlloc &alloc = ArgumentAlloc()) noexcept
	    : sherwood_v3_table(bucket_count, hash, equal, alloc) {
		insert(first, last);
	}

	template<typename It>
	constexpr sherwood_v3_table(It first, It last,
	                            const size_type bucket_count,
	                            const ArgumentAlloc &alloc) noexcept
	    : sherwood_v3_table(first, last, bucket_count, ArgumentHash(), ArgumentEqual(), alloc) {}

	template<typename It>
	constexpr sherwood_v3_table(It first, It last,
	                            const size_type bucket_count,
	                            const ArgumentHash &hash,
	                            const ArgumentAlloc &alloc) noexcept
	    : sherwood_v3_table(first, last, bucket_count, hash, ArgumentEqual(), alloc) {}

	constexpr sherwood_v3_table(std::initializer_list<T> il,
	                            const size_type bucket_count = 0,
	                            const ArgumentHash &hash = ArgumentHash(),
	                            const ArgumentEqual &equal = ArgumentEqual(),
	                            const ArgumentAlloc &alloc = ArgumentAlloc()) noexcept
	    : sherwood_v3_table(bucket_count, hash, equal, alloc) {
		if (bucket_count == 0) {
			rehash(il.size());
		}

		insert(il.begin(), il.end());
	}

	constexpr sherwood_v3_table(std::initializer_list<T> il,
	                            const size_type bucket_count,
	                            const ArgumentAlloc &alloc) noexcept
	    : sherwood_v3_table(il, bucket_count, ArgumentHash(), ArgumentEqual(), alloc) {}

	constexpr sherwood_v3_table(std::initializer_list<T> il,
	                            const size_type bucket_count,
	                            const ArgumentHash &hash,
	                            const ArgumentAlloc &alloc) noexcept
	    : sherwood_v3_table(il, bucket_count, hash, ArgumentEqual(), alloc) {}

	constexpr sherwood_v3_table(const sherwood_v3_table &other) noexcept
	    : sherwood_v3_table(other, AllocatorTraits::select_on_container_copy_construction(other.get_allocator())) {}

	constexpr sherwood_v3_table(const sherwood_v3_table &other,
	                            const ArgumentAlloc &alloc) noexcept
	    : EntryAlloc(alloc), Hasher(other), Equal(other), _max_load_factor(other._max_load_factor) {
		rehash_for_other_container(other);
		insert(other.begin(), other.end());
	}

	constexpr sherwood_v3_table(sherwood_v3_table &&other) noexcept
	    : EntryAlloc(std::move(other)), Hasher(std::move(other)), Equal(std::move(other)) {
		swap_pointers(other);
	}

	constexpr sherwood_v3_table(sherwood_v3_table &&other,
	                            const ArgumentAlloc &alloc) noexcept
	    : EntryAlloc(alloc), Hasher(std::move(other)), Equal(std::move(other)) {
		swap_pointers(other);
	}

	constexpr sherwood_v3_table &operator=(const sherwood_v3_table &other) noexcept {
		if (this == std::addressof(other)) {
			return *this;
		}

		clear();
		if (AllocatorTraits::propagate_on_container_copy_assignment::value) {
			if (static_cast<EntryAlloc &>(*this) != static_cast<const EntryAlloc &>(other)) {
				reset_to_empty_state();
			}
			AssignIfTrue<EntryAlloc, AllocatorTraits::propagate_on_container_copy_assignment::value>()(*this, other);
		}

		_max_load_factor = other._max_load_factor;
		static_cast<Hasher &>(*this) = other;
		static_cast<Equal &>(*this) = other;
		rehash_for_other_container(other);
		insert(other.begin(), other.end());
		return *this;
	}

	constexpr sherwood_v3_table &operator=(sherwood_v3_table &&other) noexcept {
		if (this == std::addressof(other)) {
			return *this;
		} else if (AllocatorTraits::propagate_on_container_move_assignment::value) {
			clear();
			reset_to_empty_state();
			AssignIfTrue<EntryAlloc, AllocatorTraits::propagate_on_container_move_assignment::value>()(*this, std::move(other));
			swap_pointers(other);
		} else if (static_cast<EntryAlloc &>(*this) == static_cast<EntryAlloc &>(other)) {
			swap_pointers(other);
		} else {
			clear();
			_max_load_factor = other._max_load_factor;
			rehash_for_other_container(other);
			for (T &elem: other)
				emplace(std::move(elem));
			other.clear();
		}

		static_cast<Hasher &>(*this) = std::move(other);
		static_cast<Equal &>(*this) = std::move(other);
		return *this;
	}

	constexpr ~sherwood_v3_table() noexcept {
		clear();
		deallocate_data(entries, num_slots_minus_one, max_lookups);
	}

	constexpr const allocator_type &get_allocator() const noexcept {
		return static_cast<const allocator_type &>(*this);
	}

	constexpr const ArgumentEqual &key_eq() const noexcept {
		return static_cast<const ArgumentEqual &>(*this);
	}

	constexpr const ArgumentHash &hash_function() const noexcept {
		return static_cast<const ArgumentHash &>(*this);
	}

	template<typename ValueType>
	struct templated_iterator {
		constexpr templated_iterator() = default;
		constexpr templated_iterator(EntryPointer current)
		    : current(current) {}
		EntryPointer current = EntryPointer();

		using iterator_category = std::forward_iterator_tag;
		using value_type = ValueType;
		using difference_type = ptrdiff_t;
		using pointer = ValueType *;
		using reference = ValueType &;

		friend bool operator==(const templated_iterator &lhs, const templated_iterator &rhs) {
			return lhs.current == rhs.current;
		}

		friend bool operator!=(const templated_iterator &lhs, const templated_iterator &rhs) {
			return !(lhs == rhs);
		}

		templated_iterator &operator++() {
			do {
				++current;
			} while (current->is_empty());
			return *this;
		}

		templated_iterator operator++(int) {
			templated_iterator copy(*this);
			++*this;
			return copy;
		}

		ValueType &operator*() const {
			return current->value;
		}

		ValueType *operator->() const {
			return std::addressof(current->value);
		}

		operator templated_iterator<const value_type>() const {
			return {current};
		}

		template<
		        class U = ValueSelect,
		        typename std::enable_if<has_mapped_type<U>::value>::type * = nullptr>
		constexpr templated_iterator value() const noexcept {
			ASSERT(false);
			// if (!current->is_empty()) {
			// 	return U()(current->value());
			// }

			// return U()(*current);
		}
	};

	using iterator = templated_iterator<value_type>;
	using const_iterator = templated_iterator<const value_type>;

	constexpr inline iterator begin() noexcept {
		for (EntryPointer it = entries;; ++it) {
			if (it->has_value()) {
				return {it};
			}
		}
	}

	constexpr inline const_iterator begin() const noexcept {
		for (EntryPointer it = entries;; ++it) {
			if (it->has_value())
				return {it};
		}
	}

	constexpr inline const_iterator cbegin() const {
		return begin();
	}

	constexpr inline iterator end() noexcept {
		return {entries + static_cast<ptrdiff_t>(num_slots_minus_one + max_lookups)};
	}

	constexpr inline const_iterator end() const noexcept {
		return {entries + static_cast<ptrdiff_t>(num_slots_minus_one + max_lookups)};
	}

	constexpr inline const_iterator cend() const noexcept {
		return end();
	}

	template<class K, class U = ValueSelect,
	         typename std::enable_if<has_mapped_type<U>::value>::type * = nullptr>
	constexpr inline ValueSelect::value_type &operator[](K &&key) noexcept {
		// TODO implement
		ASSERT(false);
	}

	constexpr inline iterator find(const FindKey &key) noexcept {
		size_t index = hash_policy.bucket_for_hash(hash_object(key));
		EntryPointer it = entries + ptrdiff_t(index);
		for (int8_t distance = 0; it->distance_from_desired >= distance; ++distance, ++it) {
			if (compares_equal(key, it->value)) {
				return {it};
			}
		}
		return end();
	}

	constexpr inline const_iterator find(const FindKey &key) const noexcept {
		return const_cast<sherwood_v3_table *>(this)->find(key);
	}

	/// TODO use hash
	template<class K>
	constexpr inline iterator find(const K &key,
	                               const std::size_t hash) noexcept {
		return find(key);
	}

	/// TODO use hash
	template<class K>
	constexpr inline const_iterator find(const K &key,
	                                     const std::size_t hash) const noexcept {
		return const_cast<sherwood_v3_table *>(this)->find(key);
	}

	constexpr inline size_t count(const FindKey &key) const noexcept {
		return find(key) == end() ? 0 : 1;
	}


	constexpr inline std::pair<iterator, iterator> equal_range(const FindKey &key) noexcept {
		iterator found = find(key);
		if (found == end())
			return {found, found};
		else
			return {found, std::next(found)};
	}

	constexpr inline std::pair<const_iterator, const_iterator> equal_range(const FindKey &key) const noexcept {
		const_iterator found = find(key);
		if (found == end())
			return {found, found};
		else
			return {found, std::next(found)};
	}

	template<typename Key, typename... Args>
	constexpr std::pair<iterator, bool> emplace(Key &&key, Args &&...args) noexcept {
		size_t index = hash_policy.bucket_for_hash(hash_object(key));
		EntryPointer current_entry = entries + ptrdiff_t(index);
		int8_t distance_from_desired = 0;
		for (; current_entry->distance_from_desired >= distance_from_desired; ++current_entry, ++distance_from_desired) {
			if (compares_equal(key, current_entry->value))
				return {{current_entry}, false};
		}

		return emplace_new_key(distance_from_desired, current_entry, std::forward<Key>(key), std::forward<Args>(args)...);
	}

	constexpr inline std::pair<iterator, bool> insert(const value_type &value) noexcept {
		return emplace(value);
	}

	constexpr inline std::pair<iterator, bool> insert(value_type &&value) noexcept {
		return emplace(std::move(value));
	}

	template<typename... Args>
	constexpr inline iterator emplace_hint(const_iterator, Args &&...args) noexcept {
		return emplace(std::forward<Args>(args)...).first;
	}

	constexpr inline iterator insert(const_iterator, const value_type &value) noexcept {
		return emplace(value).first;
	}

	constexpr inline iterator insert(const_iterator, value_type &&value) noexcept {
		return emplace(std::move(value)).first;
	}

	template<typename It>
	constexpr inline void insert(It begin, It end) noexcept {
		for (; begin != end; ++begin) {
			emplace(*begin);
		}
	}

	constexpr inline void insert(std::initializer_list<value_type> il) noexcept {
		insert(il.begin(), il.end());
	}

	constexpr void rehash(size_t num_buckets) noexcept {
		num_buckets = std::max(num_buckets, static_cast<size_t>(std::ceil(num_elements / static_cast<double>(_max_load_factor))));
		if (num_buckets == 0) {
			reset_to_empty_state();
			return;
		}

		auto new_prime_index = hash_policy.next_size_over(num_buckets);
		if (num_buckets == bucket_count())
			return;
		int8_t new_max_lookups = compute_max_lookups(num_buckets);
		EntryPointer new_buckets(AllocatorTraits::allocate(*this, num_buckets + new_max_lookups));
		EntryPointer special_end_item = new_buckets + static_cast<ptrdiff_t>(num_buckets + new_max_lookups - 1);

		for (EntryPointer it = new_buckets; it != special_end_item; ++it)
			it->distance_from_desired = -1;

		special_end_item->distance_from_desired = Entry::special_end_value;
		std::swap(entries, new_buckets);
		std::swap(num_slots_minus_one, num_buckets);
		--num_slots_minus_one;
		hash_policy.commit(new_prime_index);
		int8_t old_max_lookups = max_lookups;
		max_lookups = new_max_lookups;
		num_elements = 0;
		for (EntryPointer it = new_buckets, end = it + static_cast<ptrdiff_t>(num_buckets + old_max_lookups); it != end; ++it) {
			if (it->has_value()) {
				emplace(std::move(it->value));
				it->destroy_value();
			}
		}
		deallocate_data(new_buckets, num_buckets, old_max_lookups);
	}

	constexpr inline void reserve(size_t num_elements) noexcept {
		size_t required_buckets = num_buckets_for_reserve(num_elements);
		if (required_buckets > bucket_count())
			rehash(required_buckets);
	}

	template<class K>
	constexpr inline size_type erase(const K &key) noexcept {
		ASSERT(false);
		return 0;
	}

	// the return value is a type that can be converted to an iterator
	// the reason for doing this is that it's not free to find the
	// iterator pointing at the next element. if you care about the
	// next iterator, turn the return value into an iterator
	constexpr inline convertible_to_iterator erase(const_iterator to_erase) noexcept {
		EntryPointer current = to_erase.current;
		current->destroy_value();
		--num_elements;
		for (EntryPointer next = current + ptrdiff_t(1); !next->is_at_desired_position(); ++current, ++next) {
			current->emplace(next->distance_from_desired - 1, std::move(next->value));
			next->destroy_value();
		}
		return {to_erase.current};
	}

	iterator erase(const_iterator begin_it, const_iterator end_it) {
		if (begin_it == end_it)
			return {begin_it.current};
		for (EntryPointer it = begin_it.current, end = end_it.current; it != end; ++it) {
			if (it->has_value()) {
				it->destroy_value();
				--num_elements;
			}
		}
		if (end_it == this->end())
			return this->end();
		ptrdiff_t num_to_move = std::min(static_cast<ptrdiff_t>(end_it.current->distance_from_desired), end_it.current - begin_it.current);
		EntryPointer to_return = end_it.current - num_to_move;
		for (EntryPointer it = end_it.current; !it->is_at_desired_position();) {
			EntryPointer target = it - num_to_move;
			target->emplace(it->distance_from_desired - num_to_move, std::move(it->value));
			it->destroy_value();
			++it;
			num_to_move = std::min(static_cast<ptrdiff_t>(it->distance_from_desired), num_to_move);
		}
		return {to_return};
	}

	size_t erase(const FindKey &key) {
		auto found = find(key);
		if (found == end())
			return 0;
		else {
			erase(found);
			return 1;
		}
	}

	constexpr void clear() noexcept {
		for (EntryPointer it = entries, end = it + static_cast<ptrdiff_t>(num_slots_minus_one + max_lookups); it != end; ++it) {
			if (it->has_value())
				it->destroy_value();
		}
		num_elements = 0;
	}

	constexpr inline void shrink_to_fit() noexcept {
		rehash_for_other_container(*this);
	}

	constexpr void swap(sherwood_v3_table &other) noexcept {
		using std::swap;
		swap_pointers(other);
		swap(static_cast<ArgumentHash &>(*this), static_cast<ArgumentHash &>(other));
		swap(static_cast<ArgumentEqual &>(*this), static_cast<ArgumentEqual &>(other));
		if (AllocatorTraits::propagate_on_container_swap::value)
			swap(static_cast<EntryAlloc &>(*this), static_cast<EntryAlloc &>(other));
	}

	[[nodiscard]] constexpr inline size_t size() const noexcept {
		return num_elements;
	}

	[[nodiscard]] constexpr inline size_t max_size() const noexcept {
		return (AllocatorTraits::max_size(*this)) / sizeof(Entry);
	}

	[[nodiscard]] constexpr inline size_t bucket_count() const noexcept {
		return num_slots_minus_one ? num_slots_minus_one + 1 : 0;
	}

	[[nodiscard]] constexpr inline size_type max_bucket_count() const noexcept {
		return (AllocatorTraits::max_size(*this) - min_lookups) / sizeof(Entry);
	}

	[[nodiscard]] constexpr inline size_t bucket(const FindKey &key) const noexcept {
		return hash_policy.bucket_for_hash(hash_object(key));
	}
	[[nodiscard]] constexpr inline float load_factor() const noexcept {
		size_t buckets = bucket_count();
		if (buckets)
			return static_cast<float>(num_elements) / bucket_count();
		else
			return 0;
	}

	constexpr inline void max_load_factor(float value) noexcept {
		_max_load_factor = value;
	}

	[[nodiscard]] constexpr inline float max_load_factor() const noexcept {
		return _max_load_factor;
	}

	[[nodiscard]] constexpr inline bool empty() const noexcept {
		return num_elements == 0;
	}

private:
	EntryPointer entries = Entry::empty_default_table();
	size_t num_slots_minus_one = 0;
	// TODO generic

	cryptanalysislib::hh::fibonacci_growth_policy hash_policy;
	int8_t max_lookups = min_lookups - 1;
	float _max_load_factor = 0.5f;
	size_t num_elements = 0;

	constexpr static int8_t compute_max_lookups(size_t num_buckets) noexcept {
		int8_t desired = bits_log2(num_buckets);
		return std::max(min_lookups, desired);
	}

	[[nodiscard]] constexpr inline size_t num_buckets_for_reserve(size_t num_elements) const noexcept {
		return static_cast<size_t>(std::ceil(num_elements / std::min(0.5, static_cast<double>(_max_load_factor))));
	}

	constexpr inline void rehash_for_other_container(const sherwood_v3_table &other) noexcept {
		rehash(std::min(num_buckets_for_reserve(other.size()), other.bucket_count()));
	}

	constexpr inline void swap_pointers(sherwood_v3_table &other) {
		using std::swap;
		swap(hash_policy, other.hash_policy);
		swap(entries, other.entries);
		swap(num_slots_minus_one, other.num_slots_minus_one);
		swap(num_elements, other.num_elements);
		swap(max_lookups, other.max_lookups);
		swap(_max_load_factor, other._max_load_factor);
	}

	template<typename Key, typename... Args>
	std::pair<iterator, bool> __attribute__((noinline))
	emplace_new_key(int8_t distance_from_desired, EntryPointer current_entry, Key &&key, Args &&...args) {
		using std::swap;
		if (num_slots_minus_one == 0 || distance_from_desired == max_lookups || num_elements + 1 > (num_slots_minus_one + 1) * static_cast<double>(_max_load_factor)) {
			grow();
			return emplace(std::forward<Key>(key), std::forward<Args>(args)...);
		} else if (current_entry->is_empty()) {
			current_entry->emplace(distance_from_desired, std::forward<Key>(key), std::forward<Args>(args)...);
			++num_elements;
			return {{current_entry}, true};
		}
		value_type to_insert(std::forward<Key>(key), std::forward<Args>(args)...);
		swap(distance_from_desired, current_entry->distance_from_desired);
		swap(to_insert, current_entry->value);
		iterator result = {current_entry};
		for (++distance_from_desired, ++current_entry;; ++current_entry) {
			if (current_entry->is_empty()) {
				current_entry->emplace(distance_from_desired, std::move(to_insert));
				++num_elements;
				return {result, true};
			} else if (current_entry->distance_from_desired < distance_from_desired) {
				swap(distance_from_desired, current_entry->distance_from_desired);
				swap(to_insert, current_entry->value);
				++distance_from_desired;
			} else {
				++distance_from_desired;
				if (distance_from_desired == max_lookups) {
					swap(to_insert, result.current->value);
					grow();
					return emplace(std::move(to_insert));
				}
			}
		}
	}

	constexpr inline void grow() noexcept {
		rehash(std::max(size_t(4), 2 * bucket_count()));
	}

	constexpr inline void deallocate_data(const EntryPointer begin,
	                                      const size_t num_slots_minus_one,
	                                      const int8_t max_lookups) noexcept {
		if (begin != Entry::empty_default_table()) {
			AllocatorTraits::deallocate(*this, begin, num_slots_minus_one + max_lookups + 1);
		}
	}

	constexpr inline void reset_to_empty_state() noexcept {
		deallocate_data(entries, num_slots_minus_one, max_lookups);
		entries = Entry::empty_default_table();
		num_slots_minus_one = 0;
		hash_policy.reset();
		max_lookups = min_lookups - 1;
	}

	template<typename U>
	[[nodiscard]] constexpr inline size_t hash_object(const U &key) noexcept {
		return static_cast<Hasher &>(*this)(key);
	}

	template<typename U>
	[[nodiscard]] constexpr inline size_t hash_object(const U &key) const noexcept {
		return static_cast<const Hasher &>(*this)(key);
	}

	template<typename L, typename R>
	[[nodiscard]] constexpr inline bool compares_equal(const L &lhs, const R &rhs) noexcept {
		return static_cast<Equal &>(*this)(lhs, rhs);
	}

	struct convertible_to_iterator {
		EntryPointer it;

		operator iterator() {
			if (it->has_value())
				return {it};
			else
				return ++iterator{it};
		}

		operator const_iterator() {
			if (it->has_value())
				return {it};
			else
				return ++const_iterator{it};
		}
	};
};


template<typename T, typename Allocator>
struct sherwood_v10_entry {
	sherwood_v10_entry() {
	}
	~sherwood_v10_entry() {
	}

	using EntryPointer = typename std::allocator_traits<typename std::allocator_traits<Allocator>::template rebind_alloc<sherwood_v10_entry>>::pointer;

	EntryPointer next = nullptr;
	union {
		T value;
	};

	static EntryPointer *empty_pointer() {
		static EntryPointer result[3] = {EntryPointer(nullptr) + ptrdiff_t(1), nullptr, nullptr};
		return result + 1;
	}
};

template<typename T, typename FindKey, typename ArgumentHash, typename Hasher, typename ArgumentEqual, typename Equal, typename ArgumentAlloc, typename EntryAlloc, typename BucketAllocator>
class sherwood_v10_table : private EntryAlloc, private Hasher, private Equal, private BucketAllocator {
	using Entry = sherwood_v10_entry<T, ArgumentAlloc>;
	using AllocatorTraits = std::allocator_traits<EntryAlloc>;
	using BucketAllocatorTraits = std::allocator_traits<BucketAllocator>;
	using EntryPointer = typename AllocatorTraits::pointer;
	struct convertible_to_iterator;

public:
	using value_type = T;
	using size_type = size_t;
	using difference_type = std::ptrdiff_t;
	using hasher = ArgumentHash;
	using key_equal = ArgumentEqual;
	using allocator_type = EntryAlloc;
	using reference = value_type &;
	using const_reference = const value_type &;
	using pointer = value_type *;
	using const_pointer = const value_type *;

	constexpr sherwood_v10_table() { }

	constexpr explicit sherwood_v10_table(size_type bucket_count, const ArgumentHash &hash = ArgumentHash(), const ArgumentEqual &equal = ArgumentEqual(), const ArgumentAlloc &alloc = ArgumentAlloc())
	    : EntryAlloc(alloc), Hasher(hash), Equal(equal), BucketAllocator(alloc) {
		rehash(bucket_count);
	}

	constexpr sherwood_v10_table(size_type bucket_count, const ArgumentAlloc &alloc)
	    : sherwood_v10_table(bucket_count, ArgumentHash(), ArgumentEqual(), alloc) {}

	constexpr sherwood_v10_table(size_type bucket_count, const ArgumentHash &hash, const ArgumentAlloc &alloc)
	    : sherwood_v10_table(bucket_count, hash, ArgumentEqual(), alloc) {}

	constexpr explicit sherwood_v10_table(const ArgumentAlloc &alloc)
	    : EntryAlloc(alloc), BucketAllocator(alloc) {
	}
	template<typename It>
	constexpr sherwood_v10_table(It first, It last, size_type bucket_count = 0, const ArgumentHash &hash = ArgumentHash(), const ArgumentEqual &equal = ArgumentEqual(), const ArgumentAlloc &alloc = ArgumentAlloc())
	    : sherwood_v10_table(bucket_count, hash, equal, alloc) {
		insert(first, last);
	}

	template<typename It>
	constexpr sherwood_v10_table(It first, It last, size_type bucket_count, const ArgumentAlloc &alloc)
	    : sherwood_v10_table(first, last, bucket_count, ArgumentHash(), ArgumentEqual(), alloc) {
	}

	template<typename It>
	constexpr sherwood_v10_table(It first, It last, size_type bucket_count, const ArgumentHash &hash, const ArgumentAlloc &alloc)
	    : sherwood_v10_table(first, last, bucket_count, hash, ArgumentEqual(), alloc) {
	}

	constexpr sherwood_v10_table(std::initializer_list<T> il, size_type bucket_count = 0, const ArgumentHash &hash = ArgumentHash(), const ArgumentEqual &equal = ArgumentEqual(), const ArgumentAlloc &alloc = ArgumentAlloc())
	    : sherwood_v10_table(bucket_count, hash, equal, alloc) {
		if (bucket_count == 0) {
			reserve(il.size());
		}
		insert(il.begin(), il.end());
	}

	constexpr sherwood_v10_table(std::initializer_list<T> il, size_type bucket_count, const ArgumentAlloc &alloc)
	    : sherwood_v10_table(il, bucket_count, ArgumentHash(), ArgumentEqual(), alloc) {
	}

	constexpr sherwood_v10_table(std::initializer_list<T> il, size_type bucket_count, const ArgumentHash &hash, const ArgumentAlloc &alloc)
	    : sherwood_v10_table(il, bucket_count, hash, ArgumentEqual(), alloc) {
	}

	constexpr sherwood_v10_table(const sherwood_v10_table &other)
	    : sherwood_v10_table(other, AllocatorTraits::select_on_container_copy_construction(other.get_allocator())) {
	}

	constexpr sherwood_v10_table(const sherwood_v10_table &other, const ArgumentAlloc &alloc)
	    : EntryAlloc(alloc), Hasher(other), Equal(other), BucketAllocator(alloc), _max_load_factor(other._max_load_factor) {
		try {
			rehash_for_other_container(other);
			insert(other.begin(), other.end());
		} catch (...) {
			clear();
			deallocate_data();
			throw;
		}
	}

	constexpr sherwood_v10_table(sherwood_v10_table &&other) noexcept
	    : EntryAlloc(std::move(other)), Hasher(std::move(other)), Equal(std::move(other)), BucketAllocator(std::move(other)), _max_load_factor(other._max_load_factor) {
		swap_pointers(other);
	}

	constexpr sherwood_v10_table(sherwood_v10_table &&other, const ArgumentAlloc &alloc) noexcept
	    : EntryAlloc(alloc), Hasher(std::move(other)), Equal(std::move(other)), BucketAllocator(alloc), _max_load_factor(other._max_load_factor) {
		swap_pointers(other);
	}

	constexpr sherwood_v10_table &operator=(const sherwood_v10_table &other) {
		if (this == std::addressof(other)) {
			return *this;
		}

		clear();
		static_assert(AllocatorTraits::propagate_on_container_copy_assignment::value == BucketAllocatorTraits::propagate_on_container_copy_assignment::value, "The allocators have to behave the same way");
		if (AllocatorTraits::propagate_on_container_copy_assignment::value) {
			if (static_cast<EntryAlloc &>(*this) != static_cast<const EntryAlloc &>(other) || static_cast<BucketAllocator &>(*this) != static_cast<const BucketAllocator &>(other)) {
				reset_to_empty_state();
			}
			AssignIfTrue<EntryAlloc, AllocatorTraits::propagate_on_container_copy_assignment::value>()(*this, other);
			AssignIfTrue<BucketAllocator, BucketAllocatorTraits::propagate_on_container_copy_assignment::value>()(*this, other);
		}
		_max_load_factor = other._max_load_factor;
		static_cast<Hasher &>(*this) = other;
		static_cast<Equal &>(*this) = other;
		rehash_for_other_container(other);
		insert(other.begin(), other.end());
		return *this;
	}

	sherwood_v10_table &operator=(sherwood_v10_table &&other) noexcept {
		static_assert(AllocatorTraits::propagate_on_container_move_assignment::value == BucketAllocatorTraits::propagate_on_container_move_assignment::value, "The allocators have to behave the same way");
		if (this == std::addressof(other))
			return *this;
		else if (AllocatorTraits::propagate_on_container_move_assignment::value) {
			clear();
			reset_to_empty_state();
			AssignIfTrue<EntryAlloc, AllocatorTraits::propagate_on_container_move_assignment::value>()(*this, std::move(other));
			AssignIfTrue<BucketAllocator, BucketAllocatorTraits::propagate_on_container_move_assignment::value>()(*this, std::move(other));
			swap_pointers(other);
		} else if (static_cast<EntryAlloc &>(*this) == static_cast<EntryAlloc &>(other) && static_cast<BucketAllocator &>(*this) == static_cast<BucketAllocator &>(other)) {
			swap_pointers(other);
		} else {
			clear();
			_max_load_factor = other._max_load_factor;
			rehash_for_other_container(other);
			for (T &elem: other)
				emplace(std::move(elem));
			other.clear();
		}
		static_cast<Hasher &>(*this) = std::move(other);
		static_cast<Equal &>(*this) = std::move(other);
		return *this;
	}

	~sherwood_v10_table() {
		clear();
		deallocate_data();
	}

	constexpr const allocator_type &get_allocator() const noexcept {
		return static_cast<const allocator_type &>(*this);
	}

	constexpr const ArgumentEqual &key_eq() const noexcept {
		return static_cast<const ArgumentEqual &>(*this);
	}

	constexpr const ArgumentHash &hash_function() const noexcept {
		return static_cast<const ArgumentHash &>(*this);
	}

	template<typename ValueType>
	struct templated_iterator {
		constexpr templated_iterator() {}
		constexpr templated_iterator(EntryPointer element, EntryPointer *bucket)
		    : current_element(element), current_bucket(bucket) {
		}

		EntryPointer current_element = nullptr;
		EntryPointer *current_bucket = nullptr;

		using iterator_category = std::forward_iterator_tag;
		using value_type = ValueType;
		using difference_type = ptrdiff_t;
		using pointer = ValueType *;
		using reference = ValueType &;

		friend bool operator==(const templated_iterator &lhs, const templated_iterator &rhs) {
			return lhs.current_element == rhs.current_element;
		}
		friend bool operator!=(const templated_iterator &lhs, const templated_iterator &rhs) {
			return !(lhs == rhs);
		}

		templated_iterator &operator++() {
			if (!current_element->next) {
				do {
					--current_bucket;
				} while (!*current_bucket);
				current_element = *current_bucket;
			} else
				current_element = current_element->next;
			return *this;
		}
		templated_iterator operator++(int) {
			templated_iterator copy(*this);
			++*this;
			return copy;
		}

		ValueType &operator*() const {
			return current_element->value;
		}
		ValueType *operator->() const {
			return std::addressof(current_element->value);
		}

		operator templated_iterator<const value_type>() const {
			return {current_element, current_bucket};
		}
	};
	using iterator = templated_iterator<value_type>;
	using const_iterator = templated_iterator<const value_type>;

	constexpr inline iterator begin() noexcept {
		EntryPointer *end = entries - 1;
		for (EntryPointer *it = entries + num_slots_minus_one; it != end; --it) {
			if (*it)
				return {*it, it};
		}
		return {*end, end};
	}

	constexpr inline const_iterator begin() const noexcept {
		return const_cast<sherwood_v10_table *>(this)->begin();
	}

	constexpr inline const_iterator cbegin() const noexcept {
		return begin();
	}

	constexpr inline iterator end() noexcept {
		EntryPointer *end = entries - 1;
		return {*end, end};
	}

	constexpr inline const_iterator end() const noexcept {
		EntryPointer *end = entries - 1;
		return {*end, end};
	}

	constexpr inline const_iterator cend() const noexcept {
		return end();
	}

	constexpr inline iterator find(const FindKey &key) noexcept {
		size_t index = hash_policy.bucket_for_hash(hash_object(key));
		EntryPointer *bucket = entries + ptrdiff_t(index);
		for (EntryPointer it = *bucket; it; it = it->next) {
			if (compares_equal(key, it->value))
				return {it, bucket};
		}
		return end();
	}
	const_iterator find(const FindKey &key) const {
		return const_cast<sherwood_v10_table *>(this)->find(key);
	}
	size_t count(const FindKey &key) const {
		return find(key) == end() ? 0 : 1;
	}
	std::pair<iterator, iterator> equal_range(const FindKey &key) {
		iterator found = find(key);
		if (found == end())
			return {found, found};
		else
			return {found, std::next(found)};
	}
	std::pair<const_iterator, const_iterator> equal_range(const FindKey &key) const {
		const_iterator found = find(key);
		if (found == end())
			return {found, found};
		else
			return {found, std::next(found)};
	}

	template<typename Key, typename... Args>
	std::pair<iterator, bool> emplace(Key &&key, Args &&...args) {
		size_t index = hash_policy.bucket_for_hash(hash_object(key));
		EntryPointer *bucket = entries + ptrdiff_t(index);
		for (EntryPointer it = *bucket; it; it = it->next) {
			if (compares_equal(key, it->value))
				return {{it, bucket}, false};
		}
		return emplace_new_key(bucket, std::forward<Key>(key), std::forward<Args>(args)...);
	}

	std::pair<iterator, bool> insert(const value_type &value) {
		return emplace(value);
	}
	std::pair<iterator, bool> insert(value_type &&value) {
		return emplace(std::move(value));
	}
	template<typename... Args>
	iterator emplace_hint(const_iterator, Args &&...args) {
		return emplace(std::forward<Args>(args)...).first;
	}
	iterator insert(const_iterator, const value_type &value) {
		return emplace(value).first;
	}
	iterator insert(const_iterator, value_type &&value) {
		return emplace(std::move(value)).first;
	}

	template<typename It>
	void insert(It begin, It end) {
		for (; begin != end; ++begin) {
			emplace(*begin);
		}
	}
	void insert(std::initializer_list<value_type> il) {
		insert(il.begin(), il.end());
	}

	void rehash(size_t num_buckets) {
		num_buckets = std::max(num_buckets, static_cast<size_t>(std::ceil(num_elements / static_cast<double>(_max_load_factor))));
		if (num_buckets == 0) {
			reset_to_empty_state();
			return;
		}
		auto new_prime_index = hash_policy.next_size_over(num_buckets);
		if (num_buckets == bucket_count())
			return;
		EntryPointer *new_buckets(&*BucketAllocatorTraits::allocate(*this, num_buckets + 1));
		EntryPointer *end_it = new_buckets + static_cast<ptrdiff_t>(num_buckets + 1);
		*new_buckets = EntryPointer(nullptr) + ptrdiff_t(1);
		++new_buckets;
		std::fill(new_buckets, end_it, nullptr);
		std::swap(entries, new_buckets);
		std::swap(num_slots_minus_one, num_buckets);
		--num_slots_minus_one;
		hash_policy.commit(new_prime_index);
		if (!num_buckets)
			return;

		for (EntryPointer *it = new_buckets, *end = it + static_cast<ptrdiff_t>(num_buckets + 1); it != end; ++it) {
			for (EntryPointer e = *it; e;) {
				EntryPointer next = e->next;
				size_t index = hash_policy.bucket_for_hash(hash_object(e->value));
				EntryPointer &new_slot = entries[index];
				e->next = new_slot;
				new_slot = e;
				e = next;
			}
		}
		BucketAllocatorTraits::deallocate(*this, new_buckets - 1, num_buckets + 2);
	}

	void reserve(size_t num_elements) {
		if (!num_elements)
			return;
		num_elements = static_cast<size_t>(std::ceil(num_elements / static_cast<double>(_max_load_factor)));
		if (num_elements > bucket_count())
			rehash(num_elements);
	}

	// the return value is a type that can be converted to an iterator
	// the reason for doing this is that it's not free to find the
	// iterator pointing at the next element. if you care about the
	// next iterator, turn the return value into an iterator
	convertible_to_iterator erase(const_iterator to_erase) {
		--num_elements;
		AllocatorTraits::destroy(*this, std::addressof(to_erase.current_element->value));
		EntryPointer *previous = to_erase.current_bucket;
		while (*previous != to_erase.current_element) {
			previous = &(*previous)->next;
		}
		*previous = to_erase.current_element->next;
		AllocatorTraits::deallocate(*this, to_erase.current_element, 1);
		return {*previous, to_erase.current_bucket};
	}

	convertible_to_iterator erase(const_iterator begin_it, const_iterator end_it) {
		while (begin_it.current_bucket != end_it.current_bucket) {
			begin_it = erase(begin_it);
		}
		EntryPointer *bucket = begin_it.current_bucket;
		EntryPointer *previous = bucket;
		while (*previous != begin_it.current_element)
			previous = &(*previous)->next;
		while (*previous != end_it.current_element) {
			--num_elements;
			EntryPointer entry = *previous;
			AllocatorTraits::destroy(*this, std::addressof(entry->value));
			*previous = entry->next;
			AllocatorTraits::deallocate(*this, entry, 1);
		}
		return {*previous, bucket};
	}

	size_t erase(const FindKey &key) {
		auto found = find(key);
		if (found == end())
			return 0;
		else {
			erase(found);
			return 1;
		}
	}

	void clear() {
		if (!num_slots_minus_one)
			return;
		for (EntryPointer *it = entries, *end = it + static_cast<ptrdiff_t>(num_slots_minus_one + 1); it != end; ++it) {
			for (EntryPointer e = *it; e;) {
				EntryPointer next = e->next;
				AllocatorTraits::destroy(*this, std::addressof(e->value));
				AllocatorTraits::deallocate(*this, e, 1);
				e = next;
			}
			*it = nullptr;
		}
		num_elements = 0;
	}

	void swap(sherwood_v10_table &other) {
		using std::swap;
		swap_pointers(other);
		swap(static_cast<ArgumentHash &>(*this), static_cast<ArgumentHash &>(other));
		swap(static_cast<ArgumentEqual &>(*this), static_cast<ArgumentEqual &>(other));
		if (AllocatorTraits::propagate_on_container_swap::value)
			swap(static_cast<EntryAlloc &>(*this), static_cast<EntryAlloc &>(other));
		if (BucketAllocatorTraits::propagate_on_container_swap::value)
			swap(static_cast<BucketAllocator &>(*this), static_cast<BucketAllocator &>(other));
	}

	size_t size() const {
		return num_elements;
	}
	size_t max_size() const {
		return (AllocatorTraits::max_size(*this)) / sizeof(Entry);
	}
	size_t bucket_count() const {
		return num_slots_minus_one + 1;
	}
	size_type max_bucket_count() const {
		return (AllocatorTraits::max_size(*this) - 1) / sizeof(Entry);
	}
	size_t bucket(const FindKey &key) const {
		return hash_policy.bucket_for_hash(hash_object(key));
	}
	float load_factor() const {
		size_t buckets = bucket_count();
		if (buckets)
			return static_cast<float>(num_elements) / bucket_count();
		else
			return 0;
	}
	void max_load_factor(float value) {
		_max_load_factor = value;
	}
	float max_load_factor() const {
		return _max_load_factor;
	}

	bool empty() const {
		return num_elements == 0;
	}

private:
	EntryPointer *entries = Entry::empty_pointer();
	size_t num_slots_minus_one = 0;
	cryptanalysislib::hh::fibonacci_growth_policy hash_policy;
	float _max_load_factor = 1.0f;
	size_t num_elements = 0;

	void rehash_for_other_container(const sherwood_v10_table &other) {
		reserve(other.size());
	}

	void swap_pointers(sherwood_v10_table &other) {
		using std::swap;
		swap(hash_policy, other.hash_policy);
		swap(entries, other.entries);
		swap(num_slots_minus_one, other.num_slots_minus_one);
		swap(num_elements, other.num_elements);
		swap(_max_load_factor, other._max_load_factor);
	}

	template<typename... Args>
	std::pair<iterator, bool>
	emplace_new_key(EntryPointer *bucket, Args &&...args) {
		using std::swap;
		if (is_full()) {
			grow();
			return emplace(std::forward<Args>(args)...);
		} else {
			EntryPointer new_entry = AllocatorTraits::allocate(*this, 1);
			try {
				AllocatorTraits::construct(*this, std::addressof(new_entry->value), std::forward<Args>(args)...);
			} catch (...) {
				AllocatorTraits::deallocate(*this, new_entry, 1);
				throw;
			}
			++num_elements;
			new_entry->next = *bucket;
			*bucket = new_entry;
			return {{new_entry, bucket}, true};
		}
	}

	bool is_full() const {
		if (!num_slots_minus_one)
			return true;
		else
			return num_elements + 1 > (num_slots_minus_one + 1) * static_cast<double>(_max_load_factor);
	}

	void grow() {
		rehash(std::max(size_t(4), 2 * bucket_count()));
	}

	void deallocate_data() {
		if (entries != Entry::empty_pointer()) {
			BucketAllocatorTraits::deallocate(*this, entries - 1, num_slots_minus_one + 2);
		}
	}

	void reset_to_empty_state() {
		deallocate_data();
		entries = Entry::empty_pointer();
		num_slots_minus_one = 0;
		hash_policy.reset();
	}

	template<typename U>
	size_t hash_object(const U &key) {
		return static_cast<Hasher &>(*this)(key);
	}
	template<typename U>
	size_t hash_object(const U &key) const {
		return static_cast<const Hasher &>(*this)(key);
	}
	template<typename L, typename R>
	bool compares_equal(const L &lhs, const R &rhs) {
		return static_cast<Equal &>(*this)(lhs, rhs);
	}

	struct convertible_to_iterator {
		EntryPointer element;
		EntryPointer *bucket;

		operator iterator() {
			if (element)
				return {element, bucket};
			else {
				do {
					--bucket;
				} while (!*bucket);
				return {*bucket, bucket};
			}
		}
		operator const_iterator() {
			if (element)
				return {element, bucket};
			else {
				do {
					--bucket;
				} while (!*bucket);
				return {*bucket, bucket};
			}
		}
	};
};

#endif//CRYPTANALYSISLIB_SKA_FLAT_H
