#ifndef SMALLSECRETLWE_SIMPLE_H
#define SMALLSECRETLWE_SIMPLE_H

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <utility>

#include "atomic_primitives.h"
#include "container/hashmap/common.h"
#include "helper.h"
#include "sort/sorting_network/common.h"

///
struct SimpleHashMapConfig {
public:
	/// NOTE no constructor as not neded
	const uint64_t bucketsize;
	const uint64_t nrbuckets;
	const uint32_t threads;
};

/// NOTE: Only the indices of the list entries can be saved in here.
/// \tparam keyType		base type of the input keys
/// \tparam valueType	base type of the values to save in the list
/// \tparam config		`SimpleHashmapConfig` object
/// \tparam HashFkt		internal hash function to use.
template<
        typename keyType,
        typename valueType,
        const SimpleHashMapConfig &config,
        class Hash>
class SimpleHashMap {
public:
	using data_type = valueType;
	using key_type = keyType;
	using index_type = size_t;
	using load_type = TypeTemplate<config.bucketsize>;

	Hash hashclass = Hash{};


	// size per bucket
	constexpr static size_t bucketsize = config.bucketsize;

	// number of buckets
	constexpr static size_t nrbuckets = config.nrbuckets;

	// total number of elements in the HM
	constexpr static size_t total_size = bucketsize * nrbuckets;

	constexpr static uint32_t threads = config.threads;
	constexpr static bool multithreaded = config.threads > 1u;

	/// constructor. Zero initializing everything
	constexpr SimpleHashMap() noexcept : __internal_hashmap_array(),
	                                     __internal_load_array() {}

	/// the simple hashmap ignores the thread id.
	/// Which is nice.
	/// \param e key element (hashed down = index within the internal array)
	/// \param value element to insert
	/// \param tid (ignored) can be anything
	/// \return
	constexpr inline void insert(const keyType &e,
	                             const valueType &value,
	                             const uint32_t tid) noexcept {
		(void) tid;
		insert(e, value);
	}

	/// hashes down `e` (Element) to an index where to store
	/// the element.
	/// NOTE: Boundary checks are performed in debug mode.
	/// \param e element to insert
	/// \return nothing
	constexpr inline void insert(const keyType &e, const valueType &value) noexcept {
		const size_t index = hash(e);
		ASSERT(index < nrbuckets);

		size_t load;
		if constexpr (multithreaded) {
			load = FAA(__internal_load_array + index, 1);
			// early exit and reset
			if (load >= bucketsize) {
				__internal_load_array[index] = bucketsize;
				return;
			}
		} else {
			load = __internal_load_array[index];

			// early exit, if it's already full
			if (load == bucketsize) {
				return;
			}
		}


		// just some debugging checks
		ASSERT(load < bucketsize);
		if constexpr (!multithreaded) {
			__internal_load_array[index] += 1;
		}

		/// NOTE: this store never needs to be atomic, as the position was
		/// computed atomically.
		if constexpr (std::is_bounded_array_v<data_type>) {
			memcpy(__internal_hashmap_array[index * bucketsize + load], value, sizeof(data_type));
		} else {
			__internal_hashmap_array[index * bucketsize + load] = value;
		}
	}

	///
	/// \return
	constexpr inline valueType *ptr() noexcept {
		return __internal_hashmap_array;
	}

	///
	/// \param i
	/// \return
	using inner_data_type = std::remove_all_extents<data_type>::type;
	using ret_type = typename std::conditional<std::is_bounded_array_v<data_type>,
	                                           inner_data_type *,
	                                           valueType>::type;
	constexpr inline ret_type ptr(const index_type i) noexcept {
		ASSERT(i < total_size);
		if constexpr (std::is_bounded_array_v<data_type>) {
			return (inner_data_type *) __internal_hashmap_array[i];
		} else {
			return (valueType) __internal_hashmap_array[i];
		}
	}

	/// calls ptr
	constexpr inline ret_type operator[](const index_type i) noexcept {
		return ptr(i);
	}

	/// Quite same to `probe` but instead it will directly return
	/// the position of the element.
	/// \param e Element to hash down.
	/// \return the position within the internal const_array of `e`
	constexpr inline index_type find(const keyType &e) const noexcept {
		const index_type index = hash(e);
		ASSERT(index < nrbuckets);
		// return the index instead of the actual element, to
		// reduce the size of the returned element.
		return index * bucketsize;
	}

	///
	/// \param e
	/// \param __load
	/// \return
	constexpr inline index_type find(const keyType &e, load_type &__load) const noexcept {
		const index_type index = hash(e);
		ASSERT(index < nrbuckets);
		__load = __internal_load_array[index];
		// return the index instead of the actual element, to
		// reduce the size of the returned element.
		return index * bucketsize;
	}

	///
	/// \param e
	/// \param __load
	/// \return
	constexpr inline index_type find_without_hash(const keyType &e, load_type &__load) const noexcept {
		ASSERT(e < nrbuckets);
		__load = __internal_load_array[e];
		return e * bucketsize;
	}

	/// match the api
	constexpr inline size_t hash(const keyType &e) const noexcept {
		return hashclass(e);
	}

	/// NOTE: can be called with only a single thread
	/// overwrites the internal data const_array
	/// with zero initialized elements.
	constexpr inline void clear() noexcept {
		memset(__internal_load_array, 0, nrbuckets * sizeof(load_type));
	}

	/// multithreaded clear
	/// NOTE: cannot be `constexpr`
	inline void clear(uint32_t tid) noexcept {
		if constexpr (config.threads == 1) {
			(void) tid;
			clear();
			return;
		}

		const size_t start = tid * nrbuckets / config.threads;
		const size_t bytes = nrbuckets * sizeof(load_type) / config.threads;
		memset(__internal_load_array + start, 0, bytes);
#pragma omp barrier
	}

	/// internal function
	constexpr inline index_type load_without_hash(const keyType &e) const noexcept {
		ASSERT(e < nrbuckets);
		return __internal_load_array[e];
	}

	/// returns the load of the bucket, where the given element would hashed into
	/// \param e bucket/bucket of the element e
	/// \return the load
	constexpr inline index_type load(const keyType &e) const noexcept {
		const size_t index = hash(e);
		ASSERT(index < nrbuckets);
		return __internal_load_array[index];
	}

	/// NOTE: only single threaded.
	/// \return the load, the number of buckets which are not empty
	constexpr inline index_type load() const noexcept {
		index_type ret = index_type(0);
		for (index_type i = 0; i < nrbuckets; i++) {
			ret += __internal_load_array[i];
		}

		return ret;
	}

	/// prints some basic information about the hashmap
	constexpr void print() const noexcept {
		std::cout << "total_size:" << total_size
		          << ", total_size_byts:" << sizeof(__internal_hashmap_array) + sizeof(__internal_load_array)
		          << ", nrbuckets:" << nrbuckets
		          << ", bucketsize:" << bucketsize
		          << ", sizeof(keyType): " << sizeof(keyType)
		          << ", sizeof(valueType): " << sizeof(valueType)
		          << ", sizeof(load_type): " << sizeof(load_type)
		          << ", sizeof(index_type): " << sizeof(index_type)
		          << ", multithreaded: " << multithreaded;

		if constexpr (multithreaded) {
			std::cout << " (Threads:" << config.threads;
		}

		std::cout << std::endl;
	}

	// internal const_array
	alignas(1024) data_type __internal_hashmap_array[total_size];
	alignas(1024) load_type __internal_load_array[nrbuckets];
};

#endif//SMALLSECRETLWE_SIMPLE_H
