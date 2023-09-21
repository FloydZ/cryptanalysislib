#ifndef SMALLSECRETLWE_SIMPLE_H
#define SMALLSECRETLWE_SIMPLE_H

#include <cstdint>
#include <cstdlib>
#include <utility>
#include <cstddef>

#include "static_sort.h"
#include "container/hashmap/common.h"
#include "helper.h"

///
class SimpleHashMapConfig {
private:
	// Disable the simple constructor
	constexpr SimpleHashMapConfig() : bucketsize(0), nrbuckets(0) {};

public:
	const uint64_t bucketsize;
	const uint64_t nrbuckets;

	/// only activate the constexpr constructor
	/// \param bucketsize
	/// \param nrbuckets
	constexpr SimpleHashMapConfig(const uint64_t bucketsize,
	                              const uint64_t nrbuckets) :
			bucketsize(bucketsize), nrbuckets(nrbuckets) {};
};

/// NOTE: Only valid for SternMO, this means that no `l-part` whatsoever is saved.
/// NOTE: Only the indices of the list entries can be saved in here.
/// \tparam T			base type of the l-part (unused)
/// \tparam listType	base type of the list elements
/// \tparam config		`SimpleHashmapConfig` object
/// \tparam HashFkt		internal hash function to use.
template<typename T,
		typename listType,
		const SimpleHashMapConfig &config,
		size_t (* HashFkt)(const uint64_t)>
class SimpleHashMap {
	using data_type          = listType;
	using internal_data_type = T;
	using index_type         = size_t;
	using load_type          = uint32_t; // TODO optimize

	// sadly in the current form useless
	using cache_type = std::pair<T, listType>;
	constexpr static bool use_insert_cache = false;
	//constexpr static uint16_t insert_cache_size = 32;
	//cache_type insert_cache[insert_cache_size];
	//uint16_t insert_cache_counter = 0;
	//StaticSort<insert_cache_size> staticSort;

	// size per bucket
	constexpr static size_t bucketsize = config.bucketsize;

	// number of buckets
	constexpr static size_t nrbuckets = config.nrbuckets;

	// total number of elements in the HM
	constexpr static size_t total_size = bucketsize * nrbuckets;
public:
	/// constructor. Zero initializing everything
	constexpr SimpleHashMap() noexcept :
			__internal_hashmap_array(),
			__internal_load_array() {}


	///
	/// \return
	constexpr void insert_cache_flush() noexcept {
		if constexpr (!use_insert_cache) {
			return;
		}

		// first sort the shit
		//staticSort(insert_cache, [](const cache_type &a1, const cache_type &a2) {
		//	return a1.first < a2.first;
		//});

		//for (uint16_t i = 0; i < insert_cache_counter; i++) {
		//	const auto index = insert_cache[i].first;
		//	const auto list_index = insert_cache[i].second;
		//	size_t load = __internal_load_array[index];

		//	// early exit
		//	if (load == bucketsize-1) {
		//		continue;
		//	}

		//	__internal_load_array[index] += 1;
		//	ASSERT(load < bucketsize);

		//	__internal_hashmap_array[index*bucketsize + load] = list_index;
		//}

		//insert_cache_counter = 0;
	}


	/// hashes down `e` (Element) to an index where to store
	/// the element.
	/// NOTE: Boundary checks are performed in debug mode.
	/// \param e element to insert
	/// \return nothing
	constexpr void insert(const T &e, const listType list_index) noexcept {
		// hash down the element to the index
		const size_t index = HashFkt(e);
		ASSERT(index < nrbuckets);

		if constexpr (use_insert_cache) {
			//insert_cache[insert_cache_counter++] = cache_type{index, list_index};
			//if (insert_cache_counter >= insert_cache_size){
			//	insert_cache_flush();
			//}
			ASSERT(false);
			return;
		} else {
			size_t load = __internal_load_array[index];
			// early exit, if it's already full
			if (load == bucketsize)
				return ;

			//_mm_stream_si32((int *)__internal_load_array.data() + index, load+1);
			__internal_load_array[index] += 1;
			ASSERT(load < bucketsize);

			//_mm_stream_si32((int *)__internal_hashmap_array.data() + index*bucketsize + load, list_index);
			__internal_hashmap_array[index*bucketsize + load] = list_index;
		}
	}

	/// Quite same to `probe` but instead it will directly return
	/// the position of the element.
	/// \param e Element to hash down.
	/// \return the position within the internal array of `e`
	constexpr index_type find(const T &e) const noexcept {
		const index_type index = HashFkt(e);
		ASSERT(index < nrbuckets);
		// return the index instead of the actual element, to
		// reduce the size of the returned element.
		return index*nrbuckets;
	}

	/// prints the content of each bucket
	/// it with one thread.
	/// \return nothing
	constexpr void print() const noexcept {
		for (index_type i = 0; i < total_size; i++) {
			printf("%d %d\n",
			       __internal_hashmap_array[i].first,
			       __internal_hashmap_array[i].second);
		}
	}

	/// prints a single index
	/// \param index the element to pirnt
	/// \return nothing
	constexpr void print(const size_t index) const noexcept {
		ASSERT(index < total_size);
		printf("%d %d\n",
		       __internal_hashmap_array[index].first,
		       __internal_hashmap_array[index].second);
	}

	/// NOTE: can be called with only a single thread
	/// overwrites the internal data array
	/// with zero initialized elements.
	constexpr void clear() noexcept {
		memset(__internal_load_array, 0, nrbuckets*sizeof(load_type));
	}

	/// returns the load of the bucket, where the given element whould hashed into
	/// \param e bucket/bucket of the element e
	/// \return the load
	constexpr index_type load(const T &e) const noexcept {
		const size_t index = HashFkt(e);
		ASSERT(index < nrbuckets);
		return __internal_load_array[index];
	}

	/// NOTE: only single threaded.
	/// \return the load, the number of buckets which are not empty
	constexpr index_type load() const noexcept {
		index_type ret = index_type(0);
		for (index_type i = 0; i < nrbuckets; i++) {
			ret +=__internal_load_array[i];
		}

		return ret;
	}

	// internal array
	alignas(32) data_type __internal_hashmap_array[total_size];
	alignas(32) load_type __internal_load_array[nrbuckets];
};

#endif //SMALLSECRETLWE_SIMPLE_H
