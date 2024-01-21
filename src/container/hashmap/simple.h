#ifndef SMALLSECRETLWE_SIMPLE_H
#define SMALLSECRETLWE_SIMPLE_H

#include <cstdint>
#include <cstdlib>
#include <utility>
#include <cstddef>

#include "sort/sorting_network/common.h"
#include "container/hashmap/common.h"
#include "helper.h"

///
class SimpleHashMapConfig {
private:
	// Disable the simple constructor
	constexpr SimpleHashMapConfig() : bucketsize(0), nrbuckets(0), threads(1) {};

public:
	const uint64_t bucketsize;
	const uint64_t nrbuckets;
	const uint32_t threads;

	/// only activate the constexpr constructor
	/// \param bucketsize
	/// \param nrbuckets
	constexpr SimpleHashMapConfig(const uint64_t bucketsize,
	                              const uint64_t nrbuckets,
	                              const uint32_t threads = 1u) :
			bucketsize(bucketsize), nrbuckets(nrbuckets), threads(threads) {};
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
		size_t (* HashFkt)(const keyType)>
class SimpleHashMap {
	using data_type          = valueType;
	using index_type         = size_t;

public:
	typedef keyType 	T;

	// TODO make sure that is general enough
	typedef size_t 		LoadType;
	typedef size_t 		IndexType;

	// size per bucket
	constexpr static size_t bucketsize = config.bucketsize;

	// number of buckets
	constexpr static size_t nrbuckets = config.nrbuckets;

	// total number of elements in the HM
	constexpr static size_t total_size = bucketsize * nrbuckets;
	
	constexpr static bool multithreaded = config.threads > 1u;
	using load_type = typename std::conditional<multithreaded,
	                                            std::atomic<TypeTemplate<bucketsize>>,
	                                            TypeTemplate<bucketsize>>::type;

	/// constructor. Zero initializing everything
	constexpr SimpleHashMap() noexcept :
			__internal_hashmap_array(),
			__internal_load_array() {}

	/// the simple hashmap ignores the thread id.
	/// Which is nice.
	/// \param e key element (hashed down = index within the internal array)
	/// \param value element to insert
	/// \param tid (ignored) can be anything
	/// \return
	constexpr void insert(const keyType &e,
						  const valueType value,
						  const uint32_t tid) noexcept {
		(void)tid;
		insert(e, value);
	}

	/// hashes down `e` (Element) to an index where to store
	/// the element.
	/// NOTE: Boundary checks are performed in debug mode.
	/// \param e element to insert
	/// \return nothing
	constexpr void insert(const keyType &e, const valueType value) noexcept {
		// hash down the element to the index
		const size_t index = HashFkt(e);
		ASSERT(index < nrbuckets);


		size_t load;
		if constexpr (multithreaded) {
			load = __internal_load_array[index].fetch_add(1u);

			// early exit and reset
			if (load >= bucketsize) {
				/// NOTE maybe overkill. Is it possible without the atomic store?
				__internal_load_array[index].store(bucketsize);
				return ;
			}
		} else {
			load = __internal_load_array[index];

			// early exit, if it's already full
			if (load == bucketsize) {
				return ;
			}
		}


		// just some debugging checks
		ASSERT(load < bucketsize);
		if constexpr (! multithreaded) {
			__internal_load_array[index] += 1;
		}

		/// NOTE: this store never needs to be atomic, as the position was
		/// computed atomically.
		if constexpr (std::is_bounded_array_v<data_type>) {
			memcpy(__internal_hashmap_array[index*bucketsize + load], value, sizeof(data_type));
		} else {
			__internal_hashmap_array[index*bucketsize + load] = value;
		}

	}

	/// Quite same to `probe` but instead it will directly return
	/// the position of the element.
	/// \param e Element to hash down.
	/// \return the position within the internal const_array of `e`
	constexpr index_type find(const keyType &e) const noexcept {
		const index_type index = HashFkt(e);
		ASSERT(index < nrbuckets);
		// return the index instead of the actual element, to
		// reduce the size of the returned element.
		return index*nrbuckets;
	}

	constexpr index_type find(const keyType &e, index_type &__load) const noexcept {
		const index_type index = HashFkt(e);
		ASSERT(index < nrbuckets);
		__load = __internal_load_array[index];
		// return the index instead of the actual element, to
		// reduce the size of the returned element.
		return index*nrbuckets;
	}

	/// prints the content of each bucket
	/// it with one thread.
	/// \return nothing
	constexpr void print() const noexcept {
		for (index_type i = 0; i < nrbuckets; i++) {
			std::cout << "Bucket: " << i << ", load: " << size_t(__internal_load_array[i]) << "\n";

			for (index_type j = 0; j < bucketsize; j++) {
				print(i*bucketsize + j);
			}
		}
	}

	/// match the api
	constexpr size_t hash(const keyType &e) const noexcept {
		return HashFkt(e);
	}

	/// prints a single index
	/// \param index the element to print
	/// \return nothing
	constexpr void print(const size_t index) const noexcept {
		ASSERT(index < total_size);
		printf("%d\n", __internal_hashmap_array[index]);
	}

	/// NOTE: can be called with only a single thread
	/// overwrites the internal data const_array
	/// with zero initialized elements.
	constexpr void clear() noexcept {
		memset(__internal_load_array, 0, nrbuckets*sizeof(load_type));
	}

	/// returns the load of the bucket, where the given element would hashed into
	/// \param e bucket/bucket of the element e
	/// \return the load
	constexpr index_type load(const keyType &e) const noexcept {
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

	/// prints some basic information about the hashmap
	constexpr void info() const noexcept {
		std::cout << "total_size:" << total_size
				  << ", total_size_byts:" << sizeof(__internal_hashmap_array) + sizeof(__internal_load_array)
		          << ", nrbuckets:" << nrbuckets
				  << ", bucketsize:" << bucketsize
				  << ", multithreaded: " << multithreaded;

		if constexpr (multithreaded) {
			std::cout << " (Threads:" << config.threads
			          << ", lockfree:" << __internal_load_array[0].is_always_lock_free << ")";
		}

		std::cout << "\n";
	}

	// internal const_array
	alignas(1024) data_type __internal_hashmap_array[total_size];
	alignas(1024) load_type __internal_load_array[nrbuckets];
};

#endif //SMALLSECRETLWE_SIMPLE_H
