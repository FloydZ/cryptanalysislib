#ifndef SMALLSECRETLWE_SIMPLE2_H
#define SMALLSECRETLWE_SIMPLE2_H

#include <cstdint>
#include <cstdlib>
#include <utility>
#include <cstddef>

#include "sort/sorting_network/common.h"
#include "container/hashmap/common.h"
#include "helper.h"
#include "atomic_primitives.h"

///
class Simple2HashMapConfig {
private:
	// Disable the simple constructor
	constexpr Simple2HashMapConfig() : nrbuckets(0), threads(1) {};

public:
	const uint64_t nrbuckets;
	const uint32_t threads;

	/// only activate the constexpr constructor
	/// \param nrbuckets
	constexpr Simple2HashMapConfig(const uint64_t nrbuckets,
	                               const uint32_t threads = 1u) :
			nrbuckets(nrbuckets), threads(threads) {};
};

/// TODO hier und in der simple hashmap rin `.ptr()` einführen um auf den internal array zuzugreifen
/// NOTE: Only the indices of the list entries can be saved in here.
/// \tparam keyType		base type of the input keys
/// \tparam valueType	base type of the values to save in the list
/// \tparam config		`SimpleHashmapConfig` object
template<
        typename keyType,
        typename valueType,
		const Simple2HashMapConfig &config,
		class Hash>
class Simple2HashMap {
	using data_type          = valueType;
	using index_type         = size_t;

public:
	typedef keyType 	T;

	// TODO make sure that is general enough
	typedef size_t 		LoadType;
	typedef size_t 		IndexType;

	// size per bucket
	constexpr static size_t bucketsize = (64u/sizeof(valueType));
	constexpr static size_t internal_bucketsize = bucketsize - 1u;

	// number of buckets
	constexpr static size_t nrbuckets = config.nrbuckets;

	// total number of elements in the HM
	constexpr static size_t total_size = bucketsize * nrbuckets;
	constexpr static size_t internal_total_size = internal_bucketsize * nrbuckets;
	
	constexpr static bool multithreaded = config.threads > 1u;
	using load_type = typename std::conditional<multithreaded,
	                                            std::atomic<TypeTemplate<bucketsize>>,
	                                            TypeTemplate<bucketsize>>::type;
	
	/// constructor. Zero initializing everything
	constexpr Simple2HashMap() noexcept :
			__array() {}

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
	constexpr inline void insert(const keyType &e, const valueType value) noexcept {
		// hash down the element to the index
		const size_t index = hash(e);
		ASSERT(index < nrbuckets);


		size_t load;
		if constexpr (multithreaded) {
			load = FAA(__array.data() + index*bucketsize + internal_bucketsize, 1);

			// early exit and reset
			if (load >= internal_bucketsize) {
				return;
			}
		} else {
			load = __array[index*bucketsize + internal_bucketsize];

			// early exit, if it's already full
			if (load == internal_bucketsize) {
				return ;
			}
		}


		// just some debugging checks
		ASSERT(load < internal_bucketsize);
		if constexpr (! multithreaded) {
			__array[index*bucketsize + internal_bucketsize] += 1;
		}

		/// NOTE: this store never needs to be atomic, as the position was
		/// computed atomically.
		if constexpr (std::is_bounded_array_v<data_type>) {
			memcpy(__array[index*bucketsize + load], value, sizeof(data_type));
		} else {
			__array[index*bucketsize + load] = value;
		}

	}

	constexpr inline valueType* ptr() noexcept {
		return __array;
	}

	constexpr inline valueType ptr(const load_type i) noexcept {
		ASSERT(i < total_size);
		return __array[i];
	}

	/// Quite same to `probe` but instead it will directly return
	/// the position of the element.
	/// \param e Element to hash down.
	/// \return the position within the internal const_array of `e`
	constexpr inline index_type find(const keyType &e) const noexcept {
		const index_type index = HashFkt(e);
		ASSERT(index < nrbuckets);
		// return the index instead of the actual element, to
		// reduce the size of the returned element.
		return index*nrbuckets;
	}

	constexpr inline index_type find(const keyType &e, index_type &__load) const noexcept {
		const index_type index = HashFkt(e);
		ASSERT(index < nrbuckets);
		__load = load(index);
		// return the index instead of the actual element, to
		// reduce the size of the returned element.
		return index*nrbuckets;
	}
	
	/// match the api
	constexpr inline size_t hash(const keyType &e) const noexcept {
		return Hash::operator()(e);
	}

	/// prints the content of each bucket
	/// it with one thread.
	/// \return nothing
	constexpr inline void print() const noexcept {
		for (index_type i = 0; i < nrbuckets; i++) {
			std::cout << "Bucket: " << i << ", load: " << size_t(load(i)) << "\n";

			for (index_type j = 0; j < bucketsize; j++) {
				print(i*bucketsize + j);
			}
		}
	}

	/// prints a single index
	/// \param index the element to print
	/// \return nothing
	constexpr inline  void print(const size_t index) const noexcept {
		ASSERT(index < total_size);
		printf("%d\n", __array[index]);
	}

	/// NOTE: can be called with only a single thread
	/// overwrites the internal data const_array
	/// with zero initialized elements.
	constexpr inline void clear() noexcept {
		for (size_t i = 0; i < nrbuckets; i++) {
			__array[i*bucketsize + internal_bucketsize] = 0;
		}
	}

	/// returns the load of the bucket, where the given element would hashed into
	/// \param e bucket/bucket of the element e
	/// \return the load
	constexpr inline index_type load(const keyType &e) const noexcept {
		const size_t index = hash(e);
		ASSERT(index < nrbuckets);
		return __array[index*bucketsize + internal_bucketsize];
	}

	/// NOTE: only single threaded.
	/// \return the load, the number of buckets which are not empty
	constexpr inline index_type load() const noexcept {
		index_type ret = index_type(0);
		for (index_type i = 0; i < nrbuckets; i++) {
			ret += load(i);
		}

		return ret;
	}

	/// prints some basic information about the hashmap
	constexpr void info() const noexcept {
		std::cout << "total_size:" << total_size
				  << ", total_size_byts:" << sizeof(__array)
		          << ", nrbuckets:" << nrbuckets
				  << ", bucketsize:" << bucketsize
				  << ", multithreaded: " << multithreaded;

		if constexpr (multithreaded) {
			std::cout << " (Threads:" << config.threads << ")" << std::endl;
		}

		std::cout << "\n";
	}

	// TODO custom allocator
	// internal const_array
	alignas(1024) std::array<data_type, total_size> __array;
};

#endif //SMALLSECRETLWE_SIMPLE2_H
