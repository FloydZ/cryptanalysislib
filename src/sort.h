#ifndef SMALLSECRETLWE_SORT_H
#define SMALLSECRETLWE_SORT_H

// C++ includes
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

#if defined(SORT_PARALLEL)
#include <algorithm>
#include <execution>// parallel/sequential sort
#include <omp.h>
#endif

// Internal imports
#include "container/hashmap/common.h"
#include "container/triple.h"
#include "helper.h"
#include "list/list.h"
#include "search/search.h"
#include "sort.h"
#include "sort/ska_sort.h"
#include "thread/thread.h"


/// Sorts the labels in the list like this:
///				b0		b1		b2
///		[		|		|		]
///					^		^------ Sorted via std::sort in the buckets
///					|---- Will be sorted in the buckets.
struct ConfigParallelBucketSort {
	// is the offset of the l part of the label.
	uint32_t label_offset;// must be this value:  = (G_n - G_k - G_l); E.g. Number of bits to cut of the label
	uint32_t l;           // label `l` size. Aka dumer window

	uint64_t bucket_size;// number of elements per bucket
	uint64_t nr_buckets; // number of total buckets in this hashmap
	uint32_t nr_threads; // how many threads access this datastructure
	uint32_t nr_indices; // how many indices must each bucket element contain.

	uint16_t b0;
	uint16_t b1;
	uint16_t b2;

	uint8_t lvl;

	// number of additional l window shifts which needs to be saved in one element.
	// Note that l window is not exactly the same as the l in Dumer.
	// IM = Indyk Motwani
	uint8_t IM_nr_views;

	// for description of these flags look at the implementation of `ParallelBucketSort`
	bool STDBINARYSEARCH_SWITCH = false;
	bool INTERPOLATIONSEARCH_SWITCH = false;
	bool LINEARSEARCH_SWITCH = false;

	//
	bool USE_LOAD_IN_FIND_SWITCH = true;

	// Because of the greate abstraction of the hashing and extracting of this hashmap, this flag does not have a direct
	// influence on the code of the hashmap.
	// Still its already in place for further possible usage
	// if set to `true` the hashmap is forced to save the full 128 bit of the label, and not just the maximum b2 bits.
	bool SAVE_FULL_128BIT = false;

	// if this flag is set to `true`, the internal data of holding just the lpart and indices of the baselist, will
	// be exchanged to a triple holding the lpart, indices of the baselist and additional the label (or another part of it)
	uint8_t EXTEND_TO_TRIPLE = 0;

	// If set to true: some elements are prefetched in the `find` method to speed up the `traversing` of buckets
	bool USE_PREFETCH_SWITCH = false;

	// if set to true, the internal structure of the hashmap changes a lot.
	//      The internal `LoadType` is change to `std::atomic<LoadType>`.
	//      The above-mentioned splitting the const_array into chunks for each thread is aldo dismissed
	//      Instead we can use the const_array as the hashmap directly because, we lock the write access to each bucket through the load.
	//      Sorting is not needed afterwards
	bool USE_ATOMIC_LOAD_SWITCH = false;

	//
	bool USE_HIGH_WEIGHT_SWITCH = false;

	//
	bool USE_PACKED_SWITCH = true;

private:
	// useless empty constructor.
	constexpr ConfigParallelBucketSort() : label_offset(0),
	                                       l(0),
	                                       bucket_size(0),
	                                       nr_buckets(0),
	                                       nr_threads(0),
	                                       nr_indices(0),
	                                       b0(0),
	                                       b1(0),
	                                       b2(0),
	                                       lvl(0),
	                                       IM_nr_views(0) {}

public:
	constexpr ConfigParallelBucketSort(const uint16_t b0,            //
	                                   const uint16_t b1,            //
	                                   const uint16_t b2,            //
	                                   const uint64_t bucket_size,   //
	                                   const uint64_t nr_buckets,    //
	                                   const uint32_t nr_threads,    //
	                                   const uint32_t nr_indices,    //
	                                   const uint32_t label_offset,  //
	                                   const uint32_t l,             //
	                                   const uint8_t lvl,            //
	                                   const uint8_t nr_IM_views = 0,//
	                                   const bool SBSSW = false,     //
	                                   const bool ISSW = false,      //
	                                   const bool LSSW = false,      //
	                                   const bool USIDSW = true,     //
	                                   const bool SF128B = false,    //
	                                   const uint8_t ETT = 0,        //
	                                   const bool UPS = false,       //
	                                   const bool UALS = false,      //
	                                   const bool UHWS = false,      //
	                                   const bool UPAS = true        //
	                                   ) : label_offset(label_offset), l(l), bucket_size(bucket_size), nr_buckets(nr_buckets),
	                                       nr_threads(nr_threads), nr_indices(nr_indices), b0(b0), b1(b1), b2(b2),
	                                       lvl(lvl), IM_nr_views(nr_IM_views),
	                                       STDBINARYSEARCH_SWITCH(SBSSW), INTERPOLATIONSEARCH_SWITCH(ISSW),
	                                       LINEARSEARCH_SWITCH(LSSW), USE_LOAD_IN_FIND_SWITCH(USIDSW),
	                                       SAVE_FULL_128BIT(SF128B), EXTEND_TO_TRIPLE(ETT),
	                                       USE_PREFETCH_SWITCH(UPS), USE_ATOMIC_LOAD_SWITCH(UALS),
	                                       USE_HIGH_WEIGHT_SWITCH(UHWS), USE_PACKED_SWITCH(UPAS) {}

	/// print some information about the configuration
	void print() const {
		std::cout << "HM" << nr_indices - 1
		          << " bucket_size: " << bucket_size
		          << ", nr_buckets: " << nr_buckets
		          << ", nr_threads: " << nr_threads
		          << ", nr_indices: " << nr_indices
		          << ", b0: " << b0 << ", b1: " << b1 << ", b2: " << b2
		          << ", SAVE_FULL_128BIT: " << SAVE_FULL_128BIT
		          << ", EXTEND_TO_TRIPLE: " << unsigned(EXTEND_TO_TRIPLE)
		          << "\n";
	}
};

template<const ConfigParallelBucketSort &config,
         class ExternalList,                      //
         typename ArgumentLimbType,               // container of the `l`-part
         typename ExternalIndexType,              // container of the indices which point into the baselists
         ArgumentLimbType (*HashFkt)(__uint128_t)>//
#if __cplusplus > 201709L
    requires//HashMapListAble<ExternalList> &&
        std::is_integral<ArgumentLimbType>::value &&
        std::is_integral<ExternalIndexType>::value
#endif
        class ParallelBucketSort {
public:
	/// nomenclature:
	///		bid = bucket id
	///		tid = thread id
	///		nri	= number of indices
	///		nrt = number of threads
	///		nrb = number of buckets
	typedef typename ExternalList::ElementType Element;
	typedef typename ExternalList::LabelType Label;
	typedef typename ExternalList::ValueType Value;
	typedef typename ExternalList::LabelContainerType LabelContainerType;
	typedef ArgumentLimbType T;
	typedef ArgumentLimbType H;
	typedef ExternalIndexType IndexType;
	typedef ExternalList List;

private:
	/// Memory Layout of a bucket entry.
	/// 	The first 64/128 (depending on `ArgumentLimbType`) bit are used for the `l` part of the label. Thereafter called
	/// 	`data`. Each data chunk is split at the position `b1`. The bits [b0,..., b1) are used as the `hash` function
	/// 	for each bucket. Whereas the bits [b1, ..., b2) are used so sort each element within each bucket.
	/// 	The indices i1, ..., i4 (actually only `nri` are saved.) are used as pointers (list positions) to he 4 baselists L1, ..., L4.
	///   L1        L2        L3        L4
	/// ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
	/// │     │   │     │   │     │   │     │
	/// │     │   │     │   │     │   │     │
	/// │     │◄┐ │     |◄─┐|     │◄─┐|     │◄┐
	/// │     │ │ │     │  ││     │  ││     │ │
	/// │     │ │ │     │  ││     │  ││     │ │
	/// └─────┘ │ └─────┘  │└─────┘  │└─────┘ │
	///         │          └────┐    │        │
	///         │               │    │        │
	///         └────────────┐  │    │    ┌───┘
	///     b0    b1  b2     │  │    │    │ align + 4*32
	/// ┌────┬─────┬───┬───┬─┴─┬┴──┬─┴─┬──┴┐
	/// │    │     │   |   │   │   │   │   │
	/// │     DATA         │ i1│ i2│ i3│ i4│
	/// │    │     │   |   │   │   │   │   │
	/// └────┴─────┴───┴───┴───┴───┴───┴───┘
	constexpr static uint16_t b0 = config.b0;
	constexpr static uint16_t b1 = config.b1;
	constexpr static uint16_t b2 = config.b2;

	/// returns the number of indices needed to archive alignment
	/// \param i 					= nri
	/// \param alignment			= 128, alignment in bits
	/// \param sizeofindicesinbytes	= 32
	/// \return	number of indicies to archive alignment
	constexpr static uint32_t BucketAlignment(const uint32_t i,
	                                          const uint32_t alignment,
	                                          const uint32_t sizeofindicesinbytes,
	                                          const uint32_t sizeofarginbytes) {
		const uint32_t bytes = i * sizeofindicesinbytes;
		const uint32_t bytes2 = ((bytes + alignment - 1) / alignment) * alignment;
		const uint32_t Anri = (bytes2 - sizeofarginbytes) / sizeofindicesinbytes;
		if (Anri == 0)
			return i;
		if (Anri < i)
			return BucketAlignment(i, 2 * alignment, sizeofindicesinbytes, sizeofarginbytes);
		return Anri;
	}

	// if set to true, the hashmaps extends the number of indices it can hold, s.t. each element within the hashmap is placed
	// on address divisible by `ALIGNMENT`.
	constexpr static bool ALIGNMENT_SWITCH = false;
	constexpr static uint64_t ALIGNMENT = 64;
	constexpr static uint64_t nrb = config.nr_buckets;
	constexpr static uint32_t nrt = config.nr_threads;
	constexpr static uint32_t nri = config.nr_indices;
	// Aligned number of indices.
	constexpr static uint32_t Anri = ALIGNMENT_SWITCH ? BucketAlignment(nri, ALIGNMENT, sizeof(IndexType) * 8, sizeof(ArgumentLimbType) * 8) : nri;

	constexpr static uint64_t size_b = config.bucket_size;// size of each bucket
	constexpr static uint64_t size_t = size_b / nrt;      // size of each bucket which is accessible by one thread.
	constexpr static uint64_t chunks = nrb / nrt;         // how many buckets must each thread sort or reset.
	constexpr static uint64_t chunks_size = chunks * size_b;

public:
	// needed in certain configurations
	constexpr static ArgumentLimbType zero_element = ArgumentLimbType(-1);

	// We choose the optimal data container for every occasion.
	// This is the Type which is exported to the outside and denotes the maximum number of elements a bucket can hold.
	// Must be the same as IndexType, because `load` means always an absolute position within the const_array.
	using LoadType = IndexType;
	// In contrast to `LoadType` this type does not include the absolute position within the const_array, but only the
	// relative within the bucket. Can be smaller than `IndexType`
	using LoadInternalType = typename std::conditional<config.USE_ATOMIC_LOAD_SWITCH,
	                                                   std::atomic<TypeTemplate<size_b>>,
	                                                   TypeTemplate<size_b>>::type;

	using ArrayLoadInternalType = typename std::conditional<config.USE_ATOMIC_LOAD_SWITCH,
	                                                        std::array<LoadInternalType, nrb>,
	                                                        std::vector<LoadInternalType>>::type;
	// Number of bits needed to hold a hash
	using BucketHashType = TypeTemplate<uint64_t(1) << (config.b1 - config.b0)>;
	using BucketIndexType = TypeTemplate<nrb * size_b>;
	// Main data container
	using BucketIndexEntry = std::array<IndexType, Anri>;

	// only set, if `config.EXTEND_TO_TRIPLE` is activated.
	using TripleT = LogTypeTemplate<config.EXTEND_TO_TRIPLE>;

	// only used if `USE_PACKED_SWITCH` is set.
	struct __attribute__((packed))
	InternalPair {
		ArgumentLimbType first;
		BucketIndexEntry second;

		InternalPair() : first(0), second(BucketIndexEntry()){};
		InternalPair(ArgumentLimbType a, BucketIndexEntry b) : first(a), second(b){};
	};

	constexpr static ConfigTriple tconfig{};

	/// NOTE: probably one day i want to change this to a std::tuple.
	using BucketEntry = typename std::conditional<config.EXTEND_TO_TRIPLE != 0,
	                                              triple<ArgumentLimbType, BucketIndexEntry, TripleT, tconfig>,
	                                              typename std::conditional<config.USE_PACKED_SWITCH,
	                                                                        InternalPair,
	                                                                        std::pair<ArgumentLimbType, BucketIndexEntry>>::type>::type;

private:
	// precompute the compare masks and limbs
	constexpr static ArgumentLimbType lmask1 = (~((ArgumentLimbType(1) << b0) - 1));
	constexpr static ArgumentLimbType rmask1 = ((ArgumentLimbType(1) << b1) - 1);
	constexpr static ArgumentLimbType mask1 = lmask1 & rmask1;

	// masks for the [b1, b2] part
	constexpr static ArgumentLimbType lmask2 = (~((ArgumentLimbType(1) << b1) - 1));
	constexpr static ArgumentLimbType rmask2 = ((ArgumentLimbType(1) << b2) - 1);
	constexpr static ArgumentLimbType mask2 = lmask2 & rmask2;
	constexpr static ArgumentLimbType highbitmask = ArgumentLimbType(1) << (sizeof(ArgumentLimbType) * 8 - 1);
	constexpr static ArgumentLimbType sortrmask2 = ((ArgumentLimbType(1) << (b2 + config.lvl)) - 1) | highbitmask;

	// DO NOT USE IT.
	// if set to:   false:  `std::lower_bound` will be used to find a match in one bucket of this hashmap
	//              true: custom mono bounded binary search is used to fin a match
	constexpr static bool STDBINARYSEARCH_SWITCH = config.STDBINARYSEARCH_SWITCH;

	// replace binary search with smarter interpolation search
	constexpr static bool INTERPOLATIONSEARCH_SWITCH = config.INTERPOLATIONSEARCH_SWITCH;

	// switch to a linear search instead of binary/interpolation in special cases. Number completely arbitrary
	//((b1 != b2) && (size_t<15)) ? true : false;
	constexpr static bool LINEARSEARCH_SWITCH = config.LINEARSEARCH_SWITCH;

	// if set to true an additional mem fetch the load const_array will be made.
	constexpr static bool USE_LOAD_IN_FIND_SWITCH = config.USE_LOAD_IN_FIND_SWITCH;

	// if set to true most of the load/write instruction will be replaced with instructions which to not touch a cache
	// NOTE: each l-window entry of the data vector MUST at least 8 Byte aligned
	constexpr static bool CACHE_STREAM_SWITCH = false;

	// Set this flag to true to save the full 64 bit, regardless of the given b2 value. This allows the tree to
	// directly check if the weight threshold exceeded.
	constexpr static bool SAVE_FULL_128BIT = config.SAVE_FULL_128BIT;

	// Set this flag to true to extend the fundamental datastructures by an additional datatype (__uint128_t) to hold
	// additional information.
	constexpr static uint8_t EXTEND_TO_TRIPLE = config.EXTEND_TO_TRIPLE;

	// If set to true: The Hashmaps runs some cache prefetching instructions, before returning a found element.
	// The idea behind is to optimize the `traversing` a bucket, when a match is found.
	constexpr static bool USE_PREFETCH_SWITCH = config.USE_PREFETCH_SWITCH;

	// Get loose of the split of the `__bucket` const_array and secure the buckets' insertion process
	// by setting the load factor with an atomic store/load value.
	constexpr static bool USE_ATOMIC_LOAD_SWITCH = config.USE_ATOMIC_LOAD_SWITCH;

	// If this flag is set, the whole finding procedure is not using any access to the
	// `load` factor, but rather checking via a linear search if a element is found or not
	constexpr static bool USE_HIGH_WEIGHT_SWITCH = config.USE_HIGH_WEIGHT_SWITCH;

	// If this flag is set, the internal data structure are forced to be packed, s.t. every
	// alignment of the internal fields are disregarded.
	constexpr static bool USE_PACKED_SWITCH = config.USE_PACKED_SWITCH;

	/// NOTE: not fully implemented
	// if this flag is set, the highest bit in the load factor will be used to denote if a
	// bucket is sorted. If the bit is not set, the find function will sort it.
	constexpr static bool USE_SORTING_NETWORK_DECODED_IN_LOAD = false;

	// Indyk Motwani Nearest Neighbor Search:
	// How many additional l windows should this hashmap hold?
	constexpr static bool IM_SWITCH = config.IM_nr_views != 0;
	constexpr static uint32_t IM_nr_views = config.IM_nr_views;

	// nr of bits of each view
	constexpr static uint32_t IM_bits_view = b1;
	constexpr static uint32_t IM_bits = IM_bits_view * IM_nr_views;

	// some little helper functions:
	// returns the offset of a thread into the load const_array
	inline uint64_t thread_offset(const uint32_t tid) const noexcept { return tid * nrt; }

	inline uint64_t bucket_offset(const BucketHashType bid) const noexcept { return bid * size_b; }

	inline uint64_t bucket_offset(const uint32_t tid, const BucketHashType bid) const noexcept {
		ASSERT(bid < nrb && tid < nrt);
		uint64_t r;
		if constexpr (USE_ATOMIC_LOAD_SWITCH) {
			// In this case we break up the chunks for each thread,
			// so every thread can operate on the full length of a bucket.
			r = bid * size_b;
		} else {
			r = bid * size_b + tid * size_t;
		}

		ASSERT(r < nrb * size_b);
		ASSERT(size_t <= size_b);
		return r;
	}

	// accumulate the bucket load over all threads
	// can be called multithreading.
	inline void acc_bucket_load(const BucketHashType bid) noexcept {
		ASSERT(bid < nrb);

		if constexpr (USE_ATOMIC_LOAD_SWITCH) {
			// Do nothing.
			return;
		}

		LoadType load = 0;

		for (uint32_t tid = 0; tid < nrt; ++tid) {
			load += get_bucket_load(tid, bid);
		}

		acc_buckets_load[bid] = load;
	}

	/// \param index position to check in the __buckets const_array
	/// \return if the element is zero or not
	bool is_zero(const uint64_t index) const noexcept {
		ASSERT(index < nrb * size_b);
		return __buckets[index].first == zero_element;
	}

	/// \template insert: if set to true: find the next empty slot per bucket per thread
	///								false: find the next empty slot per bucket
	/// \param bid bucket id
	/// \param tid thread id
	/// \return the next empty slot in the const_array
	template<const bool insert>
	inline LoadType find_next_empty_slot(const BucketHashType bid, const uint32_t tid) const noexcept {
		// make sure the function is only called in the correct setting
		ASSERT((!USE_LOAD_IN_FIND_SWITCH && LINEARSEARCH_SWITCH) || USE_HIGH_WEIGHT_SWITCH);
		ASSERT(tid < nrt);

		constexpr LoadType middle = insert ? size_t / 2 : size_b / 2;
		const uint64_t offset = insert ? bucket_offset(tid, bid) : bucket_offset(bid);
		LoadType ret = middle;

		if (is_zero(offset + ret)) {
			// go down
			while (ret > 0 && is_zero(offset + ret - 1)) { ret -= 1; }
		} else {
			// go up
			while (ret < size_t - 1 && is_zero(offset + ret + 1)) { ret += 1; }
		}

		return ret;
	}

public:
	// increments the bucket load by one
	inline LoadType inc_bucket_load(const uint32_t tid, const BucketHashType bid) noexcept {
		ASSERT(tid < nrt && bid < nrb);
		if constexpr (USE_ATOMIC_LOAD_SWITCH) {
			// So here takes all the magic place. So basically we want to return and unique position where the callee
			// function can save its element.
			// We have to take care that we are not overloading the buckets
#if __cplusplus > 201709L
			// This is like super stupid. Fucking apple...
			return buckets_load[bid].fetch_add(1, std::memory_order::relaxed);
#else
			return buckets_load[bid].fetch_add(1);
#endif
		}

		buckets_load[tid * nrb + bid] += 1;
		return 0;
	}

	inline LoadType get_bucket_load(const uint32_t tid, const BucketHashType bid) const noexcept {
		ASSERT(tid < nrt && bid < nrb);
		ASSERT(config.USE_LOAD_IN_FIND_SWITCH);

		if constexpr (USE_ATOMIC_LOAD_SWITCH) {
			// because I want to reduce the atomic loads I do not check before inserting elements, which leads to an overflow of the load factor
			// do not use this function to insert elements.
			return std::min(LoadInternalType(buckets_load[bid].load()),
			                LoadInternalType(size_b));
		}

		// Normal code path
		if constexpr (nrt != 1) {
			return buckets_load[tid * nrb + bid];
		} else {
			return buckets_load[bid];
		}
	}

	// IMPORTANT: Call `acc_bucket_load` first
	inline LoadType get_bucket_load(const BucketHashType bid) const noexcept {
		ASSERT(bid < nrb);

		if constexpr (USE_ATOMIC_LOAD_SWITCH) {
			ASSERT(USE_SORTING_NETWORK_DECODED_IN_LOAD && "not impl");
			// because I want to reduce the atomic loads I do not check before inserting elements,
			// which leads to an overflow of the load factor
			// do not use this function to insert elements.
			return std::min(LoadInternalType(buckets_load[bid].load()),
			                LoadInternalType(size_b));
		}

		if constexpr (nrt != 1) {
			auto ret = acc_buckets_load[bid];
			if constexpr (USE_SORTING_NETWORK_DECODED_IN_LOAD) {
				constexpr LoadInternalType mask = (LoadInternalType(1ul) << (bits_log2(nrb) + 1u)) - LoadInternalType(1ul);
				return ret & mask;
			}

			return ret;
		} else {
			auto ret = buckets_load[bid];
			if constexpr (USE_SORTING_NETWORK_DECODED_IN_LOAD) {
				constexpr ArrayLoadInternalType mask = (ArrayLoadInternalType(1ul) << (bits_log2(nrb) + 1u)) - ArrayLoadInternalType(1ul);
				return ret & mask;
			}
			return ret;
		}
	}

	inline BucketHashType hash(const ArgumentLimbType data) const noexcept {
		const BucketHashType bid = ((data & mask1) >> config.b0);
		ASSERT(bid < nrb);
		return bid;
	}

	// internal data containers
	alignas(CUSTOM_PAGE_SIZE) std::vector<BucketEntry> __buckets;
	alignas(CUSTOM_PAGE_SIZE) ArrayLoadInternalType buckets_load;
	alignas(CUSTOM_PAGE_SIZE) std::vector<LoadInternalType> acc_buckets_load;

	ParallelBucketSort() noexcept {
		//NOTE only correct if not ternary static_assert((uint64_t(nrb)*uint64_t(size_b)) < uint64_t(std::numeric_limits<IndexType>::max()));
		if constexpr (!config.USE_ATOMIC_LOAD_SWITCH) {
			static_assert((size_b % nrt) == 0);
			static_assert(nrt <= nrb);
			static_assert(size_t <= size_b);
		}

		// NOTE this is only valid in non ternary static_assert((uint64_t(1) << uint64_t (b1 - b0)) <= uint64_t (nrb));
		static_assert(b0 < b1);
		static_assert(b1 <= b2);

		// hard limit. The internal code is not able to process more bits.
		static_assert(b2 <= 128, "Sorry nothing i can do. If you want to match on more than 128 bits, i think you have bigger problems than this limitations.");

		// make sure that the user does not want to match on more coordinates that the container hold.
		// NOTE reactivate static_assert(b2 <= sizeof(ArgumentLimbType)*8, "make sure that T is big enough...");

		// make sure that from all the given search method only one is active
		static_assert(!(LINEARSEARCH_SWITCH && INTERPOLATIONSEARCH_SWITCH));

		// make sure that we actually have some coordinates to search on, if the user wants to do a linear/different search.
		if constexpr ((LINEARSEARCH_SWITCH || INTERPOLATIONSEARCH_SWITCH) && !USE_LOAD_IN_FIND_SWITCH) {
			//static_assert(b1 < b2);
		}

		// make sure that the given size parameter are sane.
		static_assert(size_b != 0);

		// make sure that the correct element type is passed as a type to the hashmap.
		if constexpr (SAVE_FULL_128BIT) {
			static_assert(sizeof(ArgumentLimbType) == 128 / 8, "pass a `__uint128_t` to the hashmap as base type");
		}

		// only allow one of these flags activated.
		static_assert(SAVE_FULL_128BIT + bool(EXTEND_TO_TRIPLE) <= 1, "only one of these flags is allowed");

		if constexpr (nrt != 1) {
			// This restriction is only needed if we have more than one thread. Or we have to overwrite the buckets with -1.
			// The problem is, that i use lvl to get for `hm2` an extra bit
			// into the searching routine. This is needed because other wise the las element
			// of a bucket is indistinguishable from the -1 element. Especially
			// if only 1 bit or none is used in the last lvl.
			static_assert(config.lvl + config.b2 < 64);
		}

		if constexpr (IM_SWITCH) {
			// Explanation:
			//  Additionally to the l bits on which the MMT algorithm normally matches in the last lvl, we save
			//  `nr_IM_views` distinct such views. So essentially each `ArgumentLimbType` holds in fact l1 + `nr_IM_views` * l2
			// bits of a label. Or to put it more easily the last  l1 + `nr_IM_views` * l2 bits of each label
			// is copied inside.
			static_assert((IM_nr_views * IM_bits_view) <= 64);
		}

		if constexpr (USE_ATOMIC_LOAD_SWITCH) {
			// If we have atomic load, and only one thread operating on it. This would be stupid.
			static_assert(nrt > 1);
		}

		if constexpr (LINEARSEARCH_SWITCH) {
			// only allow to remove the load factor const_array if linear search is active
			static_assert(!USE_LOAD_IN_FIND_SWITCH);
		}

		if ((1 << (b2 - b1)) / nrb > size_b)
			std::cout << "WHAT DOES THIS MEAN???\n";

		__buckets.resize(nrb * size_b);

		if constexpr (!USE_ATOMIC_LOAD_SWITCH && USE_LOAD_IN_FIND_SWITCH) {
			buckets_load.resize(nrb * nrt);
		}

		if constexpr (nrt != 1 && !USE_ATOMIC_LOAD_SWITCH && USE_LOAD_IN_FIND_SWITCH) {
			acc_buckets_load.resize(nrb);
		}

		if constexpr (!USE_LOAD_IN_FIND_SWITCH && USE_SORTING_NETWORK_DECODED_IN_LOAD) {
			ASSERT(false && "makes no sense");
		}

		// Make sure the internal stuff is only printed once.
#if !defined(BENCHMARK) && !defined(NO_LOGGING) && !defined(CHALLENGE)
#pragma omp single
		{
			std::cout << "HM" << nri << "\n";
			std::cout << "\tb0=" << b0 << ", b1=" << b1 << ", b2=" << b2 << "\n";
			std::cout << "\tnr_buckets:" << nrb << "\n";
			std::cout << "\tsize_b:" << size_b << ", size_t:" << size_t << ", chunks:" << chunks << "\n";
			std::cout << "\tAnri:" << Anri << "\n";
			std::cout << "\tBucketHashType: " << sizeof(BucketHashType) * 8 << "Bits\n";
			std::cout << "\tBucketIndexType: " << sizeof(BucketIndexType) * 8 << "Bits\n";
			std::cout << "\tBucketEntry: " << sizeof(BucketEntry) * 8 << "Bits\n";
			std::cout << "\tArgumentLimbType: " << sizeof(ArgumentLimbType) * 8 << "Bits\n";
			std::cout << "\tIndexType: " << sizeof(IndexType) * 8 << "Bits\n";
			std::cout << "\tLoadType: " << sizeof(LoadType) * 8 << "Bits\n";
			std::cout << "\tLoadInternalType: " << sizeof(LoadInternalType) * 8 << "Bits\n";
			std::cout << "\tLabelType: " << sizeof(Label) * 8 << "Bits\n";
			std::cout << "\tINTERPOLATIONSEARCH_SWITCH:" << INTERPOLATIONSEARCH_SWITCH << "\n";
			std::cout << "\tSTDBINARYSEARCH_SWITCH:" << STDBINARYSEARCH_SWITCH << "\n";
			std::cout << "\tLINEAREARCH_SWITCH:" << LINEARSEARCH_SWITCH << "\n";
			std::cout << "\tUSE_LOAD_IN_FIND_SWITCH:" << USE_LOAD_IN_FIND_SWITCH << "\n";
			std::cout << "\tIM_SWITCH:" << IM_SWITCH << "\n";
			std::cout << "\tALIGNMENT_SWITCH:" << ALIGNMENT_SWITCH << "\n";
			std::cout << "\tSAVE_FULL_128BIT:" << SAVE_FULL_128BIT << "\n";
			std::cout << "\tEXTEND_TO_TRIPLE:" << int(EXTEND_TO_TRIPLE) << "\n";
			std::cout << "\tUSE_PREFETCH_SWITCH:" << USE_PREFETCH_SWITCH << "\n";
			std::cout << "\tUSE_ATOMIC_LOAD_SWITCH:" << USE_ATOMIC_LOAD_SWITCH << "\n";
			std::cout << "\tUSE_HIGH_WEIGHT_SWITCH:" << USE_HIGH_WEIGHT_SWITCH << "\n";
			std::cout << "\tUSE_PACKED_SWITCH:" << USE_PACKED_SWITCH << "\n"
			          << std::flush;
		}
#endif
		// reset the whole thing for the first usage
		reset();
	}

	/// This function returns the label at the position `i` in the list `L` but shifted to the base bit 0.
	/// This means, if ArgumentLimbType == uint128_t:
	///      0              127 bits
	///      [xxxxxxxx|0000000]  label
	///      n-k-l...n-k    label position.
	/// or, if ArgumentLimbType == uint64_t:
	///      0               64 bits
	///      [xxxxxxxx|0000000]  label
	///      n-k-l...n-k    label position.
	///		NOTE: That the last one is only valid if `l` < 64.
	/// 	NOTE: Special Indyk Motwani approach is also added. This is activated if `IM_SWITCH` is true.
	///			In this case additional `IM_bits` bits are copied from the label into the hashmap. Make sure
	///			that there is enough space.
	void hash(const List &L, const uint32_t tid) noexcept {
		ASSERT(tid < config.nr_threads);
		constexpr static uint32_t loffset = config.label_offset;
		constexpr static uint32_t loffset64 = loffset / 64;
		constexpr static uint32_t lshift = (loffset - (loffset64 * 64));  // this is also with some bitmasking possible.
		constexpr static uint32_t size_label = sizeof(LabelContainerType);// Size of an `element`. Needed to calculate the correct offset of an label within the list.
		constexpr static __uint128_t labelmask2 = ((__uint128_t(1) << config.l) - 1);

		const uint64_t b_tid = L.size() / nrt;// blocksize of each thread
		const uint64_t s_tid = tid * b_tid;   // Starting point of each process
		// Starting point of the list pointer. Points to the first Label within the list.

		// Instead of access the const_array each time in the loop, we increment an number by the length between two l-parts
		uint64_t Lptr = (uint64_t) L.data_label() + (s_tid * size_label) + (loffset64 * 8);

		ArgumentLimbType data;
		__uint128_t data2;

		IndexType pos[1];
		for (uint64_t i = s_tid; i < ((tid == nrt - 1) ? L.size() : s_tid + b_tid); ++i) {
			data2 = *((__uint128_t *) Lptr);
			data2 >>= lshift;
			data2 &= labelmask2;
			data = data2;

			// in the special case if IndykMotwani Hashing, we need to rotate the l1 and l2 windows
			if constexpr (IM_SWITCH) {
				// we do not allow searching in the IndykMotwani NN case
				static_assert(config.b2 == config.b1);

				constexpr ArgumentLimbType mask = (ArgumentLimbType(1) << config.l) - 1;
				data = ((data >> (config.l - b1)) ^ (data << b1)) & mask;
			}

			Lptr += size_label;

			pos[0] = i;
			insert(data, pos, tid);
		}
	}

	/// \tparam Extractor
	/// \param L list to hash
	/// \param load size of 'L', e.g. number of elements in the list.
	/// \param tid thread id
	/// \param e extractor function
	template<class Extractor>
	void hash(const List &L, const uint64_t load, const uint32_t tid, Extractor e) noexcept {
		ASSERT(tid < config.nr_threads);

		const std::size_t s_tid = L.start_pos(tid);
		const std::size_t e_tid = s_tid + load;

		ArgumentLimbType data;
		IndexType pos[1];
		for (std::size_t i = s_tid; i < e_tid; ++i) {
			data = e(L.data_label(i));
			pos[0] = i;
			insert(data, pos, tid);
		}
	}

	// the ase above but using an extractor and a hasher
	template<class Extractor, class Extractor2>
	void hash_extend_to_triple(const List &L, const uint64_t load, const uint32_t tid,
	                           Extractor e, Extractor2 et) noexcept {
		ASSERT(tid < config.nr_threads);

		const std::size_t s_tid = L.start_pos(tid);
		const std::size_t e_tid = s_tid + load;

		ArgumentLimbType data;
		IndexType pos[1];
		for (std::size_t i = s_tid; i < e_tid; ++i) {
			data = e(L.data_label(i));
			pos[0] = i;

			// insert the element.
			const BucketHashType bid = HashFkt(data);
			const LoadType load = !USE_HIGH_WEIGHT_SWITCH ? get_bucket_load(tid, bid) : find_next_empty_slot<1>(bid, tid);
			if (size_t - load == 0) {
				continue;
			}

			const BucketIndexType bucketOffset = bucket_offset(tid, bid) + load;
			if constexpr (!USE_HIGH_WEIGHT_SWITCH) {
				inc_bucket_load(tid, bid);
			}

			__buckets[bucketOffset].first = data;
			memcpy(&__buckets[bucketOffset].second, pos, nri * sizeof(IndexType));
			__buckets[bucketOffset].third = et(L.data_label(i));
		}
	}

	// same as above but do not touch the load const_array
	template<class Extractor>
	void traverse_hash(const List &L, const uint64_t load, const uint32_t tid, Extractor e) noexcept {
		ASSERT(tid < config.nr_threads);

		const std::size_t s_tid = L.start_pos(tid);
		const std::size_t e_tid = s_tid + load;

		ArgumentLimbType data;
		IndexType pos[1];
		for (std::size_t i = s_tid; i < e_tid; ++i) {
			data = e(L.data_label(i));
			pos[0] = i;

			// insert the element.
			const BucketHashType bid = HashFkt(data);
			const LoadType load = find_next_empty_slot<1>(bid, tid);
			if (size_t - load == 0) {
				continue;
			}

			const BucketIndexType bucketOffset = bucket_offset(tid, bid) + load;

			__buckets[bucketOffset].first = data;
			memcpy(&__buckets[bucketOffset].second, pos, nri * sizeof(IndexType));
			//__buckets[bucketOffset].third = et(L.data_label(i));
		}
	}

	/// \param data l part of the label IMPORTANT MUST BE ONE LIMB
	/// \param pos	pointer to the const_array which should be copied into the internal data structure to loop up elements in the baselists
	/// \param tid	thread_id
	void insert(const ArgumentLimbType data, const IndexType *npos, const uint32_t tid) noexcept {
		ASSERT(tid < config.nr_threads);
		const BucketHashType bid = HashFkt(data);
		LoadType load;

		if constexpr (!USE_LOAD_IN_FIND_SWITCH) {
			// in this case we do not access the 'load' const_array. This reduces the caches
			// misses by one. Instead, we do a linear search over the buckets to find a empty space
			load = find_next_empty_slot<true>(bid, tid);
		} else {
			if constexpr (USE_ATOMIC_LOAD_SWITCH) {
				load = inc_bucket_load(tid, bid);
			} else {
				load = get_bucket_load(tid, bid);
			}
		}

		// early exit if a bucket is full.
		if constexpr (USE_ATOMIC_LOAD_SWITCH) {
			if (size_b <= uint64_t(load)) {
				return;
			}
		} else {
			if (size_t - uint64_t(load) == 0) {
				return;
			}
		}

		// calculated the final index within the const_array of elements which will be returned to the tree construction.
		const BucketIndexType bucketOffset = bucket_offset(tid, bid) + load;
		if constexpr (!USE_ATOMIC_LOAD_SWITCH && USE_LOAD_IN_FIND_SWITCH) {
			// we need to increase the load factor only at this point if we do not use atomics.
			inc_bucket_load(tid, bid);
		}

		// A little if ... else ... mess. But actually we just write the element and the indices of the baselists into the correct places.
		if constexpr (CACHE_STREAM_SWITCH) {
			MM256_STREAM64(&(__buckets[bucketOffset].first), uint64_t(data));
			if constexpr (nri == 1) {
				MM256_STREAM64(&__buckets[bucketOffset].second, npos);
			} else if constexpr (nri == 2) {
				MM256_STREAM128(&__buckets[bucketOffset].second, npos);
			} else {
				memcpy(&__buckets[bucketOffset].second, npos, nri * sizeof(IndexType));
			}
		} else {
			__buckets[bucketOffset].first = data;
			if constexpr (nri == 1) {
				__buckets[bucketOffset].second[0] = npos[0];
			} else if constexpr (nri == 2) {
				__buckets[bucketOffset].second[0] = npos[0];
				__buckets[bucketOffset].second[1] = npos[1];
			} else {
				memcpy(&__buckets[bucketOffset].second, npos, nri * sizeof(IndexType));
			}
		}
	}

	/// This function is exactly the same as above, with the only change, that
	/// its allowing for a custom hash function. THis is useful, if during the
	/// algorithm your hash function changes
	/// \param data l part of the label IMPORTANT MUST BE ONE LIMB
	/// \param pos	pointer to the const_array which should be copied into the internal data structure to loop up elements in the baselists
	/// \param tid	thread_id
	template<class Hasher>
	void custom_insert(const ArgumentLimbType data,
	                   const IndexType *npos,
	                   const uint32_t tid,
	                   Hasher &CustomHashFkt) noexcept {
		ASSERT(tid < config.nr_threads);
		const BucketHashType bid = CustomHashFkt(data);
		LoadType load;

		if constexpr (!USE_LOAD_IN_FIND_SWITCH) {
			// in this case we do not access the 'load' const_array. This reduces the caches
			// misses by one. Instead, we do a linear search over the buckets to find a empty space
			load = find_next_empty_slot<true>(bid, tid);
		} else {
			if constexpr (USE_ATOMIC_LOAD_SWITCH) {
				load = inc_bucket_load(tid, bid);
			} else {
				load = get_bucket_load(tid, bid);
			}
		}

		// early exit if a bucket is full.
		if constexpr (USE_ATOMIC_LOAD_SWITCH) {
			if (size_b <= uint64_t(load)) {
				return;
			}
		} else {
			if (size_t - uint64_t(load) == 0) {
				return;
			}
		}

		// calculated the final index within the const_array of elements which will be returned to the tree construction.
		const BucketIndexType bucketOffset = bucket_offset(tid, bid) + load;
		if constexpr (!USE_ATOMIC_LOAD_SWITCH && USE_LOAD_IN_FIND_SWITCH) {
			// we need to increase the load factor only at this point if we do not use atomics.
			inc_bucket_load(tid, bid);
		}

		// A little if ... else ... mess. But actually we just write the element and the indices of the baselists into the correct places.
		if constexpr (CACHE_STREAM_SWITCH) {
			MM256_STREAM64(&(__buckets[bucketOffset].first), uint64_t(data));
			if constexpr (nri == 1) {
				MM256_STREAM64(&__buckets[bucketOffset].second, npos);
			} else if constexpr (nri == 2) {
				MM256_STREAM128(&__buckets[bucketOffset].second, npos);
			} else {
				memcpy(&__buckets[bucketOffset].second, npos, nri * sizeof(IndexType));
			}
		} else {
			__buckets[bucketOffset].first = data;
			if constexpr (nri == 1) {
				__buckets[bucketOffset].second[0] = npos[0];
			} else if constexpr (nri == 2) {
				__buckets[bucketOffset].second[0] = npos[0];
				__buckets[bucketOffset].second[1] = npos[1];
			} else {
				memcpy(&__buckets[bucketOffset].second, npos, nri * sizeof(IndexType));
			}
		}
	}

	/// Only sort a single bucket. Make sure that you call this function for every bucket.
	// Assumes more buckets than threads
	void sort_bucket(const BucketHashType bid) noexcept {
		ASSERT(bid < nrb);
		ASSERT(((bid + 1) * size_b) <= (nrb * size_b));

		const uint64_t start = bid * size_b;
		const uint64_t end = start + size_b;

		// chose the optimal sorting algorithm depending on the underlying `ArgumentLimbType`. If `__uint128_t` is needed
		// because if `l > 64` we have to fall back to the std::sort algorithm.
		if constexpr (std::is_same_v<ArgumentLimbType, __uint128_t>) {
			std::sort(__buckets.begin() + start,
			          __buckets.begin() + end,
			          [](const auto &e1, const auto &e2) -> bool {
				          // well this is actually not completely correct. This checks on the lowest b2 bits. But actually
				          // it should check on the bits [b1, b2]. I need to do this because otherwise the element -1
				          // is indistinguishable from something tha is -1 on [b1, b2].
				          // Additionally, we use the highest bit too.
				          return (e1.first & sortrmask2) < (e2.first & sortrmask2);
			          });
		} else {
			ska_sort(__buckets.begin() + start,
			         __buckets.begin() + end,
			         [](auto e) -> ArgumentLimbType {
				         // distinguish between the two cases if we are in the first/intermediate lvl
				         if constexpr (nri == 1) {
					         return e.first & sortrmask2;
				         } else {
					         return e.first & sortrmask2;
				         }
			         });
		}

		if constexpr (nrt != 1) {
			// after the sorting we can accumulate the load of each bucket over all threads.
			acc_bucket_load(bid);
		}
	}

	/// A little helper function, which maps a thread (its assumed that nr_threads <= nr_buckets) to a set of buckets,
	/// which this thread needs to sort.
	inline void sort(const uint32_t tid) noexcept {
		ASSERT(tid < config.nr_threads);

		if constexpr (USE_ATOMIC_LOAD_SWITCH) {
			// that's the whole reason, we use atomic value. So we dont have to sort.
			return;
		}

		if constexpr (!USE_LOAD_IN_FIND_SWITCH) {
			// in this case we cannot simply query the `load` const_array.
			// We need to `recalculate the load for each bucket and each on the fly new
			// Additionally, we also break up the thread `boundaries` and allow each thread
			// to access areas of other threads.

			if constexpr (b1 != b2) {
				ASSERT(0);// not implemented
				return;
			}

			if constexpr (!LINEARSEARCH_SWITCH) {
				ASSERT(0);// not implemented
				return;
			}

			for (uint64_t bid = tid * chunks; bid < ((tid + 1) * chunks); ++bid) {
				const std::size_t offset = bid * size_b;
				std::size_t load_offset = find_next_empty_slot<1>(bid, 0);
				std::size_t thread_offset = size_t;

				for (uint64_t i = 0; i < nrt - 1; i++) {
					uint64_t load2 = find_next_empty_slot<1>(bid, i + 1);

					// Note these two mem regions can overlap.
					memcpy(__buckets.data() + offset + load_offset,
					       __buckets.data() + offset + thread_offset,
					       load2 * sizeof(BucketEntry));

					load_offset += load2;
					thread_offset += size_t;
				}
			}

			// NOTE: Instead of the normal routine we do not have to accumulate the load
			//          into a separate accumulate const_array.
			return;
		}

		if constexpr ((config.nr_threads == 1) && (config.b2 == config.b1)) {
			// Fastpath. If the hash function maps onto the full length and
			// we only have one thread there is nothing to do.
			return;
		}

		if constexpr ((config.nr_threads > 1) && (config.b2 == config.b1)) {
			// in the special case that (config.b2 == config.b1) but threads > 1 we can memcpy
			// "max-load" elements down. We just have to do it in a per thread manner.
			for (uint64_t bid = tid * chunks; bid < ((tid + 1) * chunks); ++bid) {
				const uint64_t offset = bid * size_b;
				uint64_t load_offset = get_bucket_load(0, bid);
				uint64_t thread_offset = size_t;
				for (uint64_t i = 0; i < nrt - 1; i++) {
					uint64_t load2 = get_bucket_load(i + 1, bid);

					// Note these two mem regions can overlap.
					memcpy(__buckets.data() + offset + load_offset,
					       __buckets.data() + offset + thread_offset,
					       load2 * sizeof(BucketEntry));

					load_offset += load2;
					thread_offset += size_t;
				}

				// After we `sorted` everything, we have to accumulate the load over each thread.
				// but instead of using `acc_bucket_load(bid);` we can set the accumulated load to `load_offset`
				acc_buckets_load[bid] = load_offset;
			}

			return;
		}

		// Slowest path, we have to sort each bucket.
		ASSERT(tid < nrt && nrt <= nrb);
		for (uint64_t bid = tid * chunks; bid < ((tid + 1) * chunks); ++bid) {
			sort_bucket(bid);
		}
	}

	BucketIndexType traverse_find(const ArgumentLimbType &data) noexcept {
		const BucketHashType bid = HashFkt(data);
		const BucketIndexType boffset = bid * size_b;
		ASSERT(bid < nrb && boffset < nrb * size_b);
		//ignore load
		return boffset;
	}

	BucketIndexType _find(const ArgumentLimbType data, LoadType &load) const noexcept {
		return find(data, load);
	}

	// returns -1 on error/nothing found. Else the position.
	// IMPORTANT: load` is the actual load + bid*size_b
	BucketIndexType find(const ArgumentLimbType data, LoadType &load) const noexcept {
		const BucketHashType bid = HashFkt(data);
		const BucketIndexType boffset = bid * size_b;// start index of the bucket in the internal data structure

		if constexpr (!USE_LOAD_IN_FIND_SWITCH) {
			// in this case we do not access the 'load' const_array. This reduces the caches
			// misses by one. Instead, we do a linear search over the buckets to find a empty space
			load = find_next_empty_slot<false>(bid, 0);
		} else {
			// last index within the bucket.
			// IMPORTANT: Note that we need the full bucket load and not just
			// the load of a bucket for a thread.
			if constexpr ((b1 == b2) || USE_LOAD_IN_FIND_SWITCH) {
				load = get_bucket_load(bid);
			} else {
				load = size_b;
			}
		}

		ASSERT(bid < nrb && boffset < nrb * size_b && load < nrb * size_b);

		// fastpath. Meaning that there was nothing to sort on.
		if constexpr (b2 == b1) {
			if (load != 0) {
				// Prefetch data. The idea is that we know that in the special case where we dont have to sort, every element
				// in the bucket is a match.
				if constexpr (USE_PREFETCH_SWITCH) {
					__builtin_prefetch(__buckets.data() + boffset, 0, 0);
				}

				// Check if the last element is really not a -1.
				// NOTE: this check makes probably no sense after a few runs of the algorithm.
				// ASSERT(__buckets[boffset + load - 1].first != ArgumentLimbType(-1));

				load += boffset;
				return boffset;
			}

			return -1;
		}

		// Second fastpath.
		if (load == 0) { return -1; }
		load += boffset;

		constexpr static BucketIndexEntry dummy = {{0}};
		BucketEntry data_target(data, dummy);

		// on huge arrays with huge bucket size is maybe a good idea to switch to interpolation search
		if constexpr (INTERPOLATIONSEARCH_SWITCH) {
			// if this switch is enabled, we are doing an interpolation search.
			auto r = lower_bound_interpolation_search2(__buckets.begin() + boffset,
			                                           __buckets.begin() + load,
			                                           data_target,
			                                           [](const BucketEntry &e1) -> ArgumentLimbType {
				                                           return (e1.first & mask2) >> b1;// See note std::lower_bound
			                                           });

			if (r == (__buckets.begin() + load)) return -1;
			const BucketIndexType pos = distance(__buckets.begin(), r);
			if ((__buckets[pos].first & mask2) != (data & mask2)) return -1;
			return pos;
		}

		// on huge arrays with small bucket size it's maybe a good idea to switch on linear search.
		if constexpr (LINEARSEARCH_SWITCH) {
			BucketIndexType pos = boffset;
			ASSERT(pos < load);

			const ArgumentLimbType data2 = data & mask2;
			while ((__buckets[pos].first & mask2) != data2) {
				pos += 1;
			}

			if (pos == load)
				return -1;

			ASSERT(pos < load);
			return pos;
		}

		if constexpr (STDBINARYSEARCH_SWITCH) {
			//fallback implementation: binary search.
			auto r = std::lower_bound(__buckets.begin() + boffset,
			                          __buckets.begin() + load,
			                          data_target,
			                          [](const auto &e1, const auto &e2) {
				                          // Well this is actually not completely correct.
				                          // This checks on the lowest b2 bits. But actually
				                          // it should check on the bits [b1, b2].
				                          return (e1.first & mask2) < (e2.first & mask2);
			                          });

			if (r == (__buckets.begin() + load)) return -1;
			const BucketIndexType pos = distance(__buckets.begin(), r);
			if ((__buckets[pos].first & mask2) != (data & mask2)) return -1;

			ASSERT(pos < load);
			return pos;
		} else {
			auto r = lower_bound_monobound_binary_search(__buckets.begin() + boffset,
			                                             __buckets.begin() + load,
			                                             data_target,
			                                             [](const BucketEntry &e1) -> ArgumentLimbType {
				                                             return e1.first & mask2;// See note std::lower_bound
			                                             });

			if (r == (__buckets.begin() + load)) return -1;
			const BucketIndexType pos = distance(__buckets.begin(), r);
			if ((__buckets[pos].first & mask2) != (data & mask2)) return -1;

			ASSERT(pos < load);
			return pos;
		}

		ASSERT(0);
		return 0;
	}

	/// This function is exactly the same as above, with the only change, that
	/// its allowing for a custom hash function. THis is useful, if during the
	/// algorithm your hash function changes
	/// returns -1 on error/nothing found. Else the position.
	/// IMPORTANT: load` is the actual load + bid*size_b
	/// \param data
	/// \param load
	/// \return
	template<class Hasher>
	BucketIndexType find(const ArgumentLimbType &data,
	                     LoadType &load,
	                     Hasher &CustomHashFkt) const noexcept {
		const BucketHashType bid = CustomHashFkt(data);
		const BucketIndexType boffset = bid * size_b;// start index of the bucket in the internal data structure

		if constexpr (!USE_LOAD_IN_FIND_SWITCH) {
			// in this case we do not access the 'load' const_array. This reduces the caches
			// misses by one. Instead, we do a linear search over the buckets to find a empty space
			load = find_next_empty_slot<false>(bid, 0);
		} else {
			// last index within the bucket.
			// IMPORTANT: Note that we need the full bucket load and not just
			// the load of a bucket for a thread.
			if constexpr ((b1 == b2) || USE_LOAD_IN_FIND_SWITCH) {
				load = get_bucket_load(bid);
			} else {
				load = size_b;
			}
		}

		ASSERT(bid < nrb && boffset < nrb * size_b && load < nrb * size_b);

		// fastpath. Meaning that there was nothing to sort on.
		if constexpr (b2 == b1) {
			if (load != 0) {
				// Prefetch data. The idea is that we know that in the special case where we dont have to sort, every element
				// in the bucket is a match.
				if constexpr (USE_PREFETCH_SWITCH) {
					__builtin_prefetch(__buckets.data() + boffset, 0, 0);
				}

				// Check if the last element is really not a -1.
				// NOTE: this check makes probably no sense after a few runs of the algorithm.
				// ASSERT(__buckets[boffset + load - 1].first != ArgumentLimbType(-1));

				load += boffset;
				return boffset;
			}

			return -1;
		}
	}


	// Diese funktion muss
	//  - npos anpassen
	//  - pos danach inkrementieren.
	/// this function traverses from a given position until the value [b1, b2] changes. This is done by
	///		first: copy the current positions const_array into the output const_array `npos`
	/// \tparam lvl the starting point from which `npos` is copied from.
	/// \tparam ctr how many elements are copied.
	/// \param pos	current position of a match on the coordinates [b1, ..., b2) between `data` and __buckets[pos].first
	/// \param npos	output const_array if length `nri`
	/// \return
	template<uint8_t lvl, uint8_t ctr>
	ArgumentLimbType traverse(const ArgumentLimbType &data, IndexType &pos, IndexType *npos, const LoadType &load) const noexcept {
		ASSERT(lvl < 4 && npos != nullptr && pos < load);
		if (pos >= (load - 1)) {
			pos = IndexType(-1);
			return pos;
		}

		// This memcpy copies the indices (= positions of elements within the baselist) into the output const_array `npos`
		// the position and length of what needs to be copies is specified by the template parameters `lvl` and `ctr`.
		// Whereas `lvl` specifies the starting position of the memcpy and `ctr` the length.
		memcpy(&npos[lvl], __buckets[pos].second.data(), ctr * sizeof(IndexType));
		ASSERT(npos[0] != IndexType(-1) && npos[1] != IndexType(-1));

		const ArgumentLimbType ret = data ^ __buckets[pos].first;

		// check if the next element is still the same.
		const ArgumentLimbType a = __buckets[pos].first & rmask2;
		const ArgumentLimbType b = __buckets[pos + 1].first & rmask2;

		pos += 1;
		if (a != b)
			pos = IndexType(-1);
		return ret;
	}

	template<uint8_t lvl, uint8_t ctr>
	void traverse_drop(const ArgumentLimbType &data, IndexType &pos, IndexType *npos, const LoadType &load) const noexcept {
		ASSERT(npos != nullptr && pos < load);
		memcpy(&npos[lvl], __buckets[pos].second.data(), ctr * sizeof(IndexType));
		ASSERT(npos[0] != IndexType(-1) && npos[1] != IndexType(-1) && npos[2] != IndexType(-1) && npos[3] != IndexType(-1));

		if (pos >= (load - 1)) {
			pos = IndexType(-1);
			return;
		}

		ArgumentLimbType a = __buckets[pos].first & rmask2;
		ArgumentLimbType b = __buckets[pos + 1].first & rmask2;
		pos += 1;
		if (a != b)
			pos = IndexType(-1);
	}

	/// IMPORTANT: Only call this function by exactly one thread.
	void reset() noexcept {
		if constexpr (!USE_LOAD_IN_FIND_SWITCH) {
			// only in this case reset everything except the load const_array.
			memset(__buckets.data(), -1, __buckets.size() * sizeof(BucketEntry));
			return;
		}

		if constexpr (USE_HIGH_WEIGHT_SWITCH) {
			// in this case we must also reset the whole hashmap, but also the
			// load factor arrays
			memset(__buckets.data(), -1, __buckets.size() * sizeof(BucketEntry));
		}

		// for instructions please read the comment of the function `void reset(const uint32_t tid)`
		if constexpr (config.USE_ATOMIC_LOAD_SWITCH) {
			memset(buckets_load.data(), 0, nrb * sizeof(LoadInternalType));
		} else {
			memset(buckets_load.data(), 0, nrb * nrt * sizeof(LoadInternalType));
		}
		// std::fill(buckets_load.begin(), buckets_load.end(), 0);

		if constexpr (nrt == 1) {
			memset(__buckets.data(), -1, nrb * size_b * sizeof(BucketEntry));
			//std::fill(__buckets.begin(), __buckets.end(), InternalPair(-1, -1));
		}
	}

	/// Each thread resets a number of blocks
	void reset(const uint32_t tid) noexcept {
		ASSERT(tid < nrt);
		ASSERT((tid * chunks_size) < (nrb * size_b));

		if constexpr (!USE_LOAD_IN_FIND_SWITCH) {
			// in this case reset everything except the load const_array.
			memset((void *) (uint64_t(__buckets.data()) + (tid * chunks_size * sizeof(BucketEntry))),
			       -1, chunks_size * sizeof(BucketEntry));
			return;
		}

		if constexpr (USE_HIGH_WEIGHT_SWITCH) {
			// only in this case reset everything
			memset((void *) (uint64_t(__buckets.data()) + (tid * chunks_size * sizeof(BucketEntry))),
			       -1, chunks_size * sizeof(BucketEntry));
		}


		// each thread clears only the load of itself.
		memset((void *) (uint64_t(buckets_load.data()) + (tid * nrb * sizeof(LoadInternalType))),
		       0, nrb * sizeof(LoadInternalType));

		// We do not have to reset the accumulated load const_array nor the buckets, because we don't depend on the
		// data written there in the case were we have multiple threads. Only in the case were we have to go into the sorting
		// function we have to reset the buckets.
		if constexpr ((b2 != b1) && (nrt != 1)) {
			memset((void *) (uint64_t(__buckets.data()) + (tid * chunks_size * sizeof(BucketEntry))),
			       -1, chunks_size * sizeof(BucketEntry));
		}
	}

	// only print one bucket.
	void print(const uint64_t bid, const int32_t nr_elements) const noexcept {
		const uint64_t start = bid * size_b;
		const uint64_t si = nr_elements >= 0 ? start : start + size_b + nr_elements;
		const uint64_t ei = nr_elements >= 0 ? start + nr_elements : start + size_b;

		LoadType load = 0;
		for (uint64_t i = 0; i < nrt; ++i) {
			load += get_bucket_load(i, bid);
		}

		std::cout << "Content of bucket " << bid << ", load: " << unsigned(load) << "\n";
		for (uint64_t i = si; i < ei; ++i) {
			printbinary(__buckets[i].first, b1, b2);
			std::cout << ", " << i << ", ";

			std::cout << " [";
			for (uint64_t j = 0; j < nri; ++j) {
				std::cout << unsigned(__buckets[i].second[j]);
				if (j != nri - 1)
					std::cout << " ";
			}
			std::cout << "]\n";
		}
		std::cout << "\n"
		          << std::flush;
	}

	void print(const uint32_t tid) const noexcept {
		LoadType load = 0;

		// calc the load percentage of each bucket
		for (uint64_t i = tid * chunks; i < ((tid + 1) * chunks); ++i) {
			load += buckets_load[chunks + i];
		}

#pragma omp critical
		{ std::cout << "ThreadID: " << tid << ", load: " << double(load) / chunks << "\n"; }

		for (uint64_t i = 0; i < nrt; i++) {
			for (uint64_t j = 0; j < nrb; j++) {
				std::cout << "TID: " << i << " BID: " << j << " BucketLoad: " << get_bucket_load(i, j) << "\n";
			}
		}

		std::cout << "size: " << nrb * size_b * sizeof(BucketEntry) + nrb * nrt * sizeof(uint64_t) << "Byte\n";
	}

	///
	void print() const noexcept {
		LoadType load = 0;

		bool flag = false;
		// IMPORTANT: Functions is only useful if `acc_bucket_load` was called befor.
		for (uint64_t bid = 0; bid < nrb; ++bid) {
			load += (USE_LOAD_IN_FIND_SWITCH && !USE_HIGH_WEIGHT_SWITCH) ? get_bucket_load(bid) : find_next_empty_slot<0>(bid, 0);
			if (load > 0) {
				flag = true;
			}
		}

		if (!flag) {
			std::cout << "the every bucket is empty flag\n";
		}

		std::cout << "HM" << nri - 1 << " #Elements:" << load << " Avg. load per bucket : " << (double(load) / nrb) << " elements/bucket";
		std::cout << ", size: " << double(bytes()) / (1024.0 * 1024.0)
		          << "MB, load: " << (sizeof(LoadType) * buckets_load.size() >> 20)
		          << "MB, accload: " << (sizeof(LoadType) * acc_buckets_load.size() >> 20) << "MB";
		std::cout << "\n"
		          << std::flush;
	}

	// debug function.
	uint64_t get_first_non_empty_bucket(const uint32_t tid) const noexcept {
		for (uint64_t i = 0; i < nrb; i++) {
			if (get_bucket_load(tid, i) != 0) {
				return i;
			}
		}

		return -1;
	}

	// check if each bucket is correctly sorted
	// input argument is the starting position within the `__buckets` const_array
	bool check_sorted(const uint64_t start, const uint64_t load) const noexcept {
		ASSERT(start < (nrb * size_b));
		uint64_t i = start;
		// constexpr ArgumentLimbType mask = b0 == 0 ? rmask2 : rmask2&lmask1;

		if (load != 0) {
			for (; i < start + load - 1; ++i) {
				// we found the first whole -1 entry
				if (__buckets[i].first == ArgumentLimbType(-1))
					break;

				for (uint64_t j = 0; j < nri; ++j) {
					if (__buckets[i].second[j] == IndexType(-1)) {
						print(start / size_b, size_b);
						std::cout << "ERROR: -1 index in the sorted ares at index: " << i
						          << "\n"
						          << std::flush;
						return false;
					}
				}

				// check if sorted
				if ((__buckets[i].first & rmask2) >
				    (__buckets[i + 1].first & rmask2)) {
					std::cout << "\n";

					printbinary(__buckets[i].first & rmask2, b1, b2);
					std::cout << "\n";
					printbinary(__buckets[i + 1].first & rmask2, b1, b2);
					std::cout << "\n\n";
					print(start / size_b, size_b);
					std::cout << "ERROR. not sorted in bucket " << start / size_b << " at index: " << i << ":" << i + 1
					          << "\n"
					          << std::flush;
					return false;
				}
			}
		}

		// Only check if -1 are at the end of each bucket if and only if we reset the buckets after each run.
		//  And we reset the buckets only if we have more than 1 thread.
		//		if (nrt != 1) {
		//			// Now check that every following entry is completely -1
		//			// actually not completely correct, because of the minus 1. But this needs to be done or otherwise we get
		//			// errors if the bucket is full.
		//			for (; i < start + size_b - 1; ++i) {
		//				if (__buckets[i].first != ArgumentLimbType(-1)) {
		//					print(start / size_b, size_b);
		//					std::cout << "error: -1 not at the end of bucket " << start / size_b << ", pos: " << i << "\n"
		//					          << std::flush;
		//					return false;
		//				}
		//
		//				for (uint64_t j = 0; j < nri; ++j) {
		//					if (__buckets[i].second[j] != IndexType(-1)) {
		//						print(start / size_b, size_b);
		//						std::cout << "error: not -1 in indices in bucket " << start / size_b << ", pos: " << i << "\n"
		//						          << std::flush;
		//						return false;
		//					}
		//				}
		//			}
		//		}

		return true;
	}

	bool check_sorted() const noexcept {
		bool ret = true;
#pragma omp barrier

#pragma omp master
		{
			for (uint64_t bid = 0; bid < nrb; ++bid) {
				if (!check_sorted(bid * size_b, get_bucket_load(bid))) {
					ret = false;
					break;
				}
			}
		}

#pragma omp barrier
		return ret;
	}

	// NOTE: this function is not valid for bjmm hybrid tree
	// checks weather to label computation in `data` is correct or not.
	template<class List>
	bool check_label(const ArgumentLimbType data, const List &L, const uint64_t i, const uint32_t k_lower = -1, const uint32_t k_upper = -1) const noexcept {
		const bool flag = (k_lower == uint32_t(-1)) && (k_upper == uint32_t(-1));
		const uint64_t nkl = flag ? config.label_offset : k_lower;
		const uint32_t limit = flag ? config.l : k_upper - k_lower;
		ArgumentLimbType d = data;

		for (uint64_t j = 0; j < limit; ++j) {
			if ((d & 1) != L.data_label(i).data()[j + nkl]) {
				printbinary(d);
				std::cout << "  calc+extracted label\n"
				          << L.data_label(i) << "\nlabel wrong calc\n"
				          << std::flush;
				return false;
			}

			d >>= 1;
		}

		return true;
	}

	// returns the load summed over all buckets
	uint64_t load() const noexcept {
		uint64_t load = 0;

		if (Thread::get_tid() == 0) {

#pragma omp critical
			{
				for (uint64_t _nrb = 0; _nrb < nrb; ++_nrb) {
					for (uint64_t _tid = 0; _tid < nrt; ++_tid) {
						load += get_bucket_load(_tid, _nrb);
					}
				}
			}
		}

#pragma omp barrier
		return load;
	}

	/// \return the number of elements the hashmap can hold in total.
	uint64_t size() const noexcept {
		return __buckets.size();
	}

	/// \return the number of bytes the hashmap needs to hold all data.
	///				NOTE: this does not take alignment into account
	uint64_t bytes() const noexcept {
		uint64_t ret = sizeof(BucketEntry) * __buckets.size();
		if constexpr (!USE_ATOMIC_LOAD_SWITCH) {
			ret += sizeof(LoadType) * buckets_load.size();
		}
		if constexpr (!USE_ATOMIC_LOAD_SWITCH && nrt > 1) {
			ret += sizeof(LoadType) * acc_buckets_load.size();
		}
		return ret;
	}
};
#endif//SMALLSECRETLWE_SORT_H
