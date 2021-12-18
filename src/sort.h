#ifndef SMALLSECRETLWE_SORT_H
#define SMALLSECRETLWE_SORT_H

// C++ includes
#include <array>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <atomic>

// C imports
#include <omp.h>
#include <stddef.h>

#if defined(SORT_PARALLEL)
#include <execution>        // parallel/sequential sort
#include <algorithm>
#endif

// Internal imports
#include "helper.h"
#include "triple.h"
#include "list.h"
#include "search.h"
#include "sort.h"
#include "ska_sort.hpp"

template<class List>
concept HashMapListAble = requires(List l) {
	// we need some basic data types
	typename List::ValueType;
	typename List::LabelType;
	typename List::ElementType;
	typename List::MatrixType;
	typename List::LabelContainerType;

	// check that the element and therefore the value and label also fullfil all requirements.
	// requires ListAble<typename List::ElementType>;

	// The following functions are needed.
	requires requires(const size_t s){
		l.data_label(s);
		l.data_value(s);
		{ l.size() } -> std::convertible_to<size_t>;
	};
};

// LSD radix sort, taken from valentin vasseur
template<typename T, bool use_idx>
void vv_radix_sort(T *array, size_t *idx, T *aux, size_t *aux2, size_t len) {
	constexpr uint32_t BITS = sizeof(T)*8;
	constexpr uint32_t RADIX = 8;
	constexpr uint32_t BUCKETS = (1L << RADIX);

	auto DIGIT = [](T A, T B){
		return (((A) >> (BITS - ((B) + 1) * RADIX)) & (BUCKETS - 1));
	};

	for (size_t w = BITS / RADIX; w-- > 0;) {
		size_t count[BUCKETS + 1] = {0};

		for (size_t i = 0; i < len; ++i)
			++count[DIGIT(array[i], w) + 1];

		for (size_t j = 1; j < BUCKETS - 1; ++j)
			count[j + 1] += count[j];

		for (size_t i = 0; i < len; ++i) {
			size_t cnt = count[DIGIT(array[i], w)];
			aux[cnt] = array[i];
			if constexpr (use_idx) {
				aux2[cnt] = idx[i];
			}
			++count[DIGIT(array[i], w)];
		}

		for (size_t i = 0; i < len; ++i) {
			array[i] = aux[i];
			if constexpr (use_idx) {
				idx[i] = aux2[i];
			}
		}
	}
}

/// straight forward radix sort.
/// \tparam List    input list to sort
/// \tparam use_idx if set to true, additionally an array will used to restore the original sorting. Currently unusable
/// \param L
template<typename List, bool use_idx=false>
void vv_radix_sort(List &L) {
	using T = typename List::value_type;

	const uint64_t len = L.size();
	static T *aux1 = (T *) malloc(sizeof(T) * 8 * len / 8);
	static size_t *aux2;
	static size_t *idx;
	if constexpr (use_idx) {
		aux2 = (size_t *) malloc(sizeof(size_t) * len);
		idx = (size_t *) malloc(sizeof(size_t) * len);
	}
	static uint64_t old_len = 0;

	if (old_len > len) {
		aux1 = (T *) realloc(aux1, sizeof(T) * len / 8);
		if constexpr (use_idx) {
			aux2 = (size_t *) realloc(aux2, sizeof(size_t) * len);
			idx = (size_t *) realloc(idx, sizeof(size_t) * len);
		}
	}

	return vv_radix_sort<T, use_idx>(L.data(), idx, aux1, aux2, len);
}


// Std Sort Implementation.
template<class T>
class Std_Sort_Binary_Container {
private:
	Std_Sort_Binary_Container() {};

public:
	void static sort(std::vector <T> &data, const uint64_t k_lower, const uint64_t k_higher) {
#if defined(SORT_PARALLEL)
		auto exec = std::execution::par;
#endif

		if (k_higher - k_lower > 64) {
			const uint64_t lower = T::round_down_to_limb(k_lower);
			const uint64_t upper = T::round_down_to_limb(k_higher);
			const uint64_t lmask = T::higher_mask(k_lower);
			const uint64_t umask = T::lower_mask(k_higher);

			std::sort(
#if defined(SORT_PARALLEL)
					exec,
#endif
					data.begin(), data.end(),
					[lower, upper, lmask, umask](const auto &e1, const auto &e2) {
#if defined(SORT_INCREASING_ORDER)
						return e1.is_lower_ext2(e2, lower, upper, lmask, umask);
#else
						return e1.is_greater_ext2(e2, lower, upper, lmask, umask);
#endif
					}
			);
		} else {
			const uint64_t limb = T::round_down_to_limb(k_lower);
			const uint64_t mask = T::higher_mask(k_lower) & T::lower_mask(k_higher);
			std::sort(
#if defined(SORT_PARALLEL)
					exec,
#endif
					data.begin(), data.end(),
					[limb, mask](const auto &e1, const auto &e2) {
#if defined(SORT_INCREASING_ORDER)
						return e1.is_lower_simple2(e2, limb, mask);
#else
						return e1.is_greater_simple2(e2, limb, mask);
#endif
					}
			);
		}
	}
};

// T should be `BinaryContainer<XXX>` instance.
template<class T>
class Bucket_Sort_Binary_Container_Single_Limb {
private:
	constexpr static uint64_t k = 2;

	Bucket_Sort_Binary_Container_Single_Limb() {};
public:
	inline constexpr static double f(const T e, const uint64_t limb, const uint64_t mask, const uint64_t msb) {
		return double(64 - __builtin_clzll((e.data()[limb]) & mask)) / double(msb);
	}

	static void sort(std::vector <T> &data, const uint64_t k_lower, const uint64_t k_higher) {
		ASSERT(k_lower < k_higher && k_higher - k_lower < 64);

		const uint64_t limb = T::round_down_to_limb(k_lower);
		const uint64_t mask = T::higher_mask(k_lower) & T::lower_mask(k_higher);
		const uint16_t msb = 64 - __builtin_clzll(mask);

		std::array <std::vector<T>, k> buckets;

		// pre allocate expected size of each bucket
		const uint64_t pre_bucket_len = data.size() / k;
		for (int i = 0; i < k; ++i) {
			buckets[i].reserve(pre_bucket_len);
		}

		// place every element in a bucket.
		for (int i = 0; i < data.size(); ++i) {
			const auto e = data[i];
			double t1 = (f(e, limb, mask, msb) * double(k)) - double(1.0);
			buckets[t1].push_back(e);
		}

		// Now sort the buckets
		uint64_t ctr = 0;
		for (int i = 0; i < k; ++i) {
			// maybe some other sorting algorithm
			Std_Sort_Binary_Container<T>::sort(buckets[i], k_lower, k_higher);

			// copy the sorted data back.
			for (int j = 0; j < buckets[i].size(); ++j) {
				data[ctr] = buckets[i][j];
				ctr += 1;
			}
		}
	}
};

/// Sorts the labels in the list like this:
///			  n-k-l 			n-k
///				b0		b1		b2
///		[		|		|		]
///					^		^------ Sorted via std::sort in the buckets
///					|---- Will be sorted in the buckets.
/// \tparam List
/// \tparam bucket_size max. numbers of elements to store in each bucket.
/// \tparam b0
/// \tparam b1
/// \tparam b2
template<class List, const uint64_t bucket_size, const uint64_t b0, const uint64_t b1, const uint64_t b2>
class Bucket_Sort {
private:
	typedef typename List::ElementType Element;
	typedef typename List::LabelType Label;
	typedef typename List::LabelContainerType LabelContainerType;
	typedef typename LabelContainerType::ContainerLimbType LimbType;

	// precompute the compare masks and limbs
	constexpr static uint64_t mask1 = (uint64_t(1) << (b1 - b0)) - 1;

public:
	constexpr static uint64_t n_buckets = uint64_t(1) << (b1 - b0);

	std::array <std::array<std::pair < uint64_t, LimbType>, bucket_size>, n_buckets>
	buckets;
	std::array <uint32_t, n_buckets> buckets_load = {{0}};

	void hash(List &L) {
		ASSERT(Label::size() >= b2 && b0 < b1 && b1 <= b2 && b2 - b0 <= 64);

		for (int i = 0; i < n_buckets; ++i) {
			buckets_load[i] = 0;
		}

		// precompute as much as possible
		//ASSERT(0 && "i think here is a mistake. this should be b1 and not b0");
		constexpr LimbType lmask = LabelContainerType::higher_mask(b0);
		constexpr LimbType rmask = LabelContainerType::lower_mask2(b2);
		constexpr int64_t lower_limb = b0 / LabelContainerType::limb_bits_width();
		constexpr int64_t higher_limb = (b2 - 1) / LabelContainerType::limb_bits_width();
		constexpr uint64_t shift = b0 % LabelContainerType::limb_bits_width();

		MADIVE((void *) L.data(), L.get_load() * Element::size() / 8, POSIX_MADV_WILLNEED | MADV_SEQUENTIAL);
		for (uint64_t i = 0; i < L.get_load(); ++i) {
			// get the label
			const Label &e = L[i].get_label();

			// calc stuff
			//const LimbType data = e.data().get_bits(b0, b2);
			const LimbType data = e.data().get_bits(lower_limb, higher_limb, lmask, rmask, shift);
			const uint64_t j = data & mask1;
			ASSERT(j < n_buckets);

			if (buckets_load[j] >= bucket_size)
				continue;

			// store it
			// buckets[j][buckets_load[j]++] = std::pair<uint64_t, LimbType>(i, data);
			buckets[j][buckets_load[j]].first = i;
			buckets[j][buckets_load[j]].second = data;
			buckets_load[j] += 1;
		}

		// sort every bucket
		if constexpr (b1 < b2) {
			for (uint64_t i = 0; i < n_buckets; ++i) {
				std::sort(buckets[i].begin(), buckets[i].begin() + buckets_load[i],
				          [](const auto &e1, const auto &e2) {
					          return e1.second < e2.second;
				          }
				);
			}
		}
	}

	// same function as above but chooses the bits not from
	// n-k-l+l1 <-> n-k-l+l2
	// but from
	// k_lower <-> k_upper
	void hash(List &L, const uint32_t k_lower, const uint32_t k_upper) {
		ASSERT(Label::size() >= k_upper && k_upper - k_lower == b2 - b0);

		for (int i = 0; i < n_buckets; ++i) {
			buckets_load[i] = 0;
		}

		// precompute as much as possible
		const LimbType lmask = LabelContainerType::higher_mask(k_lower);
		const LimbType rmask = LabelContainerType::lower_mask2(k_upper);
		const int64_t lower_limb = k_lower / LabelContainerType::limb_bits_width();
		const int64_t higher_limb = (k_upper - 1) / LabelContainerType::limb_bits_width();
		const uint64_t shift = k_lower % LabelContainerType::limb_bits_width();

		MADIVE((void *) L.data(), L.get_load() * Element::size() / 8, POSIX_MADV_WILLNEED | MADV_SEQUENTIAL);
		for (uint64_t i = 0; i < L.get_load(); ++i) {
			// get the label
			const Label &e = L[i].get_label();

			// calc stuff
			// const LimbType data = e.data().get_bits(k_lower, k_upper);
			const LimbType data = e.data().get_bits(lower_limb, higher_limb, lmask, rmask, shift);
			const uint64_t j = data & mask1;
			ASSERT(j < n_buckets);

			if (buckets_load[j] >= bucket_size)
				continue;

			// store it
			std::pair <uint64_t, LimbType> pair1(i, data);
			buckets[j][buckets_load[j]++] = pair1;
		}

		// sort every bucket
		if (k_lower < k_upper) {
			for (uint64_t i = 0; i < n_buckets; ++i) {
				std::sort(buckets[i].begin(), buckets[i].begin() + buckets_load[i],
				          [](const auto &e1, const auto &e2) {
					          return e1.second < e2.second;
				          }
				);
			}
		}
	}

	void print_stats() {
		double avg_load = 0.0;
		uint64_t max_load = 0;
		for (int i = 0; i < n_buckets; i++) {
			avg_load += buckets_load[i];
			if (max_load < buckets_load[i])
				max_load = buckets_load[i];
		}

		avg_load /= n_buckets;

		std::cout << "Print Stats Bucket Sort:\n";
		std::cout << "avg load: " << avg_load << "\n";
		std::cout << "max load: " << max_load << "\n";
		std::cout << "n_buckets: " << n_buckets << ", bucket_size:" << bucket_size << ", size bucket: "
		          << sizeof(buckets[0]);
		std::cout << "\n";
	}


	// upper is inclusive
	// returns
	bool find(const Label &target, uint64_t *bucket, uint64_t *lower, uint64_t *upper,
	          uint16_t i0 = b0, uint16_t i2 = b2) {
		const LimbType data = target.data().get_bits(i0, i2);
		const uint64_t j = data & mask1;
		ASSERT(j < n_buckets);

		if constexpr (b2 == b1) {
			*bucket = j;
			*lower = 0;
			*upper = buckets_load[j] - 1;
			return buckets_load[j] != 0;
		}


		std::pair <uint64_t, uint64_t> data_target(0, data);

		auto r = std::lower_bound(buckets[j].begin(), buckets[j].begin() + buckets_load[j], data_target,
		                          [](const auto &e1, const auto &e2) {
			                          return e1.second < e2.second;
		                          }
		);

		const auto pos = distance(buckets[j].begin(), r);

		if (r == buckets[j].begin() + buckets_load[j])
			return false; // nothing found

		if (buckets[j][pos].second != data)
			return false;

		*bucket = j;
		*lower = pos;

		if (pos + 1 == buckets_load[j]) {
			*upper = pos;
			return true;
		}

		constexpr uint64_t window = 3;
		for (int i = pos + 1; i < MIN(pos + window, buckets_load[j]); i++) {
			if (buckets[j][i].second != data) {
				*upper = i - 1;
				return true;
			}
		}


		if (pos + window >= buckets_load[j]) {
			*upper = buckets_load[j] - 1;
			return true;
		}

		r = std::upper_bound(buckets[j].begin() + pos, buckets[j].begin() + buckets_load[j], data_target,
		                     [](const auto &e1, const auto &e2) {
			                     return e1.second < e2.second;
		                     }
		);

		const auto pos2 = distance(buckets[j].begin(), r) - 1;
		ASSERT(pos2 + 1 >= pos + window);
		*upper = pos2;
		return true;
	}

	// same as above, with the difference that we ca pass two targets which will be xored together and understood as one target.
	bool find2(const LimbType *target1, const LimbType *target2,
	           uint32_t *bucket, uint32_t *lower, uint32_t *upper) {
		std::cout << "not impl\n";
		return false;
	}

	bool find2(const LimbType *target1, const LimbType *target2, const LimbType *target3,
	           uint32_t *bucket, uint32_t *lower, uint32_t *upper) {
		std::cout << "not impl\n";
		return false;
	}

};


/// Sorts the labels in the list like this:
///				b0		b1		b2
///		[		|		|		]
///					^		^------ Sorted via std::sort in the buckets
///					|---- Will be sorted in the buckets.
struct ConfigParallelBucketSort {
	// is the offset of the l part of the label.
	uint32_t label_offset; // must be this value:  = (G_n - G_k - G_l); E.g. Number of bits to cut of the label
	uint32_t l;            // label `l` size. Aka dumer window

	uint64_t bucket_size;  // number of elements per bucket
	uint64_t nr_buckets;   // number of total buckets in this hashmap
	uint32_t nr_threads;   // how many threads access this datastructure
	uint32_t nr_indices;   // how many indices must each bucket element contain.

	uint16_t b0;
	uint16_t b1;
	uint16_t b2;

	uint8_t lvl;

	/// number of additional l window shifts which needs to be saved in one element.
	/// Note that l window is not exactly the same as the l in Dumer.
	/// IM = Infyk Motwani
	uint8_t IM_nr_views;

	// for description of these flags look at the implementation of `ParallelBucketSort`
	bool STDBINARYSEARCH_SWITCH         = false;
	bool INTERPOLATIONSEARCH_SWITCH     = false;
	bool LINEAREARCH_SWITCH             = false;
	bool USE_LOAD_IN_FIND_SWITCH        = true;

	// Because of the greate abstraction of the hashing and extracting of this hashmap, this flag does not have a direct
	// influence on the code of the hashmap.
	// Still its already in place for further possible usage
	// if set to `true` the hashmap is forced to save the full 128 bit of the label, and not just the maximum b2 bits.
	bool SAVE_FULL_128BIT               = false;

	// if this flag is set to `true`, the internal data of holding just the lpart and indices of the baselist, will
	// be exchanged to a triple holding the lpart, indices of the baselist and additional the label (or another part of it)
	bool EXTEND_TO_TRIPLE               = false;

	// useless empty constructor.
	constexpr ConfigParallelBucketSort() :
			label_offset(0),
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

	//constexpr ConfigParallelBucketSort(const ConfigParallelBucketSort &c);

	constexpr ConfigParallelBucketSort(uint16_t b0, uint16_t b1, uint16_t b2, uint64_t bucket_size, uint64_t nr_buckets,
	                                   uint32_t nr_threads, uint32_t nr_indices, uint32_t label_offset,
	                                   uint32_t l, uint8_t lvl,
	                                   uint8_t nr_IM_views=0,
	                                   bool SBSSW=false, bool ISSW=false, bool LSSW=false, bool USIDSW=true,
									   bool SF128B=false, bool ETT=false) :
			label_offset(label_offset), l(l), bucket_size(bucket_size), nr_buckets(nr_buckets),
			nr_threads(nr_threads), nr_indices(nr_indices), b0(b0), b1(b1), b2(b2),
			 lvl(lvl), IM_nr_views(nr_IM_views),
			STDBINARYSEARCH_SWITCH(SBSSW), INTERPOLATIONSEARCH_SWITCH(ISSW),
			LINEAREARCH_SWITCH(LSSW), USE_LOAD_IN_FIND_SWITCH(USIDSW),
			SAVE_FULL_128BIT(SF128B), EXTEND_TO_TRIPLE(ETT) {}

	/// print some information about the configuration
	void print() const {
		std::cout   << "HM" << nr_indices-1
					<< " bucket_size: " << bucket_size
					<< ", nr_buckets: " << nr_buckets
					<< ", nr_threads: " << nr_threads
		            << ", nr_indices: " << nr_indices
					<< ", b0: " << b0 << ", b1: " << b1 << ", b2: " << b2
					<< ", SAVE_FULL_128BIT: " << SAVE_FULL_128BIT
					<< ", EXTEND_TO_TRIPLE: " << EXTEND_TO_TRIPLE
					<< "\n";
	}
};

template<const ConfigParallelBucketSort &config,
				class ExternalList,                     // TODO
				typename ArgumentLimbType,              // container of the `l`-part
				typename ExternalIndexType,             // container of the indices which point into the baselists
				ArgumentLimbType (* HashFkt)(uint64_t)> // TODO describe
requires HashMapListAble<ExternalList> &&
         std::is_integral<ArgumentLimbType>::value &&
         std::is_integral<ExternalIndexType>::value
class ParallelBucketSort {
public:
	/// nomenclature:
	///		bid = bucket id
	///		tid = thread id
	///		nri	= number of indices
	///		nrt = number of threads
	///		nrb = number of buckets
	typedef typename ExternalList::ElementType                      Element;
	typedef typename ExternalList::LabelType                        Label;
	typedef typename ExternalList::ValueType                        Value;
	typedef typename ExternalList::LabelContainerType               LabelContainerType;
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
	constexpr static uint32_t BucketAlignment(const uint32_t i, const uint32_t alignment,
	                                          const uint32_t sizeofindicesinbytes, const uint32_t sizeofarginbytes) {
		const uint32_t bytes  = i * sizeofindicesinbytes;
		const uint32_t bytes2 = ((bytes + alignment - 1)/alignment)*alignment;
		const uint32_t Anri   = (bytes2-sizeofarginbytes)/sizeofindicesinbytes;
		if (Anri == 0)
			return i;
		if (Anri < i)
			return BucketAlignment(i, 2*alignment, sizeofindicesinbytes, sizeofarginbytes);
		return Anri;
	}

	constexpr static bool ALIGNMENT_SWITCH  = false;
	constexpr static uint64_t ALIGNMENT     = 64;
	constexpr static uint64_t nrb           = config.nr_buckets;
	constexpr static uint32_t nrt           = config.nr_threads;
	constexpr static uint32_t nri           = config.nr_indices;
	// Aligned number of indices.
	constexpr static uint32_t Anri       = ALIGNMENT_SWITCH ? BucketAlignment(nri, ALIGNMENT, sizeof(IndexType)*8, sizeof(ArgumentLimbType)*8) : nri;

	constexpr static uint64_t size_b     = config.bucket_size; // size of each bucket
	constexpr static uint64_t size_t     = size_b / nrt;       // size of each bucket which is accessible by one thread.
	constexpr static uint64_t chunks     = nrb / nrt;          // how many buckets must each thread sort or reset.
	constexpr static uint64_t chunks_size= chunks * size_b;

public:
	// We choose the optimal data container for every occasion.
	// This is the Type which is exported to the outside and denotes the maximum number of elements a bucket can hold.
	// Must be the same as IndexType, because `load` means always an absolute position within the array.
	using LoadType          = IndexType;
	// In contrast to `LoadType` this type does not include the absolute position within the array, but only the
	// relative within the bucket. Can be smaller than `IndexType`
	using LoadInternalType  = TypeTemplate<size_b>;
	// Number of bits needed to hold a hash
	using BucketHashType    = TypeTemplate<uint64_t(1) << (config.b1-config.b0)>;
	using BucketIndexType   = TypeTemplate<nrb * size_b>;
	// Main data container
	using BucketIndexEntry  = std::array<IndexType, Anri>;

	using TripleT                                        = uint64_t;

	/// TODO probably one day i want to change this to a std::tuple.
	using BucketEntry       = typename  std::conditional<config.EXTEND_TO_TRIPLE,
														 triple<ArgumentLimbType, BucketIndexEntry, TripleT>,
														 std::pair<ArgumentLimbType, BucketIndexEntry>>::type;

private:
	// precompute the compare masks and limbs
	constexpr static ArgumentLimbType lmask1    = (~((ArgumentLimbType(1) << b0) - 1));
	constexpr static ArgumentLimbType rmask1    = ((ArgumentLimbType(1) << b1) - 1);
	constexpr static ArgumentLimbType mask1     = lmask1 & rmask1;

	// masks for the [b1, b2] part
	constexpr static ArgumentLimbType lmask2        = (~((ArgumentLimbType(1) << b1) - 1));
	constexpr static ArgumentLimbType rmask2        = ((ArgumentLimbType(1) << b2) - 1);
	constexpr static ArgumentLimbType mask2         = lmask2 & rmask2;
	constexpr static ArgumentLimbType highbitmask   = ArgumentLimbType(1) << (sizeof(ArgumentLimbType)*8-1);
	constexpr static ArgumentLimbType sortrmask2    = ((ArgumentLimbType(1) << (b2 + config.lvl)) - 1) | highbitmask;

	// idea use the leftover space in one `ArgumentLimbType` to store the load.
	// 0      l      align
	// | data | load | index1 | index2 |
	// NOTE:    Cacheline = 64Byte
	constexpr static uint32_t MAXLOAD_SWITCH = 64/sizeof(ArgumentLimbType);

	// NOTE: Currently permanently disabled.
	constexpr static bool SPECIAL_LOAD_DECODING_SWITCH1         = false;//((BitSizeArgumentLimbType - b2) >= (1+BitSizeLoadInternalType)) && size_b < MAXLOAD_SWITCH;
	constexpr static uint32_t BitSizeArgumentLimbType           = sizeof(ArgumentLimbType)*8;
	constexpr static uint32_t BitSizeLoadInternalType           = sizeof(LoadInternalType)*8;
	constexpr static uint32_t DecodedLoadOffset                 = BitSizeArgumentLimbType - BitSizeLoadInternalType;
	constexpr static uint32_t DecodedLoadCheckOffset            = BitSizeArgumentLimbType - BitSizeLoadInternalType -1;
	constexpr static ArgumentLimbType DecodedLoadCheckBitMask   = ArgumentLimbType(1) << (BitSizeArgumentLimbType - BitSizeLoadInternalType - 1);
	constexpr static ArgumentLimbType DecodedLoadMask           = ~((ArgumentLimbType(1) << DecodedLoadOffset) - 1) ;
	constexpr static ArgumentLimbType DecodedLoadCheckMask      = ~((ArgumentLimbType(1) << DecodedLoadCheckOffset) - 1) ;

	inline LoadInternalType get_decoded_load1(const BucketIndexType bucket_offset) {
		// Note implicit cast here, if
		return (__buckets[bucket_offset].first & DecodedLoadMask) >> DecodedLoadOffset;
	}
	inline ArgumentLimbType get_decoded_loadcheck1(const BucketIndexType bucket_offset) {
		return (__buckets[bucket_offset].first & DecodedLoadCheckMask) >> DecodedLoadCheckOffset;
	}

	// DO NOT USE IT.
	constexpr static bool STDBINARYSEARCH_SWITCH        = config.STDBINARYSEARCH_SWITCH;

	// replace binary search with smarter interpolation search
	//size_b >= 1000
	constexpr static bool INTERPOLATIONSEARCH_SWITCH    = config.INTERPOLATIONSEARCH_SWITCH;

	// switch to a linear search insteat of binary/interpolation in special cases. Number completely arbitrary
	//((b1 != b2) && (size_t<15)) ? true : false;
	constexpr static bool LINEAREARCH_SWITCH            = config.LINEAREARCH_SWITCH;

	// if set to true an additional mem fetch the load array will be made.
	constexpr static bool USE_LOAD_IN_FIND_SWITCH       = config.USE_LOAD_IN_FIND_SWITCH;

	// if set to true most of the load/write instruction will be replaced with instructions which to not touch a cache
	// NOTE: each l-window entry of the data vector MUST at least 8 Byte aligned
	constexpr static bool CACHE_STREAM_SWITCH           = false;

	// Set this flag to true to save the full 64 bit, regardless of the given b2 value. This allows the tree to
	// directly check if the weight threshold exceeded.
	constexpr static bool SAVE_FULL_128BIT               = config.SAVE_FULL_128BIT;

	// Set this flag to true to extend the fundamental datastructure by an additional datatype (__uint128_t) to hold
	// additional information.
	constexpr static bool EXTEND_TO_TRIPLE               = config.EXTEND_TO_TRIPLE;

	// Indyk Motwani Nearest Neighbor Search:
	// How many additional l windows should this hashmap hold?
	constexpr static bool IM_SWITCH         = config.IM_nr_views != 0;
	constexpr static uint32_t IM_nr_views   = config.IM_nr_views;

	// nr of bits of each view
	constexpr static uint32_t IM_bits_view  = b1;
	constexpr static uint32_t IM_bits       = IM_bits_view*IM_nr_views;

	// some little helper functions:
	// returns the offset of a thread into the load array
	inline uint64_t thread_offset(const uint32_t tid) { return tid * nrt; }

	inline uint64_t bucket_offset(const BucketHashType bid) { return bid * size_b; }

	inline uint64_t bucket_offset(const uint32_t tid, const BucketHashType bid) {
		ASSERT(bid < nrb && tid < nrt);
		const uint64_t r = bid * size_b + tid * size_t;
		ASSERT(r < nrb * size_b);
		ASSERT(size_t <= size_b);
		return r;
	}

	// accumulate the bucket load over all threads
	// can be called multithreaded
	inline void acc_bucket_load(const BucketHashType bid) {
		ASSERT(bid < nrb);

		LoadType load = 0;

		for (uint32_t tid = 0; tid < nrt; ++tid) {
			load += get_bucket_load(tid, bid);
		}

		acc_buckets_load[bid] = load;
	}

public:

	// increments the bucket load by one
	inline void inc_bucket_load(const uint32_t tid, const BucketHashType bid) {
		ASSERT(tid < nrt && bid < nrb);
		buckets_load[tid*nrb + bid] += 1;
	}

	inline LoadType get_bucket_load(const uint32_t tid, const BucketHashType bid) {
		ASSERT(tid < nrt && bid < nrb);
		if constexpr (nrt != 1) {
			return buckets_load[tid * nrb + bid];
		} else {
			return buckets_load[bid];
		}
	}

	// IMPORTANT: Call `acc_bucket_load` first
	inline LoadType get_bucket_load(const BucketHashType bid) {
		ASSERT(bid < nrb);
		if constexpr (nrt != 1) {
			return acc_buckets_load[bid];
		} else {
			return buckets_load[bid];
		}
	}

	inline BucketHashType hash(const ArgumentLimbType data) {
		const BucketHashType bid = ((data & mask1) >> config.b0);
		ASSERT(bid < nrb);
		return bid;
	}

	// Datacontainers
	alignas(PAGE_SIZE) std::vector<BucketEntry> __buckets;
	alignas(PAGE_SIZE) std::vector<LoadInternalType> buckets_load;
	alignas(PAGE_SIZE) std::vector<LoadInternalType> acc_buckets_load; // TODO merge?

	ParallelBucketSort() {
		//TODO only correct if not ternary static_assert((uint64_t(nrb)*uint64_t(size_b)) < uint64_t(std::numeric_limits<IndexType>::max()));
		static_assert((size_b%nrt) == 0);
		static_assert(nrt <= nrb);
		static_assert(size_t <= size_b);
		//TODO this is only valid in non ternary static_assert((uint64_t(1) << uint64_t (b1 - b0)) <= uint64_t (nrb));
		static_assert(b0 < b1);
		static_assert(b1 <= b2);

		// hard limit. The internal code is not able to process more bits.
		static_assert(b2 <= 128, "Sorry nothing i can do. If you want to match on more than 128 bits, i think you have bigger problems than this limitations.");

		// make sure that the user does not want to match on more coordinates that the container hold.
		// TODO reactivate static_assert(b2 <= sizeof(ArgumentLimbType)*8, "make sure that T is big enough...");


		// make sure that from all the given search method only one is active
		static_assert(!(LINEAREARCH_SWITCH && INTERPOLATIONSEARCH_SWITCH));

		// make sure that we actually have some coordinates to search on, if the user wants to do a linear/different search.
		static_assert(!((b1 != b2) && LINEAREARCH_SWITCH));
		static_assert(!((b1 != b2) && INTERPOLATIONSEARCH_SWITCH));

		// make sure that the given size parameter are sane.
		static_assert(size_b != 0);

		// make sure that the correct element type is passed as a type to the hashmap.
		if constexpr (SAVE_FULL_128BIT) {
			static_assert(sizeof(ArgumentLimbType) == 128/8, "pass a `__uint128_t` to the hashmap as base type");
		}

		// only allow one of these flags acitvated.
		static_assert(SAVE_FULL_128BIT + EXTEND_TO_TRIPLE <= 1, "only one of these flags is allowed");

		if constexpr(nrt != 1) {
			// This restriction is only needed if we have more than one thread. Or we have to overwrite the buckets with -1.
			// The problem is, that i use lvl to get for `hm2` an extra bit
			// into the searching routine. This is needed because other wise the las element
			// of a bucket is indistinguishable from the -1 element. Especially
			// if only 1 bit or none is used in the last lvl.
			static_assert(config.lvl + config.b2 < 64);
		}

		if constexpr(IM_SWITCH) {
			// Explanation:
			//  Additionally to the l bits on which the MMT algorithm normally matches in the last lvl, we save
			//  `nr_IM_views` distinct such views. So essentially each `ArgumentLimbType` holds in fact l1 + `nr_IM_views` * l2
			// bits of a label. Or to put it more easily the last  l1 + `nr_IM_views` * l2 bits of each label
			// is copied inside.
			static_assert((IM_nr_views * IM_bits_view) <= 64);
		}

		if ((1 << (b2 - b1)) / nrb > size_b)
			std::cout << "WHAT DOES THIS MEAN???\n";

		__buckets.resize(nrb * size_b);
		buckets_load.resize(nrb * nrt);

		if constexpr(nrt != 1) {
			acc_buckets_load.resize(nrb);
		}

#if !defined(BENCHMARK) && !defined(NO_LOGGING)
		std::cout << "HM" << nri << "\n";
		std::cout << "\tb0=" << b0 << ", b1=" << b1 << ", b2=" << b2 << "\n";
		std::cout << "\tsize_b=" << size_b << ", size_t=" << size_t << ", chunks=" << chunks << "\n";
		std::cout << "\tAnri=" << Anri << "\n";
		std::cout << "\tBucketHashType: " << sizeof(BucketHashType)*8 << "Bits\n";
		std::cout << "\tBucketIndexType: " << sizeof(BucketIndexType)*8 << "Bits\n";
		std::cout << "\tBucketEntry: " << sizeof(BucketEntry)*8 << "Bits\n";
		std::cout << "\tArgumentLimbType: " << sizeof(ArgumentLimbType)*8 << "Bits\n";
		std::cout << "\tIndexType: " << sizeof(IndexType)*8 << "Bits\n";
		std::cout << "\tLoadType: " << sizeof(LoadType)*8 << "Bits\n";
		std::cout << "\tLoadInternalType: " << sizeof(LoadInternalType)*8 << "Bits\n";
		std::cout << "\tLabelType: " << sizeof(Label)*8 << "Bits\n";
		std::cout << "\tSPECIAL_LOAD_DECODING_SWITCH1:" << SPECIAL_LOAD_DECODING_SWITCH1 << "\n";
		std::cout << "\tINTERPOLATIONSEARCH_SWITCH:" << INTERPOLATIONSEARCH_SWITCH << "\n";
		std::cout << "\tSTDBINARYSEARCH_SWITCH:" << STDBINARYSEARCH_SWITCH << "\n";
		std::cout << "\tLINEAREARCH_SWITCH:" << LINEAREARCH_SWITCH << "\n";
		std::cout << "\tUSE_LOAD_IN_FIND_SWITCH:" << USE_LOAD_IN_FIND_SWITCH << "\n";
		std::cout << "\tIM_SWITCH:" << IM_SWITCH << "\n";
		std::cout << "\tALIGNMENT_SWITCH:" << ALIGNMENT_SWITCH << "\n";
#endif
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
	void hash(const List &L, const uint32_t tid) {
		ASSERT(tid < config.nr_threads);
		constexpr static uint32_t loffset               = config.label_offset;
		constexpr static uint32_t loffset64             = loffset / 64;
		constexpr static uint32_t lshift                = (loffset - (loffset64 * 64));     // this is also with some bitmasking possible.
		constexpr static uint32_t size_label            = sizeof(LabelContainerType);       // Size of an `element`. Needed to calculate the correct offset of an label within the list.
		constexpr static __uint128_t labelmask2         = ((__uint128_t(1) << config.l) - 1);

		const uint64_t b_tid = L.size() / nrt; // blocksize of each thread
		const uint64_t s_tid = tid * b_tid;        // Starting point of each process
		// Starting point of the list pointer. Points to the first Label within the list.

		// Instead of access the array each time in the loop, we increment an number by the length between two l-parts
		uint64_t Lptr = (uint64_t) L.data_label() + (s_tid * size_label) + (loffset64 *8);

		ArgumentLimbType data;
		__uint128_t data2;

		IndexType pos[1];
		for (uint64_t i = s_tid; i < ((tid == nrt-1) ? L.size() : s_tid + b_tid); ++i) {
			data2 = *((__uint128_t *) Lptr);
			data2 >>= lshift;
			data2 &= labelmask2;
			data = data2;

			// in the special case if IndykMotwani Hashing, we need to rotate the l1 and l2 windows
			if constexpr(IM_SWITCH) {
				// we do not allow searching in the IndykMotwani NN case
				static_assert(config.b2 == config.b1);

				constexpr ArgumentLimbType mask = (ArgumentLimbType(1) << config.l) - 1;
				data = ((data >> (config.l-b1)) ^ (data << b1)) & mask;
			}

			Lptr += size_label;
			//ASSERT(check_label(data, L, i));

			pos[0] = i;
			insert(data, pos, tid);
		}
	}

	/// Abstract version of
	/// \tparam H
	/// \param L
	/// \param tid
	/// \param h
	template<class H>
	void hash(const List &L, const uint64_t load, const uint32_t tid, H h) {
		ASSERT(tid < config.nr_threads);

		const std::size_t s_tid = L.start_pos(tid);
		const std::size_t e_tid = s_tid + load;

		ArgumentLimbType data;
		IndexType pos[1];
		for (uint64_t i = s_tid; i < e_tid; ++i) {
			data = h(L.data_label(i));
			pos[0] = i;
			insert(data, pos, tid);
		}
	}

	// the ase above but using an extractor and a hasher
	template<class Extractor>
	void hash1(const List &L, const uint64_t load, const uint32_t tid, Extractor e) {
		ASSERT(tid < config.nr_threads);

		const std::size_t s_tid = L.start_pos(tid);
		const std::size_t e_tid = s_tid + load;

		ArgumentLimbType data;
		IndexType pos[1];
		for (std::size_t i = s_tid; i < e_tid; ++i) {
			data = e(L.data_label(i));
			pos[0] = i;
			insert1(data, pos, tid);
		}
	}

	// the ase above but using an extractor and a hasher
	template<class Extractor, class Extractor2>
	void hash_extend_to_triple(const List &L, const uint64_t load, const uint32_t tid,
							   Extractor e, Extractor2 et) {
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
			const LoadType load      = get_bucket_load(tid, bid);
			if (size_t - load == 0) {
				continue;
			}

			const BucketIndexType bucketOffset = bucket_offset(tid, bid) + load;
			inc_bucket_load(tid, bid);

			__buckets[bucketOffset].first = data;
			memcpy(&__buckets[bucketOffset].second, pos, nri * sizeof(IndexType));
			__buckets[bucketOffset].third = et(L.data_label(i));
		}
	}

	/// \param data l part of the label IMPORTANT MUST BE ONE LIMB
	/// \param pos	pointer to the array which should be copied into the internal data structure to loop up elements in the baselists
	/// \param tid	thread_id
	/// \param add_option additional option, which can be used is various different ways, depending on what flags are activated.
	///			- SPECIAL_LOAD_DECODING_SWITCH1
	///				currently disabled because to slow
	///			- IMSWITCH
	///				unused
	///			- CACHE_STREAM_SWITCH
	///				unused
	/// returns 0 if full, 1 if successfully inserted.
	uint32_t insert(const ArgumentLimbType data, const IndexType *npos, const uint32_t tid) {
		ASSERT(tid < config.nr_threads);
		// Note this code is currently deactivated, because its tends to be slower than the normal approach.
//		if constexpr(SPECIAL_LOAD_DECODING_SWITCH1) {
//			const BucketHashType bid = hash(data);
//			BucketIndexType bucketOffset = bucket_offset(tid, bid);
//			const ArgumentLimbType loadcheck = get_decoded_loadcheck1(bucketOffset);
//			const BucketIndexType load = (loadcheck&1) == add_option ? (loadcheck >> 1) : 0;
//
//			if (size_t - load == 0) {
//				return 0;
//			}
//
//			ArgumentLimbType bla = ((((load+1) << 1) ^ add_option) << DecodedLoadCheckOffset);
//			__buckets[bucketOffset].first = (__buckets[bucketOffset].first&rmask2)^bla;;
//			bucketOffset += load;
//			__buckets[bucketOffset].first = (data&rmask2)^bla;
//			memcpy(&__buckets[bucketOffset].second, npos, nri * sizeof(IndexType));
//			return 1;
//		}

		const BucketHashType bid = hash(data);
		const LoadType load      = get_bucket_load(tid, bid);

		// early exit if a bucket is full.
		if (size_t - uint64_t (load) == 0) {
			return 0;
		}

		const BucketIndexType bucketOffset = bucket_offset(tid, bid) + load;
		inc_bucket_load(tid, bid);


		if constexpr (CACHE_STREAM_SWITCH) {
			//MM256_STREAM64(&(__buckets[bucketOffset].first), uint64_t (data))
			//MM256_STREAM64(&__buckets[bucketOffset].second, npos)
			//memcpy(&__buckets[bucketOffset].second, npos, nri * sizeof(IndexType));
		} else {
			__buckets[bucketOffset].first = data;
			memcpy(&__buckets[bucketOffset].second, npos, nri * sizeof(IndexType));
		}

		return 1;
	}

	uint32_t insert1(const ArgumentLimbType data, const IndexType *npos, const uint32_t tid) {
		ASSERT(tid < config.nr_threads);
		const BucketHashType bid = HashFkt(data);
		const LoadType load      = get_bucket_load(tid, bid);
		if (size_t - load == 0) {
			return 0;
		}

		const BucketIndexType bucketOffset = bucket_offset(tid, bid) + load;
		inc_bucket_load(tid, bid);

		__buckets[bucketOffset].first = data;
		memcpy(&__buckets[bucketOffset].second, npos, nri * sizeof(IndexType));
		return 1;
	}

	/// Only sort a single bucket. Make sure that you call this function for every bucket.
	// Assumes more buckets than threads
	void sort_bucket(const BucketHashType bid) {
		ASSERT(bid < nrb);
		ASSERT(((bid + 1) * size_b) <= (nrb * size_b));

		const uint64_t start = bid   * size_b;
		const uint64_t end   = start + size_b;

		// chose the optimal sorting algorithm depending on the underlying `ArgumentLimbType`. If `__uint128_t` is needed
		// because if `l > 64` we have to fall back to the std::sort algorithm.
		if constexpr (std::is_same_v<ArgumentLimbType, __uint128_t >) {
			std::sort(__buckets.begin() + start,
			          __buckets.begin() + end,
			          [](const auto &e1, const auto &e2) -> bool {
				          // well this is actually not completely correct. This checks on the lowest b2 bits. But actually
				          // it should check on the bits [b1, b2]. I need to do this because otherwise the element -1
				          // is indistinguishable from something tha is -1 on [b1, b2].
				          // Additionally, we use the highest bit too.
				          return (e1.first & sortrmask2) < (e2.first & sortrmask2);
			          }
			);
		} else {
			ska_sort(__buckets.begin() + start,
			         __buckets.begin() + end,
			         [](auto e) -> ArgumentLimbType {
				         // distinguish between the two cases if we are in the first/intermediate lvl
				         if constexpr(nri == 1) {
					         return e.first & sortrmask2;
				         } else {
					         return e.first & sortrmask2;
				         }
			         }
			);
		}

		if constexpr(nrt != 1) {
			// after the sorting we can accumulate the load of each bucket over all threads.
			acc_bucket_load(bid);
		}
	}

	/// A little helper function, which maps a thread (its assumed that nr_threads <= nr_buckets) to a set of buckets,
	/// which this thread needs to sort.
	inline void sort(const uint32_t tid) {
		ASSERT(tid < config.nr_threads);
		if constexpr ((config.nr_threads == 1) && (config.b2 == config.b1)) {
			// Fastpath. If the hash function maps onto the full length and
			// we only have one thread there is nothing to do.
			return;
		}

		if constexpr ((config.nr_threads > 1) && (config.b2 == config.b1)) {
			// in the special case that (config.b2 == config.b1) but threads > 1 we can memcpy
			// everything "max-load" elements down.
			// We just have to do it in a per thread manner.
			for (uint64_t bid = tid * chunks; bid < ((tid + 1) * chunks); ++bid) {
				const uint64_t offset = bid * size_b;
				uint64_t load_offset = get_bucket_load(0, bid);
				uint64_t thread_offset = size_t;
				for (uint64_t i = 0; i < nrt -1; i++) {
					uint64_t load2 = get_bucket_load(i+1, bid);

					// Note these to mem regions can overlap.
					memcpy(__buckets.data() + offset + load_offset,
					       __buckets.data() + offset + thread_offset,
					       load2 * sizeof(BucketEntry));

					load_offset += load2;
					thread_offset += size_t;
				}

				// After we `sorted` everything , we have to accumulate the load over each thread.
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


	// returns -1 on error/nothing found. Else the position.
	// IMPORTANT: load` is the actual load + bid*size_b
	BucketIndexType find(const ArgumentLimbType &data, LoadType &load) {
		const BucketHashType bid = hash(data);
		const BucketIndexType boffset = bid * size_b;   // start index of the bucket in the internal data structure

		// Depend on how we encoded the load parameter, read it.
		if constexpr(SPECIAL_LOAD_DECODING_SWITCH1) {
			// IMPORTANT: threads not implemented.
			// Note that this approach is slower
			load = get_decoded_load1(bid*size_b + 0);
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
				// Prefetch data. The idea is that we know that in the special where we dont have to sort every element
				// in the bucket is a match. So we preload it, if the whole bucket fits into one cacheline.
				//if constexpr(size_b < MAXLOAD_SWITCH) {
				//	__builtin_prefetch(__buckets.data() + boffset, 0 , 0);
				//}

				// Check if the last element is really not a -1.
				// NOTE: this check makes propably no sense after a few runs of the algorithm.
				ASSERT(__buckets[boffset + load - 1].first != ArgumentLimbType(-1));

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
		if constexpr(INTERPOLATIONSEARCH_SWITCH) {
			// if this switch is enabled, we are doing an interpolation search.
			auto r = lower_bound_interpolation_search2(__buckets.begin() + boffset, __buckets.begin() + load, data_target,
			                                           [](const BucketEntry &e1) -> ArgumentLimbType {
				                                           return (e1.first & mask2)>>b1; // See note std::lower_bound
			                                           }
			);

			if (r == (__buckets.begin() + load)) return -1;
			const BucketIndexType pos = distance(__buckets.begin(), r);
			if ((__buckets[pos].first & mask2) != (data & mask2)) return -1;
			return pos;
		}

		// on huge arrays with small bucket size it's maybe a good idea to switch on linear search.
		if constexpr(LINEAREARCH_SWITCH) {
			BucketIndexType pos = boffset;
			ASSERT(pos < load);

			const ArgumentLimbType data2 = data&mask2;
			while((__buckets[pos].first&mask2) != data2) {
				pos += 1;
			}

			if (pos == load)
				return -1;

			ASSERT(pos < load);
			return pos;
		}

		// if every other search function is not adaptable go back to the good old binary search.
		typename std::vector<BucketEntry>::iterator r;
		if constexpr(STDBINARYSEARCH_SWITCH) {
			//fallback implementation: binary search.
			r = std::lower_bound(__buckets.begin() + boffset, __buckets.begin() + load, data_target,
			                     [](const auto &e1, const auto &e2) {
				                     // well this is actually not completely correct. This checks on the lowest b2 bits. But actually
				                     // it should check on the bits [b1, b2].
				                     return (e1.first & mask2) < (e2.first & mask2);
			                     }
			);
		} else {
			r = lower_bound_monobound_binary_search(__buckets.begin() + boffset, __buckets.begin() + load, data_target,
			                                        [](const BucketEntry &e1) -> ArgumentLimbType {
				                                        return e1.first & mask2;// See note std::lower_bound
			                                        }
			);
		}



		if (r == (__buckets.begin() + load)) return -1;
		const BucketIndexType pos = distance(__buckets.begin(), r);
		if ((__buckets[pos].first & mask2) != (data & mask2)) return -1;

		ASSERT(pos < load);
		return pos;
	}

	BucketIndexType find1(const ArgumentLimbType &data, LoadType &load) {
		const BucketHashType bid = HashFkt(data);
		const BucketIndexType boffset = bid * size_b;   // start index of the bucket in the internal data structure

		// Depend on how we encoded the load parameter, read it.
		if constexpr(SPECIAL_LOAD_DECODING_SWITCH1) {
			// IMPORTANT: threads not implemented.
			// Note that this approach is slower
			load = get_decoded_load1(bid*size_b + 0);
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
				// Prefetch data. The idea is that we know that in the special where we dont have to sort every element
				// in the bucket is a match. So we preload it, if the whole bucket fits into one cacheline.
				//if constexpr(size_b < MAXLOAD_SWITCH) {
				//	__builtin_prefetch(__buckets.data() + boffset, 0 , 0);
				//}

				// Check if the last element is really not a -1.
				// NOTE: this check makes propably no sense after a few runs of the algorithm.
				// ASSERT(__buckets[boffset + load - 1].first != ArgumentLimbType(-1));

				load += boffset;
				return boffset;
			}

			// Note the implicit type cast
			return -1;
		}

		// Second fastpath.
		if (load == 0) { return -1; }
		load += boffset;

		constexpr static BucketIndexEntry dummy = {{0}};
		BucketEntry data_target(data, dummy);

		// on huge arrays with huge bucket size is maybe a good idea to switch to interpolation search
		if constexpr(INTERPOLATIONSEARCH_SWITCH) {
			// if this switch is enabled, we are doing an interpolation search.
			// return InterpolationSearch(data, boffset, load);
			auto r = lower_bound_interpolation_search2(__buckets.begin() + boffset, __buckets.begin() + load, data_target,
			                                           [](const BucketEntry &e1) -> ArgumentLimbType {
				                                           return (e1.first & mask2)>>b1; // See note std::lower_bound
			                                           }
			);

			if (r == (__buckets.begin() + load)) return -1;
			const BucketIndexType pos = distance(__buckets.begin(), r);
			if ((__buckets[pos].first & mask2) != (data & mask2)) return -1;
			return pos;
		}

		// on huge arrays with small bucket size it's maybe a good idea to switch on linear search.
		if constexpr(LINEAREARCH_SWITCH) {
			BucketIndexType pos = boffset;
			ASSERT(pos < load);

			const ArgumentLimbType data2 = data&mask2;
			while((__buckets[pos].first&mask2) != data2) {
				pos += 1;
			}

			if (pos == load)
				return -1;

			ASSERT(pos < load);
			return pos;
		}

		// if every other search function is not adaptable go back to the good old binary search.
		typename std::vector<BucketEntry>::iterator r;
		if constexpr(STDBINARYSEARCH_SWITCH) {
			//fallback implementation: binary search.
			r = std::lower_bound(__buckets.begin() + boffset, __buckets.begin() + load, data_target,
			                     [](const auto &e1, const auto &e2) {
				                     // well this is actually not completely correct. This checks on the lowest b2 bits. But actually
				                     // it should check on the bits [b1, b2].
				                     return (e1.first & mask2) < (e2.first & mask2);
			                     }
			);
		} else {
			r = lower_bound_monobound_binary_search(__buckets.begin() + boffset, __buckets.begin() + load, data_target,
			                                        [](const BucketEntry &e1) -> ArgumentLimbType {
				                                        return e1.first & mask2;// See note std::lower_bound
			                                        }
			);
		}



		if (r == (__buckets.begin() + load)) return -1;
		const BucketIndexType pos = distance(__buckets.begin(), r);
		if ((__buckets[pos].first & mask2) != (data & mask2)) return -1;

		ASSERT(pos < load);
		return pos;
	}

	// Diese funktion muss
	//  - npos anpassen
	//  - pos danach inkrementieren.
	/// this function traverses from a given position until the value [b1, b2] changes. This is done by
	///		first: copy the current positions array into the output array `npos`
	/// \tparam lvl the starting point from which `npos` is copied from.
	/// \tparam ctr how many elements are copied.
	/// \param pos	current position of a match on the coordinates [b1, ..., b2) between `data` and __buckets[pos].first
	/// \param npos	output array if length `nri`
	/// \return
	template<uint8_t lvl, uint8_t ctr>
	ArgumentLimbType traverse(const ArgumentLimbType &data, IndexType &pos, IndexType *npos, const LoadType &load) {
		ASSERT(lvl < 4 && npos != nullptr && pos < load);
		if (pos >= (load-1)) {
			pos = IndexType(-1);
			return pos;
		}

		// This memcpy copies the indices (= positions of elements within the baselist) into the output array `npos`
		// the position and length of what needs to be copies is specified by the template parameters `lvl` and `ctr`.
		// Whereas `lvl` specifies the starting position of the memcpy and `ctr` the length.
		memcpy(&npos[lvl], __buckets[pos].second.data(), ctr * sizeof(IndexType));
		ASSERT(npos[0] != IndexType(-1) && npos[1] != IndexType(-1));

		const ArgumentLimbType ret = data ^__buckets[pos].first;

		// check if the next element is still the same.
		const ArgumentLimbType a = __buckets[pos    ].first & rmask2;
		const ArgumentLimbType b = __buckets[pos + 1].first & rmask2;

		pos += 1;
		if (a != b)
			pos = IndexType(-1);

//		if (b == ArgumentLimbType(-1))
//			pos = IndexType(-1);

		return ret;
	}

	template<uint8_t lvl, uint8_t ctr>
	void traverse_drop(const ArgumentLimbType &data, IndexType &pos, IndexType *npos, const LoadType &load) {
		ASSERT(npos != nullptr && pos < load);
		memcpy(&npos[lvl], __buckets[pos].second.data(), ctr * sizeof(IndexType));
//		if (npos[0] == IndexType(-1)){
//			print(hash(data), 100);
//		}

		ASSERT(npos[0] != IndexType(-1) && npos[1] != IndexType(-1) && npos[2] != IndexType(-1) && npos[3] != IndexType(-1));

		if (pos >= (load-1)) {
			pos = IndexType(-1);
			return;
		}

		ArgumentLimbType a = __buckets[pos    ].first & rmask2;
		ArgumentLimbType b = __buckets[pos + 1].first & rmask2;
		pos += 1;
		if (a != b)
			pos = IndexType(-1);
	}

	/// IMPORTANT: Only call this function by exactly one thread.
	void reset() {
		// for instructions please read the comment of the function `void reset(const uint32_t tid)`
		memset(buckets_load.data(), 0, nrb * nrt * sizeof(LoadInternalType));
		// std::fill(buckets_load.begin(), buckets_load.end(), 0);

		if constexpr(nrt == 1){
			memset(__buckets.data(), -1, nrb * size_b * sizeof(BucketEntry));
			// std::fill(__buckets.begin(), __buckets.end(), -1);
		}
	}

	/// Each thread resets a number of blocks
	void reset(const uint32_t tid) {
		ASSERT(tid < nrt);
		ASSERT((tid * chunks_size) < (nrb*size_b));

		memset((void *) (uint64_t(buckets_load.data()) + (tid * nrb * sizeof(LoadInternalType))),
		       0, nrb * sizeof(LoadInternalType));

		// We do not have to reset the accumulated load array nor the buckets, because we don't depend on the
		// data written there in the case were we have multiple threads. Only in the case were we have to go into the sorting
		// function we have to reset the buckets.
		if constexpr((b2 != b1) && (nrt != 1)){
			memset((void *) (uint64_t(__buckets.data()) + (tid * chunks_size * sizeof(BucketEntry))),
			       -1, chunks_size * sizeof(BucketEntry));
		}
	}

	// only print one bucket.
	void print(const uint64_t bid, const int32_t nr_elements) {
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
		std::cout << "\n" << std::flush;
	}

	void print(const uint32_t tid) {
		LoadType load = 0;

		// calc the load percentage of each bucket
		for (uint64_t i = tid * chunks; i < ((tid + 1) * chunks); ++i) {
			load += buckets_load[chunks + i];
		}

#pragma omp critical
		{std::cout << "ThreadID: " << tid << ", load: " << double(load)/chunks << "\n"; }

		for (uint64_t i = 0; i < nrt; i++) {
			for (uint64_t j = 0; j < nrb; j++) {
				std::cout << "TID: " << i << " BID: " << j << " BucketLoad: " << get_bucket_load(i, j) << "\n";
			}
		}

		std::cout << "size: " << nrb * size_b * sizeof(BucketEntry) + nrb * nrt * sizeof(uint64_t) << "Byte\n";
	}

	void print() {
		LoadType load = 0;

		bool flag = false;
		// IMPORTANT: Functions is only useful if `acc_bucket_load` was called befor.
		for (uint64_t bid = 0; bid < nrb; ++bid) {
			load += get_bucket_load(bid);
			if (load > 0) {
				flag = true;
			}
		}

		if(!flag) {
			std::cout << "the every bucket is empty flag\n";
		}

		std::cout << "HM" << nri - 1 << " #Elements:" << load << " Avg. load per bucket : " << (double(load)/nrb) << " elements/bucket";
		std::cout << ", size: " << double(size()) / (1024.0 * 1024.0) << "MB";
		std::cout << "\n" << std::flush;
	}

	// debug function.
	uint64_t get_first_non_empty_bucket(const uint32_t tid) {
		for (uint64_t i = 0; i < nrb; i++) {
			if (get_bucket_load(tid, i) != 0) {
				return i;
			}
		}

		return -1;
	}

	// check if each bucket is correctly sorted
	// input argument is the starting position within the `__buckets` array
	bool check_sorted(const uint64_t start, const uint64_t load) {
		ASSERT(start < (nrb*size_b));
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
						          << "\n" << std::flush;
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
					          << "\n" << std::flush;
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

	bool check_sorted() {
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
	bool check_label(const ArgumentLimbType data, const List &L, const uint64_t i, const uint32_t k_lower=-1, const uint32_t k_upper=-1) {
		const bool flag = (k_lower == -1) && (k_upper == -1);
		const uint64_t nkl   = flag ? config.label_offset : k_lower;
		const uint32_t limit = flag ? config.l : k_upper - k_lower;
		ArgumentLimbType d = data;

		for (uint64_t j = 0; j < limit; ++j) {
			if ((d & 1) != L.data_label(i).data()[j + nkl]) {
				printbinary(d);
				std::cout << "  calc+extracted label\n" << L.data_label(i) << "\nlabel wrong calc\n" << std::flush;
				return false;
			}

			d >>= 1;
		}

		return true;
	}

	// returns the load summed over all buckets
	uint64_t load() {
		uint64_t load = 0;

		if (omp_get_thread_num() == 0) {

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

	uint64_t size() {
		return __buckets.size();
	}

	uint64_t bytes() {
		uint64_t ret = sizeof(BucketEntry) * __buckets.size();
		ret += sizeof(LoadType) * buckets_load.size();
		ret += sizeof(LoadType) * acc_buckets_load.size();
		return ret;
	}
};

template<const ConfigParallelBucketSort &config,
		class ExternalList,
        typename ArgumentLimbType,                      // container of the `l`-part
		typename ExternalIndexType,                     // container of the indices which point into the baselists
		ArgumentLimbType (* HashFkt)(ArgumentLimbType),         // hash function, the output is needed to work as a bucket index
		ArgumentLimbType (* HashSearchFkt)(ArgumentLimbType)>   // similar as the Hashfkt but with the difference its only used for search, so can be further simplified
	requires HashMapListAble<ExternalList> &&
	        std::is_integral<ArgumentLimbType>::value &&
			std::is_integral<ExternalIndexType>::value
class ParallelLockFreeBucketSort {
public:
	/// nomenclature:
	///		bid = bucket id
	///		tid = thread id
	///		nri	= number of indices
	///		nrt = number of threads
	///		nrb = number of buckets
	typedef typename ExternalList::ElementType                      Element;
	typedef typename ExternalList::LabelType                        Label;
	typedef typename ExternalList::ValueType                        Value;
	typedef typename ExternalList::LabelContainerType               LabelContainerType;
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
	constexpr static uint32_t BucketAlignment(const uint32_t i, const uint32_t alignment,
											  const uint32_t sizeofindicesinbytes, const uint32_t sizeofarginbytes) {
		const uint32_t bytes  = i * sizeofindicesinbytes;
		const uint32_t bytes2 = ((bytes + alignment - 1)/alignment)*alignment;
		const uint32_t Anri   = (bytes2-sizeofarginbytes)/sizeofindicesinbytes;
		if (Anri == 0)
			return i;
		if (Anri < i)
			return BucketAlignment(i, 2*alignment, sizeofindicesinbytes, sizeofarginbytes);
		return Anri;
	}

	constexpr static bool ALIGNMENT_SWITCH  = false;
	constexpr static uint64_t ALIGNMENT     = 64;
	constexpr static uint64_t nrb           = config.nr_buckets;
	constexpr static uint32_t nrt           = config.nr_threads;
	constexpr static uint32_t nri           = config.nr_indices;
	// Aligned number of indices.
	constexpr static uint32_t Anri       = ALIGNMENT_SWITCH ? BucketAlignment(nri, ALIGNMENT, sizeof(IndexType)*8, sizeof(ArgumentLimbType)*8) : nri;

	constexpr static uint64_t size_b     = config.bucket_size; // size of each bucket
	constexpr static uint64_t size_t     = size_b / nrt;       // size of each bucket which is accessible by one thread.
	constexpr static uint64_t chunks     = nrb / nrt;          // how many buckets must each thread sort or reset.
	constexpr static uint64_t chunks_size= chunks * size_b;

public:
	// We choose the optimal data container for every occation.
	// This is the Type which is exported to the outside and denotes the maximum number of elements a bucket can hold.
	// Must be the same as IndexType, because `load` means always an absolute position within the array.
	using LoadType          = IndexType;
	// In contrast to `LoadType` this type does not include the absolute position within the array, but only the
	// relative within the bucket. Can be smaller than `IndexType`
	using LoadInternalType  = TypeTemplate<size_b>;
	// Number of bits needed to hold a hash
	using BucketHashType    = TypeTemplate<uint64_t(1) << (config.b1-config.b0)>;
	using BucketIndexType   = TypeTemplate<nrb * size_b>;
	// Main data container
	using BucketIndexEntry  = std::array<IndexType, Anri>;
	using BucketEntry       = std::pair<ArgumentLimbType, BucketIndexEntry>;

private:
	// precompute the compare masks and limbs
	constexpr static ArgumentLimbType lmask1    = (~((ArgumentLimbType(1) << b0) - 1));
	constexpr static ArgumentLimbType rmask1    = ((ArgumentLimbType(1) << b1) - 1);
	constexpr static ArgumentLimbType mask1     = lmask1 & rmask1;

	// masks for the [b1, b2] part
	constexpr static ArgumentLimbType lmask2        = (~((ArgumentLimbType(1) << b1) - 1));
	constexpr static ArgumentLimbType rmask2        = ((ArgumentLimbType(1) << b2) - 1);
	constexpr static ArgumentLimbType mask2         = lmask2 & rmask2;
	constexpr static ArgumentLimbType highbitmask   = ArgumentLimbType(1) << (sizeof(ArgumentLimbType)*8-1);
	constexpr static ArgumentLimbType sortrmask2    = ((ArgumentLimbType(1) << (b2 + config.lvl)) - 1) | highbitmask;

	// idea use the leftover space in one `ArgumentLimbType` to store the load.
	// 0      l      align
	// | data | load | index1 | index2 |
	// NOTE:    Cacheline = 64Byte
	constexpr static uint32_t MAXLOAD_SWITCH = 64/sizeof(ArgumentLimbType);

	// NOTE: Currently permanently disabled.
	constexpr static bool SPECIAL_LOAD_DECODING_SWITCH1         = false;//((BitSizeArgumentLimbType - b2) >= (1+BitSizeLoadInternalType)) && size_b < MAXLOAD_SWITCH;
	constexpr static uint32_t BitSizeArgumentLimbType           = sizeof(ArgumentLimbType)*8;
	constexpr static uint32_t BitSizeLoadInternalType           = sizeof(LoadInternalType)*8;
	constexpr static uint32_t DecodedLoadOffset                 = BitSizeArgumentLimbType - BitSizeLoadInternalType;
	constexpr static uint32_t DecodedLoadCheckOffset            = BitSizeArgumentLimbType - BitSizeLoadInternalType -1;
	constexpr static ArgumentLimbType DecodedLoadCheckBitMask   = ArgumentLimbType(1) << (BitSizeArgumentLimbType - BitSizeLoadInternalType - 1);
	constexpr static ArgumentLimbType DecodedLoadMask           = ~((ArgumentLimbType(1) << DecodedLoadOffset) - 1) ;
	constexpr static ArgumentLimbType DecodedLoadCheckMask      = ~((ArgumentLimbType(1) << DecodedLoadCheckOffset) - 1) ;

	inline LoadInternalType get_decoded_load1(const BucketIndexType bucket_offset) {
		// Note implicit cast here, if
		return (__buckets[bucket_offset].first & DecodedLoadMask) >> DecodedLoadOffset;
	}
	inline ArgumentLimbType get_decoded_loadcheck1(const BucketIndexType bucket_offset) {
		return (__buckets[bucket_offset].first & DecodedLoadCheckMask) >> DecodedLoadCheckOffset;
	}

	// DO NOT USE IT.
	constexpr static bool STDBINARYSEARCH_SWITCH        = config.STDBINARYSEARCH_SWITCH;

	// replace binary search with smarter interpolation search
	constexpr static bool INTERPOLATIONSEARCH_SWITCH    = config.INTERPOLATIONSEARCH_SWITCH;

	// switch to a linear search instead of binary/interpolation in special cases.
	constexpr static bool LINEAREARCH_SWITCH            = config.LINEAREARCH_SWITCH;

	// if set to true an additional mem fetch the load array will be made.
	constexpr static bool USE_LOAD_IN_FIND_SWITCH       = config.USE_LOAD_IN_FIND_SWITCH;

	// if set to true most of the load/write instruction will be replaced with instructions which to not touch a cache
	// NOTE: each l-window entry of the data vector MUST at least 8 Byte aligned
	constexpr static bool CACHE_STREAM_SWITCH           = false;

	// Indyk Motwani Nearest Neighbor Search:
	// How many additional l windows should this hashmap hold?
	constexpr static bool IM_SWITCH         = config.IM_nr_views != 0;
	constexpr static uint32_t IM_nr_views   = config.IM_nr_views;

	// nr of bits of each view
	constexpr static uint32_t IM_bits_view  = b1;
	constexpr static uint32_t IM_bits       = IM_bits_view*IM_nr_views;

	inline uint64_t bucket_offset(const uint32_t tid, const BucketHashType bid) {
		ASSERT(bid < nrb && tid < nrt);
		const uint64_t r = bid * size_b + tid * size_t;
		ASSERT(r < nrb * size_b);
		ASSERT(size_t <= size_b);
		return r;
	}

	// TODO idea: Speed things up indem wir die load kopieren in ein neues array ohne locks. Damit wird find schneller.
	// ONLY CALL THIS FUNCTION TO CALCULATE THE OVERALL LOAD OF THE HM/ NEVER USE IT TO INSERT AN ELEMENT
	inline LoadType get_bucket_load(const BucketHashType bid) {
		ASSERT(bid < nrb);
		return std::min(std::size_t(__buckets_load[bid].load()), std::size_t(size_b));
	}

	// increments the bucket load by one, and return the old value
	inline LoadType inc_bucket_load(const BucketHashType bid) {
		ASSERT(bid < nrb);

		// So is all takes all the magic place. So basically we want to return and unique position where the callee
		// function can save its element.
		// We have to take care that we are not overloading the buckets
#if __cplusplus > 201709L
        // This is like super stupid. Fucking apple...
		return __buckets_load[bid].fetch_add(1, std::memory_order::relaxed);
#else
        return __buckets_load[bid].fetch_add(1);
#endif
	}

public:
	// TODO im single threaded fall den atomic datentyp weg
	// Datacontainers
	alignas(PAGE_SIZE) std::array<BucketEntry, nrb*size_b>            __buckets;
	alignas(PAGE_SIZE) std::array<std::atomic<LoadInternalType>, nrb> __buckets_load;

	ParallelLockFreeBucketSort() {
		// TODO static asserts

		if constexpr(IM_SWITCH) {
			// Explanation:
			//  Additionally to the l bits on which the MMT algorithm normally matches in the last lvl, we save
			//  `nr_IM_views` distinct such views. So essentially each `ArgumentLimbType` holds in fact l1 + `nr_IM_views` * l2
			// bits of a label. Or to put it more easily the last  l1 + `nr_IM_views` * l2 bits of each label
			// is copied inside.
			static_assert((IM_nr_views * IM_bits_view) <= 64);
		}

		if ((1 << (b2 - b1)) / nrb > size_b)
			std::cout << "WHAT DOES THIS MEAN???\n";

		reset();
	}

	// the case above but using an extractor and a hasher
	template<class Extractor>
	void hash(const List &L, const std::size_t load, const uint32_t tid, Extractor e) {
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

	/// \param data l part of the label IMPORTANT MUST BE ONE LIMB
	/// \param pos	pointer to the array which should be copied into the internal data structure to loop up elements in the baselists
	/// \param tid	thread_id
	uint32_t insert(const ArgumentLimbType data, const IndexType *npos, const uint32_t tid) {
		ASSERT(tid < config.nr_threads);
		const BucketHashType bid = HashFkt(data);

		// At this point we uniquely locked the "load" position for the elements
		const LoadType load      = inc_bucket_load(bid);

		// Already full
		if (size_t <= load) {
			return 0;
		}

		const BucketIndexType bucketOffset = bucket_offset(tid, bid) + load;

		__buckets[bucketOffset].first = data;
		memcpy(&__buckets[bucketOffset].second, npos, nri * sizeof(IndexType));
		return 1;
	}

	// returns -1 on error/nothing found. Else the position.
	// IMPORTANT: load` is the actual load + bid*size_b
	BucketIndexType find(const ArgumentLimbType &data, LoadType &load) {
		const BucketHashType bid      = HashFkt(data);
		const BucketIndexType boffset = bid * size_b;   // start index of the bucket in the internal data structure

		// Depend on how we encoded the load parameter, read it.
		if constexpr(SPECIAL_LOAD_DECODING_SWITCH1) {
			// IMPORTANT: threads not implemented.
			// Note that this approach is slower
			load = get_decoded_load1(bid*size_b + 0);
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
				// Prefetch data. The idea is that we know that in the special where we dont have to sort every element
				// in the bucket is a match. So we preload it, if the whole bucket fits into one cacheline.
				//if constexpr(size_b < MAXLOAD_SWITCH) {
				//	__builtin_prefetch(__buckets.data() + boffset, 0 , 0);
				//}

				// Check if the last element is really not a -1.
				// NOTE: this check makes propably no sense after a few runs of the algorithm.
				ASSERT(__buckets[boffset + load - 1].first != ArgumentLimbType(-1));

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

		// on huge arrays with small bucket size it's maybe a good idea to switch on linear search.
		if constexpr(LINEAREARCH_SWITCH) {
			BucketIndexType pos = boffset;
			ASSERT(pos < load);

			const ArgumentLimbType data2 = data&mask2;
			while((__buckets[pos].first&mask2) != data2) {
				pos += 1;
			}

			if (pos == load)
				return -1;

			ASSERT(pos < load);
			return pos;
		}

		// TODO simplifu the search, by using the hash function.
		// if every other search function is not adaptable go back to the good old binary search.
		typename std::array<BucketEntry, nrb*size_b>::iterator r;
		if constexpr(STDBINARYSEARCH_SWITCH) {
			//fallback implementation: binary search.
			r = std::lower_bound(__buckets.begin() + boffset, __buckets.begin() + load, data_target,
			                          [](const auto &e1, const auto &e2) {
	              // well this is actually not completely correct. This checks on the lowest b2 bits. But actually
	              // it should check on the bits [b1, b2].
	              return (e1.first & mask2) < (e2.first & mask2);
	          }
			);
		} else {
			r = lower_bound_monobound_binary_search(__buckets.begin() + boffset, __buckets.begin() + load, data_target,
                 [](const BucketEntry &e1) -> ArgumentLimbType {
                     return e1.first & mask2;
                 }
			);
		}

		if (r == (__buckets.begin() + load)) return -1;
		const BucketIndexType pos = distance(__buckets.begin(), r);
		if ((__buckets[pos].first & mask2) != (data & mask2)) return -1;

		ASSERT(pos < load);
		return pos;
	}

	// Diese funktion muss
	//  - npos anpassen
	//  - pos danach inkrementieren.
	/// this function traverses from a given position until the value [b1, b2] changes. This is done by
	///		first: copy the current positions array into the output array `npos`
	/// \tparam lvl the starting point from which `npos` is copied from.
	/// \tparam ctr how many elements are copied.
	/// \param pos	current position of a match on the coordinates [b1, ..., b2) between `data` and __buckets[pos].first
	/// \param npos	output array if length `nri`
	/// \return
	template<uint8_t lvl, uint8_t ctr>
	ArgumentLimbType traverse(const ArgumentLimbType &data, IndexType &pos, IndexType *npos, const LoadType &load) {
		ASSERT(lvl < 4 && npos != nullptr && pos < load);
		if (pos >= (load-1)) {
			pos = IndexType(-1);
			return pos;
		}

		// This memcpy copies the indices (= positions of elements within the baselist) into the output array `npos`
		// the position and length of what needs to be copies is specified by the template parameters `lvl` and `ctr`.
		// Whereas `lvl` specifies the starting position of the memcpy and `ctr` the length.
		memcpy(&npos[lvl], __buckets[pos].second.data(), ctr * sizeof(IndexType));
		ASSERT(npos[0] != IndexType(-1) && npos[1] != IndexType(-1));

		const ArgumentLimbType ret = data ^__buckets[pos].first;

		// check if the next element is still the same.
		const ArgumentLimbType a = __buckets[pos    ].first & rmask2;
		const ArgumentLimbType b = __buckets[pos + 1].first & rmask2;

		pos += 1;
		if (a != b)
			pos = IndexType(-1);

//		if (b == ArgumentLimbType(-1))
//			pos = IndexType(-1);

		return ret;
	}

	template<uint8_t lvl, uint8_t ctr>
	void traverse_drop(const ArgumentLimbType &data, IndexType &pos, IndexType *npos, const LoadType &load) {
		ASSERT(npos != nullptr && pos < load);
		memcpy(&npos[lvl], __buckets[pos].second.data(), ctr * sizeof(IndexType));
//		if (npos[0] == IndexType(-1)){
//			print(hash(data), 100);
//		}

		ASSERT(npos[0] != IndexType(-1) && npos[1] != IndexType(-1) && npos[2] != IndexType(-1) && npos[3] != IndexType(-1));

		if (pos >= (load-1)) {
			pos = IndexType(-1);
			return;
		}

		ArgumentLimbType a = __buckets[pos    ].first & rmask2;
		ArgumentLimbType b = __buckets[pos + 1].first & rmask2;
		pos += 1;
		if (a != b)
			pos = IndexType(-1);
	}

	/// IMPORTANT: Only call this function by exactly one thread.
	void reset() {
		// for instructions please read the comment of the function `void reset(const uint32_t tid)`
		memset(__buckets_load.data(), 0, nrb * nrt * sizeof(LoadInternalType));
		memset(__buckets.data(), -1, nrb * size_b * sizeof(BucketEntry));
	}

	/// Each thread resets a number of blocks
	void reset(const uint32_t tid) {
		ASSERT(tid < nrt);
		ASSERT((tid * chunks_size) < (nrb*size_b));

		memset((void *) (uint64_t(__buckets_load.data()) + (tid * nrb * sizeof(LoadInternalType))),
		       0, nrb * sizeof(LoadInternalType));

		// We do not have to reset the accumulated load array nor the buckets, because we don't depend on the
		// data written there in the case were we have multiple threads. Only in the case were we have to go into the sorting
		// function we have to reset the buckets.
		if constexpr((b2 != b1) && (nrt != 1)){
			memset((void *) (uint64_t(__buckets.data()) + (tid * chunks_size * sizeof(BucketEntry))),
			       -1, chunks_size * sizeof(BucketEntry));
		}
	}

	/// \param a
	void printbinary(ArgumentLimbType a) {
		print128(a, 128, 128);
	}

	/// \param a
	/// \param l1
	/// \param l2
	void print128(__uint128_t a, const uint16_t l1, const uint16_t l2) {
		const ArgumentLimbType mask = 1;
		for (uint16_t i = 0; i < 128; ++i) {
			if (a & mask) {
				std::cout << "1";
			} else {
				std::cout << "0";
			}
			a >>= 1;
			if ((i == l1 - 1) || (i == l2 - 1))
				std::cout << " ";
		}
	}

	/// \param bid
	void print(uint64_t bid=-1) {
		const uint64_t sbid = bid == -1 ? 0 : bid;
		const uint64_t ebid = bid == -1 ? nrb : bid+1;
		for (uint64_t i = sbid; i < ebid; ++i) {
			std::cout << "Bucket: " << i << "\n";
			for (uint64_t j = 0; j < size_b; ++j) {
				uint64_t pos = i*size_b + j;

				printbinary(__buckets[pos].first);
				std::cout << ", " << __buckets[pos].second[0] << "\n";
				//std::cout << __buckets[pos].first << ", " << __buckets[pos].second[0] << "\n";
			}
		}
	}

	// returns the load summed over all buckets
	uint64_t load() {
		uint64_t load = 0;

		if (omp_get_thread_num() == 0) {

#pragma omp critical
			{
				for (uint64_t _nrb = 0; _nrb < nrb; ++_nrb) {
					load += get_bucket_load(_nrb);
				}
			}
		}

#pragma omp barrier
		return load;
	}

	// Returns the number of total elements
	uint64_t size() {
		return size_b*nrb;
	}

	uint64_t bytes() {
		uint64_t ret = sizeof(BucketEntry) * __buckets.size();
		ret += sizeof(LoadType) * __buckets_load.size();
		return ret;
	}
};

template<class T>
class Shell_Sort_Binary_Container_Single_Limb {
private:
	constexpr static uint64_t k = 2;

	Shell_Sort_Binary_Container_Single_Limb() {};
public:
	inline constexpr static double f(const T e, const uint64_t limb, const uint64_t mask, const uint64_t msb) {
		return double(64 - __builtin_clzll((e.data()[limb]) & mask)) / double(msb);
	}

	static void sort(std::vector <T> &data, const uint64_t k_lower, const uint64_t k_higher) {
		ASSERT(k_lower < k_higher && k_higher - k_lower < 64);
		T tmp;

		const uint64_t limb = T::round_down_to_limb(k_lower);
		const uint64_t mask = T::higher_mask(k_lower) & T::lower_mask(k_higher);

		uint64_t i, j;
		for (i = 3; i < data.size(); ++i) {
			tmp = data[i];
			for (j = i; j >= 3 && (T::cmp_ternary_simple2(data[j - 3], tmp, limb, mask) > 0); j -= 3) {
				data[j] = data[j - 3];
			}
			data[j] = tmp;
		}

		for (i = 1; i < data.size(); ++i) {
			tmp = data[i];
			for (j = i; j >= 1 && (T::cmp_ternary_simple2(data[j - 1], tmp, limb, mask) > 0); j -= 1) {
				data[j] = data[j - 1];
			}
			data[j] = tmp;
		}

	}
};

template<class T>
class Radix_Sort_Binary_Container_Single_Limb {
private:
	constexpr static uint64_t stack_size = 5;
	constexpr static uint64_t cut_off = 4;

	constexpr static uint64_t char_stop = 255;
	constexpr static uint64_t SWITCH_TO_SHELL = 20;

	static auto access(const T &a, const uint64_t limb, const uint64_t mask) {
		ASSERT(limb < a.limbs());
		return a.data()[limb] & mask;
	}

public:
	static void sort(std::vector <T> &data, const uint64_t k_lower, const uint64_t k_higher) {
		const uint64_t limb = T::round_down_to_limb(k_lower);
		const uint64_t mask = T::higher_mask(k_lower) & T::lower_mask(k_higher);

		T tmp;
		uint64_t stack_pointer, target,  offset = 0;
		// uint64_t last_position, last_value, next_value, a, b;
		uint64_t counts[256] = {0}, offsets[256] = {0}, starts[256], ends[256];
		uint64_t stack[stack_size];
		const uint64_t digit = 0;

		// computing starting positions.
		for (int i = 0; i < data.size(); ++i) {
			counts[access(data[i], limb, mask) & 256] += 1;
		}

		// Compute offsets
		for (int i = 0; i < char_stop; ++i) {
			offsets[i] = offset;
			starts[i] = offsets[i];
			offset += counts[i];
		}

		for (int i = 0; i < char_stop; i++) {
			ends[i] = offsets[i + 1];
		}
		ends[char_stop] = data.size();

		for (int x = 0; x < char_stop; ++x) {
			while (offsets[x] < ends[x]) {
				if ((access(data[offsets[x]], limb, mask) + digit) == x) {
					offsets[x] += 1;
				} else {
					stack_pointer = 0;
					stack[stack_pointer] = offsets[x];
					stack_pointer += 1;
					target = access(data[offsets[x]], limb, mask) + digit;

					while (target != x && stack_pointer < stack_size) {
						stack[stack_pointer] = offsets[target];
						offsets[target] += 1;
						target = access(data[stack[stack_pointer]], limb, mask) + digit;
						stack_pointer++;
					}

					if (stack_pointer != stack_size) {
						offsets[x] += 1;
					}
					stack_pointer--;

					tmp = data[stack[stack_pointer]];

					while (stack_pointer) {
						data[stack[stack_pointer]] = data[stack[stack_pointer - 1]];
						stack_pointer--;
					}

					data[stack[0]] = tmp;
				}
			}
		}

		if (digit < cut_off) {
			for (int i = 0; i <= char_stop; ++i) {
				if (ends[i] - starts[i] > SWITCH_TO_SHELL) {
					// sort(data[starts[i]], k_lower, k_higher);
				} else {
					if (ends[i] - starts[i] <= 1)
						continue;

					// Shell_Sort_Binary_Container_Single_Limb<T>::sort(data[starts[i]], k_lower, k_higher);
				}
			}
		} else {
			for (int i = 0; i <= char_stop; ++i) {
				if (ends[i] - starts[i] > 1) {
					// Shell_Sort_Binary_Container_Single_Limb<T>::sort(data[starts[i]], k_lower, k_higher);
				}
			}
		}
	}
};

#endif //SMALLSECRETLWE_SORT_H



