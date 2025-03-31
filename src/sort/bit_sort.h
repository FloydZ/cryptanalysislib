#pragma once

#include <emmintrin.h>
#include <immintrin.h>
#include <popcntintrin.h>
#include <sys/types.h>

#include <algorithm>
#include <array>
#include <bit>
#include <bitset>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <tuple>
#include <type_traits>
#include <typeinfo>

#include "simd/simd.h"

// source: https://github.com/jonicho/simd-radix-sort/blob/main/radixSort.hpp

/// make private?
using SortIndex = ssize_t;

struct CmpSorterInsertionSort {
    /// sorts 'payloads' according to 'keys'
    /// \tparam Up[in]: if `true` sort ascending else descending
    /// \tparam K[in]: type of keys
    /// \tparam Ps[in]: type of payloads
    /// \param keys[in/out]: keys which are compared
    /// \param payloads[in/out]: values which are sorted
	template<bool Up, typename K, typename... Ps>
	constexpr static inline 
    void sort(const SortIndex left,
              const SortIndex right,
	          K *const keys, 
              Ps *const... payloads) noexcept {
		for (SortIndex i = left + 1; i <= right; i++) {
			const K key = keys[i];
			const std::tuple<Ps...> payload = std::make_tuple(payloads[i]...);
			SortIndex j = i;
			while (j > left && (Up ? key < keys[j - 1] : key > keys[j - 1])) {
				keys[j] = keys[j - 1];
				((payloads[j] = payloads[j - 1]), ...);
				j--;
			}
			keys[j] = key;
			std::apply([&](const Ps &...p) { ((payloads[j] = p), ...); }, payload);
		}
	}
};

template<typename K, typename... Ps>
struct DataElement {
	K key;
	std::tuple<Ps...> payloads;
	bool operator<(const DataElement &other) const { return key < other.key; }
	bool operator>(const DataElement &other) const { return key > other.key; }
};

// specialization of DataElement for no payloads, because an empty
// tuple still uses one byte of space, we don't want that
template<typename K>
struct DataElement<K> {
	K key;
	bool operator<(const DataElement &other) const { return key < other.key; }
	bool operator>(const DataElement &other) const { return key > other.key; }
};


///
struct BitSorterSIMD {
	template<const bool Up,
             const bool IsHighestBit, 
             const bool IsRightSide, 
             typename K,
	         typename T>
	static inline SortIndex sortBit(const std::size_t bitNo, 
                                    const SortIndex left,
	                                const SortIndex right, 
                                    K *const keys,
	                                T *const payloads) {
	    using SK = SIMDSelector<K>;
	    using ST = SIMDSelector<T>;
        constexpr size_t _numElemsPerVec = SK::size();

		const SortIndex numElems = right - left + 1;

		SortIndex readPosLeft = left;
		SortIndex readPosRight = right - _numElemsPerVec + 1;
		SortIndex writePosLeft = left;
		SortIndex writePosRight = right;

		SK keyVecStore;
		ST payloadVecStore;
		if (numElems >= _numElemsPerVec) {
            keyVecStore     = SK::unaligned_load(&keys[readPosLeft]);
            payloadVecStore = ST::unaligned_load(&payloads[readPosLeft]);
		}

		//while (readPosLeft <= readPosRight) {
		//	const auto keyVec = keyVecStore;
		//	const auto payloadVec = payloadVecStore;
		//	const auto [sortMaskLeft, sortMaskRight] =
		//	        getSortMasks<Up, IsHighestBit, IsRightSide, K, Ps...>(keyVec, bitNo);
		//	const SortIndex numElemsToLeft = simd::kpopcnt(sortMaskLeft);
		//	const SortIndex numElemsToRight = _numElemsPerVec - numElemsToLeft;
		//	const bool areEnoughElemsFreeLeft =
		//	        (readPosLeft - writePosLeft) >= numElemsToLeft;
		//	if (areEnoughElemsFreeLeft) {
		//		keyVecStore =
		//		        simd::loadu<_numElemsPerVec * sizeof(K)>(&keys[readPosRight]);
		//		payloadVecStore =
		//		        std::make_tuple(simd::loadu<_numElemsPerVec * sizeof(Ps)>(
		//		                &payloads[readPosRight])...);
		//		readPosRight -= _numElemsPerVec;
		//	} else {
		//		keyVecStore =
		//		        simd::loadu<_numElemsPerVec * sizeof(K)>(&keys[readPosLeft]);
		//		payloadVecStore =
		//		        std::make_tuple(simd::loadu<_numElemsPerVec * sizeof(Ps)>(
		//		                &payloads[readPosLeft])...);
		//		readPosLeft += _numElemsPerVec;
		//	}
		//	compress_store_left_right(
		//	        writePosLeft, writePosRight - numElemsToRight + 1, sortMaskLeft,
		//	        sortMaskRight, keyVec, payloadVec, keys, payloads...);
		//	writePosLeft += numElemsToLeft;
		//	writePosRight -= numElemsToRight;
		//}

		//const SortIndex numElemsRest = readPosRight + _numElemsPerVec - readPosLeft;

		//simd::Mask<_numElemsPerVec> restMask = 0;
		//simd::Vec<K, _numElemsPerVec * sizeof(K)> keyVecRest;
		//std::tuple<simd::Vec<Ps, _numElemsPerVec * sizeof(Ps)>...> payloadVecRest;
		//if (numElemsRest != 0) {
		//	restMask = simd::kshiftr(simd::knot(simd::Mask<_numElemsPerVec>(0)),
		//	                         _numElemsPerVec - numElemsRest);
		//	keyVecRest = simd::maskz_loadu<_numElemsPerVec * sizeof(K)>(
		//	        restMask, &keys[readPosLeft]);
		//	payloadVecRest =
		//	        std::make_tuple(simd::maskz_loadu<_numElemsPerVec * sizeof(Ps)>(
		//	                restMask, &payloads[readPosLeft])...);
		//	readPosLeft += numElemsRest;
		//}

		//if (numElems >= _numElemsPerVec) {
		//	const auto [sortMaskLeft, sortMaskRight] =
		//	        getSortMasks<Up, IsHighestBit, IsRightSide, K, Ps...>(keyVecStore,
		//	                                                              bitNo);
		//	const SortIndex numElemsToLeft = simd::kpopcnt(sortMaskLeft);
		//	const SortIndex numElemsToRight = _numElemsPerVec - numElemsToLeft;
		//	compress_store_left_right(
		//	        writePosLeft, writePosRight - numElemsToRight + 1, sortMaskLeft,
		//	        sortMaskRight, keyVecStore, payloadVecStore, keys, payloads...);
		//	writePosLeft += numElemsToLeft;
		//	writePosRight -= numElemsToRight;
		//}

		//if (numElemsRest != 0) {
		//	auto [sortMaskLeftRest, sortMaskRightRest] =
		//	        getSortMasks<Up, IsHighestBit, IsRightSide, K, Ps...>(keyVecRest,
		//	                                                              bitNo);
		//	sortMaskLeftRest = simd::kand(sortMaskLeftRest, restMask);
		//	sortMaskRightRest = simd::kand(sortMaskRightRest, restMask);
		//	const SortIndex numElemsToLeftRest = simd::kpopcnt(sortMaskLeftRest);
		//	const SortIndex numElemsToRightRest = numElemsRest - numElemsToLeftRest;
		//	compress_store_left_right(writePosLeft, writePosLeft + numElemsToLeftRest,
		//	                          sortMaskLeftRest, sortMaskRightRest, keyVecRest,
		//	                          payloadVecRest, keys, payloads...);
		//	writePosLeft += numElemsToLeftRest;
		//	writePosRight -= numElemsToRightRest;
		//}
		//return writePosLeft;
	}

private:
	// template<bool Up,
    //          bool IsHighestBit, 
    //          bool IsRightSide, typename K,
	//          typename... Ps>
	// static inline std::tuple<simd::Mask<numElemsPerVec<K, Ps...>>,
	//                          simd::Mask<numElemsPerVec<K, Ps...>>>
	// getSortMasks(const simd::Vec<K, numElemsPerVec<K, Ps...> * sizeof(K)> keyVec,
	//              const std::size_t bitNo) {
	// 	if constexpr (bitDirUp<K, Up, IsHighestBit, IsRightSide>()) {
	// 		const auto sortMaskRight = simd::test_bit(keyVec, bitNo);
	// 		const auto sortMaskLeft = simd::knot(sortMaskRight);
	// 		return std::make_tuple(sortMaskLeft, sortMaskRight);
	// 	} else {
	// 		const auto sortMaskLeft = simd::test_bit(keyVec, bitNo);
	// 		const auto sortMaskRight = simd::knot(sortMaskLeft);
	// 		return std::make_tuple(sortMaskLeft, sortMaskRight);
	// 	}
	// }
	// template<typename K, typename... Ps>
	// static inline void compress_store_left_right(
	//         const SortIndex leftPos, const SortIndex rightPos,
	//         const simd::Mask<numElemsPerVec<K, Ps...>> leftMask,
	//         const simd::Mask<numElemsPerVec<K, Ps...>> rightMask,
	//         const simd::Vec<K, numElemsPerVec<K, Ps...> * sizeof(K)> keyVec,
	//         const std::tuple<simd::Vec<Ps, numElemsPerVec<K, Ps...> * sizeof(Ps)>...>
	//                 payloadVec,
	//         K *const keys, Ps *const... payloads) {
	// 	simd::mask_compressstoreu(&keys[leftPos], leftMask, keyVec);
	// 	std::apply(
	// 	        [&](const auto... payloadVecs) {
	// 		        (simd::mask_compressstoreu(&payloads[leftPos], leftMask, payloadVecs),
	// 		         ...);
	// 	        },
	// 	        payloadVec);

	// 	simd::mask_compressstoreu(&keys[rightPos], rightMask, keyVec);
	// 	std::apply(
	// 	        [&](const auto... payloadVecs) {
	// 		        (simd::mask_compressstoreu(&payloads[rightPos], rightMask,
	// 		                                   payloadVecs),
	// 		         ...);
	// 	        },
	// 	        payloadVec);
	// }
};
