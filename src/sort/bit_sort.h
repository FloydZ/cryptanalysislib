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

/// TODO move somewhere useful
template <std::size_t X>
inline constexpr bool is_power_of_two = X > 0 && (X & (X - 1)) == 0;



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

/// \tparam K
/// \tparam Ps...
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
    // TODO: use SIMD wrapper
    template <typename K>
    static constexpr SortIndex numElemsPerVec = 64 / sizeof(K);

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

		while (readPosLeft <= readPosRight) {
			const auto keyVec = keyVecStore;
			const auto payloadVec = payloadVecStore;
			const auto [sortMaskLeft, sortMaskRight] =
			        getSortMasks<Up, IsHighestBit, IsRightSide, K, T>(keyVec, bitNo);
			const SortIndex numElemsToLeft = sortMaskLeft.popcnt();// simd::kpopcnt(sortMaskLeft);
			const SortIndex numElemsToRight = _numElemsPerVec - numElemsToLeft;
			const bool areEnoughElemsFreeLeft =
			        (readPosLeft - writePosLeft) >= numElemsToLeft;
			if (areEnoughElemsFreeLeft) {
				keyVecStore = SK::load(&keys[readPosRight]);
				        //simd::loadu<_numElemsPerVec * sizeof(K)>(&keys[readPosRight]);
				payloadVecStore = ST::load(&payloads[readPosRight]);
				        //std::make_tuple(simd::loadu<_numElemsPerVec * sizeof(Ps)>(
				        //        &payloads[readPosRight])...);
				readPosRight -= _numElemsPerVec;
			} else {
				keyVecStore = SK::load(&keys[readPosLeft]);
				        //simd::loadu<_numElemsPerVec * sizeof(K)>(&keys[readPosLeft]);
				payloadVecStore = ST::load(&payloads[readPosLeft]);
				        //std::make_tuple(simd::loadu<_numElemsPerVec * sizeof(Ps)>(
				        //        &payloads[readPosLeft])...);
				readPosLeft += _numElemsPerVec;
			}
			compress_store_left_right(
			        writePosLeft, writePosRight - numElemsToRight + 1, sortMaskLeft,
			        sortMaskRight, keyVec, payloadVec, keys, payloads);
			writePosLeft += numElemsToLeft;
			writePosRight -= numElemsToRight;
		}

		const SortIndex numElemsRest = readPosRight + _numElemsPerVec - readPosLeft;

		Mask<_numElemsPerVec> restMask = 0;
	    SK keyVecRest;
		ST payloadVecRest;
		if (numElemsRest != 0) {
			// restMask = simd::kshiftr(simd::knot(simd::Mask<_numElemsPerVec>(0)), _numElemsPerVec - numElemsRest);
            restMask = -1ul >> (_numElemsPerVec - numElemsRest);
			// keyVecRest = simd::maskz_loadu<_numElemsPerVec * sizeof(K)>(restMask, &keys[readPosLeft]);
            keyVecRest = SK::maskz_unaligned_load(&keys[readPosLeft], restMask);
			// payloadVecRest = std::make_tuple(simd::maskz_loadu<_numElemsPerVec * sizeof(Ps)>(restMask, &payloads[readPosLeft])...);
            payloadVecRest = ST::maskz_unaligned_load(&payloads[readPosLeft], restMask);
			readPosLeft += numElemsRest;
		}

		if (numElems >= _numElemsPerVec) {
			const auto [sortMaskLeft, sortMaskRight] =
			        getSortMasks<Up, IsHighestBit, IsRightSide, K, T>(keyVecStore,
			                                                              bitNo);
			const SortIndex numElemsToLeft = sortMaskLeft.popcnt();//simd::kpopcnt(sortMaskLeft);
			const SortIndex numElemsToRight = _numElemsPerVec - numElemsToLeft;
			compress_store_left_right(
			        writePosLeft, writePosRight - numElemsToRight + 1, sortMaskLeft,
			        sortMaskRight, keyVecStore, payloadVecStore, keys, payloads);
			writePosLeft += numElemsToLeft;
			writePosRight -= numElemsToRight;
		}

		if (numElemsRest != 0) {
			auto [sortMaskLeftRest, sortMaskRightRest] =
			        getSortMasks<Up, IsHighestBit, IsRightSide, K, T>(keyVecRest, bitNo);
            // simd::kand(sortMaskLeftRest, restMask);
			sortMaskLeftRest = sortMaskLeftRest & restMask;
            // simd::kand(sortMaskRightRest, restMask);
			sortMaskRightRest = sortMaskRightRest & restMask;
            // simd::kpopcnt(sortMaskLeftRest);
			const SortIndex numElemsToLeftRest = sortMaskLeftRest.popcnt(); 
			const SortIndex numElemsToRightRest = numElemsRest - numElemsToLeftRest;
			compress_store_left_right(writePosLeft, writePosLeft + numElemsToLeftRest,
			                          sortMaskLeftRest, sortMaskRightRest, keyVecRest,
			                          payloadVecRest, keys, payloads);
			writePosLeft += numElemsToLeftRest;
			writePosRight -= numElemsToRightRest;
		}
		return writePosLeft;
	}

private:
    template<bool Up,
             bool IsHighestBit, 
             bool IsRightSide,
             typename K>
    static inline std::tuple<Mask<numElemsPerVec<K>>,
                             Mask<numElemsPerVec<K>>>
    getSortMasks(const SIMDSelector<K> keyVec,
                 const std::size_t bitNo) {
       using SK = SIMDSelector<K>;
    	if constexpr (bitDirUp<K, Up, IsHighestBit, IsRightSide>()) {
           // simd::test_bit(keyVec, bitNo);
    		const auto sortMaskRight = SK::test(keyVec, bitNo);
           //simd::knot(sortMaskRight);
    		const auto sortMaskLeft = !sortMaskRight;
    		return std::make_tuple(sortMaskLeft, sortMaskRight);
    	} else {
           // simd::test_bit(keyVec, bitNo);
    		const auto sortMaskLeft = SK::test(keyVec, bitNo);
           // simd::knot(sortMaskLeft);
    		const auto sortMaskRight = !sortMaskLeft;
    		return std::make_tuple(sortMaskLeft, sortMaskRight);
    	}
    }
   
    /// \tparam K[in]
    /// \tparam T[in]
    /// \param leftPos
    /// \param rightPos
    /// \param leftMask
    /// \param rightMask
    /// \param keyVec
    /// \param payloadVec
    /// \param keys
    /// \param payloads
    template<typename K,
             typename T>
    static inline void compress_store_left_right(
                            const SortIndex leftPos, 
                            const SortIndex rightPos,
                            const Mask<numElemsPerVec<K>> leftMask,
                            const Mask<numElemsPerVec<K>> rightMask,
                            const SIMDSelector<K> keyVec,
                            const SIMDSelector<T> payloadVec,
                            K *const keys,
                            T *const payloads) {
        using SK = SIMDSelector<K>;
        using TK = SIMDSelector<T>;

        SK::compress(&keys[leftPos], keyVec, leftMask);
        std::apply(
                [&](const auto... payloadVecs) {
        	        (TK::compress(&payloads[leftPos], payloadVecs, leftMask),
        	         ...);
                },
                payloadVec);
        
        SK::compress(&keys[rightPos], keyVec, rightMask);
        std::apply(
                [&](const auto... payloadVecs) {
        	        (TK::compress(&payloads[rightPos], payloadVecs, rightMask),
        	         ...);
                },
                payloadVec);
    }
};


/// \tparam Up[in]:
/// \tparam BitSorter[in]:
/// \tparam CmpSorter[in]:
/// \tparam IsRightSide[in]:
/// \tparam IsHighestBit[in]:
/// \tparam K[in]:
/// \tparam Ps[in]:
/// \param bitNo[in]:
/// \param cmpSortThreshold[in]:
/// \param left[in]:
/// \param right[in]:
/// \param keys[in]:
/// \param payloads[in]:
template<bool Up,
         typename BitSorter,
         typename CmpSorter,
         bool IsRightSide = false,
         bool IsHighestBit = true,
         typename K,
         typename... Ps>
void radixRecursion(const std::size_t bitNo,
                    const SortIndex cmpSortThreshold,
                    const SortIndex left,
                    const SortIndex right,
                    K *const keys,
                    Ps *const... payloads) noexcept {
	if (right - left <= 0) {
		return;
	}
	if (right - left < cmpSortThreshold) {
		CmpSorter::template sort<Up, K, Ps...>(left, right, keys, payloads...);
		return;
	}

	const SortIndex split =
	        BitSorter::template sortBit<Up, IsHighestBit, IsRightSide, K, Ps...>(
	                bitNo, left, right, keys, payloads...);
	if (bitNo > 0) {
		radixRecursion<Up, BitSorter, CmpSorter, IsHighestBit ? false : IsRightSide,
		               false>(bitNo - 1, cmpSortThreshold, left, split - 1, keys,
		                      payloads...);
		radixRecursion<Up, BitSorter, CmpSorter, IsHighestBit ? true : IsRightSide,
		               false>(bitNo - 1, cmpSortThreshold, split, right, keys,
		                      payloads...);
	}
}

/// \tparam Up[in]:
/// \tparam BitSorter[in]:
/// \tparam CmpSorter[in]:
/// \tparam K[in]:
/// \tparam Ps[in]:
/// \param cmpSortThreshold[in]:
/// \param num[in]:
/// \param keys[in]:
/// \param payloads[in]:
template<bool Up = true,
         typename BitSorter = BitSorterSIMD,
         typename CmpSorter = CmpSorterInsertionSort,
         typename K,
         typename... Ps>
void sort(SortIndex cmpSortThreshold, 
          const SortIndex num,
          K *const keys,
          Ps *const... payloads) noexcept {
	radixRecursion
        <Up, BitSorter, CmpSorter>
        (sizeof(K) * 8 - 1, cmpSortThreshold, 0, num - 1, keys, payloads...);
}

/// \tparam Up[in]:
/// \tparam BitSorter[in]:
/// \tparam CmpSorter[in]:
/// \tparam K[in]:
/// \tparam Ps[in]:
/// \param cmpSortThreshold[in]:
/// \param num[in]:
/// \param elements[in]:
template<bool Up,
         typename BitSorter,
         typename CmpSorter,
         typename K,
         typename... Ps>
void sort(SortIndex cmpSortThreshold,
          const SortIndex num,
          DataElement<K, Ps...> *const elements) {
	static_assert(is_power_of_two<sizeof(DataElement<K, Ps...>)>,
	              "size of DataElement<K, Ps...> must be a power of two");
	radixRecursion
        <Up, BitSorter, CmpSorter>
        (sizeof(K) * 8 - 1, cmpSortThreshold, 0, num - 1, elements);
}

/// \tparam Up[in]:
/// \tparam K[in]:
/// \tparam Ps[in]:
/// \param num[in]:
/// \param keys[in]:
/// \param payloads[in]:
template<bool Up = true,
         typename K,
         typename... Ps>
void sort(const SortIndex num,
          K *const keys,
          Ps *const... payloads) {
	sort
        <Up, BitSorterSIMD, CmpSorterInsertionSort>
        (16, num, keys, payloads...);
}
