#ifndef CRYPTANALYSISLIB_SORT_COUNTING_SORT_H
#define CRYPTANALYSISLIB_SORT_COUNTING_SORT_H

/// original taken from https://github.com/eloj/radix-sorting
/// but with a lot optimizations from FloydZ
#include <cstdint>
#include "simd/simd.h"




///
/// \param arr
/// \param size
/// \return
constexpr static void counting_sort_u8(uint8_t *arr,
                            		  const size_t size) {
	size_t cnt[256] = { 0 };
	size_t i;

#ifdef USE_AVX2
	for (; i < size ; ++i) {
		cnt[arr[i]]++;
	}
#else
	for (i = 0 ; i < size ; ++i) {
		cnt[arr[i]]++;
	}
#endif

	i = 0;
	for (size_t a = 0 ; a < 256 ; ++a) {
		while (cnt[a]--) {
			arr[i++] = a;
		}
	}
}

constexpr static void counting_sort_u8_stable(uint8_t *output,
        									  const uint8_t *input,
                                              const size_t size) {
	size_t cnt[256] = { 0 };
	size_t i;

	// Count number of occurrences of each octet.
	for (i = 0 ; i < size ; ++i) {
		cnt[input[i]]++;
	}

	// Calculate prefix sums.
	size_t a = 0;
	for (uint32_t j = 0 ; j < 256u; ++j) {
		size_t b = cnt[j];
		cnt[j] = a;
		a += b;
	}

	// Sort elements
	for (i = 0 ; i < size; ++i) {
		// Get the key for the current entry.
		uint8_t k = input[i];
		// Find the location this entry goes into in the output array.
		size_t dst = cnt[k];
		// Copy the current entry into the right place.
		output[dst] = input[i];
		// Make it so that the next 'k' will be written after this one.
		// Since we process source entries in increasing order, this makes us a stable sort.
		cnt[k]++;
	}
}
#endif
