#ifndef CRYPTANALYSISLIB_SORT_COUNTING_SORT_H
#define CRYPTANALYSISLIB_SORT_COUNTING_SORT_H

/// original taken from https://github.com/eloj/radix-sorting
/// but with a lot optimizations from FloydZ
#include <cstdint>
#include "simd/simd.h"
#include "algorithm/histogram.h"
#include "algorithm/prefixsum.h"


using namespace cryptanalysislib::algorithm;


// this switching point was selected by a benchmark in
// `bench/algorithm/histogram.cpp`.
// Benchmarked on: AMD Ryzen 5 7600X 6-Core Processor
constexpr static size_t switch_ = 512;

/// \param arr
/// \param size
/// \return
constexpr static void counting_sort_u8(uint8_t *arr,
                            		  const size_t size) {
	size_t cnt[256] = { 0 };
	size_t i;

	if (size >= switch_) {
		histogram(cnt, arr, size);
	} else {
		for (i = 0 ; i < size ; ++i) { cnt[arr[i]]++; }
	}

	i = 0;
	for (size_t a = 0 ; a < 256 ; ++a) {
		while (cnt[a]--) {
			arr[i++] = a;
		}
	}
}

constexpr static void counting_sort_stable_u8(uint8_t *output,
        									  const uint8_t *input,
                                              const size_t size) {
	size_t cnt[256] = { 0 };
	size_t i;

	if (size >= switch_) {
		histogram(cnt, input, size);
	} else {
		for (i = 0 ; i < size ; ++i) { cnt[input[i]]++; }
	}

	// Calculate prefix sums.
	prefixsum(cnt, 256);

	// Sort elements
	for (i = 0 ; i < size; ++i) {
		uint8_t k = input[i];
		size_t dst = cnt[k];
		output[dst] = input[i];
		cnt[k]++;
	}
}
#endif
