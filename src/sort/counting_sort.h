#ifndef CRYPTANALYSISLIB_SORT_COUNTING_SORT_H
#define CRYPTANALYSISLIB_SORT_COUNTING_SORT_H

/// original taken from https://github.com/eloj/radix-sorting
/// but with a lot of optimizations from FloydZ
#include "algorithm/histogram.h"


using namespace cryptanalysislib::algorithm;


// this switching point was selected by a benchmark in
// `bench/algorithm/histogram.cpp`.
// Benchmarked on: AMD Ryzen 5 7600X 6-Core Processor
constexpr static size_t switch_ = 512;

/// TODO doc
/// \param arr
/// \param size
/// \return
constexpr static void counting_sort_u8(uint8_t *arr,
                            		  const size_t size) {
	size_t cnt[256] = {0};
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

#endif
