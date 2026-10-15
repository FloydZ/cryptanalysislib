#ifndef CRYPTANALYSISLIB_SORT_COUNTING_SORT_H
#define CRYPTANALYSISLIB_SORT_COUNTING_SORT_H

/// original taken from https://github.com/eloj/radix-sorting
/// but with a lot of optimizations from FloydZ
#include "algorithm/histogram.h"


namespace cryptanalysislib {
	namespace internal {
		/// below this size the histogram is computed with a simple loop.
		/// this switching point was selected by a benchmark in
		/// `bench/algorithm/histogram.cpp`.
		/// Benchmarked on: AMD Ryzen 5 7600X 6-Core Processor
		constexpr static size_t counting_sort_switch = 512;
	} // end namespace internal

	/// sorts `arr` in place (increasing) by counting the occurrences of every
	/// byte value and writing them back in order. O(size + 256).
	/// \param arr[in/out]: bytes to sort
	/// \param size[in]: number of bytes
	constexpr static void counting_sort_u8(uint8_t *arr,
	                                       const size_t size) noexcept {
		size_t cnt[256] = {0};
		size_t i;

		if (size >= internal::counting_sort_switch) {
			cryptanalysislib::algorithm::histogram(cnt, arr, size);
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
} // end namespace cryptanalysislib

#endif
