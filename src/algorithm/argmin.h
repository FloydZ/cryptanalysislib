#ifndef CRYPTANALYSISLIB_ALGORITHM_ARGMIN_H
#define CRYPTANALYSISLIB_ALGORITHM_ARGMIN_H

#include <concepts>
#include <cstdlib>
#include <cstdint>
#include <type_traits>

// TODO argmax
namespace cryptanalysislib {
	// TODO only < needed
	/// \tparam T
	/// \param a
	/// \param n
	/// \return
	template<typename T>
#if __cplusplus > 201709L
	    requires std::totally_ordered<T>
#endif
	[[nodiscard]] constexpr static inline size_t argmin(const T *a,
	                                                    const size_t n) noexcept {
		size_t k = 0;
		for (size_t i = 0; i < n; i++) {
			if (a[i] < a[k]) [[unlikely]] {
				k = i;
			}
		}

		return k;
	}

#ifdef USE_AVX2
#include <immintrin.h>
	// TODO write config if pointers are aligned
	// TODO write selection function in cryptanalysis namespace
	// TODO avx512

	///
	// source : https://en.algorithmica.org/hpc/algorithms/argmin/
	[[nodiscard]] constexpr static inline size_t argmin_avx2_i32(const int32_t *a,
	                                               				 const size_t n) noexcept {
		// this bound is arbitrary choosen
		if (n < 8) {
			return argmin<int32_t>(a, n);
		}

		int32_t min = INT_MAX;
		size_t idx = 0;
		__m256i p = _mm256_set1_epi32(min);

		size_t i = 0;
		for (; i+8 <= n; i += 8) {
			__m256i y = _mm256_loadu_si256((__m256i*) &a[i]);
			__m256i mask = _mm256_cmpgt_epi32(p, y);
			if (!_mm256_testz_si256(mask, mask)) { [[unlikely]]
				for (uint32_t j = i; j < i + 8; j++) {
					if (a[j] < min) {
						min = a[idx = j];
					}
				}

				p = _mm256_set1_epi32(min);
			}
		}

		// tail
		for (; i < n; i++) {
			if (a[i] < min) {
				min = a[idx = i];
			}
		}

		return idx;
	}

	/// same algorithm as `argmin_avx2_i32` but with a blocksize of 16.
	/// so 16 elements are loaded from memory in parallel.
	/// \param a
	/// \param n
	/// \return
	[[nodiscard]] constexpr static inline size_t argmin_avx2_i32_bl16(const int32_t *a,
	                                                                  const size_t n) noexcept {
		int32_t min = INT_MAX;
		__m256i p = _mm256_set1_epi32(min);
		size_t i = 0, idx = 0;
		for (; i+16 <= n; i += 16) {
			__m256i y1 = _mm256_loadu_si256((__m256i*) &a[i]);
			__m256i y2 = _mm256_loadu_si256((__m256i*) &a[i + 8]);
			__m256i y = _mm256_min_epi32(y1, y2);
			__m256i mask = _mm256_cmpgt_epi32(p, y);
			if (!_mm256_testz_si256(mask, mask)) { [[unlikely]]
				for (uint32_t j = i; j < i + 16; j++) {
					if (a[j] < min) {
						min = a[idx = j];
					}
				}

				p = _mm256_set1_epi32(min);
			}
		}

		// tail
		for (; i < n; i++) {
			if (a[i] < min) {
				min = a[idx = i];
			}
		}

		return idx;
	}

	[[nodiscard]] constexpr static inline size_t argmin_avx2_i32_bl32(const int32_t *a,
	                                                    const size_t n) noexcept {
		int32_t min = INT_MAX;
		__m256i p = _mm256_set1_epi32(min);
		size_t i = 0, idx = 0;
		for (; i+32 <= n; i += 32) {
			__m256i y1 = _mm256_loadu_si256((__m256i*) &a[i]);
			__m256i y2 = _mm256_loadu_si256((__m256i*) &a[i + 8]);
			__m256i y3 = _mm256_loadu_si256((__m256i*) &a[i + 16]);
			__m256i y4 = _mm256_loadu_si256((__m256i*) &a[i + 24]);
			y1 = _mm256_min_epi32(y1, y2);
			y3 = _mm256_min_epi32(y3, y4);
			y1 = _mm256_min_epi32(y1, y3);
			__m256i mask = _mm256_cmpgt_epi32(p, y1);
			if (!_mm256_testz_si256(mask, mask)) { [[unlikely]]
				idx = i;
				for (uint32_t j = i; j < i + 32; j++) {
					min = (a[j] < min ? a[j] : min);
				}

				p = _mm256_set1_epi32(min);
			}
		}

		size_t idx2 = idx+31;
		for (uint32_t j = idx; j < idx + 31; j++) {
			if (a[j] == min) {
				idx2 = j;
			}
		}

		for (; i < n; i++) {
			if (a[i] < min) {
				min = a[idx2 = i];
			}
		}

		return idx2;
	}

}
#endif
#endif
