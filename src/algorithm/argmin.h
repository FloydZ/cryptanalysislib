#ifndef CRYPTANALYSISLIB_ALGORITHM_ARGMIN_H
#define CRYPTANALYSISLIB_ALGORITHM_ARGMIN_H

#include <concepts>
#include <cstdlib>
#include <cstdint>
#include <type_traits>
#include <limits.h>

#include "simd/simd.h"

// TODO argmax
namespace cryptanalysislib {
    struct AlgorithmArgMinConfig {
    public:
        constexpr static size_t aligned_instructions = false;
    };
    constexpr static AlgorithmArgMinConfig algorithmArgMinConfig{};


    /// forward declaration
    template<typename T>
	[[nodiscard]] constexpr static inline size_t argmin(const T *a, const size_t n) noexcept;

    template<typename S=uint32x8_t,
            const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
	[[nodiscard]] constexpr static inline size_t argmin_simd_u32(const uint32_t *a,
	                                               				 const size_t n) noexcept {
		uint32_t min = -1u;
		size_t idx = 0;
		auto p = S::set1(min);
        
        constexpr size_t t = S::LIMBS;
		size_t i = 0;
		for (; i+t <= n; i += t) {
			auto y = S::template load<config.aligned_instructions>(a + i); 
            const uint32_t mask = S::gt(p, y);
			if (mask != 0) { [[unlikely]]
				for (uint32_t j = i; j < i + t; j++) {
					if (a[j] < min) {
						min = a[idx = j];
					}
				}

				p = S::set1(min);
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


    template<typename S=uint32x8_t,
             const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
	[[nodiscard]] constexpr static inline size_t argmin_simd_u32_bl16(const uint32_t *a,
	                                                                  const size_t n) noexcept {
        constexpr size_t t = S::LIMBS;
        constexpr size_t t2 = 2*t;
		uint32_t min = -1u;
		auto p = S::set1(min);
		size_t i = 0, idx = 0;
		for (; i+t2 <= n; i += t2) {
            const S y1 = S::template load<config.aligned_instructions>(a + i),
                    y2 = S::template load<config.aligned_instructions>(a + i + t);
			const S y = S::min(y1, y2);
			const uint32_t mask = S::gt(p, y);
			if (mask != 0) { [[unlikely]]
				for (uint32_t j = i; j < i + t2; j++) {
					if (a[j] < min) {
						min = a[idx = j];
					}
				}

				p = S::set1(min);
			}
		}

		// tail
		for (; i < n; i++) {
			if (a[i] < min) {
				min = a[idx = i];
			}
		}

		return idx;	}
 
    template<typename S=uint32x8_t,
             const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
	[[nodiscard]] constexpr static inline size_t argmin_simd_u32_bl32(const uint32_t *a,
	                                                    const size_t n) noexcept {
        constexpr size_t t = S::LIMBS;
        constexpr size_t t4 = 2*t;
		uint32_t min = -1u;
		auto p = S::set1(min);
		size_t i = 0, idx = 0;
		for (; i+t4 <= n; i += t4) {
            S y1 = S::template load<config.aligned_instructions>(a + i + 0*t),
              y2 = S::template load<config.aligned_instructions>(a + i + 1*t),
              y3 = S::template load<config.aligned_instructions>(a + i + 2*t),
              y4 = S::template load<config.aligned_instructions>(a + i + 3*t);

			y1 = S::min(y1, y2);
			y3 = S::min(y3, y4);
			y1 = S::min(y1, y3);
            const uint32_t mask = S::gt(p, y1);
			if (mask != 0) { [[unlikely]]
				idx = i;
				for (uint32_t j = i; j < i + t4; j++) {
					min = (a[j] < min ? a[j] : min);
				}

				p = S::set1(min);
			}
		}

		size_t idx2 = idx+t4-1;
		for (uint32_t j = idx; j < idx + t4-1; j++) {
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
#ifdef USE_AVX2
#include <immintrin.h>
    // TODO use simd wrapper as soon as signed version exist
	// TODO write selection function in cryptanalysis namespace

	/// source : https://en.algorithmica.org/hpc/algorithms/argmin/
    /// loads 8 concecutive elements into a single register and tests the 
    /// minimum on it.
	/// \param a array to fin the minimum in.
	/// \param n size of the array.
	/// \return the position of the minimal number within a.
    template<const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
	[[nodiscard]] constexpr static inline size_t argmin_avx2_i32(const int32_t *a,
	                                               				 const size_t n) noexcept {
		int32_t min = INT_MAX;
		size_t idx = 0;
		__m256i p = _mm256_set1_epi32(min);

		size_t i = 0;
		for (; i+8 <= n; i += 8) {
			__m256i y; 
            if constexpr (config.aligned_instructions) {
                y = _mm256_load_si256((__m256i*) &a[i]);
            } else { 
                y = _mm256_loadu_si256((__m256i*) &a[i]);
            }

			__m256i mask = _mm256_cmpgt_epi32(p, y);
            // this doesnt make sense
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
	/// \param a array to fin the minimum in.
	/// \param n size of the array.
	/// \return the position of the minimal number within a.
    template<const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
	[[nodiscard]] constexpr static inline size_t argmin_avx2_i32_bl16(const int32_t *a,
	                                                                  const size_t n) noexcept {
		int32_t min = INT_MAX;
		__m256i p = _mm256_set1_epi32(min);
		size_t i = 0, idx = 0;
		for (; i+16 <= n; i += 16) {
            __m256i y1,y2;
            if constexpr (config.aligned_instructions) {
	    		y1 = _mm256_load_si256((__m256i*) &a[i]);
	    		y2 = _mm256_load_si256((__m256i*) &a[i + 8]);
            } else {
	    		y1 = _mm256_loadu_si256((__m256i*) &a[i]);
	    		y2 = _mm256_loadu_si256((__m256i*) &a[i + 8]);
            }
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

    template<const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
	[[nodiscard]] constexpr static inline size_t argmin_avx2_i32_bl32(const int32_t *a,
	                                                    const size_t n) noexcept {
		int32_t min = INT_MAX;
		__m256i p = _mm256_set1_epi32(min);
		size_t i = 0, idx = 0;
		for (; i+32 <= n; i += 32) {
            __m256i y1,y2,y3,y4;
            if constexpr (config.aligned_instructions) {
		    	y1 = _mm256_load_si256((__m256i*) &a[i]);
		    	y2 = _mm256_load_si256((__m256i*) &a[i + 8]);
		    	y3 = _mm256_load_si256((__m256i*) &a[i + 16]);
		    	y4 = _mm256_load_si256((__m256i*) &a[i + 24]);
            } else {
		    	y1 = _mm256_loadu_si256((__m256i*) &a[i]);
		    	y2 = _mm256_loadu_si256((__m256i*) &a[i + 8]);
		    	y3 = _mm256_loadu_si256((__m256i*) &a[i + 16]);
		    	y4 = _mm256_loadu_si256((__m256i*) &a[i + 24]);
            }

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
	
    template<const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
    [[nodiscard]] constexpr static inline size_t argmin_avx2_i32_dispatch(const int32_t *a,
	                                                                    const size_t n) noexcept {

		// the boundaries is arbitrary choosen
		if (n < 8) {
			return argmin<int32_t>(a, n);
		} else if (n < 128) {
            return argmin_avx2_i32_bl16<config>(a, n);
        }
        
        return argmin_avx2_i32_bl32<config>(a, n);
    }
#endif


    /// generic fallback implementation
	/// \tparam T 
	/// \param a array you want to sort
	/// \param n number of elements in this array
	/// \return position of the min element
	template<typename T>
#if __cplusplus > 201709L
	    requires std::totally_ordered<T>
#endif
	[[nodiscard]] constexpr static inline size_t argmin(const T *a,
	                                                    const size_t n) noexcept {
#ifdef USE_AVX2
        if constexpr (std::is_same_v<T, int32_t>) {
            return argmin_avx2_i32_dispatch(a, n);
        }
#endif
		size_t k = 0;
		for (size_t i = 0; i < n; i++) {
			if (a[i] < a[k]) [[unlikely]] {
				k = i;
			}
		}

		return k;
	}

    /// helper wrapper
	template<typename T>
#if __cplusplus > 201709L
	    requires std::totally_ordered<T>
#endif
	[[nodiscard]] constexpr static inline size_t argmin(const std::vector<T> &a) noexcept {
        return argmin(a.data(), a.size());
	}

    /// helper wrapper
	template<typename T, const size_t n>
#if __cplusplus > 201709L
	    requires std::totally_ordered<T>
#endif
	[[nodiscard]] constexpr static inline size_t argmin(const std::array<T, n> &a) noexcept {
        return argmin(a.data(), n);
	}
}
#endif
