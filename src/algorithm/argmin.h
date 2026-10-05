#ifndef CRYPTANALYSISLIB_ALGORITHM_ARGMIN_H
#define CRYPTANALYSISLIB_ALGORITHM_ARGMIN_H

#include <cstdlib>
#include <cstdint>
#include <type_traits>

#include "simd/simd.h"
#include "algorithm.h"
#include "thread/thread.h"

namespace cryptanalysislib {
    struct AlgorithmArgMinConfig {
    public:
        constexpr static bool aligned_instructions = false;
        constexpr static size_t min_size_per_thread = 100000;
    };
    constexpr static AlgorithmArgMinConfig algorithmArgMinConfig{};


    namespace internal {
	    /// Find the index of the minimum value in an array using SIMD instructions
	    /// \tparam S SIMD vector type to use for operations
	    /// \tparam config Configuration parameters for the algorithm
	    /// \param a [in]: Pointer to the array to search
	    /// \param n [in]: Number of elements in the array
	    /// \return [out]: Index of the minimum value in the array
	    template<typename S=uint32x8_t,
                const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
	    [[nodiscard]] constexpr static inline size_t argmin_simd(const uint32_t *__restrict__ a,
	                                                   			 const size_t n) noexcept {
            using T = S::limb_type;
	    	T min = -1ull;
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

	    /// Find the index of the minimum value in an array using SIMD instructions with block size 16
	    /// \tparam S SIMD vector type to use for operations
	    /// \tparam config Configuration parameters for the algorithm
	    /// \param a [in]: Pointer to the array to search
	    /// \param n [in]: Number of elements in the array
	    /// \return [out]: Index of the minimum value in the array
	    template<typename S=uint32x8_t,
                 const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
	    [[nodiscard]] constexpr static inline size_t argmin_simd_bl16(const uint32_t *__restrict__ a,
	                                                                  const size_t n) noexcept {
            using T = S::limb_type;
            constexpr size_t t = S::LIMBS;
            constexpr size_t t2 = 2*t;
	    	T min = -1ull;
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

	    	return idx;	
        }

	    /// Find the index of the minimum value in an array using SIMD instructions with block size 32
	    /// \tparam S SIMD vector type to use for operations
	    /// \tparam config Configuration parameters for the algorithm
	    /// \param a [in]: Pointer to the array to search
	    /// \param n [in]: Number of elements in the array
	    /// \return [out]: Index of the minimum value in the array
	    template<typename S=uint32x8_t,
                 const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
	    [[nodiscard]] constexpr static inline size_t argmin_simd_bl32(const uint32_t *a,
	                                                                  const size_t n) noexcept {
            using T = S::limb_type;
            constexpr size_t t = S::LIMBS;
            constexpr size_t t4 = 4*t;
	    	T min = -1ull;
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

	    /// Dispatch function to select the appropriate SIMD implementation for argmin
	    /// \tparam S SIMD vector type to use for operations
	    /// \tparam config Configuration parameters for the algorithm
	    /// \param a [in]: Pointer to the array to search
	    /// \param n [in]: Number of elements in the array
	    /// \return [out]: Index of the minimum value in the array
	    template<typename S=uint32x8_t,
                 const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
	    [[nodiscard]] constexpr static inline size_t argmin_simd_dispatch(const uint32_t *__restrict__ a,
	                                                   			          const size_t n) noexcept {
            // TODO select correct one
            return argmin_simd<S, config>(a, n);
        }

// old and unused code. The SIMD class can now work with signed/unsinged types
//#ifdef USE_AVX2
//#include <immintrin.h>
//	    /// Find the index of the minimum value in an array using AVX2 instructions
//        /// Loads 8 consecutive elements into a single register and tests the minimum
//        /// Source: https://en.algorithmica.org/hpc/algorithms/argmin/
//	    /// \tparam config Configuration parameters for the algorithm
//	    /// \param a [in]: Pointer to the array to search
//	    /// \param n [in]: Number of elements in the array
//	    /// \return [out]: Index of the minimum value in the array
//        template<const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
//	    [[nodiscard]] constexpr static inline size_t argmin_avx2_i32(const int32_t *a,
//	                                                   				 const size_t n) noexcept {
//	    	int32_t min = INT_MAX;
//	    	size_t idx = 0;
//	    	__m256i p = _mm256_set1_epi32(min);
//
//	    	size_t i = 0;
//	    	for (; i+8 <= n; i += 8) {
//	    		__m256i y; 
//                if constexpr (config.aligned_instructions) {
//                    y = _mm256_load_si256((__m256i*) &a[i]);
//                } else { 
//                    y = _mm256_loadu_si256((__m256i*) &a[i]);
//                }
//
//	    		__m256i mask = _mm256_cmpgt_epi32(p, y);
//                // this doesnt make sense
//	    		if (!_mm256_testz_si256(mask, mask)) { [[unlikely]]
//	    			for (uint32_t j = i; j < i + 8; j++) {
//	    				if (a[j] < min) {
//	    					min = a[idx = j];
//	    				}
//	    			}
//
//	    			p = _mm256_set1_epi32(min);
//	    		}
//	    	}
//
//	    	// tail
//	    	for (; i < n; i++) {
//	    		if (a[i] < min) {
//	    			min = a[idx = i];
//	    		}
//	    	}
//
//	    	return idx;
//	    }
//
//	    /// same algorithm as `argmin_avx2_i32` but with a blocksize of 16.
//	    /// so 16 elements are loaded from memory in parallel.
//	    /// \param a[in]: array to fin the minimum in.
//	    /// \param n[in]: size of the array.
//	    /// \return the position of the minimal number within a.
//        template<const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
//	    [[nodiscard]] constexpr static inline size_t argmin_avx2_i32_bl16(const int32_t *a,
//	                                                                      const size_t n) noexcept {
//	    	int32_t min = INT_MAX;
//	    	__m256i p = _mm256_set1_epi32(min);
//	    	size_t i = 0, idx = 0;
//	    	for (; i+16 <= n; i += 16) {
//                __m256i y1,y2;
//                if constexpr (config.aligned_instructions) {
//	        		y1 = _mm256_load_si256((__m256i*) &a[i]);
//	        		y2 = _mm256_load_si256((__m256i*) &a[i + 8]);
//                } else {
//	        		y1 = _mm256_loadu_si256((__m256i*) &a[i]);
//	        		y2 = _mm256_loadu_si256((__m256i*) &a[i + 8]);
//                }
//	    		__m256i y = _mm256_min_epi32(y1, y2);
//	    		__m256i mask = _mm256_cmpgt_epi32(p, y);
//	    		if (!_mm256_testz_si256(mask, mask)) { [[unlikely]]
//	    			for (uint32_t j = i; j < i + 16; j++) {
//	    				if (a[j] < min) {
//	    					min = a[idx = j];
//	    				}
//	    			}
//
//	    			p = _mm256_set1_epi32(min);
//	    		}
//	    	}
//
//	    	// tail
//	    	for (; i < n; i++) {
//	    		if (a[i] < min) {
//	    			min = a[idx = i];
//	    		}
//	    	}
//
//	    	return idx;
//	    }
//
//	    /// Find the index of the minimum value in an array using AVX2 instructions with block size 32
//	    /// \tparam config Configuration parameters for the algorithm
//	    /// \param a [in]: Pointer to the array to search
//	    /// \param n [in]: Number of elements in the array
//	    /// \return [out]: Index of the minimum value in the array
//	    template<const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
//	    [[nodiscard]] constexpr static inline size_t argmin_avx2_i32_bl32(const int32_t *a,
//	                                                        const size_t n) noexcept {
//	    	int32_t min = INT_MAX;
//	    	__m256i p = _mm256_set1_epi32(min);
//	    	size_t i = 0, idx = 0;
//	    	for (; i+32 <= n; i += 32) {
//                __m256i y1,y2,y3,y4;
//                if constexpr (config.aligned_instructions) {
//	    	    	y1 = _mm256_load_si256((__m256i*) &a[i]);
//	    	    	y2 = _mm256_load_si256((__m256i*) &a[i + 8]);
//	    	    	y3 = _mm256_load_si256((__m256i*) &a[i + 16]);
//	    	    	y4 = _mm256_load_si256((__m256i*) &a[i + 24]);
//                } else {
//	    	    	y1 = _mm256_loadu_si256((__m256i*) &a[i]);
//	    	    	y2 = _mm256_loadu_si256((__m256i*) &a[i + 8]);
//	    	    	y3 = _mm256_loadu_si256((__m256i*) &a[i + 16]);
//	    	    	y4 = _mm256_loadu_si256((__m256i*) &a[i + 24]);
//                }
//
//	    		y1 = _mm256_min_epi32(y1, y2);
//	    		y3 = _mm256_min_epi32(y3, y4);
//	    		y1 = _mm256_min_epi32(y1, y3);
//	    		__m256i mask = _mm256_cmpgt_epi32(p, y1);
//	    		if (!_mm256_testz_si256(mask, mask)) { [[unlikely]]
//	    			idx = i;
//	    			for (uint32_t j = i; j < i + 32; j++) {
//	    				min = (a[j] < min ? a[j] : min);
//	    			}
//
//	    			p = _mm256_set1_epi32(min);
//	    		}
//	    	}
//
//	    	size_t idx2 = idx+31;
//	    	for (uint32_t j = idx; j < idx + 31; j++) {
//	    		if (a[j] == min) {
//	    			idx2 = j;
//	    		}
//	    	}
//
//	    	for (; i < n; i++) {
//	    		if (a[i] < min) {
//	    			min = a[idx2 = i];
//	    		}
//	    	}
//
//	    	return idx2;
//	    }
//
//	    /// Dispatch to the appropriate argmin implementation based on array size
//	    /// \tparam config Configuration parameters for the algorithm
//	    /// \param a [in]: Pointer to the array to search
//	    /// \param n [in]: Number of elements in the array
//	    /// \return [out]: Index of the minimum value in the array
//	    template<const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
//        [[nodiscard]] constexpr static inline size_t argmin_avx2_i32_dispatch(const int32_t *a,
//	                                                                        const size_t n) noexcept {
//
//	    	// NOTE the boundaries is arbitrary choosen
//	    	if (n < 128) {
//                return argmin_avx2_i32_bl16<config>(a, n);
//            }
//            
//            return argmin_avx2_i32_bl32<config>(a, n);
//        }
//#endif
    } // end namespace internal

	/// Find the index of the minimum value in a range defined by iterators
	/// \tparam Iterator Iterator type to the collection
	/// \tparam config Configuration parameters for the algorithm
	/// \param start[in]: Iterator to the beginning of the range
	/// \param end[in]: Iterator to the end of the range
	/// \return Index of the minimum value in the range
	template<class Iterator,
             const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<Iterator>
#endif
	[[nodiscard]] constexpr static inline size_t argmin(Iterator start,
														Iterator end) noexcept {
        using T = typename std::iterator_traits<Iterator>::value_type;
		const size_t len = std::distance(start, end);

		if constexpr (std::is_arithmetic_v<T>) {
		    return internal::argmin_simd(start, len);
		}
		size_t k = 0;
		for (size_t i = 1; i < len; i++) {
			if (*(start+i) < *(start + k)) [[unlikely]] {
				k = i;
			}
		}

		return k;
	}

	/// Find the index of the minimum value in a range using parallel execution policy
	/// \tparam ExecPolicy Execution policy type (sequential or parallel)
	/// \tparam RandIt Random access iterator type
	/// \tparam config Configuration parameters for the algorithm
	/// \param policy[in]: Execution policy to use (sequential or parallel)
	/// \param first[in]: Iterator to the beginning of the range
	/// \param last[in]: Iterator to the end of the range
	/// \return Index of the minimum value in the range
	template <class ExecPolicy,
			  class RandIt,
              const AlgorithmArgMinConfig &config = algorithmArgMinConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
	size_t argmin(ExecPolicy&& policy,
				  RandIt first,
				  RandIt last) noexcept {
		using T = typename RandIt::value_type;

		const auto size = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::argmin
				<RandIt, config>(first, last);
		}

		auto futures = internal::parallel_chunk_for_1(
			std::forward<ExecPolicy>(policy),
			first, last,
			cryptanalysislib::argmin<RandIt, config>,
			(size_t *)0,
			1, nthreads);

		size_t m = futures[0].get();
		T v = *(first + m);
		for (size_t i = 1; i < nthreads; i++) {
			T mm = futures[i].get();
			if (*(first + m) < v) [[unlikely]] {
				m = mm;
			}
		}

		return m;
	}
}; // end namespace cryptanalysislib
#endif
