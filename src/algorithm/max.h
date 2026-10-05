#ifndef CRYPTANALYSISLIB_ALGORITHM_MAX_H
#define CRYPTANALYSISLIB_ALGORITHM_MAX_H

#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <limits.h>
#include <type_traits>

#include "simd/simd.h"
#include "algorithm/algorithm.h"
#include "thread/thread.h"


// TODO implement max_element simd version
// TODO implement max_element parallel version


namespace cryptanalysislib {
    /// Configuration for max algorithms with SIMD and threading settings
    struct AlgorithmMaxConfig {
    public:
        const size_t aligned_instructions = false;
    	const uint32_t min_size_simd = 32;
    	const uint32_t min_size_per_thread = 16384;
    };
    constexpr static AlgorithmMaxConfig algorithmMaxConfig{};

    namespace internal {
        /// Returns maximum of two values using branchless computation
        /// Both a and b must not have the most significant bit set
        /// \tparam T Integral type for the values
        /// \param a[in]: First value to compare
        /// \param b[in]: Second value to compare
        /// \return Maximum of a and b
    	template<typename T>
            requires std::is_integral_v<T>
        constexpr static inline T max_branchless(const T a,
                                                 const T b) noexcept {
            constexpr static size_t BITS = sizeof(T) * 8u;
            T d = b - a;
            d &= (T)( (long)d >> (BITS-1) );
            // here: d ==
            // 0    if  b > a
            // b-a  if  a > b  (negative as signed type)
            return  b - d;
        }
    
    	/// SIMD-optimized maximum finding for integral arrays
    	/// \tparam T Integral type for array elements
    	/// \tparam config Algorithm configuration (default: algorithmMaxConfig)
    	/// \param a[in]: Array of integers
    	/// \param n[in]: Length of the array
    	/// \return Maximum value from a[0], ..., a[n-1]
    	template<typename T,
                 const AlgorithmMaxConfig &config = algorithmMaxConfig>
            requires std::is_integral_v<T>
    	[[nodiscard]] constexpr static inline T max_simd_uXX(const T *a,
    														 const size_t n) noexcept {
            // make sure that we actually support the integers
            static_assert(std::is_integral_v<T>);
    		using S = SIMDSelector<T>;
    
    		T m = 0;
    		auto p = S::set1(m);
    
            constexpr size_t t = S::LIMBS;
    		size_t i = 0;
    		for (; i+t <= n; i += t) {
    			auto y = S::template load<config.aligned_instructions>(a + i);
    			p = S::max(p, y);
    		}
    
            // compute the max over the register
    		for (uint32_t j = 0; j < t; j++) {
    			if (m < p[j]) {
    				m = p[j];
    			}
    		}
    
    		// tail mngt.
    		for (; i < n; i++) {
    			if (a[i] > m) [[unlikely]] {
    				m = a[i];
    			}
    		}
    
    		return m;
        }
    } // end namespace internal

	/// Finds maximum element in a range (sequential version)
	/// \tparam Iterator Forward iterator type for the range
	/// \tparam config Algorithm configuration (default: algorithmMaxConfig)
	/// \param start[in]: Iterator to the beginning of the range
	/// \param end[in]: Iterator to the end of the range
	/// \return Maximum value in the range
	template<class Iterator,
             const AlgorithmMaxConfig &config = algorithmMaxConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<Iterator>
#endif
	[[nodiscard]] constexpr static inline Iterator::value_type max(Iterator start,
																   Iterator end) noexcept {
		using T = Iterator::value_type;
		const size_t len = std::distance(start, end);
		if (std::is_integral_v<T> && (len >= config.min_size_simd)) {
			return internal::max_simd_uXX(&(*start), len);
		}

		T k = *start;
		for (size_t i = 1; i < len; i++) {
			if (*(start+i) > *(start + k)) [[unlikely]] {
				k = i;
			}
		}

		return k;
	}

	/// Finds maximum element in a range (parallel version)
	/// \tparam ExecPolicy Execution policy type for parallel execution
	/// \tparam RandIt Random access iterator type for the range
	/// \tparam config Algorithm configuration (default: algorithmMaxConfig)
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: Iterator to the beginning of the range
	/// \param last[in]: Iterator to the end of the range
	/// \return Maximum value in the range
	template <class ExecPolicy,
			  class RandIt,
              const AlgorithmMaxConfig &config = algorithmMaxConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
	RandIt::value_type
	max(ExecPolicy&& policy,
		RandIt first,
		RandIt last) noexcept {
		using T = typename RandIt::value_type;

		const auto size = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::max
				<RandIt, config>(first, last);
		}

		auto futures = internal::parallel_chunk_for_1(
			std::forward<ExecPolicy>(policy),
			first, last,
			cryptanalysislib::max<RandIt, config>,
			static_cast<T *>(nullptr),
			1, nthreads);

		size_t m = futures[0].get();
		for (size_t i = 1; i < nthreads; i++) {
			const size_t mm = futures[i].get();
			if (mm > m) {
				m = mm;
			}
		}

		return m;
	}


    /// TODO doc
    template<class ForwardIt>
    ForwardIt max_element(ForwardIt first,
                          ForwardIt last) noexcept {
        if (first == last) {
            return last;
        }
     
        ForwardIt largest = first;
     
        while (++first != last) {
            if (*largest < *first) {
                largest = first;
            }
        }
     
        return largest;
    }
    
    /// TODO doc
    template<class ForwardIt,
             class Compare>
    ForwardIt max_element(ForwardIt first,
                          ForwardIt last,
                          Compare comp) noexcept {
        if (first == last) {
            return last;
        }
     
        ForwardIt largest = first;
     
        while(++first != last) {
            if (comp(*largest, *first)) {
                largest = first;
            }
        }
     
        return largest;
    }

} // end namespace cryptanalysislib

#endif
