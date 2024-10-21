#ifndef CRYPTANALYSISLIB_ALGORITHM_ARGMAX_H
#define CRYPTANALYSISLIB_ALGORITHM_ARGMAX_H

#include "apply.h"


#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <limits.h>
#include <type_traits>

#include "simd/simd.h"
#include "thread/thread.h"

namespace cryptanalysislib {
    struct AlgorithmArgMaxConfig {
    public:
        constexpr static size_t aligned_instructions = false;
    	constexpr static uint32_t min_size_per_thread = 16384;
    };
    constexpr static AlgorithmArgMaxConfig algorithmArgMaxConfig{};


    /// forward declaration
    template<typename T, const AlgorithmApplyConfig &config>
	[[nodiscard]] constexpr static inline size_t argmax(const T *a, const size_t n) noexcept;

	/// \tparam S
	/// \tparam config
	/// \param a
	/// \param n
	/// \return
	template<typename S=uint32x8_t,
             const AlgorithmArgMaxConfig &config = algorithmArgMaxConfig>
	[[nodiscard]] constexpr static inline size_t argmax_simd_u32(const uint32_t *a,
	                                               				 const size_t n) noexcept {
		uint32_t max = 0;
		size_t idx = 0;
		auto p = S::set1(max);
        
        constexpr size_t t = S::LIMBS;
		size_t i = 0;
		for (; i+t <= n; i += t) {
			auto y = S::template load<config.aligned_instructions>(a + i); 
            const uint32_t mask = S::lt(p, y);
			if (mask != 0) { [[unlikely]]
				for (uint32_t j = i; j < i + t; j++) {
					if (a[j] > max) {
						max = a[idx = j];
					}
				}

				p = S::set1(max);
			}
		}

		// tail
		for (; i < n; i++) {
			if (a[i] > max) {
				max = a[idx = i];
			}
		}

		return idx;
    }

	/// \tparam S
	/// \tparam config
	/// \param a
	/// \param n
	/// \return
	template<typename S=uint32x8_t,
             const AlgorithmArgMaxConfig &config = algorithmArgMaxConfig>
	[[nodiscard]] constexpr static inline size_t argmax_simd_u32_bl16(const uint32_t *a,
	                                                                  const size_t n) noexcept {
        constexpr size_t t = S::LIMBS;
        constexpr size_t t2 = 2*t;
		uint32_t max = 0;
		auto p = S::set1(max);
		size_t i = 0, idx = 0;
		for (; i+t2 <= n; i += t2) {
            const S y1 = S::template load<config.aligned_instructions>(a + i),
                    y2 = S::template load<config.aligned_instructions>(a + i + t);
			const S y = S::max(y1, y2);
			const uint32_t mask = S::lt(p, y);
			if (mask != 0) { [[unlikely]]
				for (uint32_t j = i; j < i + t2; j++) {
					if (a[j] > max) {
						max = a[idx = j];
					}
				}

				p = S::set1(max);
			}
		}

		// tail
		for (; i < n; i++) {
			if (a[i] > max) {
				max = a[idx = i];
			}
		}

		return idx;	}

	/// \tparam S
	/// \tparam config
	/// \param a
	/// \param n
	/// \return
	template<typename S=uint32x8_t,
             const AlgorithmArgMaxConfig &config = algorithmArgMaxConfig>
	[[nodiscard]] constexpr static inline size_t argmax_simd_u32_bl32(const uint32_t *a,
																	  const size_t n) noexcept {
        constexpr size_t t = S::LIMBS;
        constexpr size_t t4 = 4*t;
		uint32_t max = 0;
		auto p = S::set1(max);
		size_t i = 0, idx = 0;
		for (; i+t4 <= n; i += t4) {
            S y1 = S::template load<config.aligned_instructions>(a + i + 0*t),
              y2 = S::template load<config.aligned_instructions>(a + i + 1*t),
              y3 = S::template load<config.aligned_instructions>(a + i + 2*t),
              y4 = S::template load<config.aligned_instructions>(a + i + 3*t);

			y1 = S::max(y1, y2);
			y3 = S::max(y3, y4);
			y1 = S::max(y1, y3);
            const uint32_t mask = S::gt(p, y1);
			if (mask != 0) { [[unlikely]]
				idx = i;
				for (uint32_t j = i; j < i + t4; j++) {
					max = (a[j] > max ? a[j] : max);
				}

				p = S::set1(max);
			}
		}

		size_t idx2 = idx+t4-1;
		for (uint32_t j = idx; j < idx + t4-1; j++) {
			if (a[j] == max) {
				idx2 = j;
			}
		}

		for (; i < n; i++) {
			if (a[i] > max) {
				max = a[idx2 = i];
			}
		}

		return idx2;
	}

	/// \tparam Iterator
	/// \tparam config
	/// \param start
	/// \param end
	/// \return
	template<class Iterator,
             const AlgorithmArgMaxConfig &config = algorithmArgMaxConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<Iterator>
#endif
	[[nodiscard]] constexpr static inline size_t argmax(Iterator start,
														Iterator end) noexcept {
		using T = Iterator::value_type;
		const size_t len = std::distance(start, end);
		size_t k = 0;
		for (size_t i = 1; i < len; i++) {
			if (*(start+i) > *(start + k)) [[unlikely]] {
				k = i;
			}
		}

		return k;
	}

	/// \tparam ExecPolicy
	/// \tparam RandIt
	/// \param policy
	/// \param first
	/// \param last
	/// \return
	template <class ExecPolicy,
			  class RandIt,
              const AlgorithmArgMaxConfig &config = algorithmArgMaxConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
	size_t argmax(ExecPolicy&& policy,
				  RandIt first,
				  RandIt last) noexcept {
		using T = typename RandIt::value_type;

		const auto size = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::argmax
				<RandIt, config>(first, last);
		}

		auto futures = internal::parallel_chunk_for_1(
			std::forward<ExecPolicy>(policy),
			first, last,
			cryptanalysislib::argmax<RandIt, config>,
			(size_t *)0,
			1, nthreads);

		size_t m = futures[0].get();
		T v = *(first + m);
		for (size_t i = 1; i < nthreads; i++) {
			T mm = futures[i].get();
			if (*(first + m) > v) {
				m = mm;
			}
		}

		return m;
	}
}
#endif
