#ifndef CRYPTANALYSISLIB_ALGORITHM_MAX_H
#define CRYPTANALYSISLIB_ALGORITHM_MAX_H

#include <concepts>
#include <cstdint>
#include <cstdlib>
#include <limits.h>
#include <type_traits>

#include "simd/simd.h"
#include "thread/thread.h"

namespace cryptanalysislib {
    struct AlgorithmMaxConfig {
    public:
        const size_t aligned_instructions = false;
    	const uint32_t min_size_simd = 32;
    	const uint32_t min_size_per_thread = 16384;
    };
    constexpr static AlgorithmMaxConfig algorithmMaxConfig{};

	/// \tparam T
	/// \tparam config
	/// \param a
	/// \param n
	/// \return
	template<typename T,
             const AlgorithmMaxConfig &config = algorithmMaxConfig>
	[[nodiscard]] constexpr static inline T max_simd_uXX(const T *a,
														 const size_t n) noexcept {
#ifdef USE_AVX512F
		constexpr uint32_t limbs = 64/sizeof(T);
#else
		constexpr uint32_t limbs = 32/sizeof(T);
#endif
		using S = TxN_t<T, limbs>;

		T m = 0;
		auto p = S::set1(m);

        constexpr size_t t = S::LIMBS;
		size_t i = 0;
		for (; i+t <= n; i += t) {
			auto y = S::template load<config.aligned_instructions>(a + i);
			p = S::max(p, y);
		}

		for (uint32_t j = 0; j < t; j++) {
			if (m < p[j]) {
				m = p[j];
			}
		}

		// tail
		for (; i < n; i++) {
			if (a[i] > m) {
				m = a[i];
			}
		}

		return m;
    }

	/// \tparam Iterator
	/// \tparam config
	/// \param start
	/// \param end
	/// \return
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
			return max_simd_uXX(&(*start), len);
		}

		T k = *start;
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
			(T *)0,
			1, nthreads);

		T m = futures[0].get();
		for (size_t i = 1; i < nthreads; i++) {
			T mm = futures[i].get();
			if (mm > m) {
				m = mm;
			}
		}

		return m;
	}
}

#endif
