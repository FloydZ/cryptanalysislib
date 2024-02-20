#pragma once

#include <immintrin.h>
#include <stdint.h>


///
/// NOTE: upper bound `d` is inclusive
/// \param in
/// \return
template<bool exact = false, const bool EXACT = false, const uint32_t d = 1, const uint32_t dk_bruteforce_weight = 0>
int compare_256_64(const __m256i in1, const __m256i in2) noexcept {
	if constexpr (EXACT) {
		const __m256i tmp2 = _mm256_cmpeq_epi64(in1, in2);
		return _mm256_movemask_pd((__m256d) tmp2);
	}

	const __m256i tmp1 = _mm256_xor_si256(in1, in2);
#ifdef USE_AVX512
	const __m256i pop = _mm256_popcnt_epi64(tmp1);
#else
	const __m256i pop = popcount_avx2_64(tmp1);
#endif

	/// in the special case where we want to match on a different weight to speed up
	/// the computation. This makes only sense if `dk_bruteforce_weight` < dk.
	if constexpr (dk_bruteforce_weight > 0) {
		if constexpr (EXACT) {
			const __m256i avx_exact_weight64 = _mm256_set1_epi64x(dk_bruteforce_weight);
#ifdef USE_AVX512
			return _mm256_cmp_epi64_mask(avx_exact_weight64, pop, 0);
#else
			const __m256i tmp2 = _mm256_cmpeq_epi64(avx_exact_weight64, pop);
			return _mm256_movemask_pd((__m256) tmp2);
#endif
		} else {
			const __m256i avx_weight64 = _mm256_set1_epi64x(dk_bruteforce_weight + 1);
#ifdef USE_AVX512
			return _mm256_cmp_epi64_mask(avx_weight64, pop, 6);
#else
			const __m256i tmp2 = _mm256_cmpgt_epi64(avx_weight64, pop);
			return _mm256_movemask_pd((__m256d) tmp2);
#endif
		}

		// just to make sure that the compiler will not compiler the
		// following code
		return 0;
	}

	if constexpr (EXACT) {
		const __m256i avx_exact_weight64 = _mm256_set1_epi64x(d);
#ifdef USE_AVX512
		return _mm256_cmp_epi64_mask(avx_exact_weight64, pop, 0);
#else
		const __m256i tmp2 = _mm256_cmpeq_epi64(avx_exact_weight64, pop);
		return _mm256_movemask_pd((__m256d) tmp2);
#endif
	} else {
		const __m256i avx_weight64 = _mm256_set1_epi64x(d + 1);
#ifdef USE_AVX512
		return _mm256_cmp_epi64_mask(avx_weight64, pop, 5);
#else
		const __m256i tmp2 = _mm256_cmpgt_epi64(avx_weight64, pop);
		return _mm256_movemask_pd((__m256d) tmp2);
#endif
	}
}