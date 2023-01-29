#pragma once

#include <cstdint>
#include <immintrin.h>
#include <assert.h>


/// NOTE: there is an avx2 version below
/// NOTE: w/k must be smaller than 5
/// \param a input number
/// \param n max bits
/// \param k max weight
/// \return maximal ctr < n, s.t. k+ctr over k <= a
template<typename T, const uint32_t k>
inline T opt_max_bc(const T a, const uint32_t n) {
	static_assert(k < 5, "sorry not implemented");

	if constexpr(k == 1) {
		return n - k - a;
	}

	if constexpr (k == 2) {
		const T b = 8 * a + 1;
		const double s = __builtin_sqrt(b);
		const auto s1 = s + 1;
		const auto s2 = s1 / 2.;
		const uint32_t t = __builtin_floor(s2);
		return n - t - 1;
	}

	if constexpr (k == 3) {
		if (a == 1)
			return n-k-1;

		float x = a;
		float t1 = sqrtf(729.f * x * x);
		float t2 = cbrtf(3.f * t1 + 81.f * x);
		float t3 = t2 / 2.09f;
		float ctr2 = t3;
		int ctr = int(ctr2);

		return  n - ctr - k;
	}

	if constexpr (k == 4) {
		const float x = a;
		const float t1 = __builtin_floorf(__builtin_sqrtf(24.f * x + 1.f));
		const float t2 = __builtin_floorf(__builtin_sqrtf(4.f * t1 + 5.f));
		uint32_t ctr = (t2 + 3.f)/ 2.f - 3;
		return  n - ctr - k;
	}


	// will never happen
	return -1;
}

/// same as `opt_max_bc` only doing 8 at the same time.
/// NOTE: w/k must be smaller than 5
template<typename T, const uint32_t k>
inline __m256i opt_max_bc_avx(const __m256i a, const __m256i n) {
	static_assert(k < 5, "sorry not implemented");
	if constexpr(k == 1) {
		return a;
	}

	if constexpr (k == 2) {
		// integer version
		const __m256i onei = _mm256_set1_epi32(1);
		const __m256i t1i = _mm256_slli_epi32(a, 3);
		const __m256i t2i = _mm256_add_epi32(t1i, onei);
		const __m256  t2f = _mm256_cvtepi32_ps(t2i);
		const __m256  t3f = _mm256_sqrt_ps(t2f);
		const __m256i t3i =  _mm256_cvtps_epi32(_mm256_floor_ps(t3f));
		const __m256i t4i = _mm256_add_epi32(t3i, onei);
		const __m256i t5i = _mm256_srli_epi32(t4i, 1);
		return t5i;

		// float version
		//const __m256 one = _mm256_set1_ps(1.);
		//const __m256 eight = _mm256_set1_ps(8.);
		//const __m256 af = _mm256_castsi256_ps(a);
		//const __m256 t1 = _mm256_fmadd_ps(af, eight, one);
		//const __m256 t2 = _mm256_sqrt_ps(t1);
		//const __m256 t3 = _mm256_add_ps(t2, one);
		//const __m256 t4 = _mm256_div_ps(t3, _mm256_set1_ps(2.));
		//return _mm256_floor_ps(t4);
	}

	// will never happen
	return _mm256_set1_epi32(1);
}
