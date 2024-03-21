#ifndef CRYPTANALYSISLIB_SORTING_NETWORK_AVX2_H
#define CRYPTANALYSISLIB_SORTING_NETWORK_AVX2_H

#ifndef USE_AVX2
#error "no avx"
#endif

#ifndef CRYPTANALYSISLIB_SORTING_NETWORK_H
#error "dont inlude 'sort/sorting_network/avx2.h' directly. Use `sort/sorting_network/common.h`."
#endif

#include <immintrin.h>

// SRC https://github.com/simd-sorting/fast-and-robust/blob/master/avx2_sort_demo/avx2sort.h
#define COEX32(a, b)                      \
	{                                     \
		auto vec_tmp = a;                 \
		a = _mm256_min_epi32(a, b);       \
		b = _mm256_max_epi32(vec_tmp, b); \
	}

/* shuffle 2 vectors, instruction for int is missing,
 * therefore shuffle with float */
#define SHUFFLE_2_VECS(a, b, mask)         \
	_mm256_castps_si256(_mm256_shuffle_ps( \
	        _mm256_castsi256_ps(a), _mm256_castsi256_ps(b), mask));

// optimized sorting network for two vectors, that is 16 ints
inline void sortingnetwork_sort_u32x16(__m256i &v1, __m256i &v2) noexcept {
	COEX32(v1, v2); /* step 1 */

	v2 = _mm256_shuffle_epi32(v2, _MM_SHUFFLE(2, 3, 0, 1)); /* step 2 */
	COEX32(v1, v2);

	auto tmp = v1; /* step  3 */
	v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
	COEX32(v1, v2);

	v2 = _mm256_shuffle_epi32(v2, _MM_SHUFFLE(0, 1, 2, 3)); /* step  4 */
	COEX32(v1, v2);

	tmp = v1; /* step  5 */
	v1 = SHUFFLE_2_VECS(v1, v2, 0b01000100);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b11101110);
	COEX32(v1, v2);

	tmp = v1; /* step  6 */
	v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
	COEX32(v1, v2);

	v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
	COEX32(v1, v2); /* step  7 */

	tmp = v1; /* step  8 */
	v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
	COEX32(v1, v2);

	tmp = v1; /* step  9 */
	v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
	COEX32(v1, v2);

	/* permute to make it easier to restore order */
	v1 = _mm256_permutevar8x32_epi32(v1, _mm256_setr_epi32(0, 4, 1, 5, 6, 2, 7, 3));
	v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(0, 4, 1, 5, 6, 2, 7, 3));

	tmp = v1; /* step  10 */
	v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
	COEX32(v1, v2);

	/* restore order */
	auto b2 = _mm256_shuffle_epi32(v2, 0b10110001);
	auto b1 = _mm256_shuffle_epi32(v1, 0b10110001);
	v1 = _mm256_blend_epi32(v1, b2, 0b10101010);
	v2 = _mm256_blend_epi32(b1, v2, 0b10101010);
}


/// avx single limb sorting network for uint32
__m256i sortingnetwork_sort_u32x8(__m256i v) {
	__m256i c, d, t;
	d = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);
	t = _mm256_permutevar8x32_epi32(v, d);
	COEX32(v, t);
	t = _mm256_blend_epi32(v, t, 0b10101010);

	c = _mm256_setr_epi32(3, 2, 1, 0, 7, 6, 5, 4);
	v = _mm256_permutevar8x32_epi32(t, c);
	COEX32(t, v);
	v = _mm256_blend_epi32(t, v, 0b11001100);

	t = _mm256_permutevar8x32_epi32(v, d);
	COEX32(v, t);
	t = _mm256_blend_epi32(v, t, 0b10101010);

	c = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
	v = _mm256_permutevar8x32_epi32(t, c);
	COEX32(t, v);
	v = _mm256_blend_epi32(t, v, 0b11110000);

	c = _mm256_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5);
	t = _mm256_permutevar8x32_epi32(v, c);
	COEX32(v, t);
	t = _mm256_blend_epi32(v, t, 0b11001100);

	v = _mm256_permutevar8x32_epi32(t, d);
	COEX32(t, v);
	t = _mm256_blend_epi32(t, v, 0b10101010);
	return t;
}
#endif