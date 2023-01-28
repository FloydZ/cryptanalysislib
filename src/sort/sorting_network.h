#pragma once

#include <immintrin.h>

// SRC https://github.com/simd-sorting/fast-and-robust/blob/master/avx2_sort_demo/avx2sort.h
#define COEX(a, b){                                                   		   \
    auto vec_tmp = a;                                                          \
    a = _mm256_min_epi32(a, b);                                                \
    b = _mm256_max_epi32(vec_tmp, b);}

/* shuffle 2 vectors, instruction for int is missing,
 * therefore shuffle with float */
#define SHUFFLE_2_VECS(a, b, mask)                                      \
    	_mm256_castps_si256 (_mm256_shuffle_ps(                         \
        _mm256_castsi256_ps (a), _mm256_castsi256_ps (b), mask));

// optimized sorting network for two vectors, that is 16 ints
inline void sortingnetwork_sort_16(__m256i &v1, __m256i &v2) {
	COEX(v1, v2);                                  /* step 1 */
	
	v2 = _mm256_shuffle_epi32(v2, _MM_SHUFFLE(2, 3, 0, 1)); /* step 2 */
	COEX(v1, v2);
	
	auto tmp = v1;                                          /* step  3 */
	v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
	COEX(v1, v2);
	
	v2 = _mm256_shuffle_epi32(v2, _MM_SHUFFLE(0, 1, 2, 3)); /* step  4 */
	COEX(v1, v2);
	
	tmp = v1;                                               /* step  5 */
	v1 = SHUFFLE_2_VECS(v1, v2, 0b01000100);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b11101110);
	COEX(v1, v2);
	
	tmp = v1;                                               /* step  6 */
	v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
	COEX(v1, v2);
	
	v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(7,6,5,4,3,2,1,0));
	COEX(v1, v2);                                           /* step  7 */
	
	tmp = v1;                                               /* step  8 */
	v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
	COEX(v1, v2);
	
	tmp = v1;                                               /* step  9 */
	v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
	COEX(v1, v2);
	
	/* permute to make it easier to restore order */
	v1 = _mm256_permutevar8x32_epi32(v1, _mm256_setr_epi32(0,4,1,5,6,2,7,3));
	v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(0,4,1,5,6,2,7,3));
	
	tmp = v1;                                              /* step  10 */
	v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
	COEX(v1, v2);
	
	/* restore order */
	auto b2 = _mm256_shuffle_epi32(v2,0b10110001);
	auto b1 = _mm256_shuffle_epi32(v1,0b10110001);
	v1 = _mm256_blend_epi32(v1, b2, 0b10101010);
	v2 = _mm256_blend_epi32(b1, v2, 0b10101010);
}


/// TODO finish
///
__m256i sortingnetwork_sort_8_32(__m256i v) {
	__m256i c, d, t;
	d = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);
	t = _mm256_permutevar8x32_epi32(v, d);
	COEX(v, t);
	t = _mm256_blend_epi32(v, t, 0b10101010);

	c = _mm256_setr_epi32(3, 2, 1, 0, 7, 6, 5, 4);
	v = _mm256_permutevar8x32_epi32(t, c);
	COEX(t, v);
	v = _mm256_blend_epi32(t, v, 0b11001100);

	t = _mm256_permutevar8x32_epi32(v, d);
	COEX(v, t);
	t = _mm256_blend_epi32(v, t, 0b10101010);

	c = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
	v = _mm256_permutevar8x32_epi32(t, c);
	COEX(t, v);
	v = _mm256_blend_epi32(t, v, 0b11110000);

	c = _mm256_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5);
	t = _mm256_permutevar8x32_epi32(v, c);
	COEX(v, t);
	t = _mm256_blend_epi32(v, t, 0b11001100);

	v = _mm256_permutevar8x32_epi32(t, d);
	COEX(t, v);
	t = _mm256_blend_epi32(t, v, 0b10101010);
	return t;
}
