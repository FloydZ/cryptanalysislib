#ifndef CRYPTANALYSISLIB_SORTING_NETWORK_AVX2_H
#define CRYPTANALYSISLIB_SORTING_NETWORK_AVX2_H

#ifndef USE_AVX2
#error "no avx"
#endif

#ifndef CRYPTANALYSISLIB_SORTING_NETWORK_H
#error "dont inlude 'sort/sorting_network/avx2.h' directly. Use `sort/sorting_network/common.h`."
#endif

#include <stdint.h>
#include <immintrin.h>

// SRC https://github.com/simd-sorting/fast-and-robust/blob/master/avx2_sort_demo/avx2sort.h
#define COEX32(a, b)                      \
	{                                     \
		auto vec_tmp = a;                 \
		a = _mm256_min_epi32(a, b);       \
		b = _mm256_max_epi32(vec_tmp, b); \
	}

// Signed
#define COEX8X16(a, b, tmp)               \
	{                                     \
		tmp = _mm_min_epi8(a, b);         \
		b = _mm_max_epi8(a, b);  	      \
		a = tmp;                 		  \
	}
#define COEX16X8(a, b, tmp)               \
	{                                     \
		tmp = a;                 		  \
		a = _mm_min_epi16(a, b);       	  \
		b = _mm_max_epi16(tmp, b); 	      \
	}
#define COEX32X4(a, b, tmp)               \
	{                                     \
		tmp = a;                 		  \
		a = _mm_min_epi32(a, b);       	  \
		b = _mm_max_epi32(tmp, b); 	      \
	}


#define COEX8X32(a, b, tmp)               \
	{                                     \
		tmp = a;                 		  \
		a = _mm256_min_epi8(a, b);        \
		b = _mm256_max_epi8(tmp, b);      \
	}
#define COEX16X16(a, b, tmp)              \
	{                                     \
		tmp = a;                 		  \
		a = _mm256_min_epi16(a, b);       \
		b = _mm256_max_epi16(tmp, b);     \
	}
#define COEX32X8(a, b, tmp)               \
	{                                     \
		tmp = a;                 		  \
		a = _mm256_min_epi32(a, b);       \
		b = _mm256_max_epi32(tmp, b);     \
	}




// unsigned
#define UCOEX8X16(a, b, tmp)              \
	{                                     \
		tmp = a;                 		  \
		a = _mm_min_epu8(a, b);        	  \
		b = _mm_max_epu8(tmp, b);  	      \
	}
#define UCOEX16X8(a, b, tmp)              \
	{                                     \
		tmp = a;                 		  \
		a = _mm_min_epu16(a, b);       	  \
		b = _mm_max_epu16(tmp, b); 	      \
	}
#define UCOEX32X4(a, b, tmp)              \
	{                                     \
		tmp = a;                 		  \
		a = _mm_min_epu32(a, b);       	  \
		b = _mm_max_epu32(tmp, b); 	      \
	}


#define UCOEX8X32(a, b, tmp)              \
	{                                     \
		tmp = a;                 		  \
		a = _mm256_min_epu8(a, b);        \
		b = _mm256_max_epu8(tmp, b);      \
	}
#define UCOEX16X16(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
		a = _mm256_min_epu16(a, b);       \
		b = _mm256_max_epu16(tmp, b);     \
	}
#define UCOEX32X8(a, b, tmp)              \
	{                                     \
		tmp = a;                 		  \
		a = _mm256_min_epu32(a, b);       \
		b = _mm256_max_epu32(tmp, b); \
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


/// needed by `sort_u8x16`
uint8_t layers[6][16] = {
        {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14},
        {3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12},
        {7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8},
        {2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13},
        {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0},
        {4,5,6,7,0,1,2,3,12,13,14,15, 8,9,10,11},
};

/// needed by `sort_u8x16`
int8_t blend[4][16] = {
        {0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1},
        {0, 0,-1,-1,0,0,-1,-1,0,0,-1,-1,0,0,-1,-1},
        {0,0,0,0,-1,-1,-1,-1,0,0,0,0,-1,-1,-1,-1},
        {0,0,0,0,0,0,0,0, -1,-1,-1,-1,-1,-1,-1,-1},
};

/// sorts a single SSE register
__m128i sortingnetwork_sort_u8x16(__m128i v) {
    __m128i t = v, tmp;

    // Step 1
    t = _mm_shuffle_epi8(t, _mm_load_si128((__m128i *)layers[0]));
    UCOEX8X16(t, v, tmp);
    t = _mm_blendv_epi8(t, v, _mm_load_si128((__m128i *)blend[0]));

    // Step 2
    v = _mm_shuffle_epi8(t, _mm_load_si128((__m128i *)layers[1]));
    UCOEX8X16(v, t, tmp);
    v = _mm_blendv_epi8(v, t, _mm_load_si128((__m128i *)blend[1]));

    // Step 3
    t = _mm_shuffle_epi8(v, _mm_load_si128((__m128i *)layers[0]));
    UCOEX8X16(t, v, tmp);
    t = _mm_blendv_epi8(t, v, _mm_load_si128((__m128i *)blend[0]));

    // Step 4
    v = _mm_shuffle_epi8(t, _mm_load_si128((__m128i *)layers[2]));
    UCOEX8X16(v, t, tmp);
    v = _mm_blendv_epi8(v, t, _mm_load_si128((__m128i *)blend[2]));

    // Step 5
    t = _mm_shuffle_epi8(v, _mm_load_si128((__m128i *)layers[3]));
    UCOEX8X16(t, v, tmp);
    t = _mm_blendv_epi8(t, v, _mm_load_si128((__m128i *)blend[1]));

    // Step 6
    v = _mm_shuffle_epi8(t, _mm_load_si128((__m128i *)layers[0]));
    UCOEX8X16(v, t, tmp);
    v = _mm_blendv_epi8(v, t, _mm_load_si128((__m128i *)blend[0]));

    // Step 7
    t = _mm_shuffle_epi8(v, _mm_load_si128((__m128i *)layers[4]));
    UCOEX8X16(t, v, tmp);
    t = _mm_blendv_epi8(t, v, _mm_load_si128((__m128i *)blend[3]));

    // Step 8
    v = _mm_shuffle_epi8(t, _mm_load_si128((__m128i *)layers[5]));
    UCOEX8X16(v, t, tmp);
    v = _mm_blendv_epi8(v, t, _mm_load_si128((__m128i *)blend[2]));

    // Step 9
    t = _mm_shuffle_epi8(v, _mm_load_si128((__m128i *)layers[3]));
    UCOEX8X16(t, v, tmp);
    t = _mm_blendv_epi8(t, v, _mm_load_si128((__m128i *)blend[1]));

    // Step 10
    v = _mm_shuffle_epi8(t, _mm_load_si128((__m128i *)layers[0]));
    UCOEX8X16(v, t, tmp);
    v = _mm_blendv_epi8(v, t, _mm_load_si128((__m128i *)blend[0]));
    return v;
}

void sortingnetwork_mergesort_u8x32(__m128i *a, __m128i *b) {
    __m128i H1 = _mm_shuffle_epi8(*b, _mm_load_si128((__m128i *)layers[4]));
    __m128i L1 = *a, tmp;

    UCOEX8X16(L1, H1, tmp);
    __m128i L1p = _mm_blendv_epi8(L1, _mm_bslli_si128(H1, 8), _mm_load_si128((__m128i *)blend[3]));
    __m128i H1p = _mm_blendv_epi8(_mm_bsrli_si128(L1, 8), H1, _mm_load_si128((__m128i *)blend[3]));

    UCOEX8X16(L1p, H1p, tmp);
    __m128i L2p = _mm_blendv_epi8(L1p, _mm_bslli_si128(H1p, 4), _mm_load_si128((__m128i *)blend[2]));
    __m128i H2p = _mm_blendv_epi8(_mm_bsrli_si128(L1p, 4), H1p, _mm_load_si128((__m128i *)blend[2]));

    UCOEX8X16(L2p, H2p, tmp);
    __m128i L3p = _mm_blendv_epi8(L2p, _mm_bslli_si128(H2p, 2), _mm_load_si128((__m128i *)blend[1]));
    __m128i H3p = _mm_blendv_epi8(_mm_bsrli_si128(L2p, 2), H2p, _mm_load_si128((__m128i *)blend[1]));

    UCOEX8X16(L3p, H3p, tmp);
    __m128i L4p = _mm_blendv_epi8(L3p, _mm_bslli_si128(H3p, 1), _mm_load_si128((__m128i *)blend[0]));
    __m128i H4p = _mm_blendv_epi8(_mm_bsrli_si128(L3p, 1), H3p, _mm_load_si128((__m128i *)blend[0]));

    UCOEX8X16(L4p, H4p, tmp);
    *a = _mm_unpacklo_epi8(L4p, H4p);
    *b = _mm_unpackhi_epi8(L4p, H4p);
}

void sortingnetwork_sort_u8x32(__m128i *a, __m128i *b) {
    *a = sortingnetwork_sort_u8x16(*a);
    *b = sortingnetwork_sort_u8x16(*b);
    sortingnetwork_mergesort_u8x32(a, b);
}
#endif
