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

#define COEX64X4(a,b,c,e,t)			\
	t = _mm256_cmpgt_epi64(a,b);	\
	c = _mm256_blendv_pd(a, b, t);	\
	e = _mm256_blendv_pd(b, a, t);

static inline void sortingnetwork_sort_i64x8(__m256i &a0, __m256i &b0) {
	__m256i a1,b1,t;
	COEX64X4(a0,b0,a1,b1,t)
	a0 = a1;
	b0 = _mm256_shuffle_pd(b1, b1, 0b0101);

	COEX64X4(a0,b0,a1,b1,t)
	a0 = _mm256_shuffle_pd(a1, b1, 0b1010);
	b0 = _mm256_shuffle_pd(a1, b1, 0b0101);

	COEX64X4(a0,b0,a1,b1,t)

	a1 = _mm256_permute4x64_epi64(a1, 0b11011000);
	b1 = _mm256_permute4x64_epi64(b1, 0b01100011);
	a0 = _mm256_blend_pd(a1, b1, 0b1010);
	b0 = _mm256_blend_pd(a1, b1, 0b0101);
	b0 = _mm256_permute4x64_epi64(b0, 0b01101100);

	COEX64X4(a0,b0,a1,b1,t)

	a0 = _mm256_blend_pd(a1, b1, 0b1100);
	b0 = _mm256_blend_pd(a1, b1, 0b0011);
	a0 = _mm256_permute4x64_epi64(a0, 0b10110100);
	b0 = _mm256_permute4x64_epi64(b0, 0b00011110);

	COEX64X4(a0,b0,a1,b1,t)

	b1 = _mm256_permute4x64_epi64(b1,0b10110001);
	a0 = _mm256_blend_pd(a1, b1, 0b1010);
	b0 = _mm256_blend_pd(a1, b1, 0b0101);
	b0 = _mm256_permute4x64_epi64(b0, 0b10110001);

	COEX64X4(a0,b0,a1,b1,t)

	b1 = _mm256_permute4x64_epi64(b1,0b01001110);
	a0 = _mm256_blend_pd(a1, b1, 0b1100);
	b0 = _mm256_blend_pd(a1, b1, 0b0011);

	b0 = _mm256_permute4x64_epi64(b0, 0b01110010);
	a0 = _mm256_permute4x64_epi64(a0, 0b11011000);
}

/* shuffle 2 vectors, instruction for int is missing,
 * therefore shuffle with float */
#define SHUFFLE_2_VECS(a, b, mask)         \
	_mm256_castps_si256(_mm256_shuffle_ps( \
	        _mm256_castsi256_ps(a), _mm256_castsi256_ps(b), mask));

// optimized sorting network for two vectors, that is 16 ints
static inline void sortingnetwork_sort_u32x16(__m256i &v1, __m256i &v2) noexcept {
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
static inline __m256i sortingnetwork_sort_u32x8(__m256i v) {
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


///
/// \param v1
/// \param v2
/// \return
constexpr void sortingnetowrk_sort_u32x8(__m128i &v1, __m128i &v2) noexcept {
/* shuffle 2 __m128i vectors */
#define SHUFFLE_TWO_VECS(a, b, mask)                                    \
	reinterpret_cast<__m128i>(_mm_shuffle_ps(                           \
    reinterpret_cast<__m128>(a), reinterpret_cast<__m128>(b), mask));

	__m128i t;
	/* step 1 */
	UCOEX32X4(v1, v2, t);
	v2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(2, 3, 0, 1));

	/* step 2 */
	UCOEX32X4(v1, v2, t);
	auto tmp = v1;
	v1 = SHUFFLE_TWO_VECS(v1, v2, 0b10001000);
	v2 = SHUFFLE_TWO_VECS(tmp, v2, 0b11011101);

	/* step 3 */
	UCOEX32X4(v1, v2, t);
	v2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(0, 1, 2, 3));

	/* step 4 */
	UCOEX32X4(v1, v2, t);
	tmp = v1;
	v1 = SHUFFLE_TWO_VECS(v1, v2, 0b01000100);
	v2 = SHUFFLE_TWO_VECS(tmp, v2, 0b11101110);

	/* step 5 */
	UCOEX32X4(v1, v2, t);
	tmp = v1;
	v1 = SHUFFLE_TWO_VECS(v1, v2, 0b10001000);
	v2 = SHUFFLE_TWO_VECS(tmp, v2, 0b11011101);

	/* step 6 */
	UCOEX32X4(v1, v2, t);

	/* restore order */
	tmp = _mm_shuffle_epi32(v1, _MM_SHUFFLE(2, 3, 0, 1));
	auto tmp2 = _mm_shuffle_epi32(v2, _MM_SHUFFLE(2, 3, 0, 1));
	v2 = _mm_blend_epi32(tmp, v2, 0b00001010);
	v1 = _mm_blend_epi32(v1, tmp2, 0b00001010);
}


/* merge columns without transposition */
#define ASC(a, b, c, d, e, f, g, h)                                    \
  (((h < 7) << 7) | ((g < 6) << 6) | ((f < 5) << 5) | ((e < 4) << 4) | \
      ((d < 3) << 3) | ((c < 2) << 2) | ((b < 1) << 1) | (a < 0))

#define REVERSE_VEC(vec){                                              \
    vec = _mm256_permutevar8x32_epi32(                                 \
        vec, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));}

#define COEX_SHUFFLE(vec, a, b, c, d, e, f, g, h, MASK){               \
    constexpr int shuffle_mask = _MM_SHUFFLE(d, c, b, a);              \
    __m256i shuffled = _mm256_shuffle_epi32(vec, shuffle_mask);        \
    __m256i min = _mm256_min_epi32(shuffled, vec);                     \
    __m256i max = _mm256_max_epi32(shuffled, vec);                     \
    constexpr int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask);}

#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h, MASK){               \
    __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);  \
    __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask); \
    __m256i min = _mm256_min_epi32(permuted, vec);                     \
    __m256i max = _mm256_max_epi32(permuted, vec);                     \
    constexpr int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask);}

/* sort 8 columns, each containing 16 int, with Green's 60 modules network */
#define SORT_16_INT_COLUMNS_WISE(S)							\
constexpr inline                         					\
void sort_16_int_column_wise_ ##S (__m256i* vecs) noexcept {\
	/* step 1 */											\
	COEX(vecs[0], vecs[1]); COEX(vecs[2], vecs[3]);			\
	COEX(vecs[4], vecs[5]); COEX(vecs[6], vecs[7]);			\
	COEX(vecs[8], vecs[9]); COEX(vecs[10], vecs[11]);		\
	COEX(vecs[12], vecs[13]); COEX(vecs[14], vecs[15]);		\
	/* step 2 */											\
	COEX(vecs[0], vecs[2]); COEX(vecs[1], vecs[3]);			\
	COEX(vecs[4], vecs[6]); COEX(vecs[5], vecs[7]);			\
	COEX(vecs[8], vecs[10]); COEX(vecs[9], vecs[11]);		\
	COEX(vecs[12], vecs[14]); COEX(vecs[13], vecs[15]);		\
	/* step 3 */											\
	COEX(vecs[0], vecs[4]); COEX(vecs[1], vecs[5]);			\
	COEX(vecs[2], vecs[6]); COEX(vecs[3], vecs[7]);			\
	COEX(vecs[8], vecs[12]); COEX(vecs[9], vecs[13]);		\
	COEX(vecs[10], vecs[14]); COEX(vecs[11], vecs[15]);		\
	/* step 4 */ 											\
	COEX(vecs[0], vecs[8]); COEX(vecs[1], vecs[9]);			\
	COEX(vecs[2], vecs[10]); COEX(vecs[3], vecs[11]);		\
	COEX(vecs[4], vecs[12]); COEX(vecs[5], vecs[13]);		\
	COEX(vecs[6], vecs[14]); COEX(vecs[7], vecs[15]);		\
	/* step 5 */											\
	COEX(vecs[5], vecs[10]); COEX(vecs[6], vecs[9]);		\
	COEX(vecs[3], vecs[12]); COEX(vecs[7], vecs[11]);		\
	COEX(vecs[13], vecs[14]); COEX(vecs[4], vecs[8]);		\
	COEX(vecs[1], vecs[2]);									\
	/* step 6 */											\
	COEX(vecs[1], vecs[4]); COEX(vecs[7], vecs[13]);		\
	COEX(vecs[2], vecs[8]); COEX(vecs[11], vecs[14]);		\
	/* step 7 */											\
	COEX(vecs[2], vecs[4]); COEX(vecs[5], vecs[6]);			\
	COEX(vecs[9], vecs[10]); COEX(vecs[11], vecs[13]);		\
	COEX(vecs[3], vecs[8]); COEX(vecs[7], vecs[12]);		\
	/* step 8 */											\
	COEX(vecs[3], vecs[5]); COEX(vecs[6], vecs[8]);			\
	COEX(vecs[7], vecs[9]); COEX(vecs[10], vecs[12]);		\
	/* step 9 */											\
	COEX(vecs[3], vecs[4]); COEX(vecs[5], vecs[6]);			\
	COEX(vecs[7], vecs[8]); COEX(vecs[9], vecs[10]);		\
	COEX(vecs[11], vecs[12]);								\
	/* step 10 */											\
	COEX(vecs[6], vecs[7]); COEX(vecs[8], vecs[9]);			\
}

#define CREATE_MERGE_8_COLUMNS_WITH_16_ELEMENTS(S) 												\
constexpr void inline merge_8_columns_with_16_elements_ ##S (__m256i* vecs) noexcept {			\
	vecs[8] = _mm256_shuffle_epi32(vecs[8], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[7], vecs[8]); 		\
	vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[6], vecs[9]); 		\
	vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[5], vecs[10]);   \
	vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[4], vecs[11]);   \
	vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[3], vecs[12]);   \
	vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[2], vecs[13]);   \
	vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[1], vecs[14]);   \
	vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[0], vecs[15]);   \
	vecs[4] = _mm256_shuffle_epi32(vecs[4], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[3], vecs[4]); 		\
	vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[2], vecs[5]); 		\
	vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[1], vecs[6]); 		\
	vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[0], vecs[7]); 		\
	vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[11], vecs[12]);  \
	vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[10], vecs[13]);  \
	vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[9], vecs[14]);   \
	vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[8], vecs[15]);   \
	vecs[2] = _mm256_shuffle_epi32(vecs[2], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[1], vecs[2]); 		\
	vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[0], vecs[3]); 		\
	vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[5], vecs[6]); 		\
	vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[4], vecs[7]); 		\
	vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[9], vecs[10]);   \
	vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[8], vecs[11]);   \
	vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[13], vecs[14]);  \
	vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[12], vecs[15]);  \
	vecs[1] = _mm256_shuffle_epi32(vecs[1], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[0], vecs[1]); 		\
	vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[2], vecs[3]); 		\
	vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[4], vecs[5]); 		\
	vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[6], vecs[7]); 		\
	vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[8], vecs[9]); 		\
	vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[10], vecs[11]);  \
	vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[12], vecs[13]);  \
	vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(2,3,0,1)); COEX(vecs[14], vecs[15]);  \
	COEX_SHUFFLE(vecs[0], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_SHUFFLE(vecs[1], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[2], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_SHUFFLE(vecs[3], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[4], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_SHUFFLE(vecs[5], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[6], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_SHUFFLE(vecs[7], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[8], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_SHUFFLE(vecs[9], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[10], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_SHUFFLE(vecs[11], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[12], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_SHUFFLE(vecs[13], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[14], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_SHUFFLE(vecs[15], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	vecs[8] = _mm256_shuffle_epi32(vecs[8], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[7], vecs[8]); 		\
	vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[6], vecs[9]); 		\
	vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[5], vecs[10]); 	\
	vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[4], vecs[11]); 	\
	vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[3], vecs[12]); 	\
	vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[2], vecs[13]); 	\
	vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[1], vecs[14]); 	\
	vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[0], vecs[15]); 	\
	vecs[4] = _mm256_shuffle_epi32(vecs[4], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[3], vecs[4]); 		\
	vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[2], vecs[5]); 		\
	vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[1], vecs[6]); 		\
	vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[0], vecs[7]); 		\
	vecs[12] = _mm256_shuffle_epi32(vecs[12], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[11], vecs[12]);	\
	vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[10], vecs[13]);	\
	vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[9], vecs[14]); 	\
	vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[8], vecs[15]); 	\
	vecs[2] = _mm256_shuffle_epi32(vecs[2], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[1], vecs[2]); 		\
	vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[0], vecs[3]); 		\
	vecs[6] = _mm256_shuffle_epi32(vecs[6], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[5], vecs[6]); 		\
	vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[4], vecs[7]); 		\
	vecs[10] = _mm256_shuffle_epi32(vecs[10], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[9], vecs[10]); 	\
	vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[8], vecs[11]); 	\
	vecs[14] = _mm256_shuffle_epi32(vecs[14], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[13], vecs[14]);	\
	vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[12], vecs[15]);	\
	vecs[1] = _mm256_shuffle_epi32(vecs[1], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[0], vecs[1]); 		\
	vecs[3] = _mm256_shuffle_epi32(vecs[3], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[2], vecs[3]); 		\
	vecs[5] = _mm256_shuffle_epi32(vecs[5], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[4], vecs[5]); 		\
	vecs[7] = _mm256_shuffle_epi32(vecs[7], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[6], vecs[7]); 		\
	vecs[9] = _mm256_shuffle_epi32(vecs[9], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[8], vecs[9]); 		\
	vecs[11] = _mm256_shuffle_epi32(vecs[11], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[10], vecs[11]);  \
	vecs[13] = _mm256_shuffle_epi32(vecs[13], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[12], vecs[13]);  \
	vecs[15] = _mm256_shuffle_epi32(vecs[15], _MM_SHUFFLE(0,1,2,3)); COEX(vecs[14], vecs[15]);  \
	COEX_SHUFFLE(vecs[0], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[0], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[1], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[1], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[2], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[2], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[3], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[3], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[4], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[4], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[5], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[5], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[6], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[6], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[7], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[7], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[8], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[8], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[9], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[9], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[10], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[10], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[11], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[11], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[12], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[12], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[13], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[13], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[14], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[14], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_SHUFFLE(vecs[15], 3, 2, 1, 0, 7, 6, 5, 4, ASC); COEX_SHUFFLE(vecs[15], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	REVERSE_VEC(vecs[8]); COEX(vecs[7], vecs[8]); REVERSE_VEC(vecs[9]); COEX(vecs[6], vecs[9]);				\
	REVERSE_VEC(vecs[10]); COEX(vecs[5], vecs[10]); REVERSE_VEC(vecs[11]); COEX(vecs[4], vecs[11]);			\
	REVERSE_VEC(vecs[12]); COEX(vecs[3], vecs[12]); REVERSE_VEC(vecs[13]); COEX(vecs[2], vecs[13]);			\
	REVERSE_VEC(vecs[14]); COEX(vecs[1], vecs[14]); REVERSE_VEC(vecs[15]); COEX(vecs[0], vecs[15]);			\
	REVERSE_VEC(vecs[4]); COEX(vecs[3], vecs[4]); REVERSE_VEC(vecs[5]); COEX(vecs[2], vecs[5]);				\
	REVERSE_VEC(vecs[6]); COEX(vecs[1], vecs[6]); REVERSE_VEC(vecs[7]); COEX(vecs[0], vecs[7]);				\
	REVERSE_VEC(vecs[12]); COEX(vecs[11], vecs[12]); REVERSE_VEC(vecs[13]); COEX(vecs[10], vecs[13]);		\
	REVERSE_VEC(vecs[14]); COEX(vecs[9], vecs[14]); REVERSE_VEC(vecs[15]); COEX(vecs[8], vecs[15]);			\
	REVERSE_VEC(vecs[2]); COEX(vecs[1], vecs[2]); REVERSE_VEC(vecs[3]); COEX(vecs[0], vecs[3]);				\
	REVERSE_VEC(vecs[6]); COEX(vecs[5], vecs[6]); REVERSE_VEC(vecs[7]); COEX(vecs[4], vecs[7]);				\
	REVERSE_VEC(vecs[10]); COEX(vecs[9], vecs[10]); REVERSE_VEC(vecs[11]); COEX(vecs[8], vecs[11]);			\
	REVERSE_VEC(vecs[14]); COEX(vecs[13], vecs[14]); REVERSE_VEC(vecs[15]); COEX(vecs[12], vecs[15]);		\
	REVERSE_VEC(vecs[1]); COEX(vecs[0], vecs[1]); REVERSE_VEC(vecs[3]); COEX(vecs[2], vecs[3]);				\
	REVERSE_VEC(vecs[5]); COEX(vecs[4], vecs[5]); REVERSE_VEC(vecs[7]); COEX(vecs[6], vecs[7]);				\
	REVERSE_VEC(vecs[9]); COEX(vecs[8], vecs[9]); REVERSE_VEC(vecs[11]); COEX(vecs[10], vecs[11]);			\
	REVERSE_VEC(vecs[13]); COEX(vecs[12], vecs[13]); REVERSE_VEC(vecs[15]); COEX(vecs[14], vecs[15]);		\
	COEX_PERMUTE(vecs[0], 7, 6, 5, 4, 3, 2, 1, 0, ASC); COEX_SHUFFLE(vecs[0], 2, 3, 0, 1, 6, 7, 4, 5, ASC); \
	COEX_SHUFFLE(vecs[0], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_PERMUTE(vecs[1], 7, 6, 5, 4, 3, 2, 1, 0, ASC); \
	COEX_SHUFFLE(vecs[1], 2, 3, 0, 1, 6, 7, 4, 5, ASC); COEX_SHUFFLE(vecs[1], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_PERMUTE(vecs[2], 7, 6, 5, 4, 3, 2, 1, 0, ASC); COEX_SHUFFLE(vecs[2], 2, 3, 0, 1, 6, 7, 4, 5, ASC); \
	COEX_SHUFFLE(vecs[2], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_PERMUTE(vecs[3], 7, 6, 5, 4, 3, 2, 1, 0, ASC); \
	COEX_SHUFFLE(vecs[3], 2, 3, 0, 1, 6, 7, 4, 5, ASC); COEX_SHUFFLE(vecs[3], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_PERMUTE(vecs[4], 7, 6, 5, 4, 3, 2, 1, 0, ASC); COEX_SHUFFLE(vecs[4], 2, 3, 0, 1, 6, 7, 4, 5, ASC); \
	COEX_SHUFFLE(vecs[4], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_PERMUTE(vecs[5], 7, 6, 5, 4, 3, 2, 1, 0, ASC); \
	COEX_SHUFFLE(vecs[5], 2, 3, 0, 1, 6, 7, 4, 5, ASC); COEX_SHUFFLE(vecs[5], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_PERMUTE(vecs[6], 7, 6, 5, 4, 3, 2, 1, 0, ASC); COEX_SHUFFLE(vecs[6], 2, 3, 0, 1, 6, 7, 4, 5, ASC); \
	COEX_SHUFFLE(vecs[6], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_PERMUTE(vecs[7], 7, 6, 5, 4, 3, 2, 1, 0, ASC); \
	COEX_SHUFFLE(vecs[7], 2, 3, 0, 1, 6, 7, 4, 5, ASC); COEX_SHUFFLE(vecs[7], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_PERMUTE(vecs[8], 7, 6, 5, 4, 3, 2, 1, 0, ASC); COEX_SHUFFLE(vecs[8], 2, 3, 0, 1, 6, 7, 4, 5, ASC); \
	COEX_SHUFFLE(vecs[8], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_PERMUTE(vecs[9], 7, 6, 5, 4, 3, 2, 1, 0, ASC); \
	COEX_SHUFFLE(vecs[9], 2, 3, 0, 1, 6, 7, 4, 5, ASC); COEX_SHUFFLE(vecs[9], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_PERMUTE(vecs[10], 7, 6, 5, 4, 3, 2, 1, 0, ASC); COEX_SHUFFLE(vecs[10], 2, 3, 0, 1, 6, 7, 4, 5, ASC); \
	COEX_SHUFFLE(vecs[10], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_PERMUTE(vecs[11], 7, 6, 5, 4, 3, 2, 1, 0, ASC); \
	COEX_SHUFFLE(vecs[11], 2, 3, 0, 1, 6, 7, 4, 5, ASC); COEX_SHUFFLE(vecs[11], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_PERMUTE(vecs[12], 7, 6, 5, 4, 3, 2, 1, 0, ASC); COEX_SHUFFLE(vecs[12], 2, 3, 0, 1, 6, 7, 4, 5, ASC); \
	COEX_SHUFFLE(vecs[12], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_PERMUTE(vecs[13], 7, 6, 5, 4, 3, 2, 1, 0, ASC); \
	COEX_SHUFFLE(vecs[13], 2, 3, 0, 1, 6, 7, 4, 5, ASC); COEX_SHUFFLE(vecs[13], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
	COEX_PERMUTE(vecs[14], 7, 6, 5, 4, 3, 2, 1, 0, ASC); COEX_SHUFFLE(vecs[14], 2, 3, 0, 1, 6, 7, 4, 5, ASC); \
	COEX_SHUFFLE(vecs[14], 1, 0, 3, 2, 5, 4, 7, 6, ASC); COEX_PERMUTE(vecs[15], 7, 6, 5, 4, 3, 2, 1, 0, ASC); \
	COEX_SHUFFLE(vecs[15], 2, 3, 0, 1, 6, 7, 4, 5, ASC); COEX_SHUFFLE(vecs[15], 1, 0, 3, 2, 5, 4, 7, 6, ASC); \
}

#define COEX(a, b){ __m256i c = a; a = _mm256_min_epi32(a, b); b = _mm256_max_epi32(c, b); }
SORT_16_INT_COLUMNS_WISE(i)
CREATE_MERGE_8_COLUMNS_WITH_16_ELEMENTS(i)

#undef COEX
#undef COEX_SHUFFLE
#undef COEX_PERMUTE

#define COEX_SHUFFLE(vec, a, b, c, d, e, f, g, h, MASK){               \
    constexpr int shuffle_mask = _MM_SHUFFLE(d, c, b, a);              \
    __m256i shuffled = _mm256_shuffle_epi32(vec, shuffle_mask);        \
    __m256i min = _mm256_min_epu32(shuffled, vec);                     \
    __m256i max = _mm256_max_epu32(shuffled, vec);                     \
    constexpr int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask);}

#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h, MASK){               \
    __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);  \
    __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask); \
    __m256i min = _mm256_min_epu32(permuted, vec);                     \
    __m256i max = _mm256_max_epu32(permuted, vec);                     \
    constexpr int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask);}

#define COEX(a, b){ __m256i c = a; a = _mm256_min_epu32(a, b); b = _mm256_max_epu32(c, b);}
SORT_16_INT_COLUMNS_WISE(u)
CREATE_MERGE_8_COLUMNS_WITH_16_ELEMENTS(u)

/// sorts 128 i32 elements
/// \param v
/// \return
constexpr void sortingnetwork_sort_i32x128(__m256i *v) noexcept{
	sort_16_int_column_wise_i(v);
	merge_8_columns_with_16_elements_i(v);
}

/// sorts 128 u32 elements
/// \param v
/// \return
constexpr void sortingnetwork_sort_u32x128(__m256i *v) noexcept{
	sort_16_int_column_wise_u(v);
	merge_8_columns_with_16_elements_u(v);
}

#undef COEX
#undef COEX_PERMUTE
#undef COEX_SHUFFLE
#undef REVERSE_VEC
#undef ASC

#endif
