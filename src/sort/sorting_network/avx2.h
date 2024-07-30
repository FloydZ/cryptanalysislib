#ifndef CRYPTANALYSISLIB_SORTING_NETWORK_AVX2_H
#define CRYPTANALYSISLIB_SORTING_NETWORK_AVX2_H

#include <cstdint>
#ifndef USE_AVX2
#error "no avx"
#endif

#ifndef CRYPTANALYSISLIB_SORTING_NETWORK_H
#error "dont inlude 'sort/sorting_network/avx2.h' directly. Use `sort/sorting_network/common.h`."
#endif

#include <stdint.h>
#include <immintrin.h>
#include "simd/simd.h"


#ifdef __clang__
// SRC https://github.com/sortingnetwork-sorting/fast-and-robust/blob/master/avx2_sort_demo/avx2sort.h
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
#define COEX_i32x8(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
		a = _mm256_min_epi32(a, b);       \
		b = _mm256_max_epi32(tmp, b);     \
	}

#define COEX_i32x8_(a, b, c, d)           \
	{                                     \
		c = _mm256_min_epi32(a, b);       \
		d = _mm256_max_epi32(a, b);       \
	}

// unsigned
#define COEX_u8x16(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
		a = _mm_min_epu8(a, b);        	  \
		b = _mm_max_epu8(tmp, b);  	      \
	}
#define COEX_u16x8(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
		a = _mm_min_epu16(a, b);       	  \
		b = _mm_max_epu16(tmp, b); 	      \
	}
#define COEX_u32x4(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
		a = _mm_min_epu32(a, b);       	  \
		b = _mm_max_epu32(tmp, b); 	      \
	}

#else 

#define COEX8X16(a, b, tmp)               \
	{                                     \
		tmp = _mm_min_epi8(a, b);         \
  		tmp = __builtin_ia32_pminsb128((__v16qi)a, (__v16qi)b); \
  		b = __builtin_ia32_pmaxsb128((__v16qi)a, (__v16qi)b); 	\
		a = tmp;                 		  \
	}
#define COEX16X8(a, b, tmp)               \
	{                                     \
		tmp = a;                 		  \
  		a =  (__m128i)__builtin_ia32_pminsw128 ((__v8hi)a, (__v8hi)b);\
  		b =  (__m128i)__builtin_ia32_pmaxsw128 ((__v8hi)tmp, (__v8hi)b);\
	}
#define COEX32X4(a, b, tmp)               \
	{                                     \
		tmp = a;                 		  \
  		a = (__m128i) __builtin_ia32_pminsd128 ((__v4si)a, (__v4si)b);	\
  		b = (__m128i) __builtin_ia32_pmaxsd128 ((__v4si)tmp, (__v4si)b);\
	}


#define COEX8X32(a, b, tmp)               \
	{                                     \
		tmp = a;                 		  \
  		a = (__m256i)__builtin_ia32_pminsb256 ((__v32qi)a, (__v32qi)b);	\
  		b = (__m256i)__builtin_ia32_pmaxsb256 ((__v32qi)tmp, (__v32qi)b);\
	}
#define COEX16X16(a, b, tmp)              \
	{                                     \
		tmp = a;                 		  \
  		a = (__m256i)__builtin_ia32_pminsw256 ((__v16hi)a, (__v16hi)b);\
  		b = (__m256i)__builtin_ia32_pmaxsw256 ((__v16hi)tmp, (__v16hi)b);\
	}
#define COEX_i32x8(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
  		a = (__m256i)__builtin_ia32_pminsd256 ((__v8si)a, (__v8si)b);\
  		b = (__m256i)__builtin_ia32_pmaxsd256 ((__v8si)tmp, (__v8si)b);\
	}

#define COEX_i32x8_(a, b, c, d)           \
	{                                     \
  		c = (__m256i)__builtin_ia32_pminsd256 ((__v8si)a, (__v8si)b);\
  		d = (__m256i)__builtin_ia32_pmaxsd256 ((__v8si)a, (__v8si)b);\
	}

// unsigned
#define COEX_u8x16(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
  		a = (__m128i)__builtin_ia32_pminub128 ((__v16qi)a, (__v16qi)b); \
  		b = (__m128i)__builtin_ia32_pmaxub128 ((__v16qi)tmp, (__v16qi)b);\
	}
#define COEX_u16x8(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
  		a = (__m128i) __builtin_ia32_pminuw128 ((__v8hi)a, (__v8hi)b);	\
  		b = (__m128i) __builtin_ia32_pmaxuw128 ((__v8hi)tmp, (__v8hi)b);\
	}
#define COEX_u32x4(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
  		a = (__m128i) __builtin_ia32_pminud128 ((__v4si)a, (__v4si)b); \
  		b = (__m128i) __builtin_ia32_pminud128 ((__v4si)tmp, (__v4si)b); \
	}
#endif


#ifdef __clang__
#define COEX_u8x32(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
		a = _mm256_min_epu8(a, b);        \
		b = _mm256_max_epu8(tmp, b);      \
	}
#define u_COEX_u16x16(a, b, tmp)          \
	{                                     \
		tmp = a;                 		  \
		a = _mm256_min_epu16(a, b);       \
		b = _mm256_max_epu16(tmp, b);     \
	}
#define COEX_u32x8(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
		a = _mm256_min_epu32(a, b);       \
		b = _mm256_max_epu32(tmp, b);     \
	}
#define COEX_u32x8_(a, b, c, d)           \
	{                                     \
		c = _mm256_min_epu32(a, b);       \
		d = _mm256_max_epu32(a, b);     \
	}

// float32
#define COEX_f32x8(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
		a = _mm256_min_ps(a, b);       	  \
		b = _mm256_max_ps(tmp, b);     	  \
	}
#define COEX_f32x8_(a, b, c, d)           \
	{                                     \
		c = _mm256_min_ps(a, b);       	  \
		d = _mm256_max_ps(a, b);     	  \
	}
#else

#define COEX_u8x32(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
  		a = (__m256i)__builtin_ia32_pminub256 ((__v32qi)a, (__v32qi)b); 	\
  		b = (__m256i)__builtin_ia32_pmaxub256 ((__v32qi)tmp, (__v32qi)b);	\
	}
#define u_COEX_u16x16(a, b, tmp)          \
	{                                     \
		tmp = a;                 		  \
  		a = (__m256i)__builtin_ia32_pminuw256 ((__v16hi)a, (__v16hi)b); 	\
  		b = (__m256i)__builtin_ia32_pmaxuw256 ((__v16hi)tmp, (__v16hi)b);	\
	}
#define COEX_u32x8(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
		a = (__m256i)__builtin_ia32_pminud256 ((__v8si)a, (__v8si)b); 		\
  		b = (__m256i)__builtin_ia32_pmaxud256 ((__v8si)tmp, (__v8si)b); 	\
	}

#define COEX_u32x8_(a, b, c, d)           \
	{                                     \
		c = (__m256i)__builtin_ia32_pminud256 ((__v8si)a, (__v8si)b); 		\
  		d = (__m256i)__builtin_ia32_pmaxud256 ((__v8si)a, (__v8si)b); 		\
	}

// float32
#define COEX_f32x8(a, b, tmp)             \
	{                                     \
		tmp = a;                 		  \
  		a = (__m256) __builtin_ia32_minps256 ((__v8sf)a, (__v8sf)b); 		\
  		b = (__m256) __builtin_ia32_maxps256 ((__v8sf)tmp, (__v8sf)b);		\
	}

#define COEX_f32x8_(a, b, c, d)           \
	{                                     \
  		c = (__m256) __builtin_ia32_minps256 ((__v8sf)a, (__v8sf)b); 		\
  		d = (__m256) __builtin_ia32_maxps256 ((__v8sf)a, (__v8sf)b);		\
	}
#endif



/// sorting 8 int64 at once
/// \param a0 input/output
/// \param b0 input/output
static inline void sortingnetwork_sort_i64x8(__m256i &a0,
                                                       __m256i &b0) noexcept {
#ifdef __clang__
#define COEX64X4(a,b,c,e,t)												\
	t = ((__v4di)a > (__v4di)b);										\
	c = (__m256i)_mm256_blendv_pd((__m256d)a, (__m256d)b, (__m256d)t);	\
	e = (__m256i)_mm256_blendv_pd((__m256d)b, (__m256d)a, (__m256d)t);
#else

#define COEX64X4(a,b,c,e,t)												\
	t = ((__v4di)a > (__v4di)b);										\
  	c = (__m256i)(__m256d) __builtin_ia32_blendvpd256 ((__v4df)a,(__v4df)b,(__v4df)t);\
  	e = (__m256i)(__m256d) __builtin_ia32_blendvpd256 ((__v4df)b,(__v4df)a,(__v4df)t);
#endif

	__m256i a1,b1,t;
	COEX64X4(a0,b0,a1,b1,t)
	a0 = a1;

#ifdef __clang__
	b0 = (__m256i)_mm256_shuffle_pd((__m256d)b1, (__m256d)b1, 0b0101);

	COEX64X4(a0,b0,a1,b1,t)
	a0 = (__m256i)_mm256_shuffle_pd((__m256d)a1,(__m256d)b1, 0b1010);
	b0 = (__m256i)_mm256_shuffle_pd((__m256d)a1,(__m256d)b1, 0b0101);

	COEX64X4(a0,b0,a1,b1,t)

	a1 = _mm256_permute4x64_epi64(a1, 0b11011000);
	b1 = _mm256_permute4x64_epi64(b1, 0b01100011);
	a0 = (__m256i)_mm256_blend_pd((__m256d)a1,(__m256d)b1, 0b1010);
	b0 = (__m256i)_mm256_blend_pd((__m256d)a1,(__m256d)b1, 0b0101);
	b0 = _mm256_permute4x64_epi64(b0, 0b01101100);

	COEX64X4(a0,b0,a1,b1,t)

	a0 = (__m256i)_mm256_blend_pd((__m256d)a1, (__m256d)b1, 0b1100);
	b0 = (__m256i)_mm256_blend_pd((__m256d)a1, (__m256d)b1, 0b0011);
	a0 = _mm256_permute4x64_epi64(a0, 0b10110100);
	b0 = _mm256_permute4x64_epi64(b0, 0b00011110);

	COEX64X4(a0,b0,a1,b1,t)

	b1 = _mm256_permute4x64_epi64(b1,0b10110001);
	a0 = (__m256i)_mm256_blend_pd((__m256d)a1, (__m256d)b1, 0b1010);
	b0 = (__m256i)_mm256_blend_pd((__m256d)a1, (__m256d)b1, 0b0101);
	b0 = _mm256_permute4x64_epi64(b0, 0b10110001);

	COEX64X4(a0,b0,a1,b1,t)

	b1 = _mm256_permute4x64_epi64(b1,0b01001110);
	a0 = (__m256i)_mm256_blend_pd((__m256d)a1, (__m256d)b1, 0b1100);
	b0 = (__m256i)_mm256_blend_pd((__m256d)a1, (__m256d)b1, 0b0011);

	b0 = _mm256_permute4x64_epi64(b0, 0b01110010);
	a0 = _mm256_permute4x64_epi64(a0, 0b11011000);
#else 
  	b0 = (__m256i)(__m256d) __builtin_ia32_shufpd256 ((__v4df)b1, (__v4df)b1, 0b0101);

	COEX64X4(a0,b0,a1,b1,t)
  	a0 = (__m256i)(__m256d) __builtin_ia32_shufpd256 ((__v4df)a1, (__v4df)b1, 0b1010);
  	b0 = (__m256i)(__m256d) __builtin_ia32_shufpd256 ((__v4df)a1, (__v4df)b1, 0b0101);

	COEX64X4(a0,b0,a1,b1,t)

  	a1 = (__m256i) __builtin_ia32_permdi256 ((__v4di)a1, 0b11011000);
  	b1 = (__m256i) __builtin_ia32_permdi256 ((__v4di)b1, 0b01100011);
  	a0 = (__m256i) (__m256d) __builtin_ia32_blendpd256 ((__v4df)a1, (__v4df)b1, 0b1010);
  	b0 = (__m256i) (__m256d) __builtin_ia32_blendpd256 ((__v4df)a1, (__v4df)b1, 0b0101);
	b0 = _mm256_permute4x64_epi64(b0, 0b01101100);


	COEX64X4(a0,b0,a1,b1,t)

  	a0 = (__m256i) (__m256d) __builtin_ia32_blendpd256 ((__v4df)a1, (__v4df)b1, 0b1100);
  	b0 = (__m256i) (__m256d) __builtin_ia32_blendpd256 ((__v4df)a1, (__v4df)b1, 0b0011);
  	a0 = (__m256i) __builtin_ia32_permdi256 ((__v4di)a0, 0b10110100);
  	b0 = (__m256i) __builtin_ia32_permdi256 ((__v4di)b0, 0b00011110);

	COEX64X4(a0,b0,a1,b1,t)

  	b1 = (__m256i) __builtin_ia32_permdi256 ((__v4di)b1, 0b10110001);
  	a0 = (__m256i) (__m256d) __builtin_ia32_blendpd256 ((__v4df)a1, (__v4df)b1, 0b1010);
  	b0 = (__m256i) (__m256d) __builtin_ia32_blendpd256 ((__v4df)a1, (__v4df)b1, 0b0101);
	b0 = _mm256_permute4x64_epi64(b0, 0b10110001);

	COEX64X4(a0,b0,a1,b1,t)

  	b1 = (__m256i) __builtin_ia32_permdi256 ((__v4di)b1, 0b01001110);
  	a0 = (__m256i) (__m256d) __builtin_ia32_blendpd256 ((__v4df)a1, (__v4df)b1, 0b1100);
  	b0 = (__m256i) (__m256d) __builtin_ia32_blendpd256 ((__v4df)a1, (__v4df)b1, 0b0011);

  	b0 = (__m256i) __builtin_ia32_permdi256 ((__v4di)b0, 0b01110010);
  	a0 = (__m256i) __builtin_ia32_permdi256 ((__v4di)a0, 0b11011000);
#endif

#undef COEX64X4
}

// optimized sorting network for two vectors, that is 16 ints
static inline void sortingnetwork_sort_u32x16(__m256i &v1,
													    __m256i &v2) noexcept {
#ifdef __clang__
#define SHUFFLE_2_VECS(a, b, mask)         \
	_mm256_castps_si256(_mm256_shuffle_ps( \
	        _mm256_castsi256_ps(a), _mm256_castsi256_ps(b), mask));

#define SHUFFLE_1_VEC(a, mask) _mm256_shuffle_epi32(a, mask)
#else
#define SHUFFLE_2_VECS(a, b, mask) (__m256i)(__builtin_ia32_shufps256((__v8sf)(a), (__v8sf)(b), (int)mask))
#define SHUFFLE_1_VEC(a, mask)  (__m256i)__builtin_ia32_pshufd256 ((__v8si)a,mask)
#endif
	__m256i tmp;
	COEX_u32x8(v1, v2, tmp);

	v2 = _mm256_shuffle_epi32(v2, _MM_SHUFFLE(2, 3, 0, 1));
	COEX_u32x8(v1, v2, tmp);

	tmp = v1;
	v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
	COEX_u32x8(v1, v2, tmp);

	v2 = SHUFFLE_1_VEC(v2, _MM_SHUFFLE(0, 1, 2, 3));
	COEX_u32x8(v1, v2, tmp);

	tmp = v1;
	v1 = SHUFFLE_2_VECS(v1, v2, 0b01000100);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b11101110);
	COEX_u32x8(v1, v2, tmp);

	tmp = v1;
	v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
	COEX_u32x8(v1, v2, tmp);

#ifdef __clang__
	v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));
#else
  	v2 = (__m256i) __builtin_ia32_permvarsi256 ((__v8si)v2,(__v8si){7,6,5,4,3,2,1,0});
#endif
	COEX_u32x8(v1, v2, tmp);

	tmp = v1;
	v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
	COEX_u32x8(v1, v2, tmp);

	tmp = v1;
	v1 = SHUFFLE_2_VECS(v1, v2, 0b11011000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b10001101);
	COEX_u32x8(v1, v2, tmp);

#ifdef __clang__
	v1 = _mm256_permutevar8x32_epi32(v1, _mm256_setr_epi32(0, 4, 1, 5, 6, 2, 7, 3));
	v2 = _mm256_permutevar8x32_epi32(v2, _mm256_setr_epi32(0, 4, 1, 5, 6, 2, 7, 3));
#else 
  	v1 = (__m256i) __builtin_ia32_permvarsi256 ((__v8si)v1,(__v8si){0, 4, 1, 5, 6, 2, 7, 3});
  	v2 = (__m256i) __builtin_ia32_permvarsi256 ((__v8si)v2,(__v8si){0, 4, 1, 5, 6, 2, 7, 3});
#endif

	tmp = v1;
	v1 = SHUFFLE_2_VECS(v1, v2, 0b10001000);
	v2 = SHUFFLE_2_VECS(tmp, v2, 0b11011101);
	COEX_u32x8(v1, v2, tmp);

	auto b2 = SHUFFLE_1_VEC(v2, 0b10110001);
	auto b1 = SHUFFLE_1_VEC(v1, 0b10110001);

#ifdef __clang__
	v1 = _mm256_blend_epi32(v1, b2, 0b10101010);
	v2 = _mm256_blend_epi32(b1, v2, 0b10101010);
#else
  	v1 = (__m256i) __builtin_ia32_pblendd256 ((__v8si)v1,(__v8si)b2, 0b10101010);
  	v2 = (__m256i) __builtin_ia32_pblendd256 ((__v8si)b1,(__v8si)v2, 0b10101010);
#endif

#undef SHUFFLE_2_VECS
#undef SHUFFLE_1_VEC
}


/// optimized sorting network for a single avx2 register
static inline __m256i sortingnetwork_sort_u32x8(__m256i &input) noexcept {
    __m256i perm_neigh_min,perm_neigh_max;
	{
		const __m256i perm_neigh = (__m256i)_mm256_permute_ps((__m256)input, _MM_SHUFFLE(2, 3, 0, 1));
		COEX_u32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max)
		input = (__m256i)_mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xAA);
	}
	{
		const __m256i perm_neigh = (__m256i)_mm256_permute_ps((__m256)input, _MM_SHUFFLE(0, 1, 2, 3));
		COEX_u32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max)
		input = (__m256i)_mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xCC);
	}
	{
		const __m256i perm_neigh = (__m256i)_mm256_permute_ps((__m256)input, _MM_SHUFFLE(2, 3, 0, 1));
		COEX_u32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max)
		input = (__m256i)_mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xAA);
	}
	{
		const __m256i swap = (__m256i)_mm256_permute2f128_ps((__m256)input, (__m256)input, _MM_SHUFFLE(0, 0, 1, 1));
		const __m256i perm_neigh = (__m256i)_mm256_permute_ps((__m256)swap, _MM_SHUFFLE(0, 1, 2, 3));
		COEX_u32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max)
		input = (__m256i)_mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xF0);
	}
	{
		const __m256i perm_neigh = (__m256i)_mm256_permute_ps((__m256)input, _MM_SHUFFLE(1, 0, 3, 2));
		COEX_u32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max)
		input = (__m256i)_mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xCC);
	}
	{
		const __m256i perm_neigh = (__m256i)_mm256_permute_ps((__m256)input, _MM_SHUFFLE(2, 3, 0, 1));
		COEX_u32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max)
		input = (__m256i)_mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xAA);
	}

	return input;
}

///
static inline __m256i sortingnetwork_aftermerge_u32x8(__m256i &a) noexcept {
    __m256i perm_neigh_min,perm_neigh_max;
	{
        __m256i swap = _mm256_permute2f128_si256(a, a, _MM_SHUFFLE(0, 0, 1, 1));

		COEX_u32x8_(a, swap, perm_neigh_min, perm_neigh_max)
        a = _mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xF0);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps((__m256)a, _MM_SHUFFLE(1, 0, 3, 2));
		COEX_u32x8_(a, (__m256i)perm_neigh, perm_neigh_min, perm_neigh_max)
        a = _mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xCC);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps((__m256)a, _MM_SHUFFLE(2, 3, 0, 1));
		COEX_u32x8_(a, (__m256i)perm_neigh, perm_neigh_min, perm_neigh_max)
        a = _mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xAA);
    }

    return a;
}

/// optimized sorting network for int32x8
static inline __m256i sortingnetwork_sort_i32x8(__m256i &input) noexcept {
    __m256i perm_neigh_min,perm_neigh_max;
	{
		const __m256i perm_neigh = (__m256i)_mm256_permute_ps((__m256)input, _MM_SHUFFLE(2, 3, 0, 1));
		COEX_i32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max)
		input = _mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xAA);
	}
	{
		const __m256i perm_neigh = (__m256i)_mm256_permute_ps((__m256)input, _MM_SHUFFLE(0, 1, 2, 3));
		COEX_i32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max)
		input = _mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xCC);
	}
	{
		const __m256i perm_neigh = (__m256i)_mm256_permute_ps((__m256)input, _MM_SHUFFLE(2, 3, 0, 1));
		COEX_i32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max)
		input = _mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xAA);
	}
	{
		const __m256i swap = (__m256i)_mm256_permute2f128_ps((__m256)input, (__m256)input, _MM_SHUFFLE(0, 0, 1, 1));
		const __m256i perm_neigh = (__m256i)_mm256_permute_ps((__m256)swap, _MM_SHUFFLE(0, 1, 2, 3));
		COEX_i32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max)
		input = _mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xF0);
	}
	{
		const __m256i perm_neigh = (__m256i)_mm256_permute_ps((__m256)input, _MM_SHUFFLE(1, 0, 3, 2));
		COEX_i32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max)
		input = _mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xCC);
	}
	{
		const __m256i perm_neigh = (__m256i)_mm256_permute_ps((__m256)input, _MM_SHUFFLE(2, 3, 0, 1));
		COEX_i32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max)
		input = _mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xAA);
	}

	return input;
}

///
static inline __m256i sortingnetwork_aftermerge_i32x8(__m256i &a) noexcept {
	__m256i perm_neigh_min, perm_neigh_max;
	{
		const __m256i swap = _mm256_permute2f128_si256(a, a, _MM_SHUFFLE(0, 0, 1, 1));
		COEX_i32x8_(a, swap, perm_neigh_min, perm_neigh_max)
		a = _mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xF0);
	}
	{
		const __m256 perm_neigh = _mm256_permute_ps((__m256)a, _MM_SHUFFLE(1, 0, 3, 2));
		COEX_i32x8_(a, (__m256i)perm_neigh, perm_neigh_min, perm_neigh_max)
		a = _mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xCC);
	}
	{
		const __m256 perm_neigh = _mm256_permute_ps((__m256)a, _MM_SHUFFLE(2, 3, 0, 1));
		perm_neigh_min = _mm256_min_epi32(a, (__m256i)perm_neigh);
		perm_neigh_max = _mm256_max_epi32(a, (__m256i)perm_neigh);
		a = _mm256_blend_epi32(perm_neigh_min, perm_neigh_max, 0xAA);
	}
	return a;
}

/// needed by `sort_u8x16`
constexpr static uint8_t layers[6][16] = {
        {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14},
        {3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12},
        {7,6,5,4,3,2,1,0,15,14,13,12,11,10,9,8},
        {2,3,0,1,6,7,4,5,10,11,8,9,14,15,12,13},
        {15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0},
        {4,5,6,7,0,1,2,3,12,13,14,15, 8,9,10,11},
};

/// needed by `sort_u8x16`
constexpr static int8_t blend[4][16] = {
        {0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1,0,-1},
        {0, 0,-1,-1,0,0,-1,-1,0,0,-1,-1,0,0,-1,-1},
        {0,0,0,0,-1,-1,-1,-1,0,0,0,0,-1,-1,-1,-1},
        {0,0,0,0,0,0,0,0, -1,-1,-1,-1,-1,-1,-1,-1},
};

/// sorts a single SSE register
__m128i sortingnetwork_sort_u8x16(__m128i v) noexcept {
    __m128i t = v, tmp;

    // Step 1
    t = _mm_shuffle_epi8(t, _mm_load_si128((__m128i *)layers[0]));
	COEX_u8x16(t, v, tmp);
    t = _mm_blendv_epi8(t, v, _mm_load_si128((__m128i *)blend[0]));

    // Step 2
    v = _mm_shuffle_epi8(t, _mm_load_si128((__m128i *)layers[1]));
	COEX_u8x16(v, t, tmp);
    v = _mm_blendv_epi8(v, t, _mm_load_si128((__m128i *)blend[1]));

    // Step 3
    t = _mm_shuffle_epi8(v, _mm_load_si128((__m128i *)layers[0]));
	COEX_u8x16(t, v, tmp);
    t = _mm_blendv_epi8(t, v, _mm_load_si128((__m128i *)blend[0]));

    // Step 4
    v = _mm_shuffle_epi8(t, _mm_load_si128((__m128i *)layers[2]));
	COEX_u8x16(v, t, tmp);
    v = _mm_blendv_epi8(v, t, _mm_load_si128((__m128i *)blend[2]));

    // Step 5
    t = _mm_shuffle_epi8(v, _mm_load_si128((__m128i *)layers[3]));
	COEX_u8x16(t, v, tmp);
    t = _mm_blendv_epi8(t, v, _mm_load_si128((__m128i *)blend[1]));

    // Step 6
    v = _mm_shuffle_epi8(t, _mm_load_si128((__m128i *)layers[0]));
	COEX_u8x16(v, t, tmp);
    v = _mm_blendv_epi8(v, t, _mm_load_si128((__m128i *)blend[0]));

    // Step 7
    t = _mm_shuffle_epi8(v, _mm_load_si128((__m128i *)layers[4]));
	COEX_u8x16(t, v, tmp);
    t = _mm_blendv_epi8(t, v, _mm_load_si128((__m128i *)blend[3]));

    // Step 8
    v = _mm_shuffle_epi8(t, _mm_load_si128((__m128i *)layers[5]));
	COEX_u8x16(v, t, tmp);
    v = _mm_blendv_epi8(v, t, _mm_load_si128((__m128i *)blend[2]));

    // Step 9
    t = _mm_shuffle_epi8(v, _mm_load_si128((__m128i *)layers[3]));
	COEX_u8x16(t, v, tmp);
    t = _mm_blendv_epi8(t, v, _mm_load_si128((__m128i *)blend[1]));

    // Step 10
    v = _mm_shuffle_epi8(t, _mm_load_si128((__m128i *)layers[0]));
	COEX_u8x16(v, t, tmp);
    v = _mm_blendv_epi8(v, t, _mm_load_si128((__m128i *)blend[0]));
    return v;
}

static inline void sortingnetwork_mergesort_u8x32(__m128i *a, __m128i *b) noexcept {
    __m128i H1 = _mm_shuffle_epi8(*b, _mm_load_si128((__m128i *)layers[4]));
    __m128i L1 = *a, tmp;

	COEX_u8x16(L1, H1, tmp);
    __m128i L1p = _mm_blendv_epi8(L1, _mm_bslli_si128(H1, 8), _mm_load_si128((__m128i *)blend[3]));
    __m128i H1p = _mm_blendv_epi8(_mm_bsrli_si128(L1, 8), H1, _mm_load_si128((__m128i *)blend[3]));

	COEX_u8x16(L1p, H1p, tmp);
    __m128i L2p = _mm_blendv_epi8(L1p, _mm_bslli_si128(H1p, 4), _mm_load_si128((__m128i *)blend[2]));
    __m128i H2p = _mm_blendv_epi8(_mm_bsrli_si128(L1p, 4), H1p, _mm_load_si128((__m128i *)blend[2]));

	COEX_u8x16(L2p, H2p, tmp);
    __m128i L3p = _mm_blendv_epi8(L2p, _mm_bslli_si128(H2p, 2), _mm_load_si128((__m128i *)blend[1]));
    __m128i H3p = _mm_blendv_epi8(_mm_bsrli_si128(L2p, 2), H2p, _mm_load_si128((__m128i *)blend[1]));

	COEX_u8x16(L3p, H3p, tmp);
    __m128i L4p = _mm_blendv_epi8(L3p, _mm_bslli_si128(H3p, 1), _mm_load_si128((__m128i *)blend[0]));
    __m128i H4p = _mm_blendv_epi8(_mm_bsrli_si128(L3p, 1), H3p, _mm_load_si128((__m128i *)blend[0]));

	COEX_u8x16(L4p, H4p, tmp);
    *a = _mm_unpacklo_epi8(L4p, H4p);
    *b = _mm_unpackhi_epi8(L4p, H4p);
}

static inline void sortingnetwork_sort_u8x32(__m128i *a,
						__m128i *b) noexcept{
    *a = sortingnetwork_sort_u8x16(*a);
    *b = sortingnetwork_sort_u8x16(*b);
    sortingnetwork_mergesort_u8x32(a, b);
}


/// floating point stuff
static inline __m256 sortingnetwork_sort_f32x8(__m256 &input) noexcept {
	__m256 perm_neigh_min, perm_neigh_max;
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(2, 3, 0, 1));
		COEX_f32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xAA);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(0, 1, 2, 3));
		COEX_f32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xCC);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(2, 3, 0, 1));
		COEX_f32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xAA);
    }
    {
        __m256 swap = _mm256_permute2f128_ps(input, input, _MM_SHUFFLE(0, 0, 1, 1));
        __m256 perm_neigh = _mm256_permute_ps(swap, _MM_SHUFFLE(0, 1, 2, 3));
		COEX_f32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xF0);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(1, 0, 3, 2));
		COEX_f32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xCC);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(input, _MM_SHUFFLE(2, 3, 0, 1));
		COEX_f32x8_(input, perm_neigh, perm_neigh_min, perm_neigh_max);
        input = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xAA);
    }

	return input;
}

static inline __m256 sortingnetwork_aftermerge_f32x8(__m256 &a) noexcept {
	__m256 perm_neigh_min, perm_neigh_max;
    {
        __m256 swap = _mm256_permute2f128_ps(a, a, _MM_SHUFFLE(0, 0, 1, 1));
		COEX_f32x8_(a, swap, perm_neigh_min, perm_neigh_max);
        a = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xF0);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(a, _MM_SHUFFLE(1, 0, 3, 2));
		COEX_f32x8_(a, perm_neigh, perm_neigh_min, perm_neigh_max);
        a = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xCC);
    }
    {
        __m256 perm_neigh = _mm256_permute_ps(a, _MM_SHUFFLE(2, 3, 0, 1));
		COEX_f32x8_(a, perm_neigh, perm_neigh_min, perm_neigh_max);
        a = _mm256_blend_ps(perm_neigh_min, perm_neigh_max, 0xAA);
    }

    return a;
}

/// creates a funtions which takes two `REG` variables, each of them storing 
/// a single `MULT` type, and sorts them to a single `NEW_MULT`
/// \NEW_MULT: ex: f32x16
/// \MULT: ex: f32x8
/// \REG: ex: __m256
#define sortingnetwork_aftermerge2(NEW_MULT, MULT, REG)						\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT(REG &a, REG &b) {  \
	REG tmp; 																\
    COEX_ ## MULT(a, b, tmp);                          						\
	a = sortingnetwork_aftermerge_ ## MULT (a); 							\
	b = sortingnetwork_aftermerge_ ## MULT (b); 							\
}


// REG swap = (REG)_mm256_permute2f128_ps((__m256)b, (__m256)b, _MM_SHUFFLE(0, 0, 1, 1));
// REG perm_neigh = (REG)_mm256_permute_ps((__m256)swap, _MM_SHUFFLE(0, 1, 2, 3));

#define sortingnetwork_permute_minmax2(NEW_MULT, REG, MIN_FKT, MAX_FKT )			\
static inline void sortingnetwork_permute_minmax_ ## NEW_MULT (REG &a, REG &b) noexcept { \
  	const REG swap = (REG) (__m256) __builtin_ia32_vperm2f128_ps256 ((__v8sf)b, (__v8sf)b, _MM_SHUFFLE(0, 0, 1, 1)); \
  	REG perm_neigh = (REG) (__m256) __builtin_ia32_vpermilps256 ((__v8sf)swap, _MM_SHUFFLE(0, 1, 2, 3)); 	\
	REG perm_neigh_min = MIN_FKT(a, perm_neigh);									\
	b = MAX_FKT(a, perm_neigh);														\
	a = perm_neigh_min;																\
}

#define sortingnetwork_merge_sorted2(NEW_MULT,MULT1,REG)	\
static inline void sortingnetwork_merge_sorted_ ## NEW_MULT (REG &a, REG &b) noexcept {	\
	sortingnetwork_permute_minmax_ ## NEW_MULT (a, b); 		\
	a = sortingnetwork_aftermerge_ ## MULT1 (a); 			\
	b = sortingnetwork_aftermerge_ ## MULT1 (b); 			\
}

#define sortingnetwork_sort2(NEW_MULT,MULT1,REG)		\
static inline void sortingnetwork_sort_ ## NEW_MULT (REG &a, REG &b) noexcept { 	\
	a = sortingnetwork_sort_ ## MULT1 (a); 				\
	b = sortingnetwork_sort_ ## MULT1 (b); 				\
	sortingnetwork_merge_sorted_ ## NEW_MULT(a, b); 	\
}


#define sortingnetwork_merge_sorted3(NEW_MULT,MULT2,MULT1,REG)	\
static inline void sortingnetwork_merge_sorted_ ## NEW_MULT (REG &a, REG &b, REG &c) noexcept {\
 	REG tmp;                                                    \
	sortingnetwork_permute_minmax_ ## MULT2(b, c); 				\
	COEX_ ## MULT1(a, b, tmp);									\
	a = sortingnetwork_aftermerge_ ## MULT1 (a); 				\
	b = sortingnetwork_aftermerge_ ## MULT1 (b); 				\
	c = sortingnetwork_aftermerge_ ## MULT1 (c); 				\
}

#define sortingnetwork_aftermerge_sorted3(NEW_MULT,MULT1,REG)	\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT (REG &a, REG &b, REG &c) { \
	REG tmp;                                                    \
	COEX_ ## MULT1 (a, c, tmp); 								\
	COEX_ ## MULT1 (a, b, tmp); 								\
	a = sortingnetwork_aftermerge_ ## MULT1 (a); 				\
	b = sortingnetwork_aftermerge_ ## MULT1 (b); 				\
	c = sortingnetwork_aftermerge_ ## MULT1 (c); 				\
}


#define sortingnetwork_sort3(NEW_MULT,DOUBLE_MULT,MULT1,REG)	\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c) noexcept {\
	sortingnetwork_sort_ ## DOUBLE_MULT(a, b); 					\
	c = sortingnetwork_sort_ ## MULT1 (c); 						\
	sortingnetwork_merge_sorted_ ## NEW_MULT (a, b, c);			\
}


#define sortingnetwork_merge_sorted4(NEW_MULT,DOUBLE_MULT,SINGLE_MULT,REG)	\
static inline void sortingnetwork_merge_sorted ## NEW_MULT (REG &a, REG &b, REG &c, REG &d) noexcept {\
	REG tmp;                                                \
	sortingnetwork_permute_minmax_ ## DOUBLE_MULT (a, d); 	\
	sortingnetwork_permute_minmax_ ## DOUBLE_MULT (b, c); 	\
	COEX_ ## SINGLE_MULT(a, b, tmp); 						\
	COEX_ ## SINGLE_MULT(c, d, tmp); 						\
	a = sortingnetwork_aftermerge_ ## SINGLE_MULT(a); 		\
	b = sortingnetwork_aftermerge_ ## SINGLE_MULT(b); 		\
	c = sortingnetwork_aftermerge_ ## SINGLE_MULT(c); 		\
	d = sortingnetwork_aftermerge_ ## SINGLE_MULT(d); 		\
}

#define sortingnetwork_aftermerge_sorted4(NEW_MULT,MULT1,REG)\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d) {\
	REG tmp;                                                \
	COEX_ ## MULT1 (a, c, tmp); 							\
	COEX_ ## MULT1 (b, d, tmp); 							\
	COEX_ ## MULT1 (a, b, tmp); 							\
	COEX_ ## MULT1 (c, d, tmp); 							\
	a = sortingnetwork_aftermerge_ ## MULT1 (a); 			\
	b = sortingnetwork_aftermerge_ ## MULT1 (b); 			\
	c = sortingnetwork_aftermerge_ ## MULT1 (c); 			\
	d = sortingnetwork_aftermerge_ ## MULT1 (d); 			\
}

#define sortingnetwork_aftermerge_sorted5(NEW_MULT,MULT1,REG)\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e) { \
	REG tmp;                                                \
	COEX_ ## MULT1 (a, e, tmp); 							\
	COEX_ ## MULT1 (a, c, tmp); 							\
	COEX_ ## MULT1 (b, d, tmp); 							\
	COEX_ ## MULT1 (a, b, tmp); 							\
	COEX_ ## MULT1 (c, d, tmp); 							\
	a = sortingnetwork_aftermerge_ ## MULT1 (a); 			\
	b = sortingnetwork_aftermerge_ ## MULT1 (b); 			\
	c = sortingnetwork_aftermerge_ ## MULT1 (c); 			\
	d = sortingnetwork_aftermerge_ ## MULT1 (d); 			\
	e = sortingnetwork_aftermerge_ ## MULT1 (e); 			\
}

#define sortingnetwork_aftermerge_sorted6(NEW_MULT,MULT1,REG)\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f){\
	REG tmp;                                                \
	COEX_ ## MULT1 (a, e, tmp); 							\
	COEX_ ## MULT1 (b, f, tmp); 							\
	COEX_ ## MULT1 (a, c, tmp); 							\
	COEX_ ## MULT1 (b, d, tmp); 							\
	COEX_ ## MULT1 (a, b, tmp); 							\
	COEX_ ## MULT1 (c, d, tmp); 							\
	COEX_ ## MULT1 (e, f, tmp); 							\
	a = sortingnetwork_aftermerge_ ## MULT1 (a); 			\
	b = sortingnetwork_aftermerge_ ## MULT1 (b); 			\
	c = sortingnetwork_aftermerge_ ## MULT1 (c); 			\
	d = sortingnetwork_aftermerge_ ## MULT1 (d); 			\
	e = sortingnetwork_aftermerge_ ## MULT1 (e); 			\
	f = sortingnetwork_aftermerge_ ## MULT1 (f); 			\
}

#define sortingnetwork_aftermerge_sorted7(NEW_MULT,MULT1,REG)\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT (REG &a, REG &b, REG &c, REG &d, REG& e, REG& f, REG &g) { \
	REG tmp;                                                \
	COEX_ ## MULT1 (a, e, tmp); 							\
	COEX_ ## MULT1 (b, f, tmp); 							\
	COEX_ ## MULT1 (c, g, tmp); 							\
	COEX_ ## MULT1 (a, c, tmp); 							\
	COEX_ ## MULT1 (b, d, tmp); 							\
	COEX_ ## MULT1 (a, b, tmp); 							\
	COEX_ ## MULT1 (c, d, tmp); 							\
	COEX_ ## MULT1 (e, g, tmp); 							\
	COEX_ ## MULT1 (e, f, tmp); 							\
	a = sortingnetwork_aftermerge_ ## MULT1(a); 			\
	b = sortingnetwork_aftermerge_ ## MULT1(b); 			\
	c = sortingnetwork_aftermerge_ ## MULT1(c); 			\
	d = sortingnetwork_aftermerge_ ## MULT1(d); 			\
	e = sortingnetwork_aftermerge_ ## MULT1(e); 			\
	f = sortingnetwork_aftermerge_ ## MULT1(f); 			\
	g = sortingnetwork_aftermerge_ ## MULT1(g); 			\
}



#define sortingnetwork_sort4(NEW_MULT,DOUBLE_MULT,SINGLE_MULT,REG)\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d) noexcept { \
	sortingnetwork_sort_ ## DOUBLE_MULT(a, b); 				\
	sortingnetwork_sort_ ## DOUBLE_MULT(c, d); 				\
	sortingnetwork_merge_sorted ## NEW_MULT (a, b, c, d); 	\
}

#define sortingnetwork_sort5(NEW_MULT,MULT4,MULT2,MULT1,REG)\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e) { \
	REG tmp;                                                \
	sortingnetwork_sort_ ## MULT4(a, b, c, d); 				\
	e = sortingnetwork_sort_ ## MULT1(e); 					\
	sortingnetwork_permute_minmax_ ## MULT2(d, e);			\
	COEX_ ## MULT1(a, c, tmp); 								\
	COEX_ ## MULT1(b, d, tmp); 								\
	COEX_ ## MULT1(a, b, tmp); 								\
	COEX_ ## MULT1(c, d, tmp); 								\
	a = sortingnetwork_aftermerge_ ## MULT1(a); 			\
	b = sortingnetwork_aftermerge_ ## MULT1(b); 			\
	c = sortingnetwork_aftermerge_ ## MULT1(c); 			\
	d = sortingnetwork_aftermerge_ ## MULT1(d); 			\
	e = sortingnetwork_aftermerge_ ## MULT1(e); 			\
}

#define sortingnetwork_sort6(NEW_MULT,MULT4,MULT2,MULT1,REG)\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f){ \
	REG tmp;                                        \
	sortingnetwork_sort_ ## MULT4(a, b, c, d); 		\
	sortingnetwork_sort_ ## MULT2(e, f);			\
	sortingnetwork_permute_minmax_ ## MULT2(c, f); 	\
	sortingnetwork_permute_minmax_ ## MULT2(d, e); 	\
	COEX_ ## MULT1(a, c, tmp); 						\
	COEX_ ## MULT1(b, d, tmp); 						\
	COEX_ ## MULT1(a, b, tmp); 						\
	COEX_ ## MULT1(c, d, tmp); 						\
	COEX_ ## MULT1(e, f, tmp); 						\
	a = sortingnetwork_aftermerge_ ## MULT1(a);		\
	b = sortingnetwork_aftermerge_ ## MULT1(b);		\
	c = sortingnetwork_aftermerge_ ## MULT1(c);		\
	d = sortingnetwork_aftermerge_ ## MULT1(d);		\
	e = sortingnetwork_aftermerge_ ## MULT1(e);		\
	f = sortingnetwork_aftermerge_ ## MULT1(f);		\
}

#define sortingnetwork_sort7(NEW_MULT,MULT4,MULT3,MULT2,MULT1,REG)\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g){ \
	REG tmp;                                        \
	sortingnetwork_sort_ ## MULT4(a, b, c, d); 		\
	sortingnetwork_sort_ ## MULT3(e, f, g); 		\
	sortingnetwork_permute_minmax_ ## MULT2(c, f); 	\
	sortingnetwork_permute_minmax_ ## MULT2(d, e); 	\
	sortingnetwork_permute_minmax_ ## MULT2(b, g); 	\
	COEX_ ## MULT1(a, c, tmp); 						\
	COEX_ ## MULT1(b, d, tmp); 						\
	COEX_ ## MULT1(a, b, tmp); 						\
	COEX_ ## MULT1(c, d, tmp); 						\
	COEX_ ## MULT1(e, g, tmp); 						\
	COEX_ ## MULT1(e, f, tmp); 						\
	a = sortingnetwork_aftermerge_ ## MULT1(a);		\
	b = sortingnetwork_aftermerge_ ## MULT1(b);		\
	c = sortingnetwork_aftermerge_ ## MULT1(c);		\
	d = sortingnetwork_aftermerge_ ## MULT1(d);		\
	e = sortingnetwork_aftermerge_ ## MULT1(e);		\
	f = sortingnetwork_aftermerge_ ## MULT1(f);		\
	g = sortingnetwork_aftermerge_ ## MULT1(g);		\
}

#define sortingnetwork_aftermerge8(NEW_MULT,MULT2,MULT1,REG)\
static inline void sortingnetwork_aftermerge_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h){ \
	REG tmp;                                    \
	COEX_ ## MULT1(a, e, tmp); 					\
	COEX_ ## MULT1(b, f, tmp); 					\
	COEX_ ## MULT1(c, g, tmp); 					\
	COEX_ ## MULT1(d, h, tmp); 					\
	COEX_ ## MULT1(a, c, tmp); 					\
	COEX_ ## MULT1(b, d, tmp); 					\
	COEX_ ## MULT1(a, b, tmp); 					\
	COEX_ ## MULT1(c, d, tmp); 					\
	COEX_ ## MULT1(e, g, tmp); 					\
	COEX_ ## MULT1(f, h, tmp); 					\
	COEX_ ## MULT1(e, f, tmp); 					\
	COEX_ ## MULT1(g, h, tmp); 					\
	a = sortingnetwork_aftermerge_ ## MULT1(a); \
	b = sortingnetwork_aftermerge_ ## MULT1(b); \
	c = sortingnetwork_aftermerge_ ## MULT1(c); \
	d = sortingnetwork_aftermerge_ ## MULT1(d); \
	e = sortingnetwork_aftermerge_ ## MULT1(e); \
	f = sortingnetwork_aftermerge_ ## MULT1(f); \
	g = sortingnetwork_aftermerge_ ## MULT1(g); \
	h = sortingnetwork_aftermerge_ ## MULT1(h); \
}

#define sortingnetwork_sort8(NEW_MULT,MULT4,MULT2,MULT1,REG)\
static inline void sortingnetwork_sort_ ## NEW_MULT (REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h) { \
	REG tmp;                                        \
	sortingnetwork_sort_ ## MULT4(a, b, c, d); 		\
	sortingnetwork_sort_ ## MULT4(e, f, g, h); 		\
	sortingnetwork_permute_minmax_ ## MULT2(a, h); 	\
	sortingnetwork_permute_minmax_ ## MULT2(b, g); 	\
	sortingnetwork_permute_minmax_ ## MULT2(c, f); 	\
	sortingnetwork_permute_minmax_ ## MULT2(d, e); 	\
	COEX_ ## MULT1(a, c, tmp); 						\
	COEX_ ## MULT1(b, d, tmp); 						\
	COEX_ ## MULT1(a, b, tmp); 						\
	COEX_ ## MULT1(c, d, tmp); 						\
	COEX_ ## MULT1(e, g, tmp); 						\
	COEX_ ## MULT1(f, h, tmp); 						\
	COEX_ ## MULT1(e, f, tmp); 						\
	COEX_ ## MULT1(g, h, tmp); 						\
	a = sortingnetwork_aftermerge_ ## MULT1(a);		\
	b = sortingnetwork_aftermerge_ ## MULT1(b);		\
	c = sortingnetwork_aftermerge_ ## MULT1(c);		\
	d = sortingnetwork_aftermerge_ ## MULT1(d);		\
	e = sortingnetwork_aftermerge_ ## MULT1(e);		\
	f = sortingnetwork_aftermerge_ ## MULT1(f);		\
	g = sortingnetwork_aftermerge_ ## MULT1(g);		\
	h = sortingnetwork_aftermerge_ ## MULT1(h);		\
}

#define sortingnetwork_sort9(NEW_MULT,MULT8,MULT2,MULT1,REG)		\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i) noexcept { \
	sortingnetwork_sort_ ## MULT8 (a, b, c, d, e, f, g, h); 		\
	i = sortingnetwork_sort_ ## MULT1(i); 							\
	sortingnetwork_permute_minmax_ ## MULT2(h, i);					\
	sortingnetwork_aftermerge_ ## MULT8(a, b, c, d, e, f, g, h);	\
	i = sortingnetwork_aftermerge_ ## MULT1(i); 					\
}

#define sortingnetwork_sort10(NEW_MULT,MULT8,MULT2,REG)				\
static inline void sortingnetwork_sort_ ## NEW_MULT (REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j) { \
	sortingnetwork_sort_ ## MULT8(a, b, c, d, e, f, g, h); 			\
	sortingnetwork_sort_ ## MULT2(i, j); 							\
	sortingnetwork_permute_minmax_ ## MULT2(g, j); 					\
	sortingnetwork_permute_minmax_ ## MULT2(h, i); 					\
	sortingnetwork_aftermerge_ ## MULT8(a, b, c, d, e, f, g, h); 	\
	sortingnetwork_aftermerge_ ## MULT2(i, j); 						\
}

#define sortingnetwork_sort11(NEW_MULT,MULT8,MULT3,MULT2,REG)		\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j, REG &k) { \
	sortingnetwork_sort_ ## MULT8 (a, b, c, d, e, f, g, h); 		\
	sortingnetwork_sort_ ## MULT3 (i, j, k); 						\
	sortingnetwork_permute_minmax_ ## MULT2(f, k); 					\
	sortingnetwork_permute_minmax_ ## MULT2(g, j); 					\
	sortingnetwork_permute_minmax_ ## MULT2(h, i); 					\
	sortingnetwork_aftermerge_ ## MULT8 (a, b, c, d, e, f, g, h); 	\
	sortingnetwork_aftermerge_ ## MULT3 (i, j, k); 					\
}

#define sortingnetwork_sort12(NEW_MULT,MULT8,MULT4,MULT2,REG)		\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG   &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j, REG &k, REG &l) { \
	sortingnetwork_sort_ ## MULT8(a, b, c, d, e, f, g, h); 			\
	sortingnetwork_sort_ ## MULT4(i, j, k, l); 						\
	sortingnetwork_permute_minmax_ ## MULT2(e, l); 					\
	sortingnetwork_permute_minmax_ ## MULT2(f, k); 					\
	sortingnetwork_permute_minmax_ ## MULT2(g, j); 					\
	sortingnetwork_permute_minmax_ ## MULT2(h, i); 					\
	sortingnetwork_aftermerge_ ## MULT8(a, b, c, d, e, f, g, h); 	\
	sortingnetwork_aftermerge_ ## MULT4(i, j, k, l); 				\
}

#define sortingnetwork_sort13(NEW_MULT,MULT8,MULT5,MULT2,REG)	\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j, REG &k, REG &l, REG &m) { \
	sortingnetwork_sort_ ## MULT8(a, b, c, d, e, f, g, h); 		\
	sortingnetwork_sort_ ## MULT5(i, j, k, l, m); 				\
	sortingnetwork_permute_minmax_ ## MULT2(d, m); 				\
	sortingnetwork_permute_minmax_ ## MULT2(e, l); 				\
	sortingnetwork_permute_minmax_ ## MULT2(f, k); 				\
	sortingnetwork_permute_minmax_ ## MULT2(g, j); 				\
	sortingnetwork_permute_minmax_ ## MULT2(h, i); 				\
	sortingnetwork_aftermerge_ ## MULT8(a, b, c, d, e, f, g, h);\
	sortingnetwork_aftermerge_ ## MULT5(i, j, k, l, m); 		\
}

#define sortingnetwork_sort14(NEW_MULT,MULT8,MULT6,MULT2,REG)	\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j, REG &k, REG &l, REG &m, REG &n) { \
	sortingnetwork_sort_ ## MULT8 (a, b, c, d, e, f, g, h); 	\
	sortingnetwork_sort_ ## MULT6 (i, j, k, l, m, n); 			\
	sortingnetwork_permute_minmax_ ## MULT2(c, n); 				\
	sortingnetwork_permute_minmax_ ## MULT2(d, m); 				\
	sortingnetwork_permute_minmax_ ## MULT2(e, l); 				\
	sortingnetwork_permute_minmax_ ## MULT2(f, k); 				\
	sortingnetwork_permute_minmax_ ## MULT2(g, j); 				\
	sortingnetwork_permute_minmax_ ## MULT2(h, i); 				\
	sortingnetwork_aftermerge_ ## MULT8(a, b, c, d, e, f, g, h);\
	sortingnetwork_aftermerge_ ## MULT6 (i, j, k, l, m, n); 	\
}

#define sortingnetwork_sort15(NEW_MULT,MULT8,MULT7,MULT2,REG)	\
static inline void sortingnetwork_sort_ ## NEW_MULT(REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j, REG &k, REG &l, REG &m, REG &n, REG &o){ \
	sortingnetwork_sort_ ## MULT8(a, b, c, d, e, f, g, h); 		\
	sortingnetwork_sort_ ## MULT7(i, j, k, l, m, n, o); 		\
	sortingnetwork_permute_minmax_ ## MULT2(b, o); 				\
	sortingnetwork_permute_minmax_ ## MULT2(c, n); 				\
	sortingnetwork_permute_minmax_ ## MULT2(d, m); 				\
	sortingnetwork_permute_minmax_ ## MULT2(e, l); 				\
	sortingnetwork_permute_minmax_ ## MULT2(f, k); 				\
	sortingnetwork_permute_minmax_ ## MULT2(g, j); 				\
	sortingnetwork_permute_minmax_ ## MULT2(h, i); 				\
	sortingnetwork_aftermerge_ ## MULT8(a, b, c, d, e, f, g, h);\
	sortingnetwork_aftermerge_ ## MULT7(i, j, k, l, m, n, o);	\
}

#define sortingnetwork_sort16(NEW_MULT,MULT8,MULT2,REG)	\
static inline void sortingnetwork_sort_ ## NEW_MULT (REG &a, REG &b, REG &c, REG &d, REG &e, REG &f, REG &g, REG &h, REG &i, REG &j, REG &k, REG &l, REG &m, REG &n, REG &o, REG &p){ \
	sortingnetwork_sort_ ## MULT8(a, b, c, d, e, f, g, h); \
	sortingnetwork_sort_ ## MULT8(i, j, k, l, m, n, o, p); \
	sortingnetwork_permute_minmax_ ## MULT2(a, p); \
	sortingnetwork_permute_minmax_ ## MULT2(b, o); \
	sortingnetwork_permute_minmax_ ## MULT2(c, n); \
	sortingnetwork_permute_minmax_ ## MULT2(d, m); \
	sortingnetwork_permute_minmax_ ## MULT2(e, l); \
	sortingnetwork_permute_minmax_ ## MULT2(f, k); \
	sortingnetwork_permute_minmax_ ## MULT2(g, j); \
	sortingnetwork_permute_minmax_ ## MULT2(h, i); \
	sortingnetwork_aftermerge_ ## MULT8 (a, b, c, d, e, f, g, h); \
	sortingnetwork_aftermerge_ ## MULT8 (i, j, k, l, m, n, o, p); \
}

// TODO rename?
sortingnetwork_aftermerge2(f32x16, f32x8, __m256)
sortingnetwork_aftermerge2(u32x16, u32x8, __m256i)
sortingnetwork_aftermerge2(i32x16, i32x8, __m256i)
sortingnetwork_permute_minmax2(f32x16, __m256, _mm256_min_ps, _mm256_max_ps)
sortingnetwork_permute_minmax2(u32x16, __m256i, _mm256_min_epu32, _mm256_max_epu32)
sortingnetwork_permute_minmax2(i32x16, __m256i, _mm256_min_epi32, _mm256_max_epi32)
sortingnetwork_merge_sorted2(f32x16, f32x8, __m256)
sortingnetwork_merge_sorted2(u32x16, u32x8, __m256i)
sortingnetwork_merge_sorted2(i32x16, i32x8, __m256i)
sortingnetwork_sort2(f32x16, f32x8, __m256)
sortingnetwork_sort2(i32x16, i32x8, __m256i)
// sortingnetwork_sort2(u32x16, u32x8, __m256i) // already defined just faster? todo test
sortingnetwork_merge_sorted3(f32x24,f32x16,f32x8,__m256)
sortingnetwork_merge_sorted3(u32x24,u32x16,u32x8,__m256i)
sortingnetwork_merge_sorted3(i32x24,i32x16,i32x8,__m256i)
sortingnetwork_aftermerge_sorted3(f32x24,f32x8,__m256)
sortingnetwork_aftermerge_sorted3(u32x24,u32x8,__m256i)
sortingnetwork_aftermerge_sorted3(i32x24,i32x8,__m256i)
sortingnetwork_sort3(f32x24,f32x16,f32x8,__m256)
sortingnetwork_sort3(u32x24,u32x16,u32x8,__m256i)
sortingnetwork_sort3(i32x24,i32x16,i32x8,__m256i)
sortingnetwork_merge_sorted4(f32x32,f32x16,f32x8,__m256)
sortingnetwork_merge_sorted4(u32x32,u32x16,u32x8,__m256i)
sortingnetwork_merge_sorted4(i32x32,i32x16,i32x8,__m256i)
sortingnetwork_aftermerge_sorted4(f32x32,f32x8,__m256)
sortingnetwork_aftermerge_sorted4(u32x32,u32x8,__m256i)
sortingnetwork_aftermerge_sorted4(i32x32,i32x8,__m256i)
sortingnetwork_sort4(f32x32,f32x16,f32x8,__m256)
sortingnetwork_sort4(u32x32,u32x16,u32x8,__m256i)
sortingnetwork_sort4(i32x32,i32x16,i32x8,__m256i)
sortingnetwork_aftermerge_sorted5(f32x40,f32x8,__m256)
sortingnetwork_aftermerge_sorted5(u32x40,u32x8,__m256i)
sortingnetwork_aftermerge_sorted5(i32x40,i32x8,__m256i)
sortingnetwork_sort5(f32x40,f32x32,f32x16,f32x8,__m256)
sortingnetwork_sort5(u32x40,u32x32,u32x16,u32x8,__m256i)
sortingnetwork_sort5(i32x40,i32x32,i32x16,u32x8,__m256i)
sortingnetwork_aftermerge_sorted6(f32x48,f32x8,__m256)
sortingnetwork_aftermerge_sorted6(u32x48,u32x8,__m256i)
sortingnetwork_aftermerge_sorted6(i32x48,i32x8,__m256i)
sortingnetwork_sort6(f32x48,f32x32,f32x16,f32x8,__m256)
sortingnetwork_sort6(u32x48,u32x32,u32x16,u32x8,__m256i)
sortingnetwork_sort6(i32x48,i32x32,i32x16,i32x8,__m256i)
sortingnetwork_aftermerge_sorted7(f32x56,f32x8,__m256)
sortingnetwork_aftermerge_sorted7(u32x56,u32x8,__m256i)
sortingnetwork_aftermerge_sorted7(i32x56,i32x8,__m256i)
sortingnetwork_sort7(f32x56,f32x32,f32x24,f32x16,f32x8,__m256)
sortingnetwork_sort7(u32x56,u32x32,u32x24,u32x16,u32x8,__m256i)
sortingnetwork_sort7(i32x56,i32x32,i32x24,i32x16,u32x8,__m256i)
sortingnetwork_aftermerge8(f32x64,f32x16,f32x8,__m256)
sortingnetwork_aftermerge8(u32x64,u32x16,u32x8,__m256i)
sortingnetwork_aftermerge8(i32x64,i32x16,i32x8,__m256i)
sortingnetwork_sort8(f32x64,f32x32,f32x16,f32x8,__m256)
sortingnetwork_sort8(u32x64,u32x32,u32x16,u32x8,__m256i)
sortingnetwork_sort8(i32x64,i32x32,i32x16,i32x8,__m256i)
sortingnetwork_sort9(f32x72,f32x64,f32x16,f32x8,__m256)
sortingnetwork_sort9(u32x72,u32x64,u32x16,u32x8,__m256i)
sortingnetwork_sort9(i32x72,i32x64,i32x16,i32x8,__m256i)
sortingnetwork_sort10(f32x80,f32x64,f32x16,__m256)
sortingnetwork_sort10(u32x80,u32x64,u32x16,__m256i)
sortingnetwork_sort10(i32x80,i32x64,i32x16,__m256i)
sortingnetwork_sort11(f32x88,f32x64,f32x24,f32x16,__m256)
sortingnetwork_sort11(u32x88,u32x64,u32x24,u32x16,__m256i)
sortingnetwork_sort11(i32x88,i32x64,i32x24,i32x16,__m256i)
sortingnetwork_sort12(f32x96,f32x64,f32x32,f32x16,__m256)
sortingnetwork_sort12(u32x96,u32x64,u32x32,u32x16,__m256i)
sortingnetwork_sort12(i32x96,i32x64,i32x32,i32x16,__m256i)
sortingnetwork_sort13(f32x104,f32x64,f32x40,f32x16,__m256)
sortingnetwork_sort13(u32x104,u32x64,u32x40,u32x16,__m256i)
sortingnetwork_sort13(i32x104,i32x64,i32x40,i32x16,__m256i)
sortingnetwork_sort14(f32x112,f32x64,f32x48,f32x16,__m256)
sortingnetwork_sort14(u32x112,u32x64,u32x48,u32x16,__m256i)
sortingnetwork_sort14(i32x112,i32x64,i32x48,i32x16,__m256i)
sortingnetwork_sort15(f32x120,f32x64,f32x56,f32x16,__m256)
sortingnetwork_sort15(u32x120,u32x64,u32x56,u32x16,__m256i)
sortingnetwork_sort15(i32x120,i32x64,i32x56,i32x16,__m256i)
sortingnetwork_sort16(f32x128,f32x64,f32x16,__m256)
sortingnetwork_sort16(u32x128,u32x64,u32x16,__m256i)
sortingnetwork_sort16(i32x128,i32x64,i32x16,__m256i)

/// can only sort up to 16*8 elements
/// sorts the floating point data in `array`
/// \param array base pointer to the data
/// \param element_count
/// \return true on success.
/// 		false on faile
[[nodiscard]] static bool sortingnetwork_small_f32(float* array,
									  const size_t element_count) noexcept {
	if (element_count <= 1) {
		return true;
	}

	const uint32_t full_vec_count = element_count / 8;
	const uint32_t last_vec_size = element_count - (full_vec_count * 8);
	const uint32_t last_vec_flag = last_vec_size > 0;
	if (full_vec_count > 16) {
		// too many values
		return false;
	}
	

	__m256 d[16];
	for(uint32_t i=0; i<full_vec_count; ++i) {
		d[i] = _mm256_loadu_ps(array + 8*i);
	}

	if (last_vec_size) {
		d[full_vec_count] = avx2_load_f32x8(array, full_vec_count, last_vec_size);
	}

#ifdef __clang__
	switch (full_vec_count+last_vec_flag) {
		case 1 : sortingnetwork_sort_f32x8  (d[0]); goto cleanup;
		case 2 : sortingnetwork_sort_f32x16 (d[0],d[1]); goto cleanup;
		case 3 : sortingnetwork_sort_f32x24 (d[0],d[1],d[2]); goto cleanup;
		case 4 : sortingnetwork_sort_f32x32 (d[0],d[1],d[2],d[3]); goto cleanup;
		case 5 : sortingnetwork_sort_f32x40 (d[0],d[1],d[2],d[3],d[4]); goto cleanup;
		case 6 : sortingnetwork_sort_f32x48 (d[0],d[1],d[2],d[3],d[4],d[5]); goto cleanup;
		case 7 : sortingnetwork_sort_f32x56 (d[0],d[1],d[2],d[3],d[4],d[5],d[6]); goto cleanup;
		case 8 : sortingnetwork_sort_f32x64 (d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]); goto cleanup;
		case 9 : sortingnetwork_sort_f32x72 (d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8]); goto cleanup;
		case 10: sortingnetwork_sort_f32x80 (d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9]); goto cleanup;
		case 11: sortingnetwork_sort_f32x88 (d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10]); goto cleanup;
		case 12: sortingnetwork_sort_f32x96 (d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11]); goto cleanup;
		case 13: sortingnetwork_sort_f32x104(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12]); goto cleanup;
		case 14: sortingnetwork_sort_f32x112(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13]); goto cleanup;
		case 15: sortingnetwork_sort_f32x120(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14]); goto cleanup;
		case 16: sortingnetwork_sort_f32x128(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15]); goto cleanup;
		default:
			return false;
	}
#else
	void *t[] = {
	    &&t1,  &&t2,  &&t2,  &&t3,
		&&t4,  &&t5,  &&t6,  &&t7,
		&&t8,  &&t9,  &&t10, &&t11,
		&&t12, &&t13, &&t14, &&t15, 
		&&t16
	};

	goto *t[full_vec_count+last_vec_flag - 1];
	t1 : sortingnetwork_sort_f32x8  (d[0]); goto cleanup;
	t2 : sortingnetwork_sort_f32x16 (d[0],d[1]); goto cleanup;
	t3 : sortingnetwork_sort_f32x24 (d[0],d[1],d[2]); goto cleanup;
	t4 : sortingnetwork_sort_f32x32 (d[0],d[1],d[2],d[3]); goto cleanup;
	t5 : sortingnetwork_sort_f32x40 (d[0],d[1],d[2],d[3],d[4]); goto cleanup;
	t6 : sortingnetwork_sort_f32x48 (d[0],d[1],d[2],d[3],d[4],d[5]); goto cleanup;
	t7 : sortingnetwork_sort_f32x56 (d[0],d[1],d[2],d[3],d[4],d[5],d[6]); goto cleanup;
	t8 : sortingnetwork_sort_f32x64 (d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]); goto cleanup;
	t9 : sortingnetwork_sort_f32x72 (d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8]); goto cleanup;
	t10: sortingnetwork_sort_f32x80 (d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9]); goto cleanup;
	t11: sortingnetwork_sort_f32x88 (d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10]); goto cleanup;
	t12: sortingnetwork_sort_f32x96 (d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11]); goto cleanup;
	t13: sortingnetwork_sort_f32x104(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12]); goto cleanup;
	t14: sortingnetwork_sort_f32x112(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13]); goto cleanup;
	t15: sortingnetwork_sort_f32x120(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14]); goto cleanup;
	t16: sortingnetwork_sort_f32x128(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15]); goto cleanup;
#endif

	cleanup:
    for(uint32_t i=0; i<full_vec_count; ++i) {
		_mm256_storeu_ps(array + 8*i, d[i]);
	}

    if (last_vec_size) {
		avx2_store_f32x8(array, d[full_vec_count], full_vec_count, last_vec_size);
	}

	return true;
}

/* merge columns without transposition */
#define ASC(a, b, c, d, e, f, g, h)                                    \
  (((h < 7) << 7) | ((g < 6) << 6) | ((f < 5) << 5) | ((e < 4) << 4) | \
      ((d < 3) << 3) | ((c < 2) << 2) | ((b < 1) << 1) | (a < 0))

#define REVERSE_VEC(vec){                                              \
    vec = _mm256_permutevar8x32_epi32(                                 \
        vec, _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0));}

#define COEX_SHUFFLE(vec, a, b, c, d, e, f, g, h, MASK){               \
    int shuffle_mask = _MM_SHUFFLE(d, c, b, a);              \
    __m256i shuffled = _mm256_shuffle_epi32(vec, shuffle_mask);        \
    __m256i min = _mm256_min_epi32(shuffled, vec);                     \
    __m256i max = _mm256_max_epi32(shuffled, vec);                     \
    int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask);}

#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h, MASK){               \
    __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);  \
    __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask); \
    __m256i min = _mm256_min_epi32(permuted, vec);                     \
    __m256i max = _mm256_max_epi32(permuted, vec);                     \
    int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask);}

/* sort 8 columns, each containing 16 int, with Green's 60 modules network */
#define SORT_16_INT_COLUMNS_WISE(S)							\
inline                         								\
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
void inline merge_8_columns_with_16_elements_ ##S (__m256i* vecs) noexcept {					\
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

#ifdef __clang__
#define COEX(a, b){ __m256i c = a; a = _mm256_min_epi32(a, b); b = _mm256_max_epi32(c, b); }
#else 
#define COEX(a, b){ __m256i c = a; a =  (__m256i)__builtin_ia32_pminsd256 ((__v8si)a, (__v8si)b); b = (__m256i)__builtin_ia32_pmaxsd256 ((__v8si)c, (__v8si)b); }
#endif

SORT_16_INT_COLUMNS_WISE(i)
CREATE_MERGE_8_COLUMNS_WITH_16_ELEMENTS(i)

#undef COEX
#undef COEX_SHUFFLE
#undef COEX_PERMUTE

#define COEX_SHUFFLE(vec, a, b, c, d, e, f, g, h, MASK){               \
    int shuffle_mask = _MM_SHUFFLE(d, c, b, a);              \
    __m256i shuffled = _mm256_shuffle_epi32(vec, shuffle_mask);        \
    __m256i min = _mm256_min_epu32(shuffled, vec);                     \
    __m256i max = _mm256_max_epu32(shuffled, vec);                     \
    int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask);}

#define COEX_PERMUTE(vec, a, b, c, d, e, f, g, h, MASK){               \
    __m256i permute_mask = _mm256_setr_epi32(a, b, c, d, e, f, g, h);  \
    __m256i permuted = _mm256_permutevar8x32_epi32(vec, permute_mask); \
    __m256i min = _mm256_min_epu32(permuted, vec);                     \
    __m256i max = _mm256_max_epu32(permuted, vec);                     \
    int blend_mask = MASK(a, b, c, d, e, f, g, h);           \
    vec = _mm256_blend_epi32(min, max, blend_mask);}

#ifdef __clang__
#define COEX(a, b){ __m256i c = a; a = _mm256_min_epu32(a, b); b = _mm256_max_epu32(c, b);}
#else 
#define COEX(a, b){ __m256i c = a; a = (__m256i)__builtin_ia32_pminud256 ((__v8si)a, (__v8si)b); b = (__m256i)__builtin_ia32_pmaxud256 ((__v8si)c, (__v8si)b);}
#endif
  

SORT_16_INT_COLUMNS_WISE(u)
CREATE_MERGE_8_COLUMNS_WITH_16_ELEMENTS(u)

/// sorts 128 i32 elements
/// \param v
/// \return
void sortingnetwork_sort_i32x128(__m256i *v) noexcept {
	sort_16_int_column_wise_i(v);
	merge_8_columns_with_16_elements_i(v);
}

/// sorts 128 u32 elements
/// \param v
/// \return
void sortingnetwork_sort_u32x128_2(__m256i *v) noexcept {
	sort_16_int_column_wise_u(v);
	merge_8_columns_with_16_elements_u(v);
}

#undef COEX
#undef COEX_PERMUTE
#undef COEX_SHUFFLE
#undef REVERSE_VEC
#undef ASC

#endif
