#ifndef CRYPTANALYSISLIB_POPCOUNT_X86_H
#define CRYPTANALYSISLIB_POPCOUNT_X86_H

#if !defined(USE_AVX2)
#error "no avx"
#endif

#if !defined(CRYPTANALYSISLIB_POPCOUNT_H)
#error "Do not inlcude this library directly. Use: `#include <popcount/popcount.h>`"
#endif

#include <immintrin.h>
#include "helper.h"

// small little helper macro containing some lookup definition
// which is used in all subsequent functions
#define POPCOUNT_HELPER_MACRO() 								\
constexpr __m256i lookup = __extension__ (__m256i)(__v32qi){  	\
		/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2, 			\
		/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3, 			\
		/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3, 			\
		/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4, 			\
		/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2, 			\
		/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3, 			\
		/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3, 			\
		/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4 				\
}; 																\
const __m256i low_mask =  __extension__ (__m256i)(__v32qi){ 	\
		0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf, 						\
		0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf, 						\
		0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf, 						\
		0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf, 						\
}; 																\
const __m256i lo = vec & low_mask; 								\
const __m256i hi = __builtin_ia32_psrlwi256(vec, 4) & low_mask; \
const __m256i popcnt1 = __builtin_ia32_pshufb256(lookup, lo);  	\
const __m256i popcnt2 = __builtin_ia32_pshufb256(lookup, hi);


/// special popcount which popcounts on 32 * 8u bit limbs in parallel
constexpr static __m256i popcount_avx2_8(const __m256i vec) noexcept {
	POPCOUNT_HELPER_MACRO()
    return _mm256_add_epi8(popcnt2, popcnt1);
}

/// special popcount which popcounts on 16 * 16u bit limbs in parallel
constexpr static __m256i popcount_avx2_16(const __m256i vec) noexcept {
	POPCOUNT_HELPER_MACRO()
    const __m256i local = _mm256_add_epi8(popcnt2, popcnt1);
	const __m256i mask = _mm256_set1_epi16(0xff);

	__m256i ret = _mm256_and_si256(local, mask);
	ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local,  8), mask));
	return ret;
}

/// special popcount which popcounts on 8 * 32u bit limbs in parallel
constexpr static __m256i popcount_avx2_32(const __m256i vec) noexcept {
	POPCOUNT_HELPER_MACRO()
	const __m256i local = _mm256_add_epi8(popcnt2, popcnt1);

	// not the best
	const __m256i mask = _mm256_set1_epi32(0xff);
	__m256i ret = _mm256_and_si256(local, mask);
	ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local,  8), mask));
	ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local, 16), mask));
	ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local, 24), mask));
	return ret;
}

/// special popcount which popcounts on 4 * 64 bit limbs in parallel
constexpr static __m256i popcount_avx2_64(const __m256i vec) noexcept {
	POPCOUNT_HELPER_MACRO()
    const __m256i local = _mm256_add_epi8(popcnt2, popcnt1);
	const __m256i ret = _mm256_sad_epu8(local, _mm256_setzero_si256());
	return ret;
}

#undef POPCOUNT_HELPER_MACRO
#endif
