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
#define POPCOUNT_HELPER_MACRO() 												\
constexpr __m256i lookup = __extension__ (__m256i)(__v32qi){  					\
		/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2, 							\
		/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3, 							\
		/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3, 							\
		/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4, 							\
		/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2, 							\
		/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3, 							\
		/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3, 							\
		/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4 								\
}; 																				\
const __m256i low_mask =  __extension__ (__m256i)(__v32qi){ 					\
		0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf, 										\
		0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf, 										\
		0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf, 										\
		0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf, 										\
}; 																				\
const __m256i lo = vec & low_mask; 												\
const __m256i hi = (__m256i)__builtin_ia32_psrlwi256((__v16hi)vec, 4)&low_mask;	\
const __m256i popcnt1 = (__m256i)__builtin_ia32_pshufb256((__v32qi)lookup, (__v32qi)lo); \
const __m256i popcnt2 = (__m256i)__builtin_ia32_pshufb256((__v32qi)lookup, (__v32qi)hi);


/// special popcount which popcounts on 32 * 8u bit limbs in parallel
constexpr static __m256i popcount_avx2_8(const __m256i vec) noexcept {
	POPCOUNT_HELPER_MACRO()
  	return (__m256i)((__v32qu)popcnt1 + (__v32qu)popcnt2);
}

/// special popcount which popcounts on 16 * 16u bit limbs in parallel
constexpr static __m256i popcount_avx2_16(const __m256i vec) noexcept {
	POPCOUNT_HELPER_MACRO()
  	const __m256i local = (__m256i)((__v32qu)popcnt1 + (__v32qu)popcnt2);
  	const __m256i mask = __extension__ (__m256i)(__v16hi){0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff};

  	__m256i ret = (__m256i)((__v4du)local & (__v4du)mask);
	return (__m256i)((__v32qu)ret + (__v32qu)((__v4du)((__m256i)__builtin_ia32_psrldi256((__v8si)local, 8)) & (__v4du)mask));
}

/// special popcount which popcounts on 8 * 32u bit limbs in parallel
constexpr static __m256i popcount_avx2_32(const __m256i vec) noexcept {
	POPCOUNT_HELPER_MACRO()
  	const __m256i local = (__m256i)((__v32qu)popcnt1 + (__v32qu)popcnt2);

	// not the best
  	const __m256i mask = __extension__ (__m256i)(__v16hi){0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff,0xff};

  	__m256i ret = (__m256i)((__v4du)local & (__v4du)mask);
	ret = (__m256i)((__v32qu)ret + (__v32qu)((__v4du)((__m256i)__builtin_ia32_psrldi256((__v8si)local,  8)) & (__v4du)mask));
	ret = (__m256i)((__v32qu)ret + (__v32qu)((__v4du)((__m256i)__builtin_ia32_psrldi256((__v8si)local, 16)) & (__v4du)mask));
	ret = (__m256i)((__v32qu)ret + (__v32qu)((__v4du)((__m256i)__builtin_ia32_psrldi256((__v8si)local, 24)) & (__v4du)mask));
	return ret;
}

/// special popcount which popcounts on 4 * 64 bit limbs in parallel
constexpr static __m256i popcount_avx2_64(const __m256i vec) noexcept {
	POPCOUNT_HELPER_MACRO()
  	const __m256i local = (__m256i)((__v32qu)popcnt1 + (__v32qu)popcnt2);
  	return (__m256i)__builtin_ia32_psadbw256((__v32qi)local, (__v32qi)__extension__ (__m256i)(__v4di){ 0, 0, 0, 0 });
}

/// TODO merge
static inline uint32_t hammingweight_mod2_limb256(__m256i vec) {
	POPCOUNT_HELPER_MACRO()
	const __m256i local = (__m256i)((__v32qu)popcnt1 + (__v32qu)popcnt2);
	const __m256i final = (__m256i)__builtin_ia32_psadbw256((__v32qi)local, (__v32qi)__extension__ (__m256i)(__v4di){ 0, 0, 0, 0 });

	// probably not fast
	alignas(32) static uint64_t bla[4];
	_mm256_store_si256((__m256i *)bla , final);
	return bla[0] + bla[1] + bla[2] + bla[3];
}
#undef POPCOUNT_HELPER_MACRO
#endif
