#ifndef CRYPTANALYSISLIB_POPCOUNT_X86_H
#define CRYPTANALYSISLIB_POPCOUNT_X86_H

#if !defined(USE_AVX2)
#error "no avx"
#endif

#if !defined(CRYPTANALYSISLIB_POPCOUNT_H)
#error "Do not include this file directly. Use: `#include <popcount/popcount.h>`"
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


#define POPCOUNT_HELPER_MACRO_U128() 											\
constexpr __m128i lookup = __extension__ (__m128i)(__v16qi){  					\
		/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2, 							\
		/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3, 							\
		/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3, 							\
		/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4   							\
}; 																				\
const __m128i low_mask =  __extension__ (__m128i)(__v16qi){ 					\
		0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf, 										\
		0xf,0xf,0xf,0xf,0xf,0xf,0xf,0xf, 										\
}; 																				\
const __m128i lo = vec & low_mask; 												\
const __m128i hi      = (__m128i)__builtin_ia32_psrlwi128((__v8hi)vec, 4)&low_mask;	 \
const __m128i popcnt1 = (__m128i)__builtin_ia32_pshufb128((__v16qi)lookup, (__v16qi)lo); \
const __m128i popcnt2 = (__m128i)__builtin_ia32_pshufb128((__v16qi)lookup, (__v16qi)hi);


namespace cryptanalysislib::popcount::internal {
	/// Count set bits in parallel across 16 bytes (8-bit elements) using SSE instructions
	/// \param vec [in]: 128-bit vector containing 16 bytes to count bits in
	/// \return [out]: 128-bit vector where each byte contains the popcount of the corresponding input byte
	constexpr static __m128i popcount_sse_u8x16(const __m128i vec) noexcept {
		POPCOUNT_HELPER_MACRO_U128()
		return (__m128i) ((__v16qu) popcnt1 + (__v16qu) popcnt2);
	}

	/// Count set bits in parallel across 32 bytes (8-bit elements) using AVX2 instructions
	/// \param vec [in]: 256-bit vector containing 32 bytes to count bits in
	/// \return [out]: 256-bit vector where each byte contains the popcount of the corresponding input byte
	constexpr static __m256i popcount_avx2_8(const __m256i vec) noexcept {
#ifdef USE_AVX512BITALG
        return _mm256_popcnt_epi8(vec);
#else
		POPCOUNT_HELPER_MACRO()
		return (__m256i) ((__v32qu) popcnt1 + (__v32qu) popcnt2);
#endif
	}

	/// Count set bits in parallel across 16 words (16-bit elements) using AVX2 instructions
	/// \param vec [in]: 256-bit vector containing 16 words to count bits in
	/// \return [out]: 256-bit vector where each word contains the popcount of the corresponding input word
	constexpr static __m256i popcount_avx2_16(const __m256i vec) noexcept {
#ifdef USE_AVX512BITALG
        return _mm256_popcnt_epi16(vec);
#else
		POPCOUNT_HELPER_MACRO()
		const __m256i local = (__m256i) ((__v32qu) popcnt1 + (__v32qu) popcnt2);
		const __m256i mask = __extension__(__m256i)(__v16hi){0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

		__m256i ret = (__m256i) ((__v4du) local & (__v4du) mask);
		return (__m256i) ((__v32qu) ret + (__v32qu) ((__v4du) ((__m256i) __builtin_ia32_psrldi256((__v8si) local, 8)) & (__v4du) mask));
#endif
	}

	/// Count set bits in parallel across 8 dwords (32-bit elements) using AVX2 instructions
	/// \param vec [in]: 256-bit vector containing 8 dwords to count bits in
	/// \return [out]: 256-bit vector where each dword contains the popcount of the corresponding input dword
	constexpr static __m256i popcount_avx2_32(const __m256i vec) noexcept {
#ifdef USE_AVX512BITALG
        return _mm256_popcnt_epi32(vec);
#else
		POPCOUNT_HELPER_MACRO()
		const __m256i local = (__m256i) ((__v32qu) popcnt1 + (__v32qu) popcnt2);

		// not the best
		const __m256i mask = __extension__(__m256i)(__v8si){0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

		__m256i ret = (__m256i) ((__v4du) local & (__v4du) mask);
		ret = (__m256i) ((__v32qu) ret + (__v32qu) ((__v4du) ((__m256i) __builtin_ia32_psrldi256((__v8si) local, 8)) & (__v4du) mask));
		ret = (__m256i) ((__v32qu) ret + (__v32qu) ((__v4du) ((__m256i) __builtin_ia32_psrldi256((__v8si) local, 16)) & (__v4du) mask));
		ret = (__m256i) ((__v32qu) ret + (__v32qu) ((__v4du) ((__m256i) __builtin_ia32_psrldi256((__v8si) local, 24)) & (__v4du) mask));
		return ret;
#endif
	}

	/// Count set bits in parallel across 4 qwords (64-bit elements) using AVX2 instructions
	/// \param vec [in]: 256-bit vector containing 4 qwords to count bits in
	/// \return [out]: 256-bit vector where each qword contains the popcount of the corresponding input qword
	constexpr static __m256i popcount_avx2_64(const __m256i vec) noexcept {
#ifdef USE_AVX512BITALG
        return _mm256_popcnt_epi64(vec);
#else
		POPCOUNT_HELPER_MACRO()
		const __m256i local = (__m256i) ((__v32qu) popcnt1 + (__v32qu) popcnt2);
		return (__m256i) __builtin_ia32_psadbw256((__v32qi) local, (__v32qi) __extension__(__m256i)(__v4di){0, 0, 0, 0});
#endif
	}
}

#undef POPCOUNT_HELPER_MACRO
#endif
