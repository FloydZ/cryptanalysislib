#ifndef CRYPTANALYSISLIB_SORTING_NETWORK_AVX512_H
#define CRYPTANALYSISLIB_SORTING_NETWORK_AVX512_H

#ifndef USE_AVX512F
#error "no avx512"
#endif

#ifndef CRYPTANALYSISLIB_SORTING_NETWORK_H
#error "dont inlude 'sort/sorting_network/avx2.h' directly. Use `sort/sorting_network/common.h`."
#endif

#include <immintrin.h>
#include <stdint.h>


#define compare_and_swap16(a, b, a1, b1) \
	{                                    \
		a1 = __builtin_elementwise_min(a, b);     \
		b1 = __builtin_elementwise_max(a, b);     \
	}

static int64_t __attribute__((aligned(64))) indexc[8] = {0, 8, 1, 9, 2, 10, 3, 11};
static int64_t __attribute__((aligned(64))) indexd[8] = {4, 12, 5, 13, 6, 14, 7, 15};


// only needed if less the 16 elements should be sorted in 
// `sorting_network_sort_u64x16`
// See: https://raw.githubusercontent.com/jmakino/simdsort/276be8727ae329fde686bf2b80cd2df8d37cc6f6/bitonic16.h
#define initial_copy16(data, a, b, n)                          \
	{                                                          \
		int64_t intmax = INT64_MAX;                            \
		a = _mm512_broadcastq_epi64(*((__m128i *) (&intmax))); \
		b = _mm512_broadcastq_epi64(*((__m128i *) (&intmax))); \
		__mmask8 maska = (__mmask8) ((1 << n) - 1);            \
		__mmask8 maskb = (__mmask8) (((1 << n) - 1) >> 8);     \
		a = _mm512_mask_loadu_epi64(a, maska, data);           \
		b = _mm512_mask_loadu_epi64(b, maskb, data + 8);       \
	}

constexpr static inline void sortingnetwork_sort_u64x16(__m512i &a, __m512i &b) {
	__m512i a1, b1;
	compare_and_swap16(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index0[8] = {0, 8, 2, 10, 4, 12, 6, 14};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index0), b1);
	const int64_t __attribute__((aligned(64))) index1[8] = {9, 1, 11, 3, 13, 5, 15, 7};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index1), b1);

	compare_and_swap16(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index2[8] = {0, 9, 2, 11, 4, 13, 6, 15};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index2), b1);
	const int64_t __attribute__((aligned(64))) index3[8] = {1, 8, 3, 10, 5, 12, 7, 14};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index3), b1);

	compare_and_swap16(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index4[8] = {0, 8, 1, 9, 4, 12, 5, 13};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index4), b1);
	const int64_t __attribute__((aligned(64))) index5[8] = {11, 3, 10, 2, 15, 7, 14, 6};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index5), b1);

	compare_and_swap16(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index6[8] = {0, 1, 11, 10, 4, 5, 15, 14};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index6), b1);
	const int64_t __attribute__((aligned(64))) index7[8] = {2, 3, 9, 8, 6, 7, 13, 12};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index7), b1);

	compare_and_swap16(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index8[8] = {0, 8, 2, 10, 4, 12, 6, 14};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index8), b1);
	const int64_t __attribute__((aligned(64))) index9[8] = {1, 9, 3, 11, 5, 13, 7, 15};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index9), b1);

	compare_and_swap16(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index10[8] = {0, 8, 1, 9, 2, 10, 3, 11};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index10), b1);
	const int64_t __attribute__((aligned(64))) index11[8] = {15, 7, 14, 6, 13, 5, 12, 4};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index11), b1);

	compare_and_swap16(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index12[8] = {0, 1, 2, 3, 15, 14, 13, 12};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index12), b1);
	const int64_t __attribute__((aligned(64))) index13[8] = {4, 5, 6, 7, 11, 10, 9, 8};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index13), b1);

	compare_and_swap16(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index14[8] = {0, 1, 8, 9, 4, 5, 12, 13};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index14), b1);
	const int64_t __attribute__((aligned(64))) index15[8] = {2, 3, 10, 11, 6, 7, 14, 15};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index15), b1);

	compare_and_swap16(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index16[8] = {0, 8, 2, 10, 4, 12, 6, 14};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index16), b1);
	const int64_t __attribute__((aligned(64))) index17[8] = {1, 9, 3, 11, 5, 13, 7, 15};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index17), b1);

	compare_and_swap16(a, b, a1, b1);
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(indexc), b1);
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(indexd), b1);
}
#endif
