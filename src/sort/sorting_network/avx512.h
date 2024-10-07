#ifndef CRYPTANALYSISLIB_SORTING_NETWORK_AVX512_H
#define CRYPTANALYSISLIB_SORTING_NETWORK_AVX512_H

#ifndef USE_AVX512F
#error "no avx512"
#endif

#ifndef CRYPTANALYSISLIB_SORTING_NETWORK_H
#error "dont inlude 'sort/sorting_network/avx512.h' directly. Use `sort/sorting_network/common.h`."
#endif

#include <immintrin.h>
#include <stdint.h>

#include "simd/simd.h"

///
#define COEX_u64x8(a, b, tmp)			\
	{                                   \
		tmp = a;                		\
		a = _mm512_min_epu64(a, b);     \
		b = _mm512_max_epu64(tmp, b); 	\
	}

#define COEX_u32x16(a, b, tmp)			\
	{                                   \
		tmp = a;                		\
		a = _mm512_min_epu32(a, b);     \
		b = _mm512_max_epu32(tmp, b); 	\
	}

#define COEX_u32x16_(a, b, c, d)		\
	{                                   \
		c = _mm512_min_epu32(a, b);     \
		d = _mm512_max_epu32(a, b); 	\
	}

#ifdef _clang_
// NOTE:
#define COMPAREANDSWAP_AVX512_64(a, b, a1, b1) 	\
	{                                    		\
		a1 = __builtin_elementwise_min(a, b);   \
		b1 = __builtin_elementwise_max(a, b);   \
	}
#else

#define COMPAREANDSWAP_AVX512_64(a, b, a1, b1) 	\
	{                                    		\
		a1 = _mm512_min_epu64(a, b);   \
		b1 = _mm512_max_epu64(a, b);   \
	}
#endif

#define compare_and_swap16(a, b, a1, b1) \
	{                                    \
		a1 = __builtin_ia32_pminsq512_mask ((__v8di)(a), (__v8di)(b), (__v8di)__mm512_undefined_epi32 (), (__mmask8) -1);     \
		b1 = __builtin_ia32_pmaxsq512_mask ((__v8di)(a), (__v8di)(b), (__v8di)__mm512_undefined_epi32 (), (__mmask8) -1);     \
	}


#include "sort/sorting_network/macros.h"


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

/// small little helper to be able to compile this on gcc
#ifndef __clang__ 
#define __builtin_ia32_vpermi2varq512(__A, __I, __B) 			\
(__m512i) __builtin_ia32_vpermt2varq512_mask ((__v8di)(__I),	\
						       				  (__v8di)(__A),	\
						       				  (__v8di)(__B), 	\
						       				  (__mmask8) -1);
#endif


/// sorts a and b
/// \param a first 8 elements to sort
/// \param b second 8 elements to sort
/// \return
constexpr static inline void sortingnetwork_sort_u64x16(__m512i &a, __m512i &b) {
	constexpr int64_t __attribute__((aligned(64))) sortingnetwork_av512_indexc[8] = {0, 8, 1, 9, 2, 10, 3, 11};
	constexpr int64_t __attribute__((aligned(64))) sortingnetwork_av512_indexd[8] = {4, 12, 5, 13, 6, 14, 7, 15};

	__m512i a1, b1;
	COMPAREANDSWAP_AVX512_64(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index0[8] = {0, 8, 2, 10, 4, 12, 6, 14};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index0), b1);
	const int64_t __attribute__((aligned(64))) index1[8] = {9, 1, 11, 3, 13, 5, 15, 7};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index1), b1);

	COMPAREANDSWAP_AVX512_64(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index2[8] = {0, 9, 2, 11, 4, 13, 6, 15};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index2), b1);
	const int64_t __attribute__((aligned(64))) index3[8] = {1, 8, 3, 10, 5, 12, 7, 14};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index3), b1);

	COMPAREANDSWAP_AVX512_64(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index4[8] = {0, 8, 1, 9, 4, 12, 5, 13};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index4), b1);
	const int64_t __attribute__((aligned(64))) index5[8] = {11, 3, 10, 2, 15, 7, 14, 6};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index5), b1);

	COMPAREANDSWAP_AVX512_64(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index6[8] = {0, 1, 11, 10, 4, 5, 15, 14};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index6), b1);
	const int64_t __attribute__((aligned(64))) index7[8] = {2, 3, 9, 8, 6, 7, 13, 12};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index7), b1);

	COMPAREANDSWAP_AVX512_64(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index8[8] = {0, 8, 2, 10, 4, 12, 6, 14};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index8), b1);
	const int64_t __attribute__((aligned(64))) index9[8] = {1, 9, 3, 11, 5, 13, 7, 15};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index9), b1);

	COMPAREANDSWAP_AVX512_64(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index10[8] = {0, 8, 1, 9, 2, 10, 3, 11};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index10), b1);
	const int64_t __attribute__((aligned(64))) index11[8] = {15, 7, 14, 6, 13, 5, 12, 4};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index11), b1);

	COMPAREANDSWAP_AVX512_64(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index12[8] = {0, 1, 2, 3, 15, 14, 13, 12};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index12), b1);
	const int64_t __attribute__((aligned(64))) index13[8] = {4, 5, 6, 7, 11, 10, 9, 8};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index13), b1);

	COMPAREANDSWAP_AVX512_64(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index14[8] = {0, 1, 8, 9, 4, 5, 12, 13};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index14), b1);
	const int64_t __attribute__((aligned(64))) index15[8] = {2, 3, 10, 11, 6, 7, 14, 15};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index15), b1);

	COMPAREANDSWAP_AVX512_64(a, b, a1, b1);
	const int64_t __attribute__((aligned(64))) index16[8] = {0, 8, 2, 10, 4, 12, 6, 14};
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index16), b1);
	const int64_t __attribute__((aligned(64))) index17[8] = {1, 9, 3, 11, 5, 13, 7, 15};
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(index17), b1);

	COMPAREANDSWAP_AVX512_64(a, b, a1, b1);
	a = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(sortingnetwork_av512_indexc), b1);
	b = __builtin_ia32_vpermi2varq512(a1, *(const __m512i *)(sortingnetwork_av512_indexd), b1);
}


// cleanup macros
#ifndef __clang__ 
#undef __builtin_ia32_vpermi2varq512
#endif

/// source for the avx512_sorting networks:
///	https://github.com/berenger-eu/avx-512-sort/blob/master/sort512kv.hpp#L510
/// heavily modified by FloydZ to handle key-value and different types all with
/// a single function
#define sortingnetwork_sort_x32x16_body(T, REG, MIN_FKT, MAX_FKT) \
	{																		\
		REG idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,		\
											  6, 7, 4, 5, 2, 3, 0, 1);		\
		REG permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);		\
		REG permNeighMin = MIN_FKT( input,permNeigh);						\
		REG permNeighMax = MAX_FKT(permNeigh, input);						\
        __m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax); \
    	if constexpr (kv) {                                                 \
			values = _mm512_mask_mov_epi32(                                 \
						_mm512_permutexvar_epi32(idxNoNeigh, values),       \
						_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),        \
			       		 values);\
		}                                                                   \
		input = tmp_input;													\
	}																		\
	{																		\
		REG idxNoNeigh = _mm512_set_epi32(12, 13, 14, 15, 8, 9, 10, 11,\
											  4, 5, 6, 7, 0, 1, 2, 3);\
		REG permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		REG permNeighMin = MIN_FKT( input,permNeigh);\
		REG permNeighMax = MAX_FKT(permNeigh, input);\
        __m512i tmp_input  = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);\
		if constexpr (kv) {													\
			values = _mm512_mask_mov_epi32( 								\
	                	_mm512_permutexvar_epi32(idxNoNeigh, values),		\
	                	_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
	                    values); 											\
		} 																	\
		input = tmp_input; 													\
        }																	\
	{																		\
		REG idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,		\
											  6, 7, 4, 5, 2, 3, 0, 1);		\
		REG permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		REG permNeighMin = MIN_FKT( input,permNeigh);\
		REG permNeighMax = MAX_FKT(permNeigh, input);\
		input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);\
	}\
	{\
		REG idxNoNeigh = _mm512_set_epi32(8, 9, 10, 11, 12, 13, 14, 15,\
											  0, 1, 2, 3, 4, 5, 6, 7);\
		REG permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		REG permNeighMin = MIN_FKT( input,permNeigh);\
		REG permNeighMax = MAX_FKT(permNeigh, input);\
		input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);\
	}\
	{\
		REG idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,\
											  5, 4, 7, 6, 1, 0, 3, 2);\
		REG permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		REG permNeighMin = MIN_FKT( input,permNeigh);\
		REG permNeighMax = MAX_FKT(permNeigh, input);\
		input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);\
	}\
	{\
		REG idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,\
											  6, 7, 4, 5, 2, 3, 0, 1);\
		REG permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		REG permNeighMin = MIN_FKT( input,permNeigh);\
		REG permNeighMax = MAX_FKT(permNeigh, input);\
		input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);\
	}\
	{\
		REG idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,\
											  8, 9, 10, 11, 12, 13, 14, 15);\
		REG permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		REG permNeighMin = MIN_FKT( input,permNeigh);\
		REG permNeighMax = MAX_FKT(permNeigh, input);\
		input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);\
	}\
	{\
		REG idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,\
											  3, 2, 1, 0, 7, 6, 5, 4);\
		REG permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		REG permNeighMin = MIN_FKT( input,permNeigh);\
		REG permNeighMax = MAX_FKT(permNeigh, input);\
		input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);\
	}\
	{\
		REG idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,\
											  5, 4, 7, 6, 1, 0, 3, 2);\
		REG permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		REG permNeighMin = MIN_FKT( input,permNeigh);\
		REG permNeighMax = MAX_FKT(permNeigh, input);\
		input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);\
	}\
	{\
		REG idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,\
											  6, 7, 4, 5, 2, 3, 0, 1);\
		REG permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		REG permNeighMin = MIN_FKT( input,permNeigh);\
		REG permNeighMax = MAX_FKT(permNeigh, input);\
		input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);\
	}                                                                \
	(void)values;                                                                  \
}




//inline void CoreExchangeSort2V(__m512i& input, __m512i& input2,
//                               __m512i& input_val, __m512i& input2_val){
#define sortingnetwork_exchangesort_x32x32_body(T, REG, MIN_FKT, MAX_FKT) \
	{																			\
		__m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,			\
		                                      8, 9, 10, 11, 12, 13, 14, 15);	\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input1);		\
		__m512i tmp_input = MIN_FKT( permNeigh,input2);				\
		__m512i tmp_input2 = MAX_FKT(input2, permNeigh);       		\
        if constexpr (kv) {                                                     \
			__m512i input_val_perm = _mm512_permutexvar_epi32(idxNoNeigh, value1);\
			value1 = _mm512_mask_mov_epi32(value2,								\
                            _mm512_cmp_epi32_mask(tmp_input, permNeigh, _MM_CMPINT_EQ),\
							input_val_perm);									\
			value2 = _mm512_mask_mov_epi32(input_val_perm,                     	\
							_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
							value2);											\
		}																		\
		input1 = tmp_input;\
		input2 = tmp_input2;\
	}\
	{\
		__m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,\
		                                      15, 14, 13, 12, 11, 10, 9, 8);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input1);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeighMin = MIN_FKT(input1, permNeigh);\
		__m512i permNeighMin2 = MIN_FKT(input2, permNeigh2);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input1);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);       \
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);\
		if constexpr (kv) {                                                                  \
			value1 = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, value1),     \
						_mm512_cmp_epi32_mask(tmp_input, input1, _MM_CMPINT_EQ),\
		                value1);\
			value2 = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, value2),    \
						_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                value2);\
		}\
		input1 = tmp_input;\
		input2 = tmp_input2;\
	}\
	{\
		__m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,\
		                                      3, 2, 1, 0, 7, 6, 5, 4);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input1);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeighMin = MIN_FKT( input1,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input1);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);\
 		if constexpr (kv) {\
			value1 = _mm512_mask_mov_epi32(\
					_mm512_permutexvar_epi32(idxNoNeigh, value1),\
					_mm512_cmp_epi32_mask(tmp_input, input1, _MM_CMPINT_EQ),\
					value1);\
			value2 = _mm512_mask_mov_epi32(\
					_mm512_permutexvar_epi32(idxNoNeigh, value2),\
					_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
					value2);\
		}\
		input1 = tmp_input;\
		input2 = tmp_input2;\
	}\
	{\
		__m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,\
		                                      5, 4, 7, 6, 1, 0, 3, 2);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input1);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeighMin = MIN_FKT( input1,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input1);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);\
		if constexpr (kv) {\
		    value1 = _mm512_mask_mov_epi32(\
		            	_mm512_permutexvar_epi32(idxNoNeigh, value1), \
		            	_mm512_cmp_epi32_mask(tmp_input, input1, _MM_CMPINT_EQ),\
		            	value1);\
		    value2 = _mm512_mask_mov_epi32(\
		            	_mm512_permutexvar_epi32(idxNoNeigh, value2), \
		            	_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		            	value2);\
	    }\
		input1 = tmp_input;\
		input2 = tmp_input2;\
	}\
	{\
		__m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,\
		                                      6, 7, 4, 5, 2, 3, 0, 1);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input1);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeighMin = MIN_FKT( input1,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input1);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);\
	    if constexpr (kv) {\
		    value1 = _mm512_mask_mov_epi32(\
		            	_mm512_permutexvar_epi32(idxNoNeigh, value1),\
		            	_mm512_cmp_epi32_mask(tmp_input, input1, _MM_CMPINT_EQ),\
		            	value1);\
		    value2 = _mm512_mask_mov_epi32(\
		            	_mm512_permutexvar_epi32(idxNoNeigh, value2),\
		            	_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		            	value2);\
	    }\
		input1 = tmp_input;\
		input2 = tmp_input2;\
	}                                                                        \
}


#define sortingnetwork_sort_x32x32_body(T, REG, MIN_FKT, MAX_FKT) \
	{\
		__m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,\
											   6, 7, 4, 5, 2, 3, 0, 1);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);\
 		if constexpr (kv) { \
		input_val = _mm512_mask_mov_epi32(\
						_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
						_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
						input_val);\
		input2_val = _mm512_mask_mov_epi32(\
						_mm512_permutexvar_epi32(idxNoNeigh, input2_val),\
							_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
						input2_val);\
		}                                                          \
		input = tmp_input;\
		input2 = tmp_input2;\
	} {\
	 __m512i idxNoNeigh = _mm512_set_epi32(12, 13, 14, 15, 8, 9, 10, 11,\
										   4, 5, 6, 7, 0, 1, 2, 3);\
	 __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
	 __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
	 __m512i permNeighMin = MIN_FKT( input,permNeigh);\
	 __m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
	 __m512i permNeighMax = MAX_FKT(permNeigh, input);\
	 __m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
	 __m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);\
	 __m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);\
     if constexpr (kv) {                                                             \
		input_val = _mm512_mask_mov_epi32(                              \
					   _mm512_permutexvar_epi32(idxNoNeigh, input_val),             \
					   _mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
					   input_val);\
		input2_val = _mm512_mask_mov_epi32(                             \
					   _mm512_permutexvar_epi32(idxNoNeigh, input2_val),            \
					   _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
					   input2_val);\
	 }                                                             \
	 input = tmp_input;\
	 input2 = tmp_input2;\
	} { \
	 __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,\
										   6, 7, 4, 5, 2, 3, 0, 1);\
	 __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
	 __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
	 __m512i permNeighMin = MIN_FKT( input,permNeigh);\
	 __m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
	 __m512i permNeighMax = MAX_FKT(permNeigh, input);\
	 __m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
	 __m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);\
	 __m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);\
	if constexpr (kv) {\
		input_val = _mm512_mask_mov_epi32(                              \
						_mm512_permutexvar_epi32(idxNoNeigh, input_val),            \
						_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ ),\
						input_val);\
		input2_val = _mm512_mask_mov_epi32(                             \
						_mm512_permutexvar_epi32(idxNoNeigh, input2_val),           \
						_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
						input2_val);\
	}\
	 input = tmp_input;\
	 input2 = tmp_input2;\
    }{\
	 __m512i idxNoNeigh = _mm512_set_epi32(8, 9, 10, 11, 12, 13, 14, 15,\
										   0, 1, 2, 3, 4, 5, 6, 7);\
	 __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
	 __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
	 __m512i permNeighMin = MIN_FKT( input,permNeigh);\
	 __m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
	 __m512i permNeighMax = MAX_FKT(permNeigh, input);\
	 __m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
	 __m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);\
	 __m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);\
	if constexpr (kv) {\
		input_val = _mm512_mask_mov_epi32(\
        				_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
        				_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                input_val);\
		input2_val = _mm512_mask_mov_epi32(                             \
						_mm512_permutexvar_epi32(idxNoNeigh, input2_val),           \
						_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                input2_val);\
	}\
	 input = tmp_input;\
	 input2 = tmp_input2;\
 	} {\
	 __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,\
										   5, 4, 7, 6, 1, 0, 3, 2);\
	 __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
	 __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
	 __m512i permNeighMin = MIN_FKT( input,permNeigh);\
	 __m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
	 __m512i permNeighMax = MAX_FKT(permNeigh, input);\
	 __m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
	 __m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);\
	 __m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);\
	if constexpr (kv) {\
		input_val = _mm512_mask_mov_epi32(                              \
						_mm512_permutexvar_epi32(idxNoNeigh, input_val),            \
						_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                input_val);\
		input2_val = _mm512_mask_mov_epi32(                             \
						_mm512_permutexvar_epi32(idxNoNeigh, input2_val),           \
						_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                input2_val);\
		}\
	 input = tmp_input;\
	 input2 = tmp_input2;\
	} {\
	 __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,\
										   6, 7, 4, 5, 2, 3, 0, 1);\
	 __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
	 __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
	 __m512i permNeighMin = MIN_FKT( input,permNeigh);\
	 __m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
	 __m512i permNeighMax = MAX_FKT(permNeigh, input);\
	 __m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
	 __m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);\
	 __m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);\
	if constexpr (kv) {\
		input_val = _mm512_mask_mov_epi32(\
        				_mm512_permutexvar_epi32(idxNoNeigh, input_val), \
        				_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                input_val);\
		input2_val = _mm512_mask_mov_epi32(\
                		_mm512_permutexvar_epi32(idxNoNeigh, input2_val),	\
                		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                input2_val);\
		}\
	 input = tmp_input;\
	 input2 = tmp_input2;\
 	} {\
	 __m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,\
										   8, 9, 10, 11, 12, 13, 14, 15);\
	 __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
	 __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
	 __m512i permNeighMin = MIN_FKT( input,permNeigh);\
	 __m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
	 __m512i permNeighMax = MAX_FKT(permNeigh, input);\
	 __m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
	 __m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);\
	 __m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);\
	if constexpr (kv) {\
		input_val = _mm512_mask_mov_epi32(\
        				_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
        				_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                input_val);\
		input2_val = _mm512_mask_mov_epi32(\
                		_mm512_permutexvar_epi32(idxNoNeigh, input2_val), \
                		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		               input2_val);\
    }\
	 input = tmp_input;\
	 input2 = tmp_input2;\
	} {\
	 __m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,\
										   3, 2, 1, 0, 7, 6, 5, 4);\
	 __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
	 __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
	 __m512i permNeighMin = MIN_FKT( input,permNeigh);\
	 __m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
	 __m512i permNeighMax = MAX_FKT(permNeigh, input);\
	 __m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
	 __m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);\
	 __m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);\
	if constexpr (kv) {\
		input_val = _mm512_mask_mov_epi32(\
					   _mm512_permutexvar_epi32(idxNoNeigh, input_val), \
					   _mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
					   input_val);\
		input2_val = _mm512_mask_mov_epi32(                             \
					   _mm512_permutexvar_epi32(idxNoNeigh, input2_val),            \
					   _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ ),\
					   input2_val);\
	}\
	 input = tmp_input;\
	 input2 = tmp_input2;\
	} {\
	 __m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,\
										   5, 4, 7, 6, 1, 0, 3, 2);\
	 __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
	 __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
	 __m512i permNeighMin = MIN_FKT( input,permNeigh);\
	 __m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
	 __m512i permNeighMax = MAX_FKT(permNeigh, input);\
	 __m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
	 __m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);\
	 __m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);\
	if constexpr (kv) {\
		input_val = _mm512_mask_mov_epi32(\
        				_mm512_permutexvar_epi32(idxNoNeigh, input_val), 	\
        				_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                input_val);\
		input2_val = _mm512_mask_mov_epi32(\
                		_mm512_permutexvar_epi32(idxNoNeigh, input2_val), \
                		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                input2_val);\
	}\
	 input = tmp_input;\
	 input2 = tmp_input2;\
 } {\
	 __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,\
										   6, 7, 4, 5, 2, 3, 0, 1);\
	 __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
	 __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
	 __m512i permNeighMin = MIN_FKT( input,permNeigh);\
	 __m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
	 __m512i permNeighMax = MAX_FKT(permNeigh, input);\
	 __m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
	 __m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);\
	 __m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);\
	if constexpr (kv) {\
		input_val = _mm512_mask_mov_epi32(\
        				_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
        				_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                input_val);\
		input2_val = _mm512_mask_mov_epi32(	\
                		_mm512_permutexvar_epi32(idxNoNeigh, input2_val),\
                		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                input2_val);\
	}\
	 input = tmp_input;\
	 input2 = tmp_input2;\
	}                                                            \
	avx512_sortingnetwork_exchangesort_ ##T ## 32x32<kv>(input,input2,input_val,input2_val);\
}


#define sortingnetwork_sort_x32x48_body(T, REG, MIN_FKT, MAX_FKT) \
	{\
		__m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,\
		                                      8, 9, 10, 11, 12, 13, 14, 15);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i tmp_input3 = MAX_FKT(input2, permNeigh);\
		__m512i tmp_input2 = MIN_FKT( permNeigh,input2);\
		__m512i input3_val_perm = _mm512_permutexvar_epi32(idxNoNeigh, input3_val);    \
        if constexpr (kv) {                                                         \
			input3_val = _mm512_mask_mov_epi32(input2_val,                 \
							_mm512_cmp_epi32_mask(tmp_input3, permNeigh, _MM_CMPINT_EQ ),\
							input3_val_perm);\
			input2_val = _mm512_mask_mov_epi32(input3_val_perm,\
                             _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
							input2_val);\
		}\
		input3 = tmp_input3;\
		input2 = tmp_input2;\
	} {\
		__m512i inputCopy = input;\
		__m512i tmp_input = MIN_FKT( inputCopy,input2);\
		__m512i tmp_input2 = MAX_FKT(input2, inputCopy);\
		__m512i input_val_copy = input_val;\
	    if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(input2_val,\
		                     _mm512_cmp_epi32_mask(tmp_input, inputCopy, _MM_CMPINT_EQ),\
		                     input_val_copy);\
		    input2_val = _mm512_mask_mov_epi32(input_val_copy,\
		                     _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                     input2_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,\
		                                      15, 14, 13, 12, 11, 10, 9, 8);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);\
	    if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
		            		_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                    input_val);\
		    input2_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input2_val),\
		            		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                    input2_val);\
		    input3_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input3_val),\
		            		_mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                    input3_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,\
		                                      3, 2, 1, 0, 7, 6, 5, 4);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);\
	    if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
		            		_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                    input_val);\
		    input2_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input2_val),\
		            		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                   	input2_val);\
		    input3_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input3_val),\
		            		_mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                   	input3_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,\
		                                      5, 4, 7, 6, 1, 0, 3, 2);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);\
		if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
		            		_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                    input_val);\
		    input2_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input2_val),\
		            		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                    input2_val);\
		    input3_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input3_val),\
		            		_mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                   	input3_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,\
		                                      6, 7, 4, 5, 2, 3, 0, 1);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);\
	    if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
		            		_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                    input_val);\
		    input2_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input2_val),\
		            		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                    input2_val);\
		    input3_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input3_val),\
		           		 	_mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                    input3_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
	}\
}




#define sortingnetwork_sort_x32x64_body(T, REG, MIN_FKT, MAX_FKT) \
	{\
		__m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,\
		                                      8, 9, 10, 11, 12, 13, 14, 15);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i tmp_input4 = MAX_FKT(input, permNeigh4);\
		__m512i tmp_input = MIN_FKT( permNeigh4,input);\
		__m512i tmp_input3 = MAX_FKT(input2, permNeigh3);\
		__m512i tmp_input2 = MIN_FKT( permNeigh3,input2);\
        if constexpr (kv) {\
			__m512i input4_val_perm = _mm512_permutexvar_epi32(idxNoNeigh, input4_val);\
			input4_val = _mm512_mask_mov_epi32(input_val, _mm512_cmp_epi32_mask(tmp_input4, permNeigh4, _MM_CMPINT_EQ ),\
			                                   input4_val_perm);\
			input_val = _mm512_mask_mov_epi32(input4_val_perm, _mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ ),\
		                                  input_val);\
	        __m512i input3_val_perm = _mm512_permutexvar_epi32(idxNoNeigh, input3_val);\
	        input3_val = _mm512_mask_mov_epi32(\
	                		input2_val,\
	                		_mm512_cmp_epi32_mask(tmp_input3, permNeigh3, _MM_CMPINT_EQ),\
	                        input3_val_perm);\
	        input2_val = _mm512_mask_mov_epi32(\
	                		input3_val_perm,\
	                		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
	                        input2_val);\
        }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
	} {\
		__m512i inputCopy = input;\
		__m512i tmp_input = MIN_FKT( inputCopy,input2);\
		__m512i tmp_input2 = MAX_FKT(input2, inputCopy);\
	    if constexpr (kv) {\
			__m512i input_val_copy = input_val;\
		    input_val = _mm512_mask_mov_epi32(input2_val,\
		                     _mm512_cmp_epi32_mask(tmp_input, inputCopy, _MM_CMPINT_EQ),\
		                     input_val_copy);\
		    input2_val = _mm512_mask_mov_epi32(input_val_copy,\
		                      _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                      input2_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
	} {\
		__m512i inputCopy = input3;\
		__m512i tmp_input3 = MIN_FKT( inputCopy,input4);\
		__m512i tmp_input4 = MAX_FKT(input4, inputCopy);\
	    if constexpr (kv) {\
		    __m512i input3_val_copy = input3_val;\
		    input3_val = _mm512_mask_mov_epi32(input4_val,\
		                     _mm512_cmp_epi32_mask(tmp_input3, inputCopy, _MM_CMPINT_EQ),\
		                     input3_val_copy);\
		    input4_val = _mm512_mask_mov_epi32(input3_val_copy,\
		                     _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                     input4_val);\
	    }\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,\
		                                      15, 14, 13, 12, 11, 10, 9, 8);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);\
	    if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
		            		_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                    input_val);\
		    input2_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input2_val),\
		            		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                    input2_val);\
		    input3_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input3_val),\
		            		_mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                    input3_val);\
		    input4_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input4_val),\
		            		_mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                    input4_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,\
		                                      3, 2, 1, 0, 7, 6, 5, 4);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);\
		if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
		            		_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                    input_val);\
		    input2_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input2_val),\
		            		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                    input2_val);\
		    input3_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input3_val),\
		            		_mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                    input3_val);\
		    input4_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input4_val),\
		            		_mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                    input4_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,\
		                                      5, 4, 7, 6, 1, 0, 3, 2);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);\
		if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input_val), \
		            		_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                    input_val);\
		    input2_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input2_val), \
		            		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                    input2_val);\
		    input3_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input3_val), \
		            		_mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                    input3_val);\
		    input4_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input4_val), \
		            		_mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                    input4_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
	} {\
	    __m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,\
	                                          6, 7, 4, 5, 2, 3, 0, 1);\
	    __m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
	    __m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
	    __m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
	    __m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
	    __m512i permNeighMin = MIN_FKT(input, permNeigh);\
	    __m512i permNeighMin2 = MIN_FKT(input2, permNeigh2);\
	    __m512i permNeighMin3 = MIN_FKT(input3, permNeigh3);\
	    __m512i permNeighMin4 = MIN_FKT(input4, permNeigh4);\
	    __m512i permNeighMax = MAX_FKT(permNeigh, input);\
	    __m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
	    __m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
	    __m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
	    __m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);\
	    __m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);\
	    __m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);\
	    __m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);\
	    if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
		            		_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                    input_val);\
	    	input2_val = _mm512_mask_mov_epi32(\
	            			_mm512_permutexvar_epi32(idxNoNeigh, input2_val),\
	            			_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
	    	                input2_val);\
	    	input3_val = _mm512_mask_mov_epi32(\
	                		_mm512_permutexvar_epi32(idxNoNeigh, input3_val),\
	                		_mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
	    	                input3_val);\
	    	input4_val = _mm512_mask_mov_epi32(\
	                		_mm512_permutexvar_epi32(idxNoNeigh, input4_val),\
	                		_mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
	                        input4_val);\
    	}\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
	}\
}


#define sortingnetwork_sort_x32x80_body(T, REG, MIN_FKT, MAX_FKT) \
	{\
		__m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,\
		                                      8, 9, 10, 11, 12, 13, 14, 15);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i tmp_input5 = MAX_FKT(input4, permNeigh5);\
		__m512i tmp_input4 = MIN_FKT( permNeigh5,input4);\
		__m512i input5_val_copy = _mm512_permutexvar_epi32(idxNoNeigh, input5_val);\
  		if constexpr (kv) {\
		    input5_val = _mm512_mask_mov_epi32(                         \
							input4_val,                                                \
							_mm512_cmp_epi32_mask(tmp_input5, permNeigh5, _MM_CMPINT_EQ),\
		                    input5_val_copy);\
		    input4_val = _mm512_mask_mov_epi32(\
	                        input5_val_copy, \
							_mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                    input4_val);\
	    }\
		input5 = tmp_input5;\
		input4 = tmp_input4;\
	} {\
		__m512i inputCopy = input;\
		__m512i tmp_input = MIN_FKT( inputCopy,input3);\
		__m512i tmp_input3 = MAX_FKT(input3, inputCopy);\
		if constexpr (kv) {\
		    __m512i input_val_copy = input_val;\
		    input_val = _mm512_mask_mov_epi32(                          \
							input2_val,                                                \
							_mm512_cmp_epi32_mask(tmp_input, inputCopy, _MM_CMPINT_EQ),\
		                    input_val_copy);\
		    input3_val = _mm512_mask_mov_epi32(                         \
							input_val_copy,                                            \
							_mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                    input3_val);\
	    }\
		input = tmp_input;\
		input3 = tmp_input3;\
	} { \
		__m512i inputCopy = input2;\
		__m512i tmp_input2 = MIN_FKT( inputCopy,input4);\
		__m512i tmp_input4 = MAX_FKT(input4, inputCopy);\
		if constexpr (kv) {\
		    __m512i input2_val_copy = input2_val;\
		    input2_val = _mm512_mask_mov_epi32(                         \
							input4_val,                                                \
							_mm512_cmp_epi32_mask(tmp_input2, inputCopy, _MM_CMPINT_EQ),\
		                    input2_val_copy);\
		    input4_val = _mm512_mask_mov_epi32(\
	                        input2_val_copy,                 \
							_mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                    input4_val);\
	    }\
		input2 = tmp_input2;\
		input4 = tmp_input4;\
	} {\
		__m512i inputCopy = input;\
		__m512i tmp_input = MIN_FKT( inputCopy,input2);\
		__m512i tmp_input2 = MAX_FKT(input2, inputCopy);\
		if constexpr (kv) {\
		    __m512i input_val_copy = input_val;\
		    input_val = _mm512_mask_mov_epi32(\
		            		input2_val, \
		            		_mm512_cmp_epi32_mask(tmp_input, inputCopy, _MM_CMPINT_EQ),\
		                    input_val_copy);\
		    input2_val = _mm512_mask_mov_epi32(\
		            		input_val_copy, \
		            		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                    input2_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
	} {\
		__m512i inputCopy = input3;\
		__m512i tmp_input3 = MIN_FKT( inputCopy,input4);\
		__m512i tmp_input4 = MAX_FKT(input4, inputCopy);\
		if constexpr (kv) {\
		    __m512i input3_val_copy = input3_val;\
		    input3_val = _mm512_mask_mov_epi32(\
		            		input4_val, \
		            		_mm512_cmp_epi32_mask(tmp_input3, inputCopy, _MM_CMPINT_EQ),\
		                    input3_val_copy);\
		    input4_val = _mm512_mask_mov_epi32(\
		            		input3_val_copy, \
		            		_mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                    input4_val);\
	    }\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,\
		                                      15, 14, 13, 12, 11, 10, 9, 8);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMin5 = MIN_FKT( input5,permNeigh5);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i permNeighMax5 = MAX_FKT(permNeigh5, input5);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);\
		__m512i tmp_input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xFF00, permNeighMax5);\
	    if constexpr (kv) { \
		    input_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input_val), \
		            		_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                    input_val);\
		    input2_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input2_val), \
		            		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                    input2_val);\
		    input3_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input3_val), \
		            		_mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                    input3_val);\
		    input4_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input4_val), \
		            		_mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                    input4_val);\
		    input5_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input5_val), \
		            		_mm512_cmp_epi32_mask(tmp_input5, input5, _MM_CMPINT_EQ),\
		                    input5_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
		input5 = tmp_input5;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,\
		                                      3, 2, 1, 0, 7, 6, 5, 4);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMin5 = MIN_FKT( input5,permNeigh5);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i permNeighMax5 = MAX_FKT(permNeigh5, input5);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);\
		__m512i tmp_input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xF0F0, permNeighMax5);\
		if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
		            		_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                    input_val);\
		    input2_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input2_val),\
		            		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                    input2_val);\
		    input3_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input3_val),\
		            		_mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                    input3_val);\
		    input4_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input4_val),\
		            		_mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                    input4_val);\
		    input5_val = _mm512_mask_mov_epi32( \
		            		_mm512_permutexvar_epi32(idxNoNeigh, input5_val),\
		            		_mm512_cmp_epi32_mask(tmp_input5, input5, _MM_CMPINT_EQ),\
		                    input5_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
		input5 = tmp_input5;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,\
		                                      5, 4, 7, 6, 1, 0, 3, 2);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMin5 = MIN_FKT( input5,permNeigh5);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i permNeighMax5 = MAX_FKT(permNeigh5, input5);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);\
		__m512i tmp_input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xCCCC, permNeighMax5);\
	    if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
		            		_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                    input_val);\
		    input2_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input2_val),\
		            		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                    input2_val);\
		    input3_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input3_val),\
		            		_mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                    input3_val);\
		    input4_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input4_val),\
		            		_mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                   input4_val);\
		    input5_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input5_val),\
		            		_mm512_cmp_epi32_mask(tmp_input5, input5, _MM_CMPINT_EQ),\
		                   input5_val);\
	    } \
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
		input5 = tmp_input5;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,\
		                                      6, 7, 4, 5, 2, 3, 0, 1);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMin5 = MIN_FKT( input5,permNeigh5);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i permNeighMax5 = MAX_FKT(permNeigh5, input5);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);\
		__m512i tmp_input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xAAAA, permNeighMax5);\
		if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input_val),\
		            		_mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                    input_val);\
		    input2_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input2_val),\
		            		_mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                   input2_val);\
		    input3_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input3_val),\
		            		_mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                   input3_val);\
		    input4_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input4_val),\
		            		_mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                    input4_val);\
		    input5_val = _mm512_mask_mov_epi32(\
		            		_mm512_permutexvar_epi32(idxNoNeigh, input5_val),\
		           		 	_mm512_cmp_epi32_mask(tmp_input5, input5, _MM_CMPINT_EQ),\
		                    input5_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
		input5 = tmp_input5;\
	}\
}

#define sortingnetwork_sort_x32x96_body(T, REG, MIN_FKT, MAX_FKT) \
	{\
		__m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,\
		                                      8, 9, 10, 11, 12, 13, 14, 15);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);\
		__m512i tmp_input5 = MAX_FKT(input4, permNeigh5);\
		__m512i tmp_input4 = MIN_FKT( permNeigh5,input4);\
		__m512i tmp_input6 = MAX_FKT(input3, permNeigh6);\
		__m512i tmp_input3 = MIN_FKT( permNeigh6,input3);\
		if constexpr (kv) {\
			__m512i input5_val_perm = _mm512_permutexvar_epi32(idxNoNeigh, input5_val);\
			input5_val = _mm512_mask_mov_epi32(input4_val, _mm512_cmp_epi32_mask(tmp_input5, permNeigh5, _MM_CMPINT_EQ),\
			                                   input5_val_perm);\
			input4_val = _mm512_mask_mov_epi32(input5_val_perm, _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
			                                   input4_val);\
			__m512i input6_val_perm = _mm512_permutexvar_epi32(idxNoNeigh, input6_val);\
			input6_val = _mm512_mask_mov_epi32(input3_val, _mm512_cmp_epi32_mask(tmp_input6, permNeigh6, _MM_CMPINT_EQ),\
			                                   input6_val_perm);\
			input3_val = _mm512_mask_mov_epi32(input6_val_perm, _mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
			                                   input3_val);\
		}\
		input5 = tmp_input5;\
		input4 = tmp_input4;\
		input6 = tmp_input6;\
		input3 = tmp_input3;\
	} {\
		__m512i inputCopy = input;\
		__m512i tmp_input = MIN_FKT( inputCopy,input3);\
		__m512i tmp_input3 = MAX_FKT(input3, inputCopy);\
		if constexpr (kv) {\
			__m512i input_val_copy = input_val;\
			input_val = _mm512_mask_mov_epi32(input3_val, _mm512_cmp_epi32_mask(tmp_input, inputCopy, _MM_CMPINT_EQ),\
			                                  input_val_copy);\
			input3_val = _mm512_mask_mov_epi32(input_val_copy, _mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
			                                   input3_val);\
		}\
		input = tmp_input;\
		input3 = tmp_input3;\
	} {\
		__m512i inputCopy = input2;\
		__m512i tmp_input2 = MIN_FKT( inputCopy,input4);\
		__m512i tmp_input4 = MAX_FKT(input4, inputCopy);\
		if constexpr (kv) {\
			__m512i input2_val_copy = input2_val;\
			input2_val = _mm512_mask_mov_epi32(input4_val, _mm512_cmp_epi32_mask(tmp_input2, inputCopy, _MM_CMPINT_EQ),\
			                                   input2_val_copy);\
			input4_val = _mm512_mask_mov_epi32(input2_val_copy, _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
			                                   input4_val);\
		}\
		input2 = tmp_input2;\
		input4 = tmp_input4;\
	} {\
		__m512i inputCopy = input;\
		__m512i tmp_input = MIN_FKT( inputCopy,input2);\
		__m512i tmp_input2 = MAX_FKT(input2, inputCopy);\
		if constexpr (kv) {\
			__m512i input_val_copy = input_val;\
			input_val = _mm512_mask_mov_epi32(input2_val, _mm512_cmp_epi32_mask(tmp_input, inputCopy, _MM_CMPINT_EQ),\
			                                  input_val_copy);\
			input2_val = _mm512_mask_mov_epi32(input_val_copy, _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
			                                   input2_val);\
		}\
		input = tmp_input;\
		input2 = tmp_input2;\
	} {\
		__m512i inputCopy = input3;\
		__m512i tmp_input3 = MIN_FKT( inputCopy,input4);\
		__m512i tmp_input4 = MAX_FKT(input4, inputCopy);\
		if constexpr (kv) {\
		__m512i input3_val_copy = input3_val;\
			input3_val = _mm512_mask_mov_epi32(input4_val, _mm512_cmp_epi32_mask(tmp_input3, inputCopy, _MM_CMPINT_EQ),\
			                                   input3_val_copy);\
			input4_val = _mm512_mask_mov_epi32(input3_val_copy, _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
			                                   input4_val);\
		}\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
	} {\
		__m512i inputCopy = input5;\
		__m512i tmp_input5 = MIN_FKT( inputCopy,input6);\
		__m512i tmp_input6 = MAX_FKT(input6, inputCopy);\
 		if constexpr (kv) {\
			 __m512i input5_val_copy = input5_val;\
			 input5_val = _mm512_mask_mov_epi32(input6_val, _mm512_cmp_epi32_mask(tmp_input5, inputCopy, _MM_CMPINT_EQ),\
												input5_val_copy);\
			 input6_val = _mm512_mask_mov_epi32(input5_val_copy, _mm512_cmp_epi32_mask(tmp_input6, input6, _MM_CMPINT_EQ),\
												input6_val);\
 		}\
		input5 = tmp_input5;\
		input6 = tmp_input6;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,\
		                                      15, 14, 13, 12, 11, 10, 9, 8);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMin5 = MIN_FKT( input5,permNeigh5);\
		__m512i permNeighMin6 = MIN_FKT( input6,permNeigh6);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i permNeighMax5 = MAX_FKT(permNeigh5, input5);\
		__m512i permNeighMax6 = MAX_FKT(permNeigh6, input6);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);\
		__m512i tmp_input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xFF00, permNeighMax5);\
		__m512i tmp_input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xFF00, permNeighMax6);\
		if constexpr (kv) {\
			input_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input_val), _mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
			                                  input_val);\
			input2_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input2_val), _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
			                                   input2_val);\
			input3_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input3_val), _mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
			                                   input3_val);\
			input4_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input4_val), _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
			                                   input4_val);\
			input5_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input5_val), _mm512_cmp_epi32_mask(tmp_input5, input5, _MM_CMPINT_EQ),\
			                                   input5_val);\
			input6_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input6_val), _mm512_cmp_epi32_mask(tmp_input6, input6, _MM_CMPINT_EQ),\
			                                   input6_val);\
		}\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
		input5 = tmp_input5;\
		input6 = tmp_input6;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,\
		                                      3, 2, 1, 0, 7, 6, 5, 4);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMin5 = MIN_FKT( input5,permNeigh5);\
		__m512i permNeighMin6 = MIN_FKT( input6,permNeigh6);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i permNeighMax5 = MAX_FKT(permNeigh5, input5);\
		__m512i permNeighMax6 = MAX_FKT(permNeigh6, input6);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);\
		__m512i tmp_input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xF0F0, permNeighMax5);\
		__m512i tmp_input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xF0F0, permNeighMax6);\
		if constexpr (kv) {\
			input_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input_val), _mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
			                                  input_val);\
			input2_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input2_val), _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
			                                   input2_val);\
			input3_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input3_val), _mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
			                                   input3_val);\
			input4_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input4_val), _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
			                                   input4_val);\
			input5_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input5_val), _mm512_cmp_epi32_mask(tmp_input5, input5, _MM_CMPINT_EQ),\
			                                   input5_val);\
			input6_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input6_val), _mm512_cmp_epi32_mask(tmp_input6, input6, _MM_CMPINT_EQ),\
			                                   input6_val);\
		}\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
		input5 = tmp_input5;\
		input6 = tmp_input6;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,\
		                                      5, 4, 7, 6, 1, 0, 3, 2);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMin5 = MIN_FKT( input5,permNeigh5);\
		__m512i permNeighMin6 = MIN_FKT( input6,permNeigh6);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i permNeighMax5 = MAX_FKT(permNeigh5, input5);\
		__m512i permNeighMax6 = MAX_FKT(permNeigh6, input6);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);\
		__m512i tmp_input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xCCCC, permNeighMax5);\
		__m512i tmp_input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xCCCC, permNeighMax6);\
		if constexpr (kv) {\
			input_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input_val), _mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
			                                  input_val);\
			input2_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input2_val), _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
			                                   input2_val);\
			input3_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input3_val), _mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
			                                   input3_val);\
			input4_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input4_val), _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
			                                   input4_val);\
			input5_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input5_val), _mm512_cmp_epi32_mask(tmp_input5, input5, _MM_CMPINT_EQ),\
			                                   input5_val);\
			input6_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input6_val), _mm512_cmp_epi32_mask(tmp_input6, input6, _MM_CMPINT_EQ),\
			                                   input6_val);\
		}\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
		input5 = tmp_input5;\
		input6 = tmp_input6;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,\
		                                      6, 7, 4, 5, 2, 3, 0, 1);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMin5 = MIN_FKT( input5,permNeigh5);\
		__m512i permNeighMin6 = MIN_FKT( input6,permNeigh6);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i permNeighMax5 = MAX_FKT(permNeigh5, input5);\
		__m512i permNeighMax6 = MAX_FKT(permNeigh6, input6);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);\
		__m512i tmp_input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xAAAA, permNeighMax5);\
		__m512i tmp_input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xAAAA, permNeighMax6);\
		if constexpr (kv) {\
			input_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input_val), _mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
			                                  input_val);\
			input2_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input2_val), _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
			                                   input2_val);\
			input3_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input3_val), _mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
			                                   input3_val);\
			input4_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input4_val), _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
			                                   input4_val);\
			input5_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input5_val), _mm512_cmp_epi32_mask(tmp_input5, input5, _MM_CMPINT_EQ),\
			                                   input5_val);\
			input6_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input6_val), _mm512_cmp_epi32_mask(tmp_input6, input6, _MM_CMPINT_EQ),\
			                                   input6_val);\
		}\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
		input5 = tmp_input5;\
		input6 = tmp_input6;\
	}\
}


#define sortingnetwork_sort_x32x112_body(T, REG, MIN_FKT, MAX_FKT) \
	{\
		__m512i idxNoNeigh = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7,\
		                                      8, 9, 10, 11, 12, 13, 14, 15);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);\
		__m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);\
		__m512i tmp_input5 = MAX_FKT(input4, permNeigh5);\
		__m512i tmp_input4 = MIN_FKT( permNeigh5,input4);\
		__m512i tmp_input6 = MAX_FKT(input3, permNeigh6);\
		__m512i tmp_input3 = MIN_FKT( permNeigh6,input3);\
		__m512i tmp_input7 = MAX_FKT(input2, permNeigh7);\
		__m512i tmp_input2 = MIN_FKT( permNeigh7,input2);\
		if constexpr (kv) {\
			__m512i input5_val_perm = _mm512_permutexvar_epi32(idxNoNeigh, input5_val);\
			input5_val = _mm512_mask_mov_epi32(input4_val, _mm512_cmp_epi32_mask(tmp_input5, permNeigh5, _MM_CMPINT_EQ ),\
											   input5_val_perm);\
			input4_val = _mm512_mask_mov_epi32(input5_val_perm, _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
												  input4_val);\
			__m512i input6_val_perm = _mm512_permutexvar_epi32(idxNoNeigh, input6_val);\
			input6_val = _mm512_mask_mov_epi32(input3_val, _mm512_cmp_epi32_mask(tmp_input6, permNeigh6, _MM_CMPINT_EQ),\
												  input6_val_perm);\
			input3_val = _mm512_mask_mov_epi32(input6_val_perm, _mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
												  input3_val);\
			__m512i input7_val_perm = _mm512_permutexvar_epi32(idxNoNeigh, input7_val);\
			input7_val = _mm512_mask_mov_epi32(input2_val, _mm512_cmp_epi32_mask(tmp_input7, permNeigh7, _MM_CMPINT_EQ),\
												  input7_val_perm);\
			input2_val = _mm512_mask_mov_epi32(input7_val_perm, _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
																				  input2_val);\
		}\
		input5 = tmp_input5;\
		input4 = tmp_input4;\
		input6 = tmp_input6;\
		input3 = tmp_input3;\
		input7 = tmp_input7;\
		input2 = tmp_input2;\
	} {\
		__m512i inputCopy = input;\
		__m512i tmp_input = MIN_FKT( inputCopy,input3);\
		__m512i tmp_input3 = MAX_FKT(input3, inputCopy);\
 		if constexpr (kv) {\
		    __m512i input_val_copy = input_val;\
		    input_val = _mm512_mask_mov_epi32(input3_val, _mm512_cmp_epi32_mask(tmp_input, inputCopy, _MM_CMPINT_EQ),\
		                                      input_val_copy);\
		    input3_val = _mm512_mask_mov_epi32(input_val_copy, _mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                                       input3_val);\
	    }\
		input = tmp_input;\
		input3 = tmp_input3;\
	} {\
		__m512i inputCopy = input2;\
		__m512i tmp_input2 = MIN_FKT( inputCopy,input4);\
		__m512i tmp_input4 = MAX_FKT(input4, inputCopy);\
 		if constexpr (kv) {\
		    __m512i input2_val_copy = input2_val;\
		    input2_val = _mm512_mask_mov_epi32(input4_val, _mm512_cmp_epi32_mask(tmp_input2, inputCopy, _MM_CMPINT_EQ),\
		                                       input2_val_copy);\
		    input4_val = _mm512_mask_mov_epi32(input2_val_copy, _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                                       input4_val);\
	    }\
		input2 = tmp_input2;\
		input4 = tmp_input4;\
	} {\
		__m512i inputCopy = input;\
		__m512i tmp_input = MIN_FKT( inputCopy,input2);\
		__m512i tmp_input2 = MAX_FKT(input2, inputCopy);\
		if constexpr (kv) {\
		    __m512i input_val_copy = input_val;\
		    input_val = _mm512_mask_mov_epi32(input2_val, _mm512_cmp_epi32_mask(tmp_input, inputCopy, _MM_CMPINT_EQ),\
		                                      input_val_copy);\
		    input2_val = _mm512_mask_mov_epi32(input_val_copy, _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                                       input2_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
	} {\
		__m512i inputCopy = input3;\
		__m512i tmp_input3 = MIN_FKT( inputCopy,input4);\
		__m512i tmp_input4 = MAX_FKT(input4, inputCopy);\
 		if constexpr (kv) {\
		    __m512i input3_val_copy = input3_val;\
		    input3_val = _mm512_mask_mov_epi32(input4_val, _mm512_cmp_epi32_mask(tmp_input3, inputCopy, _MM_CMPINT_EQ),\
		                                       input3_val_copy);\
		    input4_val = _mm512_mask_mov_epi32(input3_val_copy, _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                                       input4_val);\
	    }\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
	} {\
		__m512i inputCopy = input5;\
		__m512i tmp_input5 = MIN_FKT( inputCopy,input7);\
		__m512i tmp_input7 = MAX_FKT(input7, inputCopy);\
	    if constexpr (kv) {\
			 __m512i input5_val_copy = input5_val;\
			 input5_val = _mm512_mask_mov_epi32(input7_val, _mm512_cmp_epi32_mask(tmp_input5, inputCopy, _MM_CMPINT_EQ),\
												input5_val_copy);\
			 input7_val = _mm512_mask_mov_epi32(input5_val_copy, _mm512_cmp_epi32_mask(tmp_input7, input7, _MM_CMPINT_EQ),\
												input7_val);\
		 }\
		input5 = tmp_input5;\
		input7 = tmp_input7;\
	} {\
		__m512i inputCopy = input5;\
		__m512i tmp_input5 = MIN_FKT( inputCopy,input6);\
		__m512i tmp_input6 = MAX_FKT(input6, inputCopy);\
 		if constexpr (kv) {\
		    __m512i input5_val_copy = input5_val;\
		    input5_val = _mm512_mask_mov_epi32(input6_val, _mm512_cmp_epi32_mask(tmp_input5, inputCopy, _MM_CMPINT_EQ),\
		                                       input5_val_copy);\
		    input6_val = _mm512_mask_mov_epi32(input5_val_copy, _mm512_cmp_epi32_mask(tmp_input6, input6, _MM_CMPINT_EQ),\
		                                       input6_val);\
	    }\
		input5 = tmp_input5;\
		input6 = tmp_input6;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(7, 6, 5, 4, 3, 2, 1, 0,\
		                                      15, 14, 13, 12, 11, 10, 9, 8);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);\
		__m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMin5 = MIN_FKT( input5,permNeigh5);\
		__m512i permNeighMin6 = MIN_FKT( input6,permNeigh6);\
		__m512i permNeighMin7 = MIN_FKT( input7,permNeigh7);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i permNeighMax5 = MAX_FKT(permNeigh5, input5);\
		__m512i permNeighMax6 = MAX_FKT(permNeigh6, input6);\
		__m512i permNeighMax7 = MAX_FKT(permNeigh7, input7);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xFF00, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xFF00, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xFF00, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xFF00, permNeighMax4);\
		__m512i tmp_input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xFF00, permNeighMax5);\
		__m512i tmp_input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xFF00, permNeighMax6);\
		__m512i tmp_input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xFF00, permNeighMax7);\
	    if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input_val), _mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                                      input_val);\
		    input2_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input2_val), _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                                       input2_val);\
		    input3_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input3_val), _mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                                       input3_val);\
		    input4_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input4_val), _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                                       input4_val);\
		    input5_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input5_val), _mm512_cmp_epi32_mask(tmp_input5, input5, _MM_CMPINT_EQ),\
		                                       input5_val);\
		    input6_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input6_val), _mm512_cmp_epi32_mask(tmp_input6, input6, _MM_CMPINT_EQ),\
		                                       input6_val);\
		    input7_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input7_val), _mm512_cmp_epi32_mask(tmp_input7, input7, _MM_CMPINT_EQ),\
		                                       input7_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
		input5 = tmp_input5;\
		input6 = tmp_input6;\
		input7 = tmp_input7;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32( 11, 10, 9, 8, 15, 14, 13, 12,\
		                                      3, 2, 1, 0, 7, 6, 5, 4);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);\
		__m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMin5 = MIN_FKT( input5,permNeigh5);\
		__m512i permNeighMin6 = MIN_FKT( input6,permNeigh6);\
		__m512i permNeighMin7 = MIN_FKT( input7,permNeigh7);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i permNeighMax5 = MAX_FKT(permNeigh5, input5);\
		__m512i permNeighMax6 = MAX_FKT(permNeigh6, input6);\
		__m512i permNeighMax7 = MAX_FKT(permNeigh7, input7);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xF0F0, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xF0F0, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xF0F0, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xF0F0, permNeighMax4);\
		__m512i tmp_input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xF0F0, permNeighMax5);\
		__m512i tmp_input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xF0F0, permNeighMax6);\
		__m512i tmp_input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xF0F0, permNeighMax7);\
	    if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input_val), _mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                                      input_val);\
		    input2_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input2_val), _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                                       input2_val);\
		    input3_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input3_val), _mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                                       input3_val);\
		    input4_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input4_val), _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                                       input4_val);\
		    input5_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input5_val), _mm512_cmp_epi32_mask(tmp_input5, input5, _MM_CMPINT_EQ),\
		                                       input5_val);\
		    input6_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input6_val), _mm512_cmp_epi32_mask(tmp_input6, input6, _MM_CMPINT_EQ),\
		                                       input6_val);\
		    input7_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input7_val), _mm512_cmp_epi32_mask(tmp_input7, input7, _MM_CMPINT_EQ),\
		                                       input7_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
		input5 = tmp_input5;\
		input6 = tmp_input6;\
		input7 = tmp_input7;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(13, 12, 15, 14, 9, 8, 11, 10,\
		                                      5, 4, 7, 6, 1, 0, 3, 2);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);\
		__m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMin5 = MIN_FKT( input5,permNeigh5);\
		__m512i permNeighMin6 = MIN_FKT( input6,permNeigh6);\
		__m512i permNeighMin7 = MIN_FKT( input7,permNeigh7);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i permNeighMax5 = MAX_FKT(permNeigh5, input5);\
		__m512i permNeighMax6 = MAX_FKT(permNeigh6, input6);\
		__m512i permNeighMax7 = MAX_FKT(permNeigh7, input7);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xCCCC, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xCCCC, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xCCCC, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xCCCC, permNeighMax4);\
		__m512i tmp_input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xCCCC, permNeighMax5);\
		__m512i tmp_input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xCCCC, permNeighMax6);\
		__m512i tmp_input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xCCCC, permNeighMax7);\
		if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input_val), _mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                                      input_val);\
		    input2_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input2_val), _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                                       input2_val);\
		    input3_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input3_val), _mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                                       input3_val);\
		    input4_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input4_val), _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                                       input4_val);\
		    input5_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input5_val), _mm512_cmp_epi32_mask(tmp_input5, input5, _MM_CMPINT_EQ),\
		                                       input5_val);\
		    input6_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input6_val), _mm512_cmp_epi32_mask(tmp_input6, input6, _MM_CMPINT_EQ),\
		                                       input6_val);\
		    input7_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input7_val), _mm512_cmp_epi32_mask(tmp_input7, input7, _MM_CMPINT_EQ),\
		                                       input7_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
		input5 = tmp_input5;\
		input6 = tmp_input6;\
		input7 = tmp_input7;\
	} {\
		__m512i idxNoNeigh = _mm512_set_epi32(14, 15, 12, 13, 10, 11, 8, 9,\
		                                      6, 7, 4, 5, 2, 3, 0, 1);\
		__m512i permNeigh = _mm512_permutexvar_epi32(idxNoNeigh, input);\
		__m512i permNeigh2 = _mm512_permutexvar_epi32(idxNoNeigh, input2);\
		__m512i permNeigh3 = _mm512_permutexvar_epi32(idxNoNeigh, input3);\
		__m512i permNeigh4 = _mm512_permutexvar_epi32(idxNoNeigh, input4);\
		__m512i permNeigh5 = _mm512_permutexvar_epi32(idxNoNeigh, input5);\
		__m512i permNeigh6 = _mm512_permutexvar_epi32(idxNoNeigh, input6);\
		__m512i permNeigh7 = _mm512_permutexvar_epi32(idxNoNeigh, input7);\
		__m512i permNeighMin = MIN_FKT( input,permNeigh);\
		__m512i permNeighMin2 = MIN_FKT( input2,permNeigh2);\
		__m512i permNeighMin3 = MIN_FKT( input3,permNeigh3);\
		__m512i permNeighMin4 = MIN_FKT( input4,permNeigh4);\
		__m512i permNeighMin5 = MIN_FKT( input5,permNeigh5);\
		__m512i permNeighMin6 = MIN_FKT( input6,permNeigh6);\
		__m512i permNeighMin7 = MIN_FKT( input7,permNeigh7);\
		__m512i permNeighMax = MAX_FKT(permNeigh, input);\
		__m512i permNeighMax2 = MAX_FKT(permNeigh2, input2);\
		__m512i permNeighMax3 = MAX_FKT(permNeigh3, input3);\
		__m512i permNeighMax4 = MAX_FKT(permNeigh4, input4);\
		__m512i permNeighMax5 = MAX_FKT(permNeigh5, input5);\
		__m512i permNeighMax6 = MAX_FKT(permNeigh6, input6);\
		__m512i permNeighMax7 = MAX_FKT(permNeigh7, input7);\
		__m512i tmp_input = _mm512_mask_mov_epi32(permNeighMin, 0xAAAA, permNeighMax);\
		__m512i tmp_input2 = _mm512_mask_mov_epi32(permNeighMin2, 0xAAAA, permNeighMax2);\
		__m512i tmp_input3 = _mm512_mask_mov_epi32(permNeighMin3, 0xAAAA, permNeighMax3);\
		__m512i tmp_input4 = _mm512_mask_mov_epi32(permNeighMin4, 0xAAAA, permNeighMax4);\
		__m512i tmp_input5 = _mm512_mask_mov_epi32(permNeighMin5, 0xAAAA, permNeighMax5);\
		__m512i tmp_input6 = _mm512_mask_mov_epi32(permNeighMin6, 0xAAAA, permNeighMax6);\
		__m512i tmp_input7 = _mm512_mask_mov_epi32(permNeighMin7, 0xAAAA, permNeighMax7);\
		if constexpr (kv) {\
		    input_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input_val), _mm512_cmp_epi32_mask(tmp_input, input, _MM_CMPINT_EQ),\
		                                      input_val);\
		    input2_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input2_val), _mm512_cmp_epi32_mask(tmp_input2, input2, _MM_CMPINT_EQ),\
		                                       input2_val);\
		    input3_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input3_val), _mm512_cmp_epi32_mask(tmp_input3, input3, _MM_CMPINT_EQ),\
		                                       input3_val);\
		    input4_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input4_val), _mm512_cmp_epi32_mask(tmp_input4, input4, _MM_CMPINT_EQ),\
		                                       input4_val);\
		    input5_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input5_val), _mm512_cmp_epi32_mask(tmp_input5, input5, _MM_CMPINT_EQ),\
		                                       input5_val);\
		    input6_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input6_val), _mm512_cmp_epi32_mask(tmp_input6, input6, _MM_CMPINT_EQ),\
		                                       input6_val);\
		    input7_val = _mm512_mask_mov_epi32(_mm512_permutexvar_epi32(idxNoNeigh, input7_val), _mm512_cmp_epi32_mask(tmp_input7, input7, _MM_CMPINT_EQ),\
		                                       input7_val);\
	    }\
		input = tmp_input;\
		input2 = tmp_input2;\
		input3 = tmp_input3;\
		input4 = tmp_input4;\
		input5 = tmp_input5;\
		input6 = tmp_input6;\
		input7 = tmp_input7;\
	}\
}






#define sortingnetwork_sort_x32x16(T, REG, MIN_FKT, MAX_FKT) 		\
static inline void avx512_sortingnetwork_sort_ ##T ## 32x16(REG &input) { 	\
	constexpr bool kv = false;                                    	\
	REG values = input;                                             \
sortingnetwork_sort_x32x16_body(T, REG, MIN_FKT, MAX_FKT)

#define sortingnetwork_kvsort_x32x16(T, REG, MIN_FKT, MAX_FKT) 		\
static inline void avx512_sortingnetwork_kvsort_ ##T ## 32x16(REG &input, REG &values) {\
	constexpr bool kv = true;                                     	\
sortingnetwork_sort_x32x16_body(T, REG, MIN_FKT, MAX_FKT)



#define sortingnetwork_exchangesort_x32x32(T, REG, MIN_FKT, MAX_FKT) 				\
template<const bool kv>                                                             \
static inline void avx512_sortingnetwork_exchangesort_ ## T ## 32x32(REG &input1,	\
	                                                                 REG &input2,	\
																	 REG &value1,	\
																	 REG &value2){ 	\
	if constexpr (kv) {                                                             \
		(void) value1; (void) value2;                                             	\
	}                                                                         		\
sortingnetwork_exchangesort_x32x32_body(T, REG, MIN_FKT, MAX_FKT)




#define sortingnetwork_sort_x32x32(T, REG, MIN_FKT, MAX_FKT) 				\
static inline void avx512_sortingnetwork_sort_ ##T ## 32x32(REG &input, 	\
	                                                        REG &input2) { 	\
	constexpr bool kv = false;                                     			\
	REG input_val = input; REG input2_val = input;                     		\
sortingnetwork_sort_x32x32_body(T, REG, MIN_FKT, MAX_FKT)

#define sortingnetwork_kvsort_x32x32(T, REG, MIN_FKT, MAX_FKT) 					\
static inline void avx512_sortingnetwork_kvsort_ ##T ## 32x32(REG &input, 		\
															  REG &input2, 		\
	                                                          REG &input_val,	\
	                                                          REG &input2_val){	\
	constexpr bool kv = true;                                     				\
sortingnetwork_sort_x32x32_body(T, REG, MIN_FKT, MAX_FKT)


#define sortingnetwork_sort_x32x48(T, REG, MIN_FKT, MAX_FKT) 				\
static inline void avx512_sortingnetwork_sort_ ##T ## 32x48(REG &input, 	\
	                                                        REG &input2, 	\
															REG &input3) { 	\
	constexpr bool kv = false;                                     			\
	REG input_val=input,input2_val = input,input3_val=input;                \
	avx512_sortingnetwork_sort_ ## T ## 32x32(input, input2);				\
	avx512_sortingnetwork_sort_ ## T ## 32x16(input3);						\
sortingnetwork_sort_x32x48_body(T, REG, MIN_FKT, MAX_FKT)

#define sortingnetwork_kvsort_x32x48(T, REG, MIN_FKT, MAX_FKT) 					\
static inline void avx512_sortingnetwork_kvsort_ ##T ## 32x48(REG &input, 		\
															  REG &input2, 		\
															  REG &input3, 		\
	                                                          REG &input_val,	\
															  REG &input2_val,	\
	                                                          REG &input3_val){	\
	constexpr bool kv = true;                                     				\
	avx512_sortingnetwork_kvsort_ ## T ## 32x32(input, input2, input_val, input2_val);\
	avx512_sortingnetwork_kvsort_ ## T ## 32x16(input3, input3_val);\
sortingnetwork_sort_x32x48_body(T, REG, MIN_FKT, MAX_FKT)



#define sortingnetwork_sort_x32x64(T, REG, MIN_FKT, MAX_FKT) 				\
static inline void avx512_sortingnetwork_sort_ ##T ## 32x64(REG &input, 	\
	                                                        REG &input2, 	\
															REG &input3, 	\
															REG &input4) { 	\
	constexpr bool kv = false;                                     			\
	REG input_val=input,input2_val=input,input3_val=input,input4_val=input; \
	avx512_sortingnetwork_sort_ ## T ## 32x32(input, input2);				\
	avx512_sortingnetwork_sort_ ## T ## 32x32(input3, input4);				\
sortingnetwork_sort_x32x64_body(T, REG, MIN_FKT, MAX_FKT)

#define sortingnetwork_kvsort_x32x64(T, REG, MIN_FKT, MAX_FKT) 					\
static inline void avx512_sortingnetwork_kvsort_ ##T ## 32x64(REG &input, 		\
	                                                          REG &input2, 		\
															  REG &input3, 		\
															  REG &input4,  	\
															  REG &input_val, 	\
															  REG &input2_val,	\
															  REG &input3_val,	\
															  REG &input4_val) {\
	constexpr bool kv = true;                                     				\
	avx512_sortingnetwork_kvsort_ ## T ## 32x32(input, input2,         			\
		                                        input_val, input2_val);			\
	avx512_sortingnetwork_kvsort_ ## T ## 32x32(input3, input4,        			\
		                                        input3_val, input4_val);		\
sortingnetwork_sort_x32x64_body(T, REG, MIN_FKT, MAX_FKT)


#define sortingnetwork_sort_x32x80(T, REG, MIN_FKT, MAX_FKT) 				\
static inline void avx512_sortingnetwork_sort_ ##T ## 32x80(REG &input, 	\
	                                                        REG &input2, 	\
															REG &input3, 	\
															REG &input4, 	\
															REG &input5) { 	\
	constexpr bool kv = false;                                     			\
	REG input_val=input,input2_val=input,input3_val=input,input4_val=input, \
		input5_val=input;													\
	avx512_sortingnetwork_sort_ ## T ## 32x64(input, input2,        		\
		                                      input3, input4);				\
	avx512_sortingnetwork_sort_ ## T ## 32x16(input5);						\
sortingnetwork_sort_x32x80_body(T, REG, MIN_FKT, MAX_FKT)

#define sortingnetwork_kvsort_x32x80(T, REG, MIN_FKT, MAX_FKT) 				 \
static inline void avx512_sortingnetwork_kvsort_ ##T ## 32x80(REG &input, 	 \
	                                                        REG &input2, 	 \
															REG &input3, 	 \
															REG &input4, 	 \
															REG &input5,     \
															REG &input_val,  \
															REG &input2_val, \
															REG &input3_val, \
															REG &input4_val, \
															REG &input5_val){\
	constexpr bool kv = true;                                     			 \
	avx512_sortingnetwork_kvsort_ ## T ## 32x64(input, input2,         		 \
											    input3, input4,        		 \
		                                        input_val, input2_val,    	 \
		                                        input3_val, input4_val);	 \
	avx512_sortingnetwork_kvsort_ ## T ## 32x16(input5, input5_val);		 \
sortingnetwork_sort_x32x80_body(T, REG, MIN_FKT, MAX_FKT)


#define sortingnetwork_sort_x32x96(T, REG, MIN_FKT, MAX_FKT) 				\
static inline void avx512_sortingnetwork_sort_ ##T ## 32x96(REG &input, 	\
	                                                        REG &input2, 	\
															REG &input3, 	\
															REG &input4, 	\
															REG &input5, 	\
															REG &input6) { 	\
	constexpr bool kv = false;                                     			\
	REG input_val=input,input2_val=input,input3_val=input,input4_val=input, \
		input5_val=input,input6_val=input;									\
	avx512_sortingnetwork_sort_ ## T ## 32x64(input, input2,        		\
		                                      input3, input4);				\
	avx512_sortingnetwork_sort_ ## T ## 32x32(input5, input6);				\
sortingnetwork_sort_x32x96_body(T, REG, MIN_FKT, MAX_FKT)

#define sortingnetwork_kvsort_x32x96(T, REG, MIN_FKT, MAX_FKT) 				 \
static inline void avx512_sortingnetwork_kvsort_ ##T ## 32x96(REG &input, 	 \
	                                                        REG &input2, 	 \
															REG &input3, 	 \
															REG &input4, 	 \
															REG &input5,     \
															REG &input6,     \
															REG &input_val,  \
															REG &input2_val, \
															REG &input3_val, \
															REG &input4_val, \
															REG &input5_val, \
															REG &input6_val){\
	constexpr bool kv = true;                                     			 \
	avx512_sortingnetwork_kvsort_ ## T ## 32x64(input, input2,         		 \
											    input3, input4,        		 \
		                                        input_val, input2_val,    	 \
		                                        input3_val, input4_val);	 \
	avx512_sortingnetwork_kvsort_ ## T ## 32x32(input5, input6,        		 \
		                                        input5_val,input6_val);		 \
sortingnetwork_sort_x32x96_body(T, REG, MIN_FKT, MAX_FKT)



#define sortingnetwork_sort_x32x112(T, REG, MIN_FKT, MAX_FKT) 				\
static inline void avx512_sortingnetwork_sort_ ##T ## 32x112(REG &input, 	\
	                                                         REG &input2, 	\
															 REG &input3, 	\
															 REG &input4, 	\
															 REG &input5, 	\
															 REG &input6, 	\
															 REG &input7) { \
	constexpr bool kv = false;                                     			\
	REG input_val=input,input2_val=input,input3_val=input,input4_val=input, \
		input5_val=input,input6_val=input,input7_val=input;					\
	avx512_sortingnetwork_sort_ ## T ## 32x64(input, input2,        		\
		                                      input3, input4);				\
	avx512_sortingnetwork_sort_ ## T ## 32x48(input5, input6,input7);		\
sortingnetwork_sort_x32x112_body(T, REG, MIN_FKT, MAX_FKT)

#define sortingnetwork_kvsort_x32x112(T, REG, MIN_FKT, MAX_FKT) 			 \
static inline void avx512_sortingnetwork_kvsort_ ##T ## 32x112(REG &input, 	 \
	                                                    	REG &input2, 	 \
															REG &input3, 	 \
															REG &input4, 	 \
															REG &input5,     \
															REG &input6,     \
															REG &input7,     \
															REG &input_val,  \
															REG &input2_val, \
															REG &input3_val, \
															REG &input4_val, \
															REG &input5_val, \
															REG &input6_val, \
															REG &input7_val){\
	constexpr bool kv = true;                                     			 \
	avx512_sortingnetwork_kvsort_ ## T ## 32x64(input, input2,         		 \
											    input3, input4,        		 \
		                                        input_val, input2_val,    	 \
		                                        input3_val, input4_val);	 \
	avx512_sortingnetwork_kvsort_ ## T ## 32x48(input5, input6, input7,      \
		                                        input5_val,input6_val,input7_val);\
sortingnetwork_sort_x32x112_body(T, REG, MIN_FKT, MAX_FKT)




sortingnetwork_sort_x32x16(i, __m512i, _mm512_min_epi32, _mm512_max_epi32)
sortingnetwork_kvsort_x32x16(i, __m512i, _mm512_min_epi32, _mm512_max_epi32)
sortingnetwork_sort_x32x16(u, __m512i, _mm512_min_epu32, _mm512_max_epu32)
sortingnetwork_kvsort_x32x16(u, __m512i, _mm512_min_epu32, _mm512_max_epu32)
sortingnetwork_sort_x32x16(f, __m512, _mm512_min_ps, _mm512_max_ps)
// sortingnetwork_kvsort_x32x16(f, __m512, _mm512_min_ps, _mm512_max_ps)

sortingnetwork_exchangesort_x32x32(i, __m512i, _mm512_min_epi32, _mm512_max_epi32)
sortingnetwork_exchangesort_x32x32(u, __m512i, _mm512_min_epu32, _mm512_max_epu32)
//sortingnetwork_exchangesort_x32x32(f, __m512, _mm512_min_ps, _mm512_max_ps)


sortingnetwork_sort_x32x32(i, __m512i, _mm512_min_epi32, _mm512_max_epi32)
sortingnetwork_kvsort_x32x32(i, __m512i, _mm512_min_epi32, _mm512_max_epi32)
sortingnetwork_sort_x32x32(u, __m512i, _mm512_min_epu32, _mm512_max_epu32)
sortingnetwork_kvsort_x32x32(u, __m512i, _mm512_min_epu32, _mm512_max_epu32)
//sortingnetwork_sort_x32x32(f, __m512, _mm512_min_ps, _mm512_max_ps)
//sortingnetwork_kvsort_x32x32(f, __m512, _mm512_min_ps, _mm512_max_ps)


sortingnetwork_sort_x32x48(i, __m512i, _mm512_min_epi32, _mm512_max_epi32)
sortingnetwork_kvsort_x32x48(i, __m512i, _mm512_min_epi32, _mm512_max_epi32)
sortingnetwork_sort_x32x48(u, __m512i, _mm512_min_epu32, _mm512_max_epu32)
sortingnetwork_kvsort_x32x48(u, __m512i, _mm512_min_epu32, _mm512_max_epu32)
//sortingnetwork_sort_x32x48(f, __m512, _mm512_min_ps, _mm512_max_ps)
//sortingnetwork_kvsort_x32x48(f, __m512, _mm512_min_ps, _mm512_max_ps)


sortingnetwork_sort_x32x64(i, __m512i, _mm512_min_epi32, _mm512_max_epi32)
sortingnetwork_kvsort_x32x64(i, __m512i, _mm512_min_epi32, _mm512_max_epi32)
sortingnetwork_sort_x32x64(u, __m512i, _mm512_min_epu32, _mm512_max_epu32)
sortingnetwork_kvsort_x32x64(u, __m512i, _mm512_min_epu32, _mm512_max_epu32)
//sortingnetwork_sort_x32x64(f, __m512, _mm512_min_ps, _mm512_max_ps)
//sortingnetwork_kvsort_x32x64(f, __m512, _mm512_min_ps, _mm512_max_ps)


sortingnetwork_sort_x32x80(i, __m512i, _mm512_min_epi32, _mm512_max_epi32)
sortingnetwork_kvsort_x32x80(i, __m512i, _mm512_min_epi32, _mm512_max_epi32)
sortingnetwork_sort_x32x80(u, __m512i, _mm512_min_epu32, _mm512_max_epu32)
sortingnetwork_kvsort_x32x80(u, __m512i, _mm512_min_epu32, _mm512_max_epu32)
//sortingnetwork_sort_x32x80(f, __m512, _mm512_min_ps, _mm512_max_ps)
//sortingnetwork_kvsort_x32x80(f, __m512, _mm512_min_ps, _mm512_max_ps)


sortingnetwork_sort_x32x96(i, __m512i, _mm512_min_epi32, _mm512_max_epi32)
sortingnetwork_kvsort_x32x96(i, __m512i, _mm512_min_epi32, _mm512_max_epi32)
sortingnetwork_sort_x32x96(u, __m512i, _mm512_min_epu32, _mm512_max_epu32)
sortingnetwork_kvsort_x32x96(u, __m512i, _mm512_min_epu32, _mm512_max_epu32)
//sortingnetwork_sort_x32x96(f, __m512, _mm512_min_ps, _mm512_max_ps)
//sortingnetwork_kvsort_x32x96(f, __m512, _mm512_min_ps, _mm512_max_ps)




#define avx512_sortingnetwork_small(T, N, REG)						\
[[nodiscard]] static bool avx512_sortingnetwork_small_ ## N(N* array,	\
			const size_t element_count) noexcept {						\
	if (element_count <= 1) { return true; }							\
	constexpr size_t s = 16;											\
	const uint32_t full_vec_count = element_count/s;					\
	const uint32_t last_vec_size = element_count-(full_vec_count*s);	\
	const uint32_t last_vec_flag = last_vec_size > 0;					\
	if (full_vec_count > 6) { return false; }							\
	REG d[16];															\
	for(uint32_t i=0; i<full_vec_count; ++i) {							\
		d[i] = *((REG *)(array + s*i));									\
	}                                                      				\
    N tmp[s];                                           				\
	if (last_vec_size) {                                   				\
		for(uint32_t i=0; i<last_vec_size; ++i) {              			\
    		tmp[i] = array[full_vec_count*s + i];                       \
		}                                                     			\
		d[full_vec_count] = *((REG *)(tmp));							\
	}																	\
	switch (full_vec_count+last_vec_flag) {								\
		case 1 : avx512_sortingnetwork_sort_ ## T ## 32x16 (d[0]); goto cleanup;\
		case 2 : avx512_sortingnetwork_sort_ ## T ## 32x32 (d[0],d[1]); goto cleanup;\
		case 3 : avx512_sortingnetwork_sort_ ## T ## 32x48 (d[0],d[1],d[2]); goto cleanup;\
		case 4 : avx512_sortingnetwork_sort_ ## T ## 32x64 (d[0],d[1],d[2],d[3]); goto cleanup;\
		case 5 : avx512_sortingnetwork_sort_ ## T ## 32x80 (d[0],d[1],d[2],d[3],d[4]); goto cleanup;\
		case 6 : avx512_sortingnetwork_sort_ ## T ## 32x96 (d[0],d[1],d[2],d[3],d[4],d[5]); goto cleanup;\
		default:														\
			return false;												\
	}																	\
	cleanup:															\
	for(uint32_t i=0; i<full_vec_count; ++i) {							\
		*((REG *)(array + s*i)) = d[i];									\
	}																	\
	if (last_vec_size) {												\
		d[full_vec_count] = *((REG *)(tmp));							\
		for(uint32_t i=0; i<last_vec_size; ++i) {              			\
    		array[full_vec_count*s + i] = tmp[i];                       \
		}                                                     			\
	}																	\
	return true;														\
}

avx512_sortingnetwork_small(u, uint32_t, __m512i);
avx512_sortingnetwork_small(i, int32_t, __m512i);
//avx512_sortingnetwork_small(f, float, __m512);
#endif
