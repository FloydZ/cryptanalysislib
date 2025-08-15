#pragma once

#include <stdint.h>
#include <immintrin.h>

#include "popcount.h"

using namespace cryptanalysislib::popcount;

// TODO tests and benches

/// Source: https://github.com/WojciechMula/toys/blob/master/simd-pdep-pext/pext_avx2.cpp
/// \param
template <const uint32_t MAX_MASK_BITS,
          const bool EARLY_EXIT>
void avx2_pext_u32_reference(const uint32_t* data_arr,
                             const uint32_t* mask_arr,
                             uint32_t* out_arr,
                             size_t n) {
    static_assert(MAX_MASK_BITS > 0);
    static_assert(MAX_MASK_BITS <= 32);

    const __m256i one  = _mm256_set1_epi32(1);
    const __m256i zero = _mm256_set1_epi32(0);
    for (size_t i=0; i < n; i += 8) {
        __m256i data = _mm256_loadu_si256((const __m256i*)(&data_arr[i]));
        __m256i mask = _mm256_loadu_si256((const __m256i*)(&mask_arr[i]));
        __m256i out  = _mm256_set1_epi32(0);

        __m256i bit = one;

        for (int j=0; j < MAX_MASK_BITS; j++) {
            // 1. isolate the first non-zero bit set of mask
            const __m256i m0 = _mm256_sub_epi32(mask, one);
            const __m256i m1 = _mm256_and_si256(mask, m0);
            const __m256i m2 = _mm256_xor_si256(mask, m1);

            // 2. isolate that bit from data word
            const __m256i d0 = _mm256_and_si256(data, m2);

            // 3. move that bit on the next position in out
            const __m256i o0 = _mm256_min_epu32(d0, bit);
            out = _mm256_or_si256(out, o0);

            // 4. reset selected bit in mask (we already done it)
            mask = m1;

            // 5. advance to the next bit in out
            bit = _mm256_add_epi32(bit, bit);

            // 6. all are zeros?
            if (EARLY_EXIT && _mm256_testc_si256(zero, mask)) {
                break;
            }
        }

        _mm256_storeu_si256((__m256i*)(&out_arr[i]), out);
    }
}

/// Source: https://github.com/WojciechMula/toys/blob/master/simd-pdep-pext/pext_avx512.cpp
template <const uint32_t MAX_MASK_BITS,
          const bool EARLY_EXIT>
void avx512_pext_u32_reference(const uint32_t* data_arr,
                               const uint32_t* mask_arr,
                               uint32_t* out_arr,
                               size_t n) {
    static_assert(MAX_MASK_BITS > 0);
    static_assert(MAX_MASK_BITS <= 32);

    const __m512i one  = _mm512_set1_epi32(1);
    const __m512i zero = _mm512_set1_epi32(0);
    for (size_t i=0; i < n; i += 16) {
        __m512i data = _mm512_loadu_si512((const __m512i*)(&data_arr[i]));
        __m512i mask = _mm512_loadu_si512((const __m512i*)(&mask_arr[i]));
        __m512i out  = _mm512_set1_epi32(0);

        __m512i bit = one;

        for (int j=0; j < MAX_MASK_BITS; j++) {
            // 1. isolate the first bit set of mask

            //                                                 mask = [0101_1001_1100_0000|...]
            const __m512i m0 = _mm512_sub_epi32(mask, one); // m0   = [0101_1001_1011_1111|...]
            const __m512i m1 = _mm512_and_si512(mask, m0);  // m1   = [0101_1001_1000_0000|...]
            const __m512i m2 = _mm512_xor_si512(mask, m1);  // m2   = [0000_0000_0100_0000|...]
            // the above and & xor should be fused to a single ternarylogic instruction

            // 2. isolate that bit from data                   data = [1100_0000_1111_0000|...]
            const __m512i d0 = _mm512_and_si512(data, m2);  //   d0 = [0000_0000_0100_0000|...]

            // 3. move that bit on the next position in out
            const __m512i o0 = _mm512_min_epu32(d0, bit);   //   o0 = [0000_0000_0000_0001|...]
            out = _mm512_or_si512(out, o0);

            // 4. reset selected bit in mask
            mask = m1;

            // 5. the next bit to set
            bit = _mm512_add_epi32(bit, bit);

            // 6. all are zeros?
            if (EARLY_EXIT && (_mm512_cmpeq_epi32_mask(zero, mask) == 0xffff)) {
                break;
            }
        }

        _mm512_storeu_si512((__m512i*)(&out_arr[i]), out);
    }
}


/// TODO
unsigned int pext32_emu(unsigned int v, unsigned int m) {
	uint32_t ret = 0, pc = popcount(m);
	switch (pc) {
		case 0:
			ret = 0;
			break;
		case 1:
			ret = (v & m) != 0;
			break;
		case 2: {
				unsigned int msb = _bextr_u32(v, (31 - _lzcnt_u32(m)), 1);
				unsigned int lsb = _bextr_u32(v, _tzcnt_u32(m), 1);
				ret = (msb << 1) | lsb;
			}
		   break;
		case 3: {
				const unsigned int lz = 31 - _lzcnt_u32(m);
				const unsigned int tz = _tzcnt_u32(m);
				unsigned int msb = _bextr_u32(v, lz, 1);
				unsigned int lsb = _bextr_u32(v, tz, 1);
				m = _blsr_u32(m);
				unsigned int csb = _bextr_u32(v, _tzcnt_u32(m), 1);
				ret = (msb << 2) | (csb << 1) | lsb;
			}
			break;
		case 4: {
				const unsigned int lz = 31 - _lzcnt_u32(m);
				const unsigned int tz = _tzcnt_u32(m);
				unsigned int msb1 = _bextr_u32(v, lz, 1);
				unsigned int lsb1 = _bextr_u32(v, tz, 1);
				m &= ~((1 << lz) | (1 << tz));
				unsigned int msb0 = _bextr_u32(v, 31 - _lzcnt_u32(m), 1);
				unsigned int lsb0 = _bextr_u32(v, _tzcnt_u32(m), 1);
				ret = (msb1 << 3) | (msb0 << 2) | (lsb0 << 1) | lsb1;
			break;
		}
		default: {
			__m128i mm		= _mm_cvtsi32_si128(~m);
			__m128i mtwo	= _mm_set1_epi64x((~0ULL) - 1);
			__m128i clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
	unsigned int	bit0	= _mm_cvtsi128_si32(clmul);
	unsigned int	a		= v & m;
					a		= (~bit0 & a) | ((bit0 & a) >> 1);
					mm		= _mm_and_si128(mm, clmul);
					clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
	unsigned int	bit1	= _mm_cvtsi128_si32(clmul);
					a		= (~bit1 & a) | ((bit1 & a) >> 2);
					mm		= _mm_and_si128(mm, clmul);
					clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
	unsigned int	bit2	= _mm_cvtsi128_si32(clmul);
					a		= (~bit2 & a) | ((bit2 & a) >> 4);
					mm		= _mm_and_si128(mm, clmul);
					clmul	= _mm_clmulepi64_si128(mm, mtwo, 0);
	unsigned int	bit3	= _mm_cvtsi128_si32(clmul);
					a		= (~bit3 & a) | ((bit3 & a) >> 8);
					mm		= _mm_and_si128(mm, clmul);
					clmul	= _mm_sub_epi64(_mm_setzero_si128(), mm);
	unsigned int	bit4	= _mm_cvtsi128_si32(clmul);
					bit4	+= bit4;
					ret		= (unsigned int)((~bit4 & a) | ((bit4 & a) >> 16));
			break;
		}
	}
	return ret;
};
