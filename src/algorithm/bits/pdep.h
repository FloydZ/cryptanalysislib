#pragma once 

#include <stdint.h>
#include <immintrin.h>

#include "popcount.h"

using namespace cryptanalysislib::popcount;

// TODO tests and benches

/// Source:  https://github.com/WojciechMula/toys/blob/master/simd-pdep-pext/pdep_avx2.cpp
template <const uint32_t MAX_MASK_BITS, 
          const bool EARLY_EXIT>
void avx2_pdep_u32_reference(const uint32_t* data_arr,
                             const uint32_t* mask_arr,
                             uint32_t* out_arr,
                             const size_t n) {
    static_assert(MAX_MASK_BITS > 0);
    static_assert(MAX_MASK_BITS <= 32);

    const __m256i one  = _mm256_set1_epi32(1);
    const __m256i zero = _mm256_set1_epi32(0);
    for (size_t i=0; i < n; i += 8) {
        __m256i data = _mm256_loadu_si256((const __m256i*)(&data_arr[i]));
        __m256i mask = _mm256_loadu_si256((const __m256i*)(&mask_arr[i]));
        __m256i out  = _mm256_set1_epi32(0);

        __m256i bit = one;

        /*  for m = 0 .. 31 loop
                if mask[m] == 1 then
                    out[m] = data[k]
                    k := k + 1
                fi
            end
        */
        for (int j=0; j < MAX_MASK_BITS; j++) {
            // 1. isolate the first non-zoro bit set of mask (at m)
            //                                                       mask = [0101_1001_1100_0000|0000_1110_1100_1000|...]
            const __m256i m0 = _mm256_sub_epi32(mask, one); // m0   = [0101_1001_1011_1111|0000_1110_1100_0111|...]
            const __m256i m1 = _mm256_and_si256(mask, m0);  // m1   = [0101_1001_1000_0000|0000_1110_1100_0000|...]
            const __m256i m2 = _mm256_xor_si256(mask, m1);  // m2   = [0000_0000_0100_0000|0000_0000_0000_1000|...]

            // 2. isolate k-th bit from data                   data = [1100_0000_1111_1110|0000_0000_1000_0000|...]
            //                                                  bit = [0000_0000_0001_0000]0000_0000_0001_0000|...]
            const __m256i d0 = _mm256_and_si256(data, bit); //   d0 = [0000_0000_0001_0000|0000_0000_0000_0000|...]
                                                            //                      ^                   ^
            // 4. fill word with *negation* of data bit
            const __m256i d1 = _mm256_cmpeq_epi32(d0, zero);//   d1 = [0000_0000_0000_0000|1111_1111_1111_1111|...]

            // 5. keep the mask bit, iff data[k] == 1
            const __m256i m3 = _mm256_andnot_si256(d1, m2); //   m3 = [0000_0000_0100_0000|0000_0000_0000_0000|...]

            // 6. update the out
            out = _mm256_or_si256(out, m3);
            mask = m1;

            // 7. the next bit in data to check
            bit = _mm256_add_epi32(bit, bit);

            // 8. all are zeros?
            if (EARLY_EXIT && _mm256_testc_si256(zero, mask)) {
                break;
            }
        }

        _mm256_storeu_si256((__m256i*)(&out_arr[i]), out);
    }
}

/// Source: https://github.com/WojciechMula/toys/blob/master/simd-pdep-pext/pdep_avx512.cpp
template <const uint32_t MAX_MASK_BITS,
          const bool EARLY_EXIT>
void avx512_pdep_u32_reference(const uint32_t* data_arr,
                               const uint32_t* mask_arr,
                               uint32_t* out_arr,
                               const size_t n) {
    static_assert(MAX_MASK_BITS > 0);
    static_assert(MAX_MASK_BITS <= 32);

    const __m512i one  = _mm512_set1_epi32(1);
    const __m512i zero = _mm512_set1_epi32(0);
    for (size_t i=0; i < n; i += 16) {
        __m512i data = _mm512_loadu_si512((const __m512i*)(&data_arr[i]));
        __m512i mask = _mm512_loadu_si512((const __m512i*)(&mask_arr[i]));
        __m512i out  = _mm512_set1_epi32(0);

        __m512i bit = one;

        /*  for m = 0 .. 31 loop
                if mask[m] == 1 then
                    out[m] = data[k]
                    k := k + 1          -- invariant: k is never greater than m
                fi
            end
        */
        for (int j=0; j < MAX_MASK_BITS; j++) {
            // 1. isolate the first non-zoro bit set of mask (at m)

            //                                                 mask = [0101_1001_1100_0000|0000_1110_1100_1000|...]
            const __m512i m0 = _mm512_sub_epi32(mask, one); // m0   = [0101_1001_1011_1111|0000_1110_1100_0111|...]
            const __m512i m1 = _mm512_and_si512(mask, m0);  // m1   = [0101_1001_1000_0000|0000_1110_1100_0000|...]
            const __m512i m2 = _mm512_xor_si512(mask, m1);  // m2   = [0000_0000_0100_0000|0000_0000_0000_1000|...]
            // the above and & xor should be fused to a single ternarylogic instruction


            // 2. isolate k-th bit from data                   data = [1100_0000_1111_1110|0000_0000_1000_0000|...]
            //                                                  bit = [0000_0000_0001_0000]0000_0000_0001_0000|...]
            const __m512i d0 = _mm512_and_si512(data, bit); //   d0 = [0000_0000_0001_0000|0000_0000_0000_0000|...]
                                                            //                      ^                   ^
            // 4. move k-th bit to n-th position, possible since k <= n
            const __m512i d1 = _mm512_add_epi32(d0, m0);    //   d1 = [0101_1001_1100_1111|0000_1110_1100_0111|...]
            const __m512i d2 = _mm512_and_si512(d1, m2);    //   d2 = [0000_0000_0100_0000|0000_0000_0000_0000|...]

            // 6. update the out
            out = _mm512_or_si512(out, d2);
            // the above and & or should be fused to a single ternarylogic instruction
            mask = m1;

            // 7. the next bit in data to check
            bit = _mm512_add_epi32(bit, bit);

            // 8. all are zeros?
            if (EARLY_EXIT && (_mm512_cmpeq_epi32_mask(zero, mask) == 0xffff)) {
                break;
            }
        }

        _mm512_storeu_si512((__m512i*)(&out_arr[i]), out);
    }
}

unsigned int pdep32_emu(unsigned int v, unsigned int m) {
	unsigned int ret = 0, pc = popcount(m);
	switch (pc) {
		case 0:
			ret = 0;
			break;
		case 1:
			ret = (v & 1) << _tzcnt_u32(m);
			break;
		case 2:
			ret = (((v << (32 - pc)) & 0x80000000) >> _lzcnt_u32(m)) | ((v & 1) << _tzcnt_u32(m));
			break;
		case 3:
		case 4:
		case 5:
		case 6:
		case 7:
		case 8: 
		case 9:
		case 10: 
		case 11:
		case 12:
		case 13: {
			unsigned int lsb = 0, msb = 0;
			unsigned int v1 = v << (32 - pc);
			for (unsigned int i = 0; i < pc / 2  ; i++) {
				const unsigned int tz = _tzcnt_u32(m);
				const unsigned int lz = _lzcnt_u32(m);
				m &= ~((0x80000000 >> lz) | (1 << tz));
				msb = (v1 & 0x80000000) >> lz;
				lsb = (v & 1) << tz;
				ret |= (msb | lsb);
				v >>= 1;
				v1 <<= 1;
			}
			ret |= ((pc & 1) & v) << _tzcnt_u32(m);
			break;
		}
		default: {
			__m128i mtwo	= _mm_set1_epi64x((~0ULL) - 1);
			__m128i mm		= _mm_cvtsi32_si128(~m);
			__m128i bit0	= _mm_clmulepi64_si128(mm, mtwo, 0);
					mm		= _mm_and_si128(mm, bit0);
			__m128i bit1	= _mm_clmulepi64_si128(mm, mtwo, 0);
					mm		= _mm_and_si128(mm, bit1);
			__m128i bit2	= _mm_clmulepi64_si128(mm, mtwo, 0);
					mm		= _mm_and_si128(mm, bit2);
			__m128i bit3	= _mm_clmulepi64_si128(mm, mtwo, 0);
					mm		= _mm_and_si128(mm, bit3);
			__m128i bit4	= _mm_sub_epi64(_mm_setzero_si128(), mm);
					bit4	= _mm_add_epi64(bit4, bit4);
			__m128i a		= _mm_cvtsi32_si128(_bzhi_u32(v, pc));

					bit4	= _mm_srli_epi64(bit4, 16);
					a		= _mm_add_epi64(_mm_andnot_si128(bit4, a),_mm_slli_epi64(_mm_and_si128(bit4, a), 16));
					bit3	= _mm_srli_epi64(bit3, 8);
					a		= _mm_add_epi64(_mm_andnot_si128(bit3, a),_mm_slli_epi64(_mm_and_si128(bit3, a), 8));
					bit2	= _mm_srli_epi64(bit2, 4);
					a		= _mm_add_epi64(_mm_andnot_si128(bit2, a),_mm_slli_epi64(_mm_and_si128(bit2, a), 4));
					bit1	= _mm_srli_epi64(bit1, 2);
					a		= _mm_add_epi64(_mm_andnot_si128(bit1, a),_mm_slli_epi64(_mm_and_si128(bit1, a), 2));
					bit0	= _mm_srli_epi64(bit0, 1);
					a		= _mm_add_epi64(_mm_andnot_si128(bit0, a),_mm_slli_epi64(_mm_and_si128(bit0, a), 1));
			ret = _mm_cvtsi128_si32(a);
		}
		break;
	}
	return ret;
};
