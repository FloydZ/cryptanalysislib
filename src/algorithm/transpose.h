#ifndef CRYPTANALYSISLIB_ALGORITHM_TRANSPOSE_H
#define CRYPTANALYSISLIB_ALGORITHM_TRANSPOSE_H

#include <cstdint>
#include <cstdlib>

/// Transpose 8x8 bit array packed into a single quadword
/// 		   input
///  0 						 7 bit
///  [a0,a1,a2,a3,a4,a5,a6,a7] 7
///  [b0,b1,b2,b3,b4,b5,b6,b7] 15
///  [c0,c1,c2,c3,c4,c5,c6,c7] 23
///  [d0,d1,d2,d3,d4,d5,d6,d7]
///  [e0,e1,e2,e3,e4,e5,e6,e7]
///  [f0,f1,f2,f3,f4,f5,f6,f7]
///  [g0,g1,g2,g3,g4,g5,g6,g7]
///  [h0,h1,h2,h3,h4,h5,h6,h7] 63
///
/// 			output
///  [a0,b0,c0,d0,e0,f0,g0,h0] 7 bit
///  [a1,b1,c1,d1,e1,f1,g1,h1] 15
///  [a2,b2,c2,d2,e2,f2,g2,h2] 23
///  [a3,b3,c3,d3,e3,f3,g3,h3]
///  [a4,b4,c4,d4,e4,f4,g4,h4]
///  [a5,b5,c5,d5,e5,f5,g5,h5]
///  [a6,b6,c6,d6,e6,f6,g6,h6]
///  [a7,b7,c7,d7,e7,f7,g7,h7] 63
constexpr inline uint64_t transpose_b8x8(const uint64_t x_) noexcept {
	uint64_t x = x_, t;
    t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AALL;
    x = x ^ t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCLL;
    x = x ^ t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0LL;
    x = x ^ t ^ (t << 28);
	return x;
}

/// Transpose 8x8 bit array along the diagonal from upper right
/// 		   input
///  0 						 7 bit
///  [a0,a1,a2,a3,a4,a5,a6,a7] 7
///  [b0,b1,b2,b3,b4,b5,b6,b7] 15
///  [c0,c1,c2,c3,c4,c5,c6,c7] 23
///  [d0,d1,d2,d3,d4,d5,d6,d7]
///  [e0,e1,e2,e3,e4,e5,e6,e7]
///  [f0,f1,f2,f3,f4,f5,f6,f7]
///  [g0,g1,g2,g3,g4,g5,g6,g7]
///  [h0,h1,h2,h3,h4,h5,h6,h7] 63
///
/// out
///  [h7,g7,f7,e7,d7,c7,b7,a7] 7
///  [h6,g6,f6,e6,d6,c6,b6,a6]
///  [h5,g5,f5,e5,d5,c5,b5,a5]
///  [h4,g4,f4,e4,d4,c4,b4,a4]
///  [h3,g3,f3,e3,d3,c3,b3,a3]
///  [h2,g2,f2,e2,d2,c2,b2,a2]
///  [h1,g1,f1,e1,d1,c1,b1,a1]
///  [h0,g0,f0,e0,d0,c0,b0,a0] 63 bit
constexpr inline uint64_t transpose_b8x8_be(const uint64_t x_) noexcept {
	uint64_t x = x_, t;
    t = (x ^ (x >> 9)) & 0x0055005500550055LL;
    x = x ^ t ^ (t << 9);
    t = (x ^ (x >> 18)) & 0x0000333300003333LL;
    x = x ^ t ^ (t << 18);
    t = (x ^ (x >> 36)) & 0x000000000F0F0F0FLL;
    x = x ^ t ^ (t << 36);
	return x;
}

/// \param A
/// \param m
/// \param n
/// \param B
void transpose8(uint32_t A[8],
				int m,
				int n,
				uint32_t B[8]) noexcept {
	unsigned x, y, t;

	// Load the array and pack it into x and y.

	x = (A[0] << 24) | (A[m] << 16) | (A[2 * m] << 8) | A[3 * m];
	y = (A[4 * m] << 24) | (A[5 * m] << 16) | (A[6 * m] << 8) | A[7 * m];

	t = (x ^ (x >> 7)) & 0x00AA00AA;
	x = x ^ t ^ (t << 7);
	t = (y ^ (y >> 7)) & 0x00AA00AA;
	y = y ^ t ^ (t << 7);

	t = (x ^ (x >> 14)) & 0x0000CCCC;
	x = x ^ t ^ (t << 14);
	t = (y ^ (y >> 14)) & 0x0000CCCC;
	y = y ^ t ^ (t << 14);

	t = (x & 0xF0F0F0F0) | ((y >> 4) & 0x0F0F0F0F);
	y = ((x << 4) & 0xF0F0F0F0) | (y & 0x0F0F0F0F);
	x = t;

	B[0] = x >> 24;
	B[n] = x >> 16;
	B[2 * n] = x >> 8;
	B[3 * n] = x;
	B[4 * n] = y >> 24;
	B[5 * n] = y >> 16;
	B[6 * n] = y >> 8;
	B[7 * n] = y;
}

/// inplace
/// transpose of a 64x64 matrix over gf(2)
inline void transpose_b64x64_inplace(uint64_t a[64]) noexcept {
	for (uint64_t j = 32, m = 0x00000000FFFFFFFF; j; j >>= 1, m ^= m << j) {
		for (uint64_t k = 0; k < 64; k = ((k | j) + 1) & ~j) {
			const uint64_t t = (a[k] ^ (a[k | j] >> j)) & m;
			a[k] ^= t;
			a[k | j] ^= (t << j);
		}
	}
}


/// \param dst[out]: out  data
/// \param src[in] input bytes 8x8 matrix
/// \param src_stride[in] in bytes
/// \param dst_stride[in] in bytes
void transpose_u8_8x8(uint8_t* dst,
                      const uint8_t* src,
                      const size_t src_stride,
                      const size_t dst_stride) {
    // load rows of src matrix
    const uint64_t a0 = *((uint64_t*)(src+0*src_stride));
    const uint64_t a1 = *((uint64_t*)(src+1*src_stride));
    const uint64_t a2 = *((uint64_t*)(src+2*src_stride));
    const uint64_t a3 = *((uint64_t*)(src+3*src_stride));
    const uint64_t a4 = *((uint64_t*)(src+4*src_stride));
    const uint64_t a5 = *((uint64_t*)(src+5*src_stride));
    const uint64_t a6 = *((uint64_t*)(src+6*src_stride));
    const uint64_t a7 = *((uint64_t*)(src+7*src_stride));

    // 2x2 block matrices
    const uint64_t b0 = (a0 & 0x00ff00ff00ff00ffULL) | ((a1 << 8) & 0xff00ff00ff00ff00ULL);
    const uint64_t b1 = (a1 & 0xff00ff00ff00ff00ULL) | ((a0 >> 8) & 0x00ff00ff00ff00ffULL);
    const uint64_t b2 = (a2 & 0x00ff00ff00ff00ffULL) | ((a3 << 8) & 0xff00ff00ff00ff00ULL);
    const uint64_t b3 = (a3 & 0xff00ff00ff00ff00ULL) | ((a2 >> 8) & 0x00ff00ff00ff00ffULL);
    const uint64_t b4 = (a4 & 0x00ff00ff00ff00ffULL) | ((a5 << 8) & 0xff00ff00ff00ff00ULL);
    const uint64_t b5 = (a5 & 0xff00ff00ff00ff00ULL) | ((a4 >> 8) & 0x00ff00ff00ff00ffULL);
    const uint64_t b6 = (a6 & 0x00ff00ff00ff00ffULL) | ((a7 << 8) & 0xff00ff00ff00ff00ULL);
    const uint64_t b7 = (a7 & 0xff00ff00ff00ff00ULL) | ((a6 >> 8) & 0x00ff00ff00ff00ffULL);

    // 4x4 block matrices
    const uint64_t c0 = (b0 & 0x0000ffff0000ffffULL) | ((b2 << 16) & 0xffff0000ffff0000ULL);
    const uint64_t c1 = (b1 & 0x0000ffff0000ffffULL) | ((b3 << 16) & 0xffff0000ffff0000ULL);
    const uint64_t c2 = (b2 & 0xffff0000ffff0000ULL) | ((b0 >> 16) & 0x0000ffff0000ffffULL);
    const uint64_t c3 = (b3 & 0xffff0000ffff0000ULL) | ((b1 >> 16) & 0x0000ffff0000ffffULL);
    const uint64_t c4 = (b4 & 0x0000ffff0000ffffULL) | ((b6 << 16) & 0xffff0000ffff0000ULL);
    const uint64_t c5 = (b5 & 0x0000ffff0000ffffULL) | ((b7 << 16) & 0xffff0000ffff0000ULL);
    const uint64_t c6 = (b6 & 0xffff0000ffff0000ULL) | ((b4 >> 16) & 0x0000ffff0000ffffULL);
    const uint64_t c7 = (b7 & 0xffff0000ffff0000ULL) | ((b5 >> 16) & 0x0000ffff0000ffffULL);

    // 8x8 block matrix
    const uint64_t d0 = (c0 & 0x00000000ffffffffULL) | ((c4 << 32) & 0xffffffff00000000ULL);
    const uint64_t d1 = (c1 & 0x00000000ffffffffULL) | ((c5 << 32) & 0xffffffff00000000ULL);
    const uint64_t d2 = (c2 & 0x00000000ffffffffULL) | ((c6 << 32) & 0xffffffff00000000ULL);
    const uint64_t d3 = (c3 & 0x00000000ffffffffULL) | ((c7 << 32) & 0xffffffff00000000ULL);
    const uint64_t d4 = (c4 & 0xffffffff00000000ULL) | ((c0 >> 32) & 0x00000000ffffffffULL);
    const uint64_t d5 = (c5 & 0xffffffff00000000ULL) | ((c1 >> 32) & 0x00000000ffffffffULL);
    const uint64_t d6 = (c6 & 0xffffffff00000000ULL) | ((c2 >> 32) & 0x00000000ffffffffULL);
    const uint64_t d7 = (c7 & 0xffffffff00000000ULL) | ((c3 >> 32) & 0x00000000ffffffffULL);

    // write to dst matrix
    *(uint64_t*)(dst + 0*dst_stride) = d0;
    *(uint64_t*)(dst + 1*dst_stride) = d1;
    *(uint64_t*)(dst + 2*dst_stride) = d2;
    *(uint64_t*)(dst + 3*dst_stride) = d3;
    *(uint64_t*)(dst + 4*dst_stride) = d4;
    *(uint64_t*)(dst + 5*dst_stride) = d5;
    *(uint64_t*)(dst + 6*dst_stride) = d6;
    *(uint64_t*)(dst + 7*dst_stride) = d7;
}

#ifdef USE_AVX2
#include <immintrin.h>
/// \param dst_origin[out]: output matrix
/// \param src_origin[in]: input matrix
/// \param prf_origin[in]: lookahead pointer to prefetch it
/// \param src_stride[in]:
/// \param dst_stride[in]:
static inline
void matrix_transpose_u8_32x32(uint8_t* dst_origin,
                               const uint8_t* src_origin,
                               const uint8_t* prf_origin,
                               const size_t src_stride,
                               const size_t dst_stride) {

    static const uint32_t matrix_transpose_table[] __attribute__((aligned(32))) = {
        0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15
    };
    (void)prf_origin;
    __m256i t[32];
    for (uint32_t i = 0; i < 32; i++) {
        t[i] = _mm256_loadu_si256((const __m256i *)(src_origin + i*src_stride));
    }

    #pragma unroll
    for (uint32_t i = 0; i < 32; i+=2) {
        const __m256i t0 = _mm256_unpacklo_epi8(t[i+0], t[i+1]);
        const __m256i t1 = _mm256_unpackhi_epi8(t[i+0], t[i+1]);
        t[i+0] = t0;
        t[i+1] = t1;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 32; i+=4) {
        const __m256i t0 = _mm256_unpacklo_epi16(t[i+0], t[i+2]);
        const __m256i t1 = _mm256_unpacklo_epi16(t[i+1], t[i+3]);
        const __m256i t2 = _mm256_unpackhi_epi16(t[i+0], t[i+2]);
        const __m256i t3 = _mm256_unpackhi_epi16(t[i+1], t[i+3]);
        t[i+0] = t0;
        t[i+1] = t1;
        t[i+2] = t2;
        t[i+3] = t3;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 32; i+=8) {
        const __m256i t0 = _mm256_unpacklo_epi32(t[i+0], t[i+4]);
        const __m256i t1 = _mm256_unpacklo_epi32(t[i+1], t[i+5]);
        const __m256i t2 = _mm256_unpacklo_epi32(t[i+2], t[i+6]);
        const __m256i t3 = _mm256_unpacklo_epi32(t[i+3], t[i+7]);
        const __m256i t4 = _mm256_unpackhi_epi32(t[i+0], t[i+4]);
        const __m256i t5 = _mm256_unpackhi_epi32(t[i+1], t[i+5]);
        const __m256i t6 = _mm256_unpackhi_epi32(t[i+2], t[i+6]);
        const __m256i t7 = _mm256_unpackhi_epi32(t[i+3], t[i+7]);
        t[i+0] = t0;
        t[i+1] = t1;
        t[i+2] = t2;
        t[i+3] = t3;
        t[i+4] = t4;
        t[i+5] = t5;
        t[i+6] = t6;
        t[i+7] = t7;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        const __m256i t0 = _mm256_unpacklo_epi64(t[i+ 0], t[i+ 8]);
        const __m256i t1 = _mm256_unpackhi_epi64(t[i+ 0], t[i+ 8]);
        const __m256i t2 = _mm256_unpacklo_epi64(t[i+16], t[i+24]);
        const __m256i t3 = _mm256_unpackhi_epi64(t[i+16], t[i+24]);
        t[i+ 0] = t0;
        t[i+ 8] = t1;
        t[i+16] = t2;
        t[i+24] = t3;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 16; i++) {
        const __m256i t0 = _mm256_permute2x128_si256(t[i+0], t[i+16], 0b100000);
        const __m256i t1 = _mm256_permute2x128_si256(t[i+0], t[i+16], 0b110001);
        t[i+ 0] = t0;
        t[i+16] = t1;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 16; i++) {
        const uint32_t pos = matrix_transpose_table[i];
        _mm256_storeu_si256((__m256i *)(dst_origin + i*dst_stride), t[pos]);
    }

    for (uint32_t i = 0; i < 16; i++) {
        const uint32_t pos = matrix_transpose_table[i];
        _mm256_storeu_si256((__m256i *)(dst_origin + (i+16)*dst_stride), t[pos+16]);
    }
}
#endif // USE_AVX2

#ifdef USE_NEON 

/// \param dst_origin[out]: output matrix
/// \param src_origin[in]: input matrix
/// \param prf_origin[in]: lookahead pointer to prefetch it (unused in neon impl.)
/// \param src_stride[in]: number of bytes (including alignment) in each row for source matrix
/// \param dst_stride[in]: number of bytes (including alignment) in each row for destination matrix
void matrix_transpose_u8_32x32(uint8_t* dst_origin,
                           const uint8_t* src_origin,
                           const uint8_t* prf_origin,
                           const size_t src_stride,
                           const size_t dst_stride) {
    (void)prf_origin;
    const vec256_t rnd_0_0 = *((vec256_t *)(src_origin + 0*src_stride));
    const vec256_t rnd_0_1 = *((vec256_t *)(src_origin + 1*src_stride));
    vec256_t rnd_1_0; rnd_1_0.v[0] = vtrn1q_u8(rnd_0_0.v[0], rnd_0_1.v[0]); rnd_1_0.v[1] = vtrn1q_u8(rnd_0_0.v[1], rnd_0_1.v[1]);
    vec256_t rnd_1_1; rnd_1_1.v[0] = vtrn2q_u8(rnd_0_0.v[0], rnd_0_1.v[0]); rnd_1_1.v[1] = vtrn2q_u8(rnd_0_0.v[1], rnd_0_1.v[1]);
    const vec256_t rnd_0_2 = *((vec256_t *)(src_origin + 2*src_stride));
    const vec256_t rnd_0_3 = *((vec256_t *)(src_origin + 3*src_stride));
    vec256_t rnd_1_2; rnd_1_2.v[0] = vtrn1q_u8(rnd_0_2.v[0], rnd_0_3.v[0]); rnd_1_2.v[1] = vtrn1q_u8(rnd_0_2.v[1], rnd_0_3.v[1]);
    vec256_t rnd_1_3; rnd_1_3.v[0] = vtrn2q_u8(rnd_0_2.v[0], rnd_0_3.v[0]); rnd_1_3.v[1] = vtrn2q_u8(rnd_0_2.v[1], rnd_0_3.v[1]);
    const vec256_t rnd_0_4 = *((vec256_t *)(src_origin + 4*src_stride));
    const vec256_t rnd_0_5 = *((vec256_t *)(src_origin + 5*src_stride));
    vec256_t rnd_1_4; rnd_1_4.v[0] = vtrn1q_u8(rnd_0_4.v[0], rnd_0_5.v[0]); rnd_1_4.v[1] = vtrn1q_u8(rnd_0_4.v[1], rnd_0_5.v[1]);
    vec256_t rnd_1_5; rnd_1_5.v[0] = vtrn2q_u8(rnd_0_4.v[0], rnd_0_5.v[0]); rnd_1_5.v[1] = vtrn2q_u8(rnd_0_4.v[1], rnd_0_5.v[1]);
    const vec256_t rnd_0_6 = *((vec256_t *)(src_origin + 6*src_stride));
    const vec256_t rnd_0_7 = *((vec256_t *)(src_origin + 7*src_stride));
    vec256_t rnd_1_6; rnd_1_6.v[0] = vtrn1q_u8(rnd_0_6.v[0], rnd_0_7.v[0]); rnd_1_6.v[1] = vtrn1q_u8(rnd_0_6.v[1], rnd_0_7.v[1]);
    vec256_t rnd_1_7; rnd_1_7.v[0] = vtrn2q_u8(rnd_0_6.v[0], rnd_0_7.v[0]); rnd_1_7.v[1] = vtrn2q_u8(rnd_0_6.v[1], rnd_0_7.v[1]);
    const vec256_t rnd_0_8 = *((vec256_t *)(src_origin + 8*src_stride));
    const vec256_t rnd_0_9 = *((vec256_t *)(src_origin + 9*src_stride));
    vec256_t rnd_1_8; rnd_1_8.v[0] = vtrn1q_u8(rnd_0_8.v[0], rnd_0_9.v[0]); rnd_1_8.v[1] = vtrn1q_u8(rnd_0_8.v[1], rnd_0_9.v[1]);
    vec256_t rnd_1_9; rnd_1_9.v[0] = vtrn2q_u8(rnd_0_8.v[0], rnd_0_9.v[0]); rnd_1_9.v[1] = vtrn2q_u8(rnd_0_8.v[1], rnd_0_9.v[1]);
    const vec256_t rnd_0_10 = *((vec256_t *)(src_origin + 10*src_stride));
    const vec256_t rnd_0_11 = *((vec256_t *)(src_origin + 11*src_stride));
    vec256_t rnd_1_10; rnd_1_10.v[0] = vtrn1q_u8(rnd_0_10.v[0], rnd_0_11.v[0]); rnd_1_10.v[1] = vtrn1q_u8(rnd_0_10.v[1], rnd_0_11.v[1]);
    vec256_t rnd_1_11; rnd_1_11.v[0] = vtrn2q_u8(rnd_0_10.v[0], rnd_0_11.v[0]); rnd_1_11.v[1] = vtrn2q_u8(rnd_0_10.v[1], rnd_0_11.v[1]);
    const vec256_t rnd_0_12 = *((vec256_t *)(src_origin + 12*src_stride));
    const vec256_t rnd_0_13 = *((vec256_t *)(src_origin + 13*src_stride));
    vec256_t rnd_1_12; rnd_1_12.v[0] = vtrn1q_u8(rnd_0_12.v[0], rnd_0_13.v[0]); rnd_1_12.v[1] = vtrn1q_u8(rnd_0_12.v[1], rnd_0_13.v[1]);
    vec256_t rnd_1_13; rnd_1_13.v[0] = vtrn2q_u8(rnd_0_12.v[0], rnd_0_13.v[0]); rnd_1_13.v[1] = vtrn2q_u8(rnd_0_12.v[1], rnd_0_13.v[1]);
    const vec256_t rnd_0_14 = *((vec256_t *)(src_origin + 14*src_stride));
    const vec256_t rnd_0_15 = *((vec256_t *)(src_origin + 15*src_stride));
    vec256_t rnd_1_14; rnd_1_14.v[0] = vtrn1q_u8(rnd_0_14.v[0], rnd_0_15.v[0]); rnd_1_14.v[1] = vtrn1q_u8(rnd_0_14.v[1], rnd_0_15.v[1]);
    vec256_t rnd_1_15; rnd_1_15.v[0] = vtrn2q_u8(rnd_0_14.v[0], rnd_0_15.v[0]); rnd_1_15.v[1] = vtrn2q_u8(rnd_0_14.v[1], rnd_0_15.v[1]);
    const vec256_t rnd_0_16 = *((vec256_t *)(src_origin + 16*src_stride));
    const vec256_t rnd_0_17 = *((vec256_t *)(src_origin + 17*src_stride));
    vec256_t rnd_1_16; rnd_1_16.v[0] = vtrn1q_u8(rnd_0_16.v[0], rnd_0_17.v[0]); rnd_1_16.v[1] = vtrn1q_u8(rnd_0_16.v[1], rnd_0_17.v[1]);
    vec256_t rnd_1_17; rnd_1_17.v[0] = vtrn2q_u8(rnd_0_16.v[0], rnd_0_17.v[0]); rnd_1_17.v[1] = vtrn2q_u8(rnd_0_16.v[1], rnd_0_17.v[1]);
    const vec256_t rnd_0_18 = *((vec256_t *)(src_origin + 18*src_stride));
    const vec256_t rnd_0_19 = *((vec256_t *)(src_origin + 19*src_stride));
    vec256_t rnd_1_18; rnd_1_18.v[0] = vtrn1q_u8(rnd_0_18.v[0], rnd_0_19.v[0]); rnd_1_18.v[1] = vtrn1q_u8(rnd_0_18.v[1], rnd_0_19.v[1]);
    vec256_t rnd_1_19; rnd_1_19.v[0] = vtrn2q_u8(rnd_0_18.v[0], rnd_0_19.v[0]); rnd_1_19.v[1] = vtrn2q_u8(rnd_0_18.v[1], rnd_0_19.v[1]);
    const vec256_t rnd_0_20 = *((vec256_t *)(src_origin + 20*src_stride));
    const vec256_t rnd_0_21 = *((vec256_t *)(src_origin + 21*src_stride));
    vec256_t rnd_1_20; rnd_1_20.v[0] = vtrn1q_u8(rnd_0_20.v[0], rnd_0_21.v[0]); rnd_1_20.v[1] = vtrn1q_u8(rnd_0_20.v[1], rnd_0_21.v[1]);
    vec256_t rnd_1_21; rnd_1_21.v[0] = vtrn2q_u8(rnd_0_20.v[0], rnd_0_21.v[0]); rnd_1_21.v[1] = vtrn2q_u8(rnd_0_20.v[1], rnd_0_21.v[1]);
    const vec256_t rnd_0_22 = *((vec256_t *)(src_origin + 22*src_stride));
    const vec256_t rnd_0_23 = *((vec256_t *)(src_origin + 23*src_stride));
    vec256_t rnd_1_22; rnd_1_22.v[0] = vtrn1q_u8(rnd_0_22.v[0], rnd_0_23.v[0]); rnd_1_22.v[1] = vtrn1q_u8(rnd_0_22.v[1], rnd_0_23.v[1]);
    vec256_t rnd_1_23; rnd_1_23.v[0] = vtrn2q_u8(rnd_0_22.v[0], rnd_0_23.v[0]); rnd_1_23.v[1] = vtrn2q_u8(rnd_0_22.v[1], rnd_0_23.v[1]);
    const vec256_t rnd_0_24 = *((vec256_t *)(src_origin + 24*src_stride));
    const vec256_t rnd_0_25 = *((vec256_t *)(src_origin + 25*src_stride));
    vec256_t rnd_1_24; rnd_1_24.v[0] = vtrn1q_u8(rnd_0_24.v[0], rnd_0_25.v[0]); rnd_1_24.v[1] = vtrn1q_u8(rnd_0_24.v[1], rnd_0_25.v[1]);
    vec256_t rnd_1_25; rnd_1_25.v[0] = vtrn2q_u8(rnd_0_24.v[0], rnd_0_25.v[0]); rnd_1_25.v[1] = vtrn2q_u8(rnd_0_24.v[1], rnd_0_25.v[1]);
    const vec256_t rnd_0_26 = *((vec256_t *)(src_origin + 26*src_stride));
    const vec256_t rnd_0_27 = *((vec256_t *)(src_origin + 27*src_stride));
    vec256_t rnd_1_26; rnd_1_26.v[0] = vtrn1q_u8(rnd_0_26.v[0], rnd_0_27.v[0]); rnd_1_26.v[1] = vtrn1q_u8(rnd_0_26.v[1], rnd_0_27.v[1]);
    vec256_t rnd_1_27; rnd_1_27.v[0] = vtrn2q_u8(rnd_0_26.v[0], rnd_0_27.v[0]); rnd_1_27.v[1] = vtrn2q_u8(rnd_0_26.v[1], rnd_0_27.v[1]);
    const vec256_t rnd_0_28 = *((vec256_t *)(src_origin + 28*src_stride));
    const vec256_t rnd_0_29 = *((vec256_t *)(src_origin + 29*src_stride));
    vec256_t rnd_1_28; rnd_1_28.v[0] = vtrn1q_u8(rnd_0_28.v[0], rnd_0_29.v[0]); rnd_1_28.v[1] = vtrn1q_u8(rnd_0_28.v[1], rnd_0_29.v[1]);
    vec256_t rnd_1_29; rnd_1_29.v[0] = vtrn2q_u8(rnd_0_28.v[0], rnd_0_29.v[0]); rnd_1_29.v[1] = vtrn2q_u8(rnd_0_28.v[1], rnd_0_29.v[1]);
    const vec256_t rnd_0_30 = *((vec256_t *)(src_origin + 30*src_stride));
    const vec256_t rnd_0_31 = *((vec256_t *)(src_origin + 31*src_stride));
    vec256_t rnd_1_30; rnd_1_30.v[0] = vtrn1q_u8(rnd_0_30.v[0], rnd_0_31.v[0]); rnd_1_30.v[1] = vtrn1q_u8(rnd_0_30.v[1], rnd_0_31.v[1]);
    vec256_t rnd_1_31; rnd_1_31.v[0] = vtrn2q_u8(rnd_0_30.v[0], rnd_0_31.v[0]); rnd_1_31.v[1] = vtrn2q_u8(rnd_0_30.v[1], rnd_0_31.v[1]);

    vec256_t rnd_2_0; rnd_2_0.v[0] = vtrn1q_u16(rnd_1_0.v[0], rnd_1_2.v[0]); rnd_2_0.v[1] = vtrn1q_u16(rnd_1_0.v[1], rnd_1_2.v[1]);
    vec256_t rnd_2_2; rnd_2_2.v[0] = vtrn2q_u16(rnd_1_0.v[0], rnd_1_2.v[0]); rnd_2_2.v[1] = vtrn2q_u16(rnd_1_0.v[1], rnd_1_2.v[1]);
    vec256_t rnd_2_1; rnd_2_1.v[0] = vtrn1q_u16(rnd_1_1.v[0], rnd_1_3.v[0]); rnd_2_1.v[1] = vtrn1q_u16(rnd_1_1.v[1], rnd_1_3.v[1]);
    vec256_t rnd_2_3; rnd_2_3.v[0] = vtrn2q_u16(rnd_1_1.v[0], rnd_1_3.v[0]); rnd_2_3.v[1] = vtrn2q_u16(rnd_1_1.v[1], rnd_1_3.v[1]);
    vec256_t rnd_2_4; rnd_2_4.v[0] = vtrn1q_u16(rnd_1_4.v[0], rnd_1_6.v[0]); rnd_2_4.v[1] = vtrn1q_u16(rnd_1_4.v[1], rnd_1_6.v[1]);
    vec256_t rnd_2_6; rnd_2_6.v[0] = vtrn2q_u16(rnd_1_4.v[0], rnd_1_6.v[0]); rnd_2_6.v[1] = vtrn2q_u16(rnd_1_4.v[1], rnd_1_6.v[1]);
    vec256_t rnd_2_5; rnd_2_5.v[0] = vtrn1q_u16(rnd_1_5.v[0], rnd_1_7.v[0]); rnd_2_5.v[1] = vtrn1q_u16(rnd_1_5.v[1], rnd_1_7.v[1]);
    vec256_t rnd_2_7; rnd_2_7.v[0] = vtrn2q_u16(rnd_1_5.v[0], rnd_1_7.v[0]); rnd_2_7.v[1] = vtrn2q_u16(rnd_1_5.v[1], rnd_1_7.v[1]);
    vec256_t rnd_2_8 ; rnd_2_8.v[0]  = vtrn1q_u16(rnd_1_8.v[0], rnd_1_10.v[0]); rnd_2_8.v[1]  = vtrn1q_u16(rnd_1_8.v[1], rnd_1_10.v[1]);
    vec256_t rnd_2_10; rnd_2_10.v[0] = vtrn2q_u16(rnd_1_8.v[0], rnd_1_10.v[0]); rnd_2_10.v[1] = vtrn2q_u16(rnd_1_8.v[1], rnd_1_10.v[1]);
    vec256_t rnd_2_9 ; rnd_2_9.v[0]  = vtrn1q_u16(rnd_1_9.v[0], rnd_1_11.v[0]); rnd_2_9.v[1]  = vtrn1q_u16(rnd_1_9.v[1], rnd_1_11.v[1]);
    vec256_t rnd_2_11; rnd_2_11.v[0] = vtrn2q_u16(rnd_1_9.v[0], rnd_1_11.v[0]); rnd_2_11.v[1] = vtrn2q_u16(rnd_1_9.v[1], rnd_1_11.v[1]);
    vec256_t rnd_2_12; rnd_2_12.v[0] = vtrn1q_u16(rnd_1_12.v[0], rnd_1_14.v[0]); rnd_2_12.v[1] = vtrn1q_u16(rnd_1_12.v[1], rnd_1_14.v[1]);
    vec256_t rnd_2_14; rnd_2_14.v[0] = vtrn2q_u16(rnd_1_12.v[0], rnd_1_14.v[0]); rnd_2_14.v[1] = vtrn2q_u16(rnd_1_12.v[1], rnd_1_14.v[1]);
    vec256_t rnd_2_13; rnd_2_13.v[0] = vtrn1q_u16(rnd_1_13.v[0], rnd_1_15.v[0]); rnd_2_13.v[1] = vtrn1q_u16(rnd_1_13.v[1], rnd_1_15.v[1]);
    vec256_t rnd_2_15; rnd_2_15.v[0] = vtrn2q_u16(rnd_1_13.v[0], rnd_1_15.v[0]); rnd_2_15.v[1] = vtrn2q_u16(rnd_1_13.v[1], rnd_1_15.v[1]);
    vec256_t rnd_2_16; rnd_2_16.v[0] = vtrn1q_u16(rnd_1_16.v[0], rnd_1_18.v[0]); rnd_2_16.v[1] = vtrn1q_u16(rnd_1_16.v[1], rnd_1_18.v[1]);
    vec256_t rnd_2_18; rnd_2_18.v[0] = vtrn2q_u16(rnd_1_16.v[0], rnd_1_18.v[0]); rnd_2_18.v[1] = vtrn2q_u16(rnd_1_16.v[1], rnd_1_18.v[1]);
    vec256_t rnd_2_17; rnd_2_17.v[0] = vtrn1q_u16(rnd_1_17.v[0], rnd_1_19.v[0]); rnd_2_17.v[1] = vtrn1q_u16(rnd_1_17.v[1], rnd_1_19.v[1]);
    vec256_t rnd_2_19; rnd_2_19.v[0] = vtrn2q_u16(rnd_1_17.v[0], rnd_1_19.v[0]); rnd_2_19.v[1] = vtrn2q_u16(rnd_1_17.v[1], rnd_1_19.v[1]);
    vec256_t rnd_2_20; rnd_2_20.v[0] = vtrn1q_u16(rnd_1_20.v[0], rnd_1_22.v[0]); rnd_2_20.v[1] = vtrn1q_u16(rnd_1_20.v[1], rnd_1_22.v[1]);
    vec256_t rnd_2_22; rnd_2_22.v[0] = vtrn2q_u16(rnd_1_20.v[0], rnd_1_22.v[0]); rnd_2_22.v[1] = vtrn2q_u16(rnd_1_20.v[1], rnd_1_22.v[1]);
    vec256_t rnd_2_21; rnd_2_21.v[0] = vtrn1q_u16(rnd_1_21.v[0], rnd_1_23.v[0]); rnd_2_21.v[1] = vtrn1q_u16(rnd_1_21.v[1], rnd_1_23.v[1]);
    vec256_t rnd_2_23; rnd_2_23.v[0] = vtrn2q_u16(rnd_1_21.v[0], rnd_1_23.v[0]); rnd_2_23.v[1] = vtrn2q_u16(rnd_1_21.v[1], rnd_1_23.v[1]);
    vec256_t rnd_2_24; rnd_2_24.v[0] = vtrn1q_u16(rnd_1_24.v[0], rnd_1_26.v[0]); rnd_2_24.v[1] = vtrn1q_u16(rnd_1_24.v[1], rnd_1_26.v[1]);
    vec256_t rnd_2_26; rnd_2_26.v[0] = vtrn2q_u16(rnd_1_24.v[0], rnd_1_26.v[0]); rnd_2_26.v[1] = vtrn2q_u16(rnd_1_24.v[1], rnd_1_26.v[1]);
    vec256_t rnd_2_25; rnd_2_25.v[0] = vtrn1q_u16(rnd_1_25.v[0], rnd_1_27.v[0]); rnd_2_25.v[1] = vtrn1q_u16(rnd_1_25.v[1], rnd_1_27.v[1]);
    vec256_t rnd_2_27; rnd_2_27.v[0] = vtrn2q_u16(rnd_1_25.v[0], rnd_1_27.v[0]); rnd_2_27.v[1] = vtrn2q_u16(rnd_1_25.v[1], rnd_1_27.v[1]);
    vec256_t rnd_2_28; rnd_2_28.v[0] = vtrn1q_u16(rnd_1_28.v[0], rnd_1_30.v[0]); rnd_2_28.v[1] = vtrn1q_u16(rnd_1_28.v[1], rnd_1_30.v[1]);
    vec256_t rnd_2_30; rnd_2_30.v[0] = vtrn2q_u16(rnd_1_28.v[0], rnd_1_30.v[0]); rnd_2_30.v[1] = vtrn2q_u16(rnd_1_28.v[1], rnd_1_30.v[1]);
    vec256_t rnd_2_29; rnd_2_29.v[0] = vtrn1q_u16(rnd_1_29.v[0], rnd_1_31.v[0]); rnd_2_29.v[1] = vtrn1q_u16(rnd_1_29.v[1], rnd_1_31.v[1]);
    vec256_t rnd_2_31; rnd_2_31.v[0] = vtrn2q_u16(rnd_1_29.v[0], rnd_1_31.v[0]); rnd_2_31.v[1] = vtrn2q_u16(rnd_1_29.v[1], rnd_1_31.v[1]);

    vec256_t rnd_3_0; rnd_3_0.v[0] = vtrn1q_u32(rnd_2_0.v[0], rnd_2_4.v[0]); rnd_3_0.v[1] = vtrn1q_u32(rnd_2_0.v[1], rnd_2_4.v[1]);
    vec256_t rnd_3_4; rnd_3_4.v[0] = vtrn2q_u32(rnd_2_0.v[0], rnd_2_4.v[0]); rnd_3_4.v[1] = vtrn2q_u32(rnd_2_0.v[1], rnd_2_4.v[1]);
    vec256_t rnd_3_1; rnd_3_1.v[0] = vtrn1q_u32(rnd_2_1.v[0], rnd_2_5.v[0]); rnd_3_1.v[1] = vtrn1q_u32(rnd_2_1.v[1], rnd_2_5.v[1]);
    vec256_t rnd_3_5; rnd_3_5.v[0] = vtrn2q_u32(rnd_2_1.v[0], rnd_2_5.v[0]); rnd_3_5.v[1] = vtrn2q_u32(rnd_2_1.v[1], rnd_2_5.v[1]);
    vec256_t rnd_3_2; rnd_3_2.v[0] = vtrn1q_u32(rnd_2_2.v[0], rnd_2_6.v[0]); rnd_3_2.v[1] = vtrn1q_u32(rnd_2_2.v[1], rnd_2_6.v[1]);
    vec256_t rnd_3_6; rnd_3_6.v[0] = vtrn2q_u32(rnd_2_2.v[0], rnd_2_6.v[0]); rnd_3_6.v[1] = vtrn2q_u32(rnd_2_2.v[1], rnd_2_6.v[1]);
    vec256_t rnd_3_3; rnd_3_3.v[0] = vtrn1q_u32(rnd_2_3.v[0], rnd_2_7.v[0]); rnd_3_3.v[1] = vtrn1q_u32(rnd_2_3.v[1], rnd_2_7.v[1]);
    vec256_t rnd_3_7; rnd_3_7.v[0] = vtrn2q_u32(rnd_2_3.v[0], rnd_2_7.v[0]); rnd_3_7.v[1] = vtrn2q_u32(rnd_2_3.v[1], rnd_2_7.v[1]);
    vec256_t rnd_3_8 ; rnd_3_8.v[0]  = vtrn1q_u32(rnd_2_8.v[0], rnd_2_12.v[0]); rnd_3_8.v[1]  = vtrn1q_u32(rnd_2_8.v[1], rnd_2_12.v[1]);
    vec256_t rnd_3_12; rnd_3_12.v[0] = vtrn2q_u32(rnd_2_8.v[0], rnd_2_12.v[0]); rnd_3_12.v[1] = vtrn2q_u32(rnd_2_8.v[1], rnd_2_12.v[1]);
    vec256_t rnd_3_9;  rnd_3_9.v[0] =  vtrn1q_u32(rnd_2_9.v[0], rnd_2_13.v[0]); rnd_3_9.v[1] =  vtrn1q_u32(rnd_2_9.v[1], rnd_2_13.v[1]);
    vec256_t rnd_3_13; rnd_3_13.v[0] = vtrn2q_u32(rnd_2_9.v[0], rnd_2_13.v[0]); rnd_3_13.v[1] = vtrn2q_u32(rnd_2_9.v[1], rnd_2_13.v[1]);
    vec256_t rnd_3_10; rnd_3_10.v[0] = vtrn1q_u32(rnd_2_10.v[0], rnd_2_14.v[0]); rnd_3_10.v[1] = vtrn1q_u32(rnd_2_10.v[1], rnd_2_14.v[1]);
    vec256_t rnd_3_14; rnd_3_14.v[0] = vtrn2q_u32(rnd_2_10.v[0], rnd_2_14.v[0]); rnd_3_14.v[1] = vtrn2q_u32(rnd_2_10.v[1], rnd_2_14.v[1]);
    vec256_t rnd_3_11; rnd_3_11.v[0] = vtrn1q_u32(rnd_2_11.v[0], rnd_2_15.v[0]); rnd_3_11.v[1] = vtrn1q_u32(rnd_2_11.v[1], rnd_2_15.v[1]);
    vec256_t rnd_3_15; rnd_3_15.v[0] = vtrn2q_u32(rnd_2_11.v[0], rnd_2_15.v[0]); rnd_3_15.v[1] = vtrn2q_u32(rnd_2_11.v[1], rnd_2_15.v[1]);
    vec256_t rnd_3_16; rnd_3_16.v[0] = vtrn1q_u32(rnd_2_16.v[0], rnd_2_20.v[0]); rnd_3_16.v[1] = vtrn1q_u32(rnd_2_16.v[1], rnd_2_20.v[1]);
    vec256_t rnd_3_20; rnd_3_20.v[0] = vtrn2q_u32(rnd_2_16.v[0], rnd_2_20.v[0]); rnd_3_20.v[1] = vtrn2q_u32(rnd_2_16.v[1], rnd_2_20.v[1]);
    vec256_t rnd_3_17; rnd_3_17.v[0] = vtrn1q_u32(rnd_2_17.v[0], rnd_2_21.v[0]); rnd_3_17.v[1] = vtrn1q_u32(rnd_2_17.v[1], rnd_2_21.v[1]);
    vec256_t rnd_3_21; rnd_3_21.v[0] = vtrn2q_u32(rnd_2_17.v[0], rnd_2_21.v[0]); rnd_3_21.v[1] = vtrn2q_u32(rnd_2_17.v[1], rnd_2_21.v[1]);
    vec256_t rnd_3_18; rnd_3_18.v[0] = vtrn1q_u32(rnd_2_18.v[0], rnd_2_22.v[0]); rnd_3_18.v[1] = vtrn1q_u32(rnd_2_18.v[1], rnd_2_22.v[1]);
    vec256_t rnd_3_22; rnd_3_22.v[0] = vtrn2q_u32(rnd_2_18.v[0], rnd_2_22.v[0]); rnd_3_22.v[1] = vtrn2q_u32(rnd_2_18.v[1], rnd_2_22.v[1]);
    vec256_t rnd_3_19; rnd_3_19.v[0] = vtrn1q_u32(rnd_2_19.v[0], rnd_2_23.v[0]); rnd_3_19.v[1] = vtrn1q_u32(rnd_2_19.v[1], rnd_2_23.v[1]);
    vec256_t rnd_3_23; rnd_3_23.v[0] = vtrn2q_u32(rnd_2_19.v[0], rnd_2_23.v[0]); rnd_3_23.v[1] = vtrn2q_u32(rnd_2_19.v[1], rnd_2_23.v[1]);
    vec256_t rnd_3_24; rnd_3_24.v[0] = vtrn1q_u32(rnd_2_24.v[0], rnd_2_28.v[0]); rnd_3_24.v[1] = vtrn1q_u32(rnd_2_24.v[1], rnd_2_28.v[1]);
    vec256_t rnd_3_28; rnd_3_28.v[0] = vtrn2q_u32(rnd_2_24.v[0], rnd_2_28.v[0]); rnd_3_28.v[1] = vtrn2q_u32(rnd_2_24.v[1], rnd_2_28.v[1]);
    vec256_t rnd_3_25; rnd_3_25.v[0] = vtrn1q_u32(rnd_2_25.v[0], rnd_2_29.v[0]); rnd_3_25.v[1] = vtrn1q_u32(rnd_2_25.v[1], rnd_2_29.v[1]);
    vec256_t rnd_3_29; rnd_3_29.v[0] = vtrn2q_u32(rnd_2_25.v[0], rnd_2_29.v[0]); rnd_3_29.v[1] = vtrn2q_u32(rnd_2_25.v[1], rnd_2_29.v[1]);
    vec256_t rnd_3_26; rnd_3_26.v[0] = vtrn1q_u32(rnd_2_26.v[0], rnd_2_30.v[0]); rnd_3_26.v[1] = vtrn1q_u32(rnd_2_26.v[1], rnd_2_30.v[1]);
    vec256_t rnd_3_30; rnd_3_30.v[0] = vtrn2q_u32(rnd_2_26.v[0], rnd_2_30.v[0]); rnd_3_30.v[1] = vtrn2q_u32(rnd_2_26.v[1], rnd_2_30.v[1]);
    vec256_t rnd_3_27; rnd_3_27.v[0] = vtrn1q_u32(rnd_2_27.v[0], rnd_2_31.v[0]); rnd_3_27.v[1] = vtrn1q_u32(rnd_2_27.v[1], rnd_2_31.v[1]);
    vec256_t rnd_3_31; rnd_3_31.v[0] = vtrn2q_u32(rnd_2_27.v[0], rnd_2_31.v[0]); rnd_3_31.v[1] = vtrn2q_u32(rnd_2_27.v[1], rnd_2_31.v[1]);

    vec256_t rnd_4_0; rnd_4_0.v[0] = vtrn1q_u64(rnd_3_0.v[0], rnd_3_8.v[0]); rnd_4_0.v[1] = vtrn1q_u64(rnd_3_0.v[1], rnd_3_8.v[1]);
    vec256_t rnd_4_8; rnd_4_8.v[0] = vtrn2q_u64(rnd_3_0.v[0], rnd_3_8.v[0]); rnd_4_8.v[1] = vtrn2q_u64(rnd_3_0.v[1], rnd_3_8.v[1]);
    vec256_t rnd_4_1; rnd_4_1.v[0] = vtrn1q_u64(rnd_3_1.v[0], rnd_3_9.v[0]); rnd_4_1.v[1] = vtrn1q_u64(rnd_3_1.v[1], rnd_3_9.v[1]);
    vec256_t rnd_4_9; rnd_4_9.v[0] = vtrn2q_u64(rnd_3_1.v[0], rnd_3_9.v[0]); rnd_4_9.v[1] = vtrn2q_u64(rnd_3_1.v[1], rnd_3_9.v[1]);
    vec256_t rnd_4_2 ; rnd_4_2.v[0]  = vtrn1q_u64(rnd_3_2.v[0], rnd_3_10.v[0]); rnd_4_2.v[1]  = vtrn1q_u64(rnd_3_2.v[1], rnd_3_10.v[1]);
    vec256_t rnd_4_10; rnd_4_10.v[0] = vtrn2q_u64(rnd_3_2.v[0], rnd_3_10.v[0]); rnd_4_10.v[1] = vtrn2q_u64(rnd_3_2.v[1], rnd_3_10.v[1]);
    vec256_t rnd_4_3;  rnd_4_3.v[0]  = vtrn1q_u64(rnd_3_3.v[0], rnd_3_11.v[0]); rnd_4_3.v[1]  = vtrn1q_u64(rnd_3_3.v[1], rnd_3_11.v[1]);
    vec256_t rnd_4_11; rnd_4_11.v[0] = vtrn2q_u64(rnd_3_3.v[0], rnd_3_11.v[0]); rnd_4_11.v[1] = vtrn2q_u64(rnd_3_3.v[1], rnd_3_11.v[1]);
    vec256_t rnd_4_4;  rnd_4_4.v[0]  = vtrn1q_u64(rnd_3_4.v[0], rnd_3_12.v[0]); rnd_4_4.v[1]  = vtrn1q_u64(rnd_3_4.v[1], rnd_3_12.v[1]);
    vec256_t rnd_4_12; rnd_4_12.v[0] = vtrn2q_u64(rnd_3_4.v[0], rnd_3_12.v[0]); rnd_4_12.v[1] = vtrn2q_u64(rnd_3_4.v[1], rnd_3_12.v[1]);
    vec256_t rnd_4_5;  rnd_4_5.v[0]  = vtrn1q_u64(rnd_3_5.v[0], rnd_3_13.v[0]); rnd_4_5.v[1]  = vtrn1q_u64(rnd_3_5.v[1], rnd_3_13.v[1]);
    vec256_t rnd_4_13; rnd_4_13.v[0] = vtrn2q_u64(rnd_3_5.v[0], rnd_3_13.v[0]); rnd_4_13.v[1] = vtrn2q_u64(rnd_3_5.v[1], rnd_3_13.v[1]);
    vec256_t rnd_4_6;  rnd_4_6.v[0]  = vtrn1q_u64(rnd_3_6.v[0], rnd_3_14.v[0]); rnd_4_6.v[1]  = vtrn1q_u64(rnd_3_6.v[1], rnd_3_14.v[1]);
    vec256_t rnd_4_14; rnd_4_14.v[0] = vtrn2q_u64(rnd_3_6.v[0], rnd_3_14.v[0]); rnd_4_14.v[1] = vtrn2q_u64(rnd_3_6.v[1], rnd_3_14.v[1]);
    vec256_t rnd_4_7;  rnd_4_7.v[0]  = vtrn1q_u64(rnd_3_7.v[0], rnd_3_15.v[0]); rnd_4_7.v[1]  = vtrn1q_u64(rnd_3_7.v[1], rnd_3_15.v[1]);
    vec256_t rnd_4_15; rnd_4_15.v[0] = vtrn2q_u64(rnd_3_7.v[0], rnd_3_15.v[0]); rnd_4_15.v[1] = vtrn2q_u64(rnd_3_7.v[1], rnd_3_15.v[1]);
    vec256_t rnd_4_16; rnd_4_16.v[0] = vtrn1q_u64(rnd_3_16.v[0], rnd_3_24.v[0]); rnd_4_16.v[1] = vtrn1q_u64(rnd_3_16.v[1], rnd_3_24.v[1]);
    vec256_t rnd_4_24; rnd_4_24.v[0] = vtrn2q_u64(rnd_3_16.v[0], rnd_3_24.v[0]); rnd_4_24.v[1] = vtrn2q_u64(rnd_3_16.v[1], rnd_3_24.v[1]);
    vec256_t rnd_4_17; rnd_4_17.v[0] = vtrn1q_u64(rnd_3_17.v[0], rnd_3_25.v[0]); rnd_4_17.v[1] = vtrn1q_u64(rnd_3_17.v[1], rnd_3_25.v[1]);
    vec256_t rnd_4_25; rnd_4_25.v[0] = vtrn2q_u64(rnd_3_17.v[0], rnd_3_25.v[0]); rnd_4_25.v[1] = vtrn2q_u64(rnd_3_17.v[1], rnd_3_25.v[1]);
    vec256_t rnd_4_18; rnd_4_18.v[0] = vtrn1q_u64(rnd_3_18.v[0], rnd_3_26.v[0]); rnd_4_18.v[1] = vtrn1q_u64(rnd_3_18.v[1], rnd_3_26.v[1]);
    vec256_t rnd_4_26; rnd_4_26.v[0] = vtrn2q_u64(rnd_3_18.v[0], rnd_3_26.v[0]); rnd_4_26.v[1] = vtrn2q_u64(rnd_3_18.v[1], rnd_3_26.v[1]);
    vec256_t rnd_4_19; rnd_4_19.v[0] = vtrn1q_u64(rnd_3_19.v[0], rnd_3_27.v[0]); rnd_4_19.v[1] = vtrn1q_u64(rnd_3_19.v[1], rnd_3_27.v[1]);
    vec256_t rnd_4_27; rnd_4_27.v[0] = vtrn2q_u64(rnd_3_19.v[0], rnd_3_27.v[0]); rnd_4_27.v[1] = vtrn2q_u64(rnd_3_19.v[1], rnd_3_27.v[1]);
    vec256_t rnd_4_20; rnd_4_20.v[0] = vtrn1q_u64(rnd_3_20.v[0], rnd_3_28.v[0]); rnd_4_20.v[1] = vtrn1q_u64(rnd_3_20.v[1], rnd_3_28.v[1]);
    vec256_t rnd_4_28; rnd_4_28.v[0] = vtrn2q_u64(rnd_3_20.v[0], rnd_3_28.v[0]); rnd_4_28.v[1] = vtrn2q_u64(rnd_3_20.v[1], rnd_3_28.v[1]);
    vec256_t rnd_4_21; rnd_4_21.v[0] = vtrn1q_u64(rnd_3_21.v[0], rnd_3_29.v[0]); rnd_4_21.v[1] = vtrn1q_u64(rnd_3_21.v[1], rnd_3_29.v[1]);
    vec256_t rnd_4_29; rnd_4_29.v[0] = vtrn2q_u64(rnd_3_21.v[0], rnd_3_29.v[0]); rnd_4_29.v[1] = vtrn2q_u64(rnd_3_21.v[1], rnd_3_29.v[1]);
    vec256_t rnd_4_22; rnd_4_22.v[0] = vtrn1q_u64(rnd_3_22.v[0], rnd_3_30.v[0]); rnd_4_22.v[1] = vtrn1q_u64(rnd_3_22.v[1], rnd_3_30.v[1]);
    vec256_t rnd_4_30; rnd_4_30.v[0] = vtrn2q_u64(rnd_3_22.v[0], rnd_3_30.v[0]); rnd_4_30.v[1] = vtrn2q_u64(rnd_3_22.v[1], rnd_3_30.v[1]);
    vec256_t rnd_4_23; rnd_4_23.v[0] = vtrn1q_u64(rnd_3_23.v[0], rnd_3_31.v[0]); rnd_4_23.v[1] = vtrn1q_u64(rnd_3_23.v[1], rnd_3_31.v[1]);
    vec256_t rnd_4_31; rnd_4_31.v[0] = vtrn2q_u64(rnd_3_23.v[0], rnd_3_31.v[0]); rnd_4_31.v[1] = vtrn2q_u64(rnd_3_23.v[1], rnd_3_31.v[1]);

    // NOTE: maybe It's useful to reoptimize the following code. As we are just shuffling some register we could do
    // already do this in round 4, just above.
    vec256_t rnd_5_0;  rnd_5_0.v[0]  = rnd_4_0.v[0]; rnd_5_0.v[1]  = rnd_4_16.v[0];
    vec256_t rnd_5_16; rnd_5_16.v[0] = rnd_4_0.v[1]; rnd_5_16.v[1] = rnd_4_16.v[1];
    *((vec256_t *)(dst_origin +  0*dst_stride)) = rnd_5_0;
    *((vec256_t *)(dst_origin + 16*dst_stride)) = rnd_5_16;
    vec256_t rnd_5_1;  rnd_5_1.v[0]  = rnd_4_1.v[0]; rnd_5_1.v[1]  = rnd_4_17.v[0];
    vec256_t rnd_5_17; rnd_5_17.v[0] = rnd_4_1.v[1]; rnd_5_17.v[1] = rnd_4_17.v[1];
    *((vec256_t *)(dst_origin +  1*dst_stride)) = rnd_5_1;
    *((vec256_t *)(dst_origin + 17*dst_stride)) = rnd_5_17;
    vec256_t rnd_5_2;  rnd_5_2.v[0]  = rnd_4_2.v[0];  rnd_5_2.v[1] = rnd_4_18.v[0];
    vec256_t rnd_5_18; rnd_5_18.v[0] = rnd_4_2.v[1]; rnd_5_18.v[1] = rnd_4_18.v[1];
    *((vec256_t *)(dst_origin +  2*dst_stride)) = rnd_5_2;
    *((vec256_t *)(dst_origin + 18*dst_stride)) = rnd_5_18;
    vec256_t rnd_5_3;  rnd_5_3.v[0]  = rnd_4_3.v[0];  rnd_5_3.v[1] = rnd_4_19.v[0];
    vec256_t rnd_5_19; rnd_5_19.v[0] = rnd_4_3.v[1]; rnd_5_19.v[1] = rnd_4_19.v[1];
    *((vec256_t *)(dst_origin +  3*dst_stride)) = rnd_5_3;
    *((vec256_t *)(dst_origin + 19*dst_stride)) = rnd_5_19;
    vec256_t rnd_5_4;  rnd_5_4.v[0]  = rnd_4_4.v[0];  rnd_5_4.v[1] = rnd_4_20.v[0];
    vec256_t rnd_5_20; rnd_5_20.v[0] = rnd_4_4.v[1]; rnd_5_20.v[1] = rnd_4_20.v[1];
    *((vec256_t *)(dst_origin +  4*dst_stride)) = rnd_5_4;
    *((vec256_t *)(dst_origin + 20*dst_stride)) = rnd_5_20;
    vec256_t rnd_5_5;  rnd_5_5.v[0]  = rnd_4_5.v[0];  rnd_5_5.v[1] = rnd_4_21.v[0];
    vec256_t rnd_5_21; rnd_5_21.v[0] = rnd_4_5.v[1]; rnd_5_21.v[1] = rnd_4_21.v[1];
    *((vec256_t *)(dst_origin +  5*dst_stride)) = rnd_5_5;
    *((vec256_t *)(dst_origin + 21*dst_stride)) = rnd_5_21;
    vec256_t rnd_5_6;  rnd_5_6.v[0]  = rnd_4_6.v[0];  rnd_5_6.v[1] = rnd_4_22.v[0];
    vec256_t rnd_5_22; rnd_5_22.v[0] = rnd_4_6.v[1]; rnd_5_22.v[1] = rnd_4_22.v[1];
    *((vec256_t *)(dst_origin +  6*dst_stride)) = rnd_5_6;
    *((vec256_t *)(dst_origin + 22*dst_stride)) = rnd_5_22;
    vec256_t rnd_5_7;  rnd_5_7.v[0]  = rnd_4_7.v[0];  rnd_5_7.v[1] = rnd_4_23.v[0];
    vec256_t rnd_5_23; rnd_5_23.v[0] = rnd_4_7.v[1]; rnd_5_23.v[1] = rnd_4_23.v[1];
    *((vec256_t *)(dst_origin +  7*dst_stride)) = rnd_5_7;
    *((vec256_t *)(dst_origin + 23*dst_stride)) = rnd_5_23;
    vec256_t rnd_5_8;  rnd_5_8.v[0]  = rnd_4_8.v[0];  rnd_5_8.v[1] = rnd_4_24.v[0];
    vec256_t rnd_5_24; rnd_5_24.v[0] = rnd_4_8.v[1]; rnd_5_24.v[1] = rnd_4_24.v[1];
    *((vec256_t *)(dst_origin +  8*dst_stride)) = rnd_5_8;
    *((vec256_t *)(dst_origin + 24*dst_stride)) = rnd_5_24;
    vec256_t rnd_5_9;  rnd_5_9.v[0]  = rnd_4_9.v[0];  rnd_5_9.v[1] = rnd_4_25.v[0];
    vec256_t rnd_5_25; rnd_5_25.v[0] = rnd_4_9.v[1]; rnd_5_25.v[1] = rnd_4_25.v[1];
    *((vec256_t *)(dst_origin +  9*dst_stride)) = rnd_5_9;
    *((vec256_t *)(dst_origin + 25*dst_stride)) = rnd_5_25;
    vec256_t rnd_5_10; rnd_5_10.v[0] = rnd_4_10.v[0]; rnd_5_10.v[1] = rnd_4_26.v[0];
    vec256_t rnd_5_26; rnd_5_26.v[0] = rnd_4_10.v[1]; rnd_5_26.v[1] = rnd_4_26.v[1];
    *((vec256_t *)(dst_origin + 10*dst_stride)) = rnd_5_10;
    *((vec256_t *)(dst_origin + 26*dst_stride)) = rnd_5_26;
    vec256_t rnd_5_11; rnd_5_11.v[0] = rnd_4_11.v[0]; rnd_5_11.v[1] = rnd_4_27.v[0];
    vec256_t rnd_5_27; rnd_5_27.v[0] = rnd_4_11.v[1]; rnd_5_27.v[1] = rnd_4_27.v[1];
    *((vec256_t *)(dst_origin + 11*dst_stride)) = rnd_5_11;
    *((vec256_t *)(dst_origin + 27*dst_stride)) = rnd_5_27;
    vec256_t rnd_5_12; rnd_5_12.v[0] = rnd_4_12.v[0]; rnd_5_12.v[1] = rnd_4_28.v[0];
    vec256_t rnd_5_28; rnd_5_28.v[0] = rnd_4_12.v[1]; rnd_5_28.v[1] = rnd_4_28.v[1];
    *((vec256_t *)(dst_origin + 12*dst_stride)) = rnd_5_12;
    *((vec256_t *)(dst_origin + 28*dst_stride)) = rnd_5_28;
    vec256_t rnd_5_13; rnd_5_13.v[0] = rnd_4_13.v[0]; rnd_5_13.v[1] = rnd_4_29.v[0];
    vec256_t rnd_5_29; rnd_5_29.v[0] = rnd_4_13.v[1]; rnd_5_29.v[1] = rnd_4_29.v[1];
    *((vec256_t *)(dst_origin + 13*dst_stride)) = rnd_5_13;
    *((vec256_t *)(dst_origin + 29*dst_stride)) = rnd_5_29;
    vec256_t rnd_5_14; rnd_5_14.v[0] = rnd_4_14.v[0]; rnd_5_14.v[1] = rnd_4_30.v[0];
    vec256_t rnd_5_30; rnd_5_30.v[0] = rnd_4_14.v[1]; rnd_5_30.v[1] = rnd_4_30.v[1];
    *((vec256_t *)(dst_origin + 14*dst_stride)) = rnd_5_14;
    *((vec256_t *)(dst_origin + 30*dst_stride)) = rnd_5_30;
    vec256_t rnd_5_15; rnd_5_15.v[0] = rnd_4_15.v[0]; rnd_5_15.v[1] = rnd_4_31.v[0];
    vec256_t rnd_5_31; rnd_5_31.v[0] = rnd_4_15.v[1]; rnd_5_31.v[1] = rnd_4_31.v[1];
    *((vec256_t *)(dst_origin + 15*dst_stride)) = rnd_5_15;
    *((vec256_t *)(dst_origin + 31*dst_stride)) = rnd_5_31;
}
#endif
#ifdef USE_AVX512


/// \param dst_origin[out]: output matrix
/// \param src_origin[in]: input matrix
/// \param prf_origin[in]: lookahead pointer to prefetch it
/// \param src_stride[in]:
/// \param dst_stride[in]:
void matrix_transpose_u8_64x64(uint8_t* dst_origin,
                            const uint8_t* src_origin,
                            const uint8_t* prf_origin,
                            const size_t src_stride,
                            const size_t dst_stride) {
    static const uint32_t matrix_transpose_table[] __attribute__((aligned(32))) = {
        0,8,4,12,2,10,6,14,1,9,5,13,3,11,7,15
    };
    const __m512i m1 = _mm512_setr_epi64(0b0000, 0b0001, 0b1000, 0b1001, 0b0100, 0b0101, 0b1100, 0b1101);
    const __m512i m2 = _mm512_setr_epi64(0b0010, 0b0011, 0b1010, 0b1011, 0b0110, 0b0111, 0b1110, 0b1111);

    const __m512i m3 = _mm512_setr_epi64(0b0000, 0b0001, 0b0010, 0b0011, 0b1000, 0b1001, 0b1010, 0b1011);
    const __m512i m4 = _mm512_setr_epi64(0b0100, 0b0101, 0b0110, 0b0111, 0b1100, 0b1101, 0b1110, 0b1111);

    (void)prf_origin;
    __m512i t[64];
    for (uint32_t i = 0; i < 64; i++) {
        t[i] = _mm512_loadu_si512((const __m512i *)(src_origin + i*src_stride));
    }

    #pragma unroll
    for (uint32_t i = 0; i < 64; i+=2) {
        const __m512i t0 = _mm512_unpacklo_epi8(t[i+0], t[i+1]);
        const __m512i t1 = _mm512_unpackhi_epi8(t[i+0], t[i+1]);
        t[i+0] = t0;
        t[i+1] = t1;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 64; i+=4) {
        const __m512i t0 = _mm512_unpacklo_epi16(t[i+0], t[i+2]);
        const __m512i t1 = _mm512_unpacklo_epi16(t[i+1], t[i+3]);
        const __m512i t2 = _mm512_unpackhi_epi16(t[i+0], t[i+2]);
        const __m512i t3 = _mm512_unpackhi_epi16(t[i+1], t[i+3]);
        t[i+0] = t0;
        t[i+1] = t1;
        t[i+2] = t2;
        t[i+3] = t3;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 64; i+=8) {
        const __m512i t0 = _mm512_unpacklo_epi32(t[i+0], t[i+4]);
        const __m512i t1 = _mm512_unpacklo_epi32(t[i+1], t[i+5]);
        const __m512i t2 = _mm512_unpacklo_epi32(t[i+2], t[i+6]);
        const __m512i t3 = _mm512_unpacklo_epi32(t[i+3], t[i+7]);
        const __m512i t4 = _mm512_unpackhi_epi32(t[i+0], t[i+4]);
        const __m512i t5 = _mm512_unpackhi_epi32(t[i+1], t[i+5]);
        const __m512i t6 = _mm512_unpackhi_epi32(t[i+2], t[i+6]);
        const __m512i t7 = _mm512_unpackhi_epi32(t[i+3], t[i+7]);
        t[i+0] = t0;
        t[i+1] = t1;
        t[i+2] = t2;
        t[i+3] = t3;
        t[i+4] = t4;
        t[i+5] = t5;
        t[i+6] = t6;
        t[i+7] = t7;
    }

    #pragma unroll
    for (uint32_t i = 0; i < 8; i++) {
        const __m512i t0 = _mm512_unpacklo_epi64(t[i+ 0], t[i+ 8]);
        const __m512i t1 = _mm512_unpackhi_epi64(t[i+ 0], t[i+ 8]);
        const __m512i t2 = _mm512_unpacklo_epi64(t[i+16], t[i+24]);
        const __m512i t3 = _mm512_unpackhi_epi64(t[i+16], t[i+24]);
        const __m512i t4 = _mm512_unpacklo_epi64(t[i+32], t[i+40]);
        const __m512i t5 = _mm512_unpackhi_epi64(t[i+32], t[i+40]);
        const __m512i t6 = _mm512_unpacklo_epi64(t[i+48], t[i+56]);
        const __m512i t7 = _mm512_unpackhi_epi64(t[i+48], t[i+56]);
        t[i+ 0] = t0;
        t[i+ 8] = t1;
        t[i+16] = t2;
        t[i+24] = t3;
        t[i+32] = t4;
        t[i+40] = t5;
        t[i+48] = t6;
        t[i+56] = t7;
    }

    // swap 128 bit limbs
    #pragma unroll
    for (uint32_t i = 0; i < 16; i++) {
        const __m512i t0 = _mm512_permutex2var_epi64(t[i+ 0], m1, t[i+16]);
        const __m512i t1 = _mm512_permutex2var_epi64(t[i+ 0], m2, t[i+16]);
        const __m512i t2 = _mm512_permutex2var_epi64(t[i+32], m1, t[i+48]);
        const __m512i t3 = _mm512_permutex2var_epi64(t[i+32], m2, t[i+48]);
        t[i+ 0] = t0;
        t[i+16] = t1;
        t[i+32] = t2;
        t[i+48] = t3;
    }

    // swap 256 bit limbs
    #pragma unroll
    for (uint32_t i = 0; i < 32; i++) {
        const __m512i t0 = _mm512_permutex2var_epi64(t[i+0], m3, t[i+32]);
        const __m512i t1 = _mm512_permutex2var_epi64(t[i+0], m4, t[i+32]);
        t[i+ 0] = t0;
        t[i+32] = t1;
    }


    #pragma unroll
    for (uint32_t j = 0; j < 4; j++) {
        const uint32_t off = j*16;

        #pragma unroll
        for (uint32_t i = 0; i < 16; i++) {
            const uint32_t pos = matrix_transpose_table[i];
            _mm512_storeu_si512((__m512i *)(dst_origin + (off+i)*dst_stride), t[off+pos]);
        }
    }
}
#endif // USE_AVX512



#ifdef USE_AVX2

// NOTE: stride in bytes
static void transpose_u4_32x32_avx2(uint8_t *B,
                                      const uint8_t *const A,
                                      const uint32_t stride) {
    __m128i M[32];
    const __m128i mask1 = _mm_set1_epi8  (0x0F),
                  mask2 = _mm_set1_epi16 (0x00FF),
                  mask3 = _mm_set1_epi32 (0x0000FFFF),
                  mask4 = _mm_set1_epi64x(0x00000000FFFFFFFF),
                  mask5 = _mm_setr_epi64((__m64)0xFFFFFFFFFFFFFFFF, (__m64)0ul);

    for (uint32_t i = 0; i < 32; i++) {
        M[i] = _mm_loadu_si128((__m128i *)(A+i*16));
    }

    for (uint32_t i = 0; i < 32; i+=2) {
        const __m128i t = (_mm_srli_epi64(M[i], 4) ^ M[i+1]) & mask1;
        M[i+0] ^= _mm_slli_epi64(t, 4);
        M[i+1] ^= t;
    }

    for (uint32_t i = 0; i < 32; i+=4) {
        const __m128i t0 = (_mm_srli_epi64(M[i+0], 8) ^ M[i+2]) & mask2;
        const __m128i t1 = (_mm_srli_epi64(M[i+1], 8) ^ M[i+3]) & mask2;
        M[i+0] ^= _mm_slli_epi64(t0, 8);
        M[i+1] ^= _mm_slli_epi64(t1, 8);
        M[i+2] ^= t0;
        M[i+3] ^= t1;
    }

    for (uint32_t i = 0; i < 32; i+=8) {
        const __m128i t0 = (_mm_srli_epi64(M[i+0], 16) ^ M[i+4]) & mask3;
        const __m128i t1 = (_mm_srli_epi64(M[i+1], 16) ^ M[i+5]) & mask3;
        const __m128i t2 = (_mm_srli_epi64(M[i+2], 16) ^ M[i+6]) & mask3;
        const __m128i t3 = (_mm_srli_epi64(M[i+3], 16) ^ M[i+7]) & mask3;
        M[i+0] ^= _mm_slli_epi64(t0, 16);
        M[i+1] ^= _mm_slli_epi64(t1, 16);
        M[i+2] ^= _mm_slli_epi64(t2, 16);
        M[i+3] ^= _mm_slli_epi64(t3, 16);
        M[i+4] ^= t0;
        M[i+5] ^= t1;
        M[i+6] ^= t2;
        M[i+7] ^= t3;
    }

    for (uint32_t i = 0; i < 32; i+=16) {
        const __m128i t0 = (_mm_srli_epi64(M[i+0], 32) ^ M[i+ 8]) & mask4;
        const __m128i t1 = (_mm_srli_epi64(M[i+1], 32) ^ M[i+ 9]) & mask4;
        const __m128i t2 = (_mm_srli_epi64(M[i+2], 32) ^ M[i+10]) & mask4;
        const __m128i t3 = (_mm_srli_epi64(M[i+3], 32) ^ M[i+11]) & mask4;
        const __m128i t4 = (_mm_srli_epi64(M[i+4], 32) ^ M[i+12]) & mask4;
        const __m128i t5 = (_mm_srli_epi64(M[i+5], 32) ^ M[i+13]) & mask4;
        const __m128i t6 = (_mm_srli_epi64(M[i+6], 32) ^ M[i+14]) & mask4;
        const __m128i t7 = (_mm_srli_epi64(M[i+7], 32) ^ M[i+15]) & mask4;
        M[i+ 0] ^= _mm_slli_epi64(t0, 32);
        M[i+ 1] ^= _mm_slli_epi64(t1, 32);
        M[i+ 2] ^= _mm_slli_epi64(t2, 32);
        M[i+ 3] ^= _mm_slli_epi64(t3, 32);
        M[i+ 4] ^= _mm_slli_epi64(t4, 32);
        M[i+ 5] ^= _mm_slli_epi64(t5, 32);
        M[i+ 6] ^= _mm_slli_epi64(t6, 32);
        M[i+ 7] ^= _mm_slli_epi64(t7, 32);
        M[i+ 8] ^= t0;
        M[i+ 9] ^= t1;
        M[i+10] ^= t2;
        M[i+11] ^= t3;
        M[i+12] ^= t4;
        M[i+13] ^= t5;
        M[i+14] ^= t6;
        M[i+15] ^= t7;
    }

    for (uint32_t i = 0; i < 16; i++) {
        const __m128i t0 = (_mm_srli_si128(M[i+ 0], 8) ^ M[i+16]) & mask5;
        M[i   ] ^= _mm_slli_si128(t0, 8);
        M[i+16] ^= t0;
    }

    // write out
    for (uint32_t i = 0; i < 32; i++) {
        _mm_storeu_si128((__m128i *)(B + i*stride), M[i]);
    }
}

// NOTE: stride in bytes
static void transpose_u4_64x64_avx2(uint8_t *B,
                                      const uint8_t *const A,
                                      const uint32_t src_stride,
                                      const uint32_t dst_stride) {
    __m256i M[64];
    const __m256i mask1 = _mm256_set1_epi8  (0x0F),
                  mask2 = _mm256_set1_epi16 (0x00FF),
                  mask3 = _mm256_set1_epi32 (0x0000FFFF),
                  mask4 = _mm256_set1_epi64x(0x00000000FFFFFFFF),
                  mask5 = _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0, 0xFFFFFFFFFFFFFFFF, 0),
                  mask6 = _mm256_setr_epi64x(0xFFFFFFFFFFFFFFFF, 0xFFFFFFFFFFFFFFFF, 0, 0);
    for (uint32_t i = 0; i < 64; i++) {
        M[i] = _mm256_loadu_si256((__m256i *)(A+i*src_stride));
    }

    for (uint32_t i = 0; i < 64; i+=2) {
        const __m256i t = (_mm256_srli_epi64(M[i], 4) ^ M[i+1]) & mask1;
        M[i+0] ^= _mm256_slli_epi64(t, 4);
        M[i+1] ^= t;
    }

    for (uint32_t i = 0; i < 64; i+=4) {
        const __m256i t0 = (_mm256_srli_epi64(M[i+0], 8) ^ M[i+2]) & mask2;
        const __m256i t1 = (_mm256_srli_epi64(M[i+1], 8) ^ M[i+3]) & mask2;
        M[i+0] ^= _mm256_slli_epi64(t0, 8);
        M[i+1] ^= _mm256_slli_epi64(t1, 8);
        M[i+2] ^= t0;
        M[i+3] ^= t1;
    }

    for (uint32_t i = 0; i < 64; i+=8) {
        const __m256i t0 = (_mm256_srli_epi64(M[i+0], 16) ^ M[i+4]) & mask3;
        const __m256i t1 = (_mm256_srli_epi64(M[i+1], 16) ^ M[i+5]) & mask3;
        const __m256i t2 = (_mm256_srli_epi64(M[i+2], 16) ^ M[i+6]) & mask3;
        const __m256i t3 = (_mm256_srli_epi64(M[i+3], 16) ^ M[i+7]) & mask3;
        M[i+0] ^= _mm256_slli_epi64(t0, 16);
        M[i+1] ^= _mm256_slli_epi64(t1, 16);
        M[i+2] ^= _mm256_slli_epi64(t2, 16);
        M[i+3] ^= _mm256_slli_epi64(t3, 16);
        M[i+4] ^= t0;
        M[i+5] ^= t1;
        M[i+6] ^= t2;
        M[i+7] ^= t3;
    }

    for (uint32_t i = 0; i < 64; i+=16) {
        const __m256i t0 = (_mm256_srli_epi64(M[i+0], 32) ^ M[i+ 8]) & mask4;
        const __m256i t1 = (_mm256_srli_epi64(M[i+1], 32) ^ M[i+ 9]) & mask4;
        const __m256i t2 = (_mm256_srli_epi64(M[i+2], 32) ^ M[i+10]) & mask4;
        const __m256i t3 = (_mm256_srli_epi64(M[i+3], 32) ^ M[i+11]) & mask4;
        const __m256i t4 = (_mm256_srli_epi64(M[i+4], 32) ^ M[i+12]) & mask4;
        const __m256i t5 = (_mm256_srli_epi64(M[i+5], 32) ^ M[i+13]) & mask4;
        const __m256i t6 = (_mm256_srli_epi64(M[i+6], 32) ^ M[i+14]) & mask4;
        const __m256i t7 = (_mm256_srli_epi64(M[i+7], 32) ^ M[i+15]) & mask4;
        M[i+ 0] ^= _mm256_slli_epi64(t0, 32);
        M[i+ 1] ^= _mm256_slli_epi64(t1, 32);
        M[i+ 2] ^= _mm256_slli_epi64(t2, 32);
        M[i+ 3] ^= _mm256_slli_epi64(t3, 32);
        M[i+ 4] ^= _mm256_slli_epi64(t4, 32);
        M[i+ 5] ^= _mm256_slli_epi64(t5, 32);
        M[i+ 6] ^= _mm256_slli_epi64(t6, 32);
        M[i+ 7] ^= _mm256_slli_epi64(t7, 32);
        M[i+ 8] ^= t0;
        M[i+ 9] ^= t1;
        M[i+10] ^= t2;
        M[i+11] ^= t3;
        M[i+12] ^= t4;
        M[i+13] ^= t5;
        M[i+14] ^= t6;
        M[i+15] ^= t7;
    }

    for (uint32_t i = 0; i < 16; i++) {
        const __m256i t0 = (_mm256_srli_si256(M[i+ 0], 8) ^ M[i+16]) & mask5;
        const __m256i t1 = (_mm256_srli_si256(M[i+32], 8) ^ M[i+48]) & mask5;
        M[i   ] ^= _mm256_slli_si256(t0, 8);
        M[i+32] ^= _mm256_slli_si256(t1, 8);
        M[i+16] ^= t0;
        M[i+48] ^= t1;
    }

    for (uint32_t i = 0; i < 32; i++) {
        const __m256i t = (_mm256_permute2x128_si256(M[i+0], M[i+0], 0b10000001) ^ M[i+32]) & mask6;
        M[i   ] ^= _mm256_permute2x128_si256(t, t, 0b01000); //
        M[i+32] ^= t;
    }

    // write out
    for (uint32_t i = 0; i < 64; i++) {
        _mm256_storeu_si256((__m256i *)(B + i*dst_stride), M[i]);
    }
}

/// every element is a u64
/// out: [a0 b0 c0 d0]
///      [a1 b1 c1 d1]
///      [a2 b2 c2 d2]
///      [a1 b1 c1 d1]
/// in : [a0 a1 a2 a3]
///      [b0 b1 b2 b3]
///      [c0 c1 c2 c3]
///      [d0 d1 d2 d3]
static inline
void transpose_u64_4x4_avx2(uint64_t *out,
                           const uint64_t *in) {
    const __m256i a0 = _mm256_loadu_si256((const __m256i *)(in +  0));
    const __m256i a1 = _mm256_loadu_si256((const __m256i *)(in +  4));
    const __m256i a2 = _mm256_loadu_si256((const __m256i *)(in +  8));
    const __m256i a3 = _mm256_loadu_si256((const __m256i *)(in + 12));

    const __m256i b0 = _mm256_unpacklo_epi64(a0, a1);
    const __m256i b1 = _mm256_unpackhi_epi64(a0, a1);
    const __m256i b2 = _mm256_unpacklo_epi64(a2, a3);
    const __m256i b3 = _mm256_unpackhi_epi64(a2, a3);

    const __m256i t0 = _mm256_permute2x128_si256(b0, b2, 0x20);
    const __m256i t1 = _mm256_permute2x128_si256(b1, b3, 0x20);
    const __m256i t2 = _mm256_permute2x128_si256(b0, b2, 0x31);
    const __m256i t3 = _mm256_permute2x128_si256(b1, b3, 0x31);

    _mm256_storeu_si256((__m256i *)(out +  0), t0);
    _mm256_storeu_si256((__m256i *)(out +  4), t1);
    _mm256_storeu_si256((__m256i *)(out +  8), t2);
    _mm256_storeu_si256((__m256i *)(out + 12), t3);
}

/// every element is a u64
/// out: [a0 b0 c0 d0]
///      [a1 b1 c1 d1]
///      [a2 b2 c2 d2]
///      [a1 b1 c1 d1]
/// in : [a0 a1 a2 a3]
///      [b0 b1 b2 b3]
///      [c0 c1 c2 c3]
///      [d0 d1 d2 d3]
static inline
void transpose_u64_4x4_avx2_(__m256i a[4]) {
    const __m256i b0 = _mm256_unpacklo_epi64(a[0], a[1]);
    const __m256i b1 = _mm256_unpackhi_epi64(a[0], a[1]);
    const __m256i b2 = _mm256_unpacklo_epi64(a[2], a[3]);
    const __m256i b3 = _mm256_unpackhi_epi64(a[2], a[3]);

    a[0] = _mm256_permute2x128_si256(b0, b2, 0x20);
    a[1] = _mm256_permute2x128_si256(b1, b3, 0x20);
    a[2] = _mm256_permute2x128_si256(b0, b2, 0x31);
    a[3] = _mm256_permute2x128_si256(b1, b3, 0x31);
}


#ifdef USE_AVX512 

/// every element is a u64
/// out: [a0 b0 c0 d0 e0 f0 g0 h0]
///      [a1 b1 c1 d1 e1 f1 g1 h1]
///      [a2 b2 c2 d2 e2 f2 g2 h2]
///      [a3 b3 c3 d3 e3 f3 g3 h3]
///      [a4 b4 c4 d4 e4 f4 g4 h4]
///      [a5 b5 c5 d5 e5 f5 g5 h5]
///      [a6 b6 c6 d6 e6 f6 g6 h6]
///      [a7 b7 c7 d7 e7 f7 g7 h7]
/// in : [a0 a1 a2 a3 a4 a5 a6 a7]
///      [b0 b1 b2 b7 b4 b5 b6 b7]
///      [c0 c1 c2 c3 c4 c5 c6 c7]
///      [d0 d1 d2 d3 d4 d5 d6 d7]
///      [e0 e1 e2 e3 e4 e5 e6 e7]
///      [f0 f1 f2 f3 f4 f5 f6 f7]
///      [g0 g1 g2 g3 g4 g5 g6 g7]
///      [h0 h1 h2 h3 h4 h5 h6 h7]
void transpose_u64_8x8_avx512_(__m512i a[8]) {
    const __m512i m1 = _mm512_setr_epi64(0b0000, 0b0001, 0b1000, 0b1001, 0b0100, 0b0101, 0b1100, 0b1101);
    const __m512i m2 = _mm512_setr_epi64(0b0010, 0b0011, 0b1010, 0b1011, 0b0110, 0b0111, 0b1110, 0b1111);

    const __m512i m3 = _mm512_setr_epi64(0b0000, 0b0001, 0b0010, 0b0011, 0b1000, 0b1001, 0b1010, 0b1011);
    const __m512i m4 = _mm512_setr_epi64(0b0100, 0b0101, 0b0110, 0b0111, 0b1100, 0b1101, 0b1110, 0b1111);

    const __m512i b0 = _mm512_unpacklo_epi64(a[0], a[1]);
    const __m512i b1 = _mm512_unpackhi_epi64(a[0], a[1]);
    const __m512i b2 = _mm512_unpacklo_epi64(a[2], a[3]);
    const __m512i b3 = _mm512_unpackhi_epi64(a[2], a[3]);
    const __m512i b4 = _mm512_unpacklo_epi64(a[4], a[5]);
    const __m512i b5 = _mm512_unpackhi_epi64(a[4], a[5]);
    const __m512i b6 = _mm512_unpacklo_epi64(a[6], a[7]);
    const __m512i b7 = _mm512_unpackhi_epi64(a[6], a[7]);

    const __m512i t0 = _mm512_permutex2var_epi64(b0, m1, b2);
    const __m512i t1 = _mm512_permutex2var_epi64(b1, m2, b3);
    const __m512i t2 = _mm512_permutex2var_epi64(b0, m1, b2);
    const __m512i t3 = _mm512_permutex2var_epi64(b1, m2, b3);
    const __m512i t4 = _mm512_permutex2var_epi64(b4, m1, b6);
    const __m512i t5 = _mm512_permutex2var_epi64(b5, m2, b7);
    const __m512i t6 = _mm512_permutex2var_epi64(b4, m1, b6);
    const __m512i t7 = _mm512_permutex2var_epi64(b5, m2, b7);

    a[0] = _mm512_permutex2var_epi64(b0, m3, b4);
    a[1] = _mm512_permutex2var_epi64(b1, m3, b5);
    a[2] = _mm512_permutex2var_epi64(b2, m3, b6);
    a[3] = _mm512_permutex2var_epi64(b3, m3, b7);
    a[4] = _mm512_permutex2var_epi64(b4, m4, b6);
    a[5] = _mm512_permutex2var_epi64(b5, m4, b7);
    a[6] = _mm512_permutex2var_epi64(b6, m4, b6);
    a[7] = _mm512_permutex2var_epi64(b7, m4, b7);
}
// source: https://stackoverflow.com/questions/29519222/how-to-transpose-a-16x16-matrix-using-simd-instructions
inline static
void tran_new2(uint32_t* mat, uint32_t* matT) noexcept {
    __m512i t0, t1, t2, t3, t4, t5, t6, t7, t8, t9, ta, tb, tc, td, te, tf;
    __m512i r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, ra, rb, rc, rd, re, rf;
    
    int mask;
    int64_t idx1[8] __attribute__((aligned(64))) = {2, 3, 0, 1, 6, 7, 4, 5}; 
    int64_t idx2[8] __attribute__((aligned(64))) = {1, 0, 3, 2, 5, 4, 7, 6}; 
    int32_t idx3[16] __attribute__((aligned(64))) = {1, 0, 3, 2, 5 ,4 ,7 ,6 ,9 ,8 , 11, 10, 13, 12 ,15, 14};
    __m512i vidx1 = _mm512_load_epi64(idx1);
    __m512i vidx2 = _mm512_load_epi64(idx2);
    __m512i vidx3 = _mm512_load_epi32(idx3);
    
    t0 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 0*16+0])), _mm256_load_si256((__m256i*)&mat[ 8*16+0]), 1);
    t1 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 1*16+0])), _mm256_load_si256((__m256i*)&mat[ 9*16+0]), 1);
    t2 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 2*16+0])), _mm256_load_si256((__m256i*)&mat[10*16+0]), 1);
    t3 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 3*16+0])), _mm256_load_si256((__m256i*)&mat[11*16+0]), 1);
    t4 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 4*16+0])), _mm256_load_si256((__m256i*)&mat[12*16+0]), 1);
    t5 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 5*16+0])), _mm256_load_si256((__m256i*)&mat[13*16+0]), 1);
    t6 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 6*16+0])), _mm256_load_si256((__m256i*)&mat[14*16+0]), 1);
    t7 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 7*16+0])), _mm256_load_si256((__m256i*)&mat[15*16+0]), 1);
    
    t8 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 0*16+8])), _mm256_load_si256((__m256i*)&mat[ 8*16+8]), 1);
    t9 = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 1*16+8])), _mm256_load_si256((__m256i*)&mat[ 9*16+8]), 1);
    ta = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 2*16+8])), _mm256_load_si256((__m256i*)&mat[10*16+8]), 1);
    tb = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 3*16+8])), _mm256_load_si256((__m256i*)&mat[11*16+8]), 1);
    tc = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 4*16+8])), _mm256_load_si256((__m256i*)&mat[12*16+8]), 1);
    td = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 5*16+8])), _mm256_load_si256((__m256i*)&mat[13*16+8]), 1);
    te = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 6*16+8])), _mm256_load_si256((__m256i*)&mat[14*16+8]), 1);
    tf = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_load_si256((__m256i*)&mat[ 7*16+8])), _mm256_load_si256((__m256i*)&mat[15*16+8]), 1);
    
    mask= 0xcc;
    r0 = _mm512_mask_permutexvar_epi64(t0, (__mmask8)mask, vidx1, t4);
    r1 = _mm512_mask_permutexvar_epi64(t1, (__mmask8)mask, vidx1, t5);
    r2 = _mm512_mask_permutexvar_epi64(t2, (__mmask8)mask, vidx1, t6);
    r3 = _mm512_mask_permutexvar_epi64(t3, (__mmask8)mask, vidx1, t7);
    r8 = _mm512_mask_permutexvar_epi64(t8, (__mmask8)mask, vidx1, tc);
    r9 = _mm512_mask_permutexvar_epi64(t9, (__mmask8)mask, vidx1, td);
    ra = _mm512_mask_permutexvar_epi64(ta, (__mmask8)mask, vidx1, te);
    rb = _mm512_mask_permutexvar_epi64(tb, (__mmask8)mask, vidx1, tf);
    
    mask= 0x33;
    r4 = _mm512_mask_permutexvar_epi64(t4, (__mmask8)mask, vidx1, t0);
    r5 = _mm512_mask_permutexvar_epi64(t5, (__mmask8)mask, vidx1, t1);
    r6 = _mm512_mask_permutexvar_epi64(t6, (__mmask8)mask, vidx1, t2);
    r7 = _mm512_mask_permutexvar_epi64(t7, (__mmask8)mask, vidx1, t3);
    rc = _mm512_mask_permutexvar_epi64(tc, (__mmask8)mask, vidx1, t8);
    rd = _mm512_mask_permutexvar_epi64(td, (__mmask8)mask, vidx1, t9);
    re = _mm512_mask_permutexvar_epi64(te, (__mmask8)mask, vidx1, ta);
    rf = _mm512_mask_permutexvar_epi64(tf, (__mmask8)mask, vidx1, tb);
    
    mask = 0xaa;
    t0 = _mm512_mask_permutexvar_epi64(r0, (__mmask8)mask, vidx2, r2);
    t1 = _mm512_mask_permutexvar_epi64(r1, (__mmask8)mask, vidx2, r3);
    t4 = _mm512_mask_permutexvar_epi64(r4, (__mmask8)mask, vidx2, r6);
    t5 = _mm512_mask_permutexvar_epi64(r5, (__mmask8)mask, vidx2, r7);
    t8 = _mm512_mask_permutexvar_epi64(r8, (__mmask8)mask, vidx2, ra);
    t9 = _mm512_mask_permutexvar_epi64(r9, (__mmask8)mask, vidx2, rb);
    tc = _mm512_mask_permutexvar_epi64(rc, (__mmask8)mask, vidx2, re);
    td = _mm512_mask_permutexvar_epi64(rd, (__mmask8)mask, vidx2, rf);
    
    mask = 0x55;
    t2 = _mm512_mask_permutexvar_epi64(r2, (__mmask8)mask, vidx2, r0);
    t3 = _mm512_mask_permutexvar_epi64(r3, (__mmask8)mask, vidx2, r1);
    t6 = _mm512_mask_permutexvar_epi64(r6, (__mmask8)mask, vidx2, r4);
    t7 = _mm512_mask_permutexvar_epi64(r7, (__mmask8)mask, vidx2, r5);
    ta = _mm512_mask_permutexvar_epi64(ra, (__mmask8)mask, vidx2, r8);
    tb = _mm512_mask_permutexvar_epi64(rb, (__mmask8)mask, vidx2, r9);
    te = _mm512_mask_permutexvar_epi64(re, (__mmask8)mask, vidx2, rc);
    tf = _mm512_mask_permutexvar_epi64(rf, (__mmask8)mask, vidx2, rd);
    
    mask = 0xaaaa;
    r0 = _mm512_mask_permutexvar_epi32(t0, (__mmask16)mask, vidx3, t1);
    r2 = _mm512_mask_permutexvar_epi32(t2, (__mmask16)mask, vidx3, t3);
    r4 = _mm512_mask_permutexvar_epi32(t4, (__mmask16)mask, vidx3, t5);
    r6 = _mm512_mask_permutexvar_epi32(t6, (__mmask16)mask, vidx3, t7);
    r8 = _mm512_mask_permutexvar_epi32(t8, (__mmask16)mask, vidx3, t9);
    ra = _mm512_mask_permutexvar_epi32(ta, (__mmask16)mask, vidx3, tb);
    rc = _mm512_mask_permutexvar_epi32(tc, (__mmask16)mask, vidx3, td);
    re = _mm512_mask_permutexvar_epi32(te, (__mmask16)mask, vidx3, tf);    
    
    mask = 0x5555;
    r1 = _mm512_mask_permutexvar_epi32(t1, (__mmask16)mask, vidx3, t0);
    r3 = _mm512_mask_permutexvar_epi32(t3, (__mmask16)mask, vidx3, t2);
    r5 = _mm512_mask_permutexvar_epi32(t5, (__mmask16)mask, vidx3, t4);
    r7 = _mm512_mask_permutexvar_epi32(t7, (__mmask16)mask, vidx3, t6);
    r9 = _mm512_mask_permutexvar_epi32(t9, (__mmask16)mask, vidx3, t8);  
    rb = _mm512_mask_permutexvar_epi32(tb, (__mmask16)mask, vidx3, ta);  
    rd = _mm512_mask_permutexvar_epi32(td, (__mmask16)mask, vidx3, tc);
    rf = _mm512_mask_permutexvar_epi32(tf, (__mmask16)mask, vidx3, te);
    
    _mm512_store_epi32(&matT[ 0*16], r0);
    _mm512_store_epi32(&matT[ 1*16], r1);
    _mm512_store_epi32(&matT[ 2*16], r2);
    _mm512_store_epi32(&matT[ 3*16], r3);
    _mm512_store_epi32(&matT[ 4*16], r4);
    _mm512_store_epi32(&matT[ 5*16], r5);
    _mm512_store_epi32(&matT[ 6*16], r6);
    _mm512_store_epi32(&matT[ 7*16], r7);
    _mm512_store_epi32(&matT[ 8*16], r8);
    _mm512_store_epi32(&matT[ 9*16], r9);
    _mm512_store_epi32(&matT[10*16], ra);
    _mm512_store_epi32(&matT[11*16], rb);
    _mm512_store_epi32(&matT[12*16], rc);
    _mm512_store_epi32(&matT[13*16], rd);
    _mm512_store_epi32(&matT[14*16], re);
    _mm512_store_epi32(&matT[15*16], rf);
    int* tmp = mat;
    mat = matT;
    matT = tmp;
}
#endif
