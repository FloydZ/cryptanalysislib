#ifndef CRYPTANALYSISLIB_ALGORITHM_TRANSPOSE_H
#define CRYPTANALYSISLIB_ALGORITHM_TRANSPOSE_H

#include <cstdint>

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

/// input: in, a 64x64 matrix over GF(2)
/// output: out, transpose of in
constexpr void transpose_b64x64(uint64_t out[64],
							    const uint64_t in[64]) noexcept {
	constexpr uint64_t masks[6][2] = {
			{0x5555555555555555, 0xAAAAAAAAAAAAAAAA},
			{0x3333333333333333, 0xCCCCCCCCCCCCCCCC},
			{0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0},
			{0x00FF00FF00FF00FF, 0xFF00FF00FF00FF00},
			{0x0000FFFF0000FFFF, 0xFFFF0000FFFF0000},
			{0x00000000FFFFFFFF, 0xFFFFFFFF00000000}};

	for (uint64_t i = 0; i < 64; i++) {
		out[i] = in[i];
	}

	for (int32_t d = 5; d >= 0; d--) {
		const uint32_t s = 1 << d;

		for (uint32_t i = 0; i < 64; i += s * 2) {
			for (uint32_t j = i; j < i + s; j++) {
				const uint64_t x = (out[j] & masks[d][0]) | ((out[j + s] & masks[d][0]) << s);
				const uint64_t y = ((out[j] & masks[d][1]) >> s) | (out[j + s] & masks[d][1]);
				out[j + 0] = x;
				out[j + s] = y;
			}
		}
	}
}

/// transpose of a 64x64 matrix over gf(2)
/// inplace
inline void transpose_b64x64_inplace(uint64_t a[64]) noexcept {
	for (uint64_t j = 32, m = 0x00000000FFFFFFFF; j; j >>= 1, m ^= m << j) {
		for (uint64_t k = 0; k < 64; k = ((k | j) + 1) & ~j) {
			uint64_t t = (a[k] ^ (a[k | j] >> j)) & m;
			a[k] ^= t;
			a[k | j] ^= (t << j);
		}
	}
}
#endif
