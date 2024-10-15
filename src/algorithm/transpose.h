#ifndef CRYPTANALYSISLIB_ALGORITHM_TRANSPOSE_H
#define CRYPTANALYSISLIB_ALGORITHM_TRANSPOSE_H

#include <cstdint>

// Transpose 8x8 bit array packed into a single quadword 
constexpr uint64_t transpose_b8x8(const uint64_t x_) noexcept {
	uint64_t x = x_, t;
    t = (x ^ (x >> 7)) & 0x00AA00AA00AA00AALL;
    x = x ^ t ^ (t << 7);
    t = (x ^ (x >> 14)) & 0x0000CCCC0000CCCCLL;
    x = x ^ t ^ (t << 14);
    t = (x ^ (x >> 28)) & 0x00000000F0F0F0F0LL;
    x = x ^ t ^ (t << 28);
	return x;
}

// Transpose 8x8 bit array along the diagonal from upper right
constexpr uint64_t transpose_b8x8_be(const uint64_t x_) noexcept {
	uint64_t x = x_, t;
    t = (x ^ (x >> 9)) & 0x0055005500550055LL;
    x = x ^ t ^ (t << 9);
    t = (x ^ (x >> 18)) & 0x0000333300003333LL;
    x = x ^ t ^ (t << 18);
    t = (x ^ (x >> 36)) & 0x000000000F0F0F0FLL;
    x = x ^ t ^ (t << 36);
	return x;
}


// TODO integrate
void transpose8(uint32_t A[8], int m, int n,
				uint32_t B[8]) {
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
void transpose_64x64(uint64_t *out, uint64_t *in) {
	const static uint64_t masks[6][2] = {
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


// inplace
inline void transpose64(uint64_t a[64]) noexcept {
	for (uint64_t j = 32, m = 0x00000000FFFFFFFF; j; j >>= 1, m ^= m << j) {
		for (uint64_t k = 0; k < 64; k = ((k | j) + 1) & ~j) {
			uint64_t t = (a[k] ^ (a[k | j] >> j)) & m;
			a[k] ^= t;
			a[k | j] ^= (t << j);
		}
	}
}
#endif
