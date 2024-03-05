#ifndef CRYPTANALYSISLIB_SIMD_MATRIX_SIMPLE_H
#define CRYPTANALYSISLIB_SIMD_MATRIX_SIMPLE_H

class uint1x64x64_T {
	uint64_t data[64];

	constexpr static void transpose(uint64_t *out, uint64_t *in) noexcept {
		constexpr static uint64_t masks[6][2] = {
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
			const uint32_t s = 1u << d;

			for (uint32_t i = 0; i < 64u; i += s * 2u) {
				for (uint32_t j = i; j < i + s; j++) {
					const uint64_t x = (out[j] & masks[d][0]) | ((out[j + s] & masks[d][0]) << s);
					const uint64_t y = ((out[j] & masks[d][1]) >> s) | (out[j + s] & masks[d][1]);
					out[j + 0] = x;
					out[j + s] = y;
				}
			}
		}
	}

	/// inplace transpose
	constexpr void transpose() noexcept {
		for (uint64_t j = 32, m = 0x00000000FFFFFFFF; j; j >>= 1, m ^= m << j) {
			for (uint64_t k = 0; k < 64; k = ((k | j) + 1) & ~j) {
				uint64_t t = (data[k] ^ (data[k | j] >> j)) & m;
				data[k] ^= t;
				data[k | j] ^= (t << j);
			}
		}
	}

	constexpr void mul() {
	}
};


class uint8x32x32_t {
	uint32_t data[32];

	/// inlpace
	constexpr void transpose() noexcept {
#if 1
		// TODO not correct,
		// taken from https://github.com/pqov/pqov-paper/blob/main/src/avx2/blas_matrix_avx2.c
		alignas(32) uint64x4_t mat[32];

		// load
		for (size_t i = 0; i < 32; i++) {
			mat[i] = uint64x4_t::load(data + i);
		}

		// swap 16x16 blocks
		for (size_t i = 0; i < 16; i++) {
			uint64x4_t tmp = uint64x4_t::template permute<0x20>(mat[i], mat[i + 16]);
			mat[i + 16] = uint64x4_t::template permute<0x31>(mat[i], mat[i + 16]);
			mat[i] = tmp;
		}

		// swap 8x8 blocks
		for (size_t i = 0; i < 2; i++) {
			for (size_t j = 0; j < 8; j++) {
				size_t r = 16 * i + j;
				uint64x4_t tmp = uint64x4_t::unpacklo(mat[r], mat[r + 8]);
				mat[r + 8] = uint64x4_t::unpackhi(mat[r], mat[r + 8]);
				mat[r] = tmp;
			}
		}

		// swap 4x4 blocks
		for (size_t i = 0; i < 4; i++) {
			for (size_t j = 0; j < 4; j++) {
				size_t r = 8 * i + j;
				uint32x8_t tmp = uint32x8_t::template blend<0xaa>(mat[r], uint128x2_t::template slli<4>(mat[r + 4]));
				mat[r + 4] = uint32x8_t::template blend<0xaa>(uint128x2_t::template srli<4>(mat[r]), mat[r + 4]);
				mat[r] = tmp;
			}
		}

		// swap 2x2 blocks
		for (size_t i = 0; i < 8; i++) {
			for (size_t j = 0; j < 2; j++) {
				size_t r = 4 * i + j;
				uint16x16_t tmp = uint16x16_t::template blend<0xaa>(mat[r], uint128x2_t::template slli<2>(mat[r + 2]));
				mat[r + 2] = uint16x16_t::template blend<0xaa>(uint128x2_t::template srli<0xaa>(mat[r]), mat[r + 2]);
				mat[r] = tmp;
			}
		}

		// swap last bytes
		for (size_t i = 0; i < 16; i++) {
			size_t r = 2 * i;
			const uint16x16_t blend_mask = uint16x16_t::set1(0xFF00);
			const uint64x4_t tmp = uint8x32_t::blend(mat[r], uint128x2_t::template slli<1>(mat[r + 1]), blend_mask);
			mat[r + 1] = uint8x32_t::blend(uint128x2_t::template srli<1>(mat[r]), mat[r + 1], blend_mask);
			mat[r] = tmp;
		}
		// store result
		for (size_t i = 0; i < 32; i++) {
			uint64x4_t::store(data + i * 64, mat[i]);
		}
#else
		for (unsigned i = 0; i < 32; i++) {
			for (unsigned j = i + 1; j < 32; j++) {
				uint8_t tmp = mat[j * 64 + i];
				mat[j * 64 + i] = mat[i * 64 + j];
				mat[i * 64 + j] = tmp;
			}
		}
#endif
	}
};

#endif//CRYPTANALYSISLIB_SIMD_MATRIX_SIMPLE_H
