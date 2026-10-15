#ifndef CRYPTANALYSISLIB_ALGORITHM_BITS_BASE2_H
#define CRYPTANALYSISLIB_ALGORITHM_BITS_BASE2_H

#include <cstddef>
#include <cstdint>

#if (defined(USE_AVX512BITALG) && defined(USE_AVX512BW)) || defined(USE_BMI2)
#include <immintrin.h>
#endif

/// Conversion between bytes and their base2 ascii representation, i.e. every
/// byte is written as 8 chars `'0'`/`'1'`, the most significant bit first:
/// 	"01001000" <-> 0x48 ('H')
/// source: https://gist.github.com/Wunkolo/98cee1ab725d3d0b92f47eee2acfc7e4
/// With AVX512-BITALG `vpshufbitqmb` (`_mm512_bitshuffle_epi64_mask`) converts
/// 8 bytes at once: every 64 bit lane is a lookup table of 64 bits, from which
/// the bits at the 8 indices of the other operand are gathered into 8 bits of
/// a mask register.
namespace cryptanalysislib {
namespace internal {
	/// \param w[in]: 8 ascii chars '0'/'1', the first one in the lowest byte
	/// \return the byte, the first char is the most significant bit
	[[nodiscard]] constexpr inline uint8_t base2_decode_word(const uint64_t w) noexcept {
#ifdef USE_BMI2
		if !consteval {
			return uint8_t(_pext_u64(__builtin_bswap64(w), 0x0101010101010101ull));
		}
#endif
		// byte i of `w & 0x01..01` times `2^j` in byte j: lands at bit 8(i+j) + j,
		// for i + j = 7 these are the bits [56, 63], without carries
		return uint8_t(((w & 0x0101010101010101ull) * 0x8040201008040201ull) >> 56u);
	}

	/// \param b[in]: byte
	/// \return 8 ascii chars '0'/'1', the first one (the most significant
	/// 	bit of `b`) in the lowest byte
	[[nodiscard]] constexpr inline uint64_t base2_encode_word(const uint8_t b) noexcept {
#ifdef USE_BMI2
		if !consteval {
			return __builtin_bswap64(_pdep_u64(b, 0x0101010101010101ull)) | 0x3030303030303030ull;
		}
#endif
		// byte j: bit 7-j of `b`, normalized to 0/1
		const uint64_t t = (b * 0x0101010101010101ull) & 0x0102040810204080ull;
		return (((t + 0x7f7f7f7f7f7f7f7full) & 0x8080808080808080ull) >> 7u) | 0x3030303030303030ull;
	}
} // end namespace internal

/// \param out[out]: `n` bytes
/// \param in[in]: `8n` chars '0'/'1', the most significant bit of every byte
/// 	first. Other chars are not detected, only their lowest bit is used.
/// \param n[in]: number of bytes
constexpr inline void base2_decode(uint8_t *out,
                                   const char *in,
                                   const size_t n) noexcept {
	size_t i = 0;
#if defined(USE_AVX512BITALG) && defined(USE_AVX512BW)
	if !consteval {
		// lane byte k selects bit 8(7-k): the lowest bit of char 7-k
		const __m512i idx = _mm512_set1_epi64(0x0008101820283038ll);
		for (; (i + 8) <= n; i += 8) {
			const __m512i a = _mm512_loadu_si512((const void *)(in + 8*i));
			const __mmask64 m = _mm512_bitshuffle_epi64_mask(a, idx);
			_store_mask64((__mmask64 *)(out + i), m);
		}

		if (i < n) {
			const size_t r = n - i;
			const __m512i a = _mm512_maskz_loadu_epi8((1ull << (8*r)) - 1ull, in + 8*i);
			const uint64_t m = _cvtmask64_u64(_mm512_bitshuffle_epi64_mask(a, idx));
			__builtin_memcpy(out + i, &m, r);
		}
		return;
	}
#endif

	for (; i < n; ++i) {
		uint64_t w = 0;
		if consteval {
			for (uint32_t j = 0; j < 8; ++j) {
				w |= uint64_t(uint8_t(in[8*i + j])) << (8u*j);
			}
		} else {
			__builtin_memcpy(&w, in + 8*i, 8);
		}
		out[i] = internal::base2_decode_word(w);
	}
}

/// \param out[out]: `8n` chars '0'/'1', the most significant bit of every
/// 	byte first. Not null terminated.
/// \param in[in]: `n` bytes
/// \param n[in]: number of bytes
constexpr inline void base2_encode(char *out,
                                   const uint8_t *in,
                                   const size_t n) noexcept {
	size_t i = 0;
#if defined(USE_AVX512BITALG) && defined(USE_AVX512BW)
	if !consteval {
		// lane k (output chars 8k..8k+7) selects the bits 7..0 of byte k
		const __m512i idx = _mm512_set_epi64(
			0x0001020304050607ll + 0x0101010101010101ll * 0x38,
			0x0001020304050607ll + 0x0101010101010101ll * 0x30,
			0x0001020304050607ll + 0x0101010101010101ll * 0x28,
			0x0001020304050607ll + 0x0101010101010101ll * 0x20,
			0x0001020304050607ll + 0x0101010101010101ll * 0x18,
			0x0001020304050607ll + 0x0101010101010101ll * 0x10,
			0x0001020304050607ll + 0x0101010101010101ll * 0x08,
			0x0001020304050607ll);
		const __m512i zero = _mm512_set1_epi8('0'), one = _mm512_set1_epi8('1');
		for (; (i + 8) <= n; i += 8) {
			uint64_t w;
			__builtin_memcpy(&w, in + i, 8);
			const __mmask64 m = _mm512_bitshuffle_epi64_mask(_mm512_set1_epi64(w), idx);
			_mm512_storeu_si512((void *)(out + 8*i), _mm512_mask_blend_epi8(m, zero, one));
		}

		if (i < n) {
			const size_t r = n - i;
			uint64_t w = 0;
			__builtin_memcpy(&w, in + i, r);
			const __mmask64 m = _mm512_bitshuffle_epi64_mask(_mm512_set1_epi64(w), idx);
			_mm512_mask_storeu_epi8(out + 8*i, (1ull << (8*r)) - 1ull, _mm512_mask_blend_epi8(m, zero, one));
		}
		return;
	}
#endif

	for (; i < n; ++i) {
		const uint64_t w = internal::base2_encode_word(in[i]);
		if consteval {
			for (uint32_t j = 0; j < 8; ++j) {
				out[8*i + j] = char(w >> (8u*j));
			}
		} else {
			__builtin_memcpy(out + 8*i, &w, 8);
		}
	}
}
} // end namespace cryptanalysislib
#endif
