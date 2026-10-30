#ifndef CRYPTANALYSISLIB_HASH_ADLER32_H
#define CRYPTANALYSISLIB_HASH_ADLER32_H

#include <cstdlib>
#include <cstdint>

#include "dispatch.h"

#if defined(USE_AVX2) || defined(USE_AVX512F)
#include <immintrin.h>
#include "simd/simd.h"
#endif

namespace cryptanalysislib::hash::adler32::internal {
	constexpr static uint32_t MOD = 65521u;
	constexpr static uint32_t NMAX = 5552u;
};// namespace cryptanalysislib::hash::adler32::internal

// NOTE: in `cryptanalysislib`, as zlib declares a global `adler32`
namespace cryptanalysislib {
/// Updates an Adler-32 checksum with new data
/// Implements the standard Adler-32 algorithm with optimized loop unrolling
/// 
/// \param a[in] Lower 16 bits of the current Adler-32 value
/// \param b[in] Upper 16 bits of the current Adler-32 value
/// \param in[in] Pointer to input data
/// \param in_len_[in] Length of input data in bytes
/// \return Updated Adler-32 checksum (b << 16 | a)
constexpr static uint32_t adler32_update(uint32_t a,
										 uint32_t b,
                                  		 const uint8_t *in,
                                  		 const size_t in_len_) noexcept {
	using cryptanalysislib::hash::adler32::internal::MOD;
	using cryptanalysislib::hash::adler32::internal::NMAX;
	
	uint32_t blk_len, i;
	size_t in_len = in_len_;

	blk_len = in_len % NMAX;
	while (in_len) {
		// loop unroll factor: 8
		for (i = 0; i + 7 < blk_len; i += 8) {
			a += in[0]; b += a;
			a += in[1]; b += a;
			a += in[2]; b += a;
			a += in[3]; b += a;
			a += in[4]; b += a;
			a += in[5]; b += a;
			a += in[6]; b += a;
			a += in[7]; b += a;
			in += 8;
		}
		
		for (; i < blk_len; ++i) {
			a += *in++, b += a;
		}

		a %= MOD;
		b %= MOD;
		in_len -= blk_len;
		blk_len = NMAX;
	}

	return (uint32_t)(b << 16u) + (uint32_t)a;
}

/// Calculates or updates an Adler-32 checksum for a block of data
/// A wrapper around adler32_update that handles the initial value splitting
/// 
/// \param val[in] Initial Adler-32 value (typically 1 for new checksums)
/// \param in[in] Pointer to input data
/// \param in_len_[in] Length of input data in bytes
/// \return Calculated Adler-32 checksum
constexpr static uint32_t adler32(uint32_t val,
                                  const uint8_t *in,
                                  const size_t in_len_) noexcept {
	uint32_t a = val & 0xffffu;
	uint32_t b = val >> 16u;
	return adler32_update(a, b, in, in_len_);
}

#ifdef USE_AVX2
#include <immintrin.h>

/// AVX2-optimized version of Adler-32 checksum calculation
/// Based on implementation from: https://github.com/mcountryman/simd-adler32/blob/main/src/imp/avx2.rs
/// Uses SIMD operations to process 32 bytes at a time
/// 
/// \param val[in] Initial Adler-32 value (typically 1 for new checksums)
/// \param in[in] Pointer to input data
/// \param in_len[in] Length of input data in bytes
/// \return Calculated Adler-32 checksum
constexpr static uint32_t avx2_adler32(const uint32_t val,
                                  	   const uint8_t *in,
                                  	   const size_t in_len) noexcept {
	using cryptanalysislib::hash::adler32::internal::MOD;
	using cryptanalysislib::hash::adler32::internal::NMAX;

	// NOTE: bytes per AVX2 register, fixed by the algorithm
	constexpr size_t BLOCK_SIZE = 32u;

	if (in_len < BLOCK_SIZE) {
		return adler32(val, in, in_len);
	}
	
	const size_t blocks = in_len / BLOCK_SIZE;
	const size_t blocks_remainder = in_len % BLOCK_SIZE;


    const __m256i one_v = _mm256_set1_epi16(1);
    const __m256i zero_v = _mm256_setzero_si256();
    const __m256i weights =  _mm256_set_epi8(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
	    17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32);

	uint32_t a = val & 0xffffu;
	uint32_t b = val >> 16u;

	// NOTE: at most NMAX bytes are summed up before a and b are reduced,
	// otherwise the 32-bit lanes overflow.
	constexpr size_t CHUNK = NMAX / BLOCK_SIZE;
	size_t blocks_left = blocks;
	const uint8_t *ptr = in;
	while (blocks_left) {
		const size_t nb = blocks_left < CHUNK ? blocks_left : CHUNK;
		__m256i p_v = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, a*nb);
		__m256i a_v = _mm256_setzero_si256();
		__m256i b_v = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, b);

		for (size_t i = 0; i < nb; i++) {
			const __m256i block = _mm256_loadu_si256((__m256i *)ptr);
			p_v = _mm256_add_epi32(p_v, a_v);
			a_v = _mm256_add_epi32(a_v, _mm256_sad_epu8(block, zero_v));
			const __m256i mad = _mm256_maddubs_epi16(block, weights);
			b_v = _mm256_add_epi32(b_v, _mm256_madd_epi16(mad, one_v));
			ptr += BLOCK_SIZE;
		}

		b_v = _mm256_add_epi32(b_v, _mm256_slli_epi32(p_v, 5));
		uint32x8_t ta, tb;
		ta.v256 = a_v; tb.v256 = b_v;
		a = (a + uint32x8_t::reduce_add(ta)) % MOD;
		b = uint32x8_t::reduce_add(tb) % MOD;
		blocks_left -= nb;
	}

	return adler32_update(a, b, ptr, blocks_remainder);
}
#endif

#ifdef USE_AVX512F 

/// AVX512-optimized version of Adler-32 checksum calculation
/// Requires both AVX512F and AVX512BW instruction set extensions
/// Uses SIMD operations to process 64 bytes at a time for maximum throughput
/// 
/// \param val[in] Initial Adler-32 value (typically 1 for new checksums)
/// \param in[in] Pointer to input data
/// \param in_len[in] Length of input data in bytes
/// \return Calculated Adler-32 checksum
constexpr static uint32_t avx512_adler32(uint32_t val,
                                  		 const uint8_t *in,
                                  		 const size_t in_len) noexcept {
	constexpr uint32_t BLOCK_SIZE = 64u;
	if (in_len < BLOCK_SIZE) {
		return avx2_adler32(val, in, in_len);
	}

	using cryptanalysislib::hash::adler32::internal::MOD;
	using cryptanalysislib::hash::adler32::internal::NMAX;
    const __m512i weights = _mm512_set_epi8(
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
      45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64
    );

	const size_t blocks = in_len / BLOCK_SIZE;
	const size_t blocks_remainder = in_len % BLOCK_SIZE;

    const __m512i one_v = _mm512_set1_epi16(1);
    const __m512i zero_v = _mm512_setzero_si512();

	uint32_t a = val & 0xffffu;
	uint32_t b = val >> 16u;

	// NOTE: at most NMAX bytes are summed up before a and b are reduced,
	// otherwise the 32-bit lanes overflow.
	constexpr size_t CHUNK = NMAX / BLOCK_SIZE;
	size_t blocks_left = blocks;
	const uint8_t *ptr = in;
	while (blocks_left) {
		const size_t nb = blocks_left < CHUNK ? blocks_left : CHUNK;
		__m512i p_v = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, a*nb);
		__m512i a_v = _mm512_setzero_si512();
		__m512i b_v = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b);

		for (size_t i = 0; i < nb; i++) {
			// NOTE: 64-byte stride (was 32)
			const __m512i block = _mm512_loadu_si512((__m512i *)ptr);
			p_v = _mm512_add_epi32(p_v, a_v);
			a_v = _mm512_add_epi32(a_v, _mm512_sad_epu8(block, zero_v));
			const __m512i mad = _mm512_maddubs_epi16(block, weights);
			b_v = _mm512_add_epi32(b_v, _mm512_madd_epi16(mad, one_v));
			ptr += BLOCK_SIZE;
		}

		b_v = _mm512_add_epi32(b_v, _mm512_slli_epi32(p_v, 6));
		uint32x16_t ta, tb;
		ta.v512 = a_v; tb.v512 = b_v;
		a = (a + uint32x16_t::reduce_add(ta)) % MOD;
		b = uint32x16_t::reduce_add(tb) % MOD;
		blocks_left -= nb;
	}

	return adler32_update(a, b, ptr, blocks_remainder);
}
#endif

/// Adler-32 with the fastest available implementation (scalar, AVX2,
/// AVX-512): on the first call all candidates are benchmarked once on a
/// fixed 4 KiB buffer (like `binary_search_dispatch` in `search/binary.h`),
/// the choice is a function local static (thread safe initialization).
/// \param val[in] initial Adler-32 value (1 for a new checksum)
/// \param in[in] input data
/// \param in_len[in] length of the input in bytes
/// \return the Adler-32 checksum, identical for all implementations
static inline uint32_t adler32_dispatch(const uint32_t val,
                                        const uint8_t *in,
                                        const size_t in_len) noexcept {
	using FF = uint32_t (*)(uint32_t, const uint8_t *, size_t);
	static FF functions[] = {
		adler32,
#ifdef USE_AVX2
		avx2_adler32,
#endif
#ifdef USE_AVX512F
		avx512_adler32,
#endif
	};

	static const FF best = []() noexcept {
		static uint8_t buf[4096];
		for (size_t i = 0; i < sizeof(buf); ++i) { buf[i] = uint8_t(i * 131u + 7u); }
		FF out = functions[0];
		generic_dispatch(out, functions, sizeof(functions) / sizeof(functions[0]),
		                 uint32_t(1), (const uint8_t *)buf, sizeof(buf));
		return out;
	}();

	return best(val, in, in_len);
}
} // namespace cryptanalysislib
#endif
