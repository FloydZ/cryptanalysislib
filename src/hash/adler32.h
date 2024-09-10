#ifndef CRYPTANALYSISLIB_HASH_ADLER32_H
#define CRYPTANALYSISLIB_HASH_ADLER32_H

#include <cstdlib>
#include <cstdint>

#if defined(USE_AVX2) || defined(USE_AVX512F)
#include <immintrin.h>
#endif

/// TODO SSE and neon translation from:
/// 	https://github.com/mcountryman/simd-adler32

namespace cryptanalysislib::hash::adler32::internal {
	constexpr static uint32_t MOD = 65521u;
	constexpr static uint32_t NMAX = 5552u;

#if defined(USE_AVX2) || defined(USE_AVX512F)
	static uint32_t avx2_hadd_adler32(const __m256i v) noexcept {
		__m128i tmp[2];
		_mm256_store_si256((__m256i *)tmp, v);
	    auto sum = _mm_add_epi32(tmp[0], tmp[1]);
	    auto hi = _mm_unpackhi_epi64(sum, sum);
	
	    sum = _mm_add_epi32(hi, sum);
	    hi = _mm_shuffle_epi32(sum, _MM_SHUFFLE(2, 3, 0, 1));
	
	    sum = _mm_add_epi32(sum, hi);
	    return _mm_cvtsi128_si32(sum);
	}
#endif
};


constexpr static uint32_t adler32_update(uint16_t a,
										 uint16_t b,
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

///
///
constexpr static uint32_t adler32(uint32_t val,
                                  const uint8_t *in,
                                  const size_t in_len_) noexcept {
	using cryptanalysislib::hash::adler32::internal::MOD;
	using cryptanalysislib::hash::adler32::internal::NMAX;

	uint32_t a = val & 0xffffu;
	uint32_t b = val >> 16u;
	return adler32_update(a, b, in, in_len_);
}

#ifdef USE_AVX2
#include <immintrin.h>
 /// translation from: https://github.com/mcountryman/simd-adler32/blob/main/src/imp/avx2.rs
/// \param val
/// \param in
/// \param in_len
/// \return
constexpr static uint32_t avx2_adler32(const uint32_t val,
                                  	   const uint8_t *in,
                                  	   const size_t in_len) noexcept {
	using cryptanalysislib::hash::adler32::internal::MOD;
	using cryptanalysislib::hash::adler32::internal::NMAX;
	using cryptanalysislib::hash::adler32::internal::avx2_hadd_adler32;
	constexpr static size_t BLOCK_SIZE = 32u;

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
    __m256i p_v = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, a*blocks);
    __m256i a_v = _mm256_setzero_si256();
    __m256i b_v = _mm256_set_epi32(0, 0, 0, 0, 0, 0, 0, b);
	
	for (uint32_t i = 0; i < blocks; i++) {
        const __m256i block = _mm256_loadu_si256((__m256i *)(in + i*32));
        p_v = _mm256_add_epi32(p_v, a_v);
        a_v = _mm256_add_epi32(a_v, _mm256_sad_epu8(block, zero_v));
        const __m256i mad = _mm256_maddubs_epi16(block, weights);
        b_v = _mm256_add_epi32(b_v, _mm256_madd_epi16(mad, one_v));
	}

    b_v = _mm256_add_epi32(b_v, _mm256_slli_epi32(p_v, 5));
    a += avx2_hadd_adler32(a_v);
    b  = avx2_hadd_adler32(b_v);

	return adler32_update(a, b, in + blocks*BLOCK_SIZE, blocks_remainder);
}
#endif

#ifdef USE_AVX512F 
/// NOTE: needs AVX512F and AVX512BW
constexpr static uint32_t avx512_adler32(uint32_t adler32,
                                  		 const uint8_t *in,
                                  		 const size_t in_len_) noexcept {
	constexpr static uint32_t BLOCK_SIZE = 64u;
	if (in_len_ < BLOCK_SIZE) {
		return avx2_adler32(adler32, in, in_len_);
	}

	using cryptanalysislib::hash::adler32::internal::MOD;
	using cryptanalysislib::hash::adler32::internal::NMAX;
    _mm512_set_epi8(
      1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
      24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
      45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64,
    )

		  let blocks = chunk.chunks_exact(BLOCK_SIZE);
    let blocks_remainder = blocks.remainder();

    let one_v = _mm512_set1_epi16(1);
    let zero_v = _mm512_setzero_si512();
    let weights = get_weights();

    let p_v = (*a * blocks.len() as u32) as _;
    let mut p_v = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, p_v);
    let mut a_v = _mm512_setzero_si512();
    let mut b_v = _mm512_set_epi32(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, *b as _);

    for block in blocks {
      let block_ptr = block.as_ptr() as *const _;
      let block = _mm512_loadu_si512(block_ptr);

      p_v = _mm512_add_epi32(p_v, a_v);

      a_v = _mm512_add_epi32(a_v, _mm512_sad_epu8(block, zero_v));
      let mad = _mm512_maddubs_epi16(block, weights);
      b_v = _mm512_add_epi32(b_v, _mm512_madd_epi16(mad, one_v));
    }

    b_v = _mm512_add_epi32(b_v, _mm512_slli_epi32(p_v, 6));

    *a += reduce_add(a_v);
    *b = reduce_add(b_v);
}
#endif
#endif
