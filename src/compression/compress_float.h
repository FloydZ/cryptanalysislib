#ifndef CRYPTANALYSISLIB_COMPRESSION_FLOAT_H
#define CRYPTANALYSISLIB_COMPRESSION_FLOAT_H

#include <cstddef>
#include <cstdint>

#if defined(__AVX512F__) && defined(__AVX512BF16__) && defined(__AVX512VL__)
#include <immintrin.h>
#define CRYPTANALYSISLIB_COMPRESS_FLOAT_AVX512BF16
#endif

/// Lossy compression of doubles to 16 bit: bfloat16 (8 exponent bits, 7
/// mantissa bits, i.e. the upper half of a float), relative error <= 2^-8.
/// NOTE: despite the names `to_float16`/`from_float16`, the format is
/// 	bfloat16, not IEEE half precision.
/// Source original from:
/// https://github.com/lemire/Code-used-on-Daniel-Lemire-s-blog/blob/master/2024/09/02/include/avx512compress.h
/// NOTE: the AVX-512 versions need AVX512-BF16 (e.g. Zen 4, Sapphire Rapids),
/// 	else the scalar versions below are used (bit-identical results).
namespace cryptanalysislib {
	namespace internal {
		/// double -> bfloat16 like `_mm512_cvt_roundpd_ps` (round to nearest
		/// even) followed by `_mm256_cvtneps_pbh` (round to nearest even,
		/// denormal floats are treated as zero, NaNs are quieted)
		inline uint16_t double_to_bfloat16(const double d) noexcept {
			const float f = static_cast<float>(d);
			uint32_t u = __builtin_bit_cast(uint32_t, f);
			if ((u & 0x7fffffffu) > 0x7f800000u) {
				// NaN: keep sign and payload top bits, set the quiet bit
				return static_cast<uint16_t>((u >> 16u) | 0x40u);
			}
			if ((u & 0x7f800000u) == 0) {
				// zero or denormal: (signed) zero
				return static_cast<uint16_t>((u & 0x80000000u) >> 16u);
			}
			u += 0x7fffu + ((u >> 16u) & 1u);
			return static_cast<uint16_t>(u >> 16u);
		}

		/// bfloat16 -> double (exact)
		inline double bfloat16_to_double(const uint16_t h) noexcept {
			const uint32_t u = static_cast<uint32_t>(h) << 16u;
			const float f = __builtin_bit_cast(float, u);
			return static_cast<double>(f);
		}
	} // end namespace internal

	/// converts `length` doubles to bfloat16
	/// \param dst[out]: `length` bfloat16 values
	/// \param src[in]: `length` doubles
	/// \param length[in]: number of values
	inline void to_float16(uint16_t *dst,
	                       const double *src,
	                       const size_t length) noexcept {
		size_t i = 0;
#ifdef CRYPTANALYSISLIB_COMPRESS_FLOAT_AVX512BF16
		// Process 8 elements at a time
		for (; i + 7 < length; i += 8) {
			// Load 8 double-precision floats
			const __m512d src_vec = _mm512_loadu_pd(&src[i]);
			// Convert to 16-bit floats with rounding
			const __m128bh dst_vec = _mm256_cvtneps_pbh(_mm512_cvt_roundpd_ps(src_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
			_mm_storeu_si128((__m128i *) &dst[i], (__m128i) dst_vec);
		}

		// Handle remaining elements
		if (i < length) {
			const __mmask8 mask = (1u << (length - i)) - 1u;
			const __m512d src_vec = _mm512_maskz_loadu_pd(mask, &src[i]);
			const __m128bh dst_vec = _mm256_cvtneps_pbh(_mm512_cvt_roundpd_ps(src_vec, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC));
			_mm_mask_storeu_epi16(&dst[i], mask, (__m128i) dst_vec);
		}
		return;
#endif
		for (; i < length; ++i) {
			dst[i] = internal::double_to_bfloat16(src[i]);
		}
	}

	/// converts `length` bfloat16 values to doubles (exact)
	/// \param dst[out]: `length` doubles
	/// \param src[in]: `length` bfloat16 values
	/// \param length[in]: number of values
	inline void from_float16(double *dst,
	                         const uint16_t *src,
	                         const size_t length) noexcept {
		size_t i = 0;
#ifdef CRYPTANALYSISLIB_COMPRESS_FLOAT_AVX512BF16
		// Process 8 elements at a time
		for (; i + 7 < length; i += 8) {
			const __m128i src_vec = _mm_loadu_si128((const __m128i *) &src[i]);
			const __m512d dst_vec = _mm512_cvtps_pd(_mm256_cvtpbh_ps((__m128bh) src_vec));
			_mm512_storeu_pd(&dst[i], dst_vec);
		}

		// Handle remaining elements
		if (i < length) {
			const __mmask8 mask = (1u << (length - i)) - 1u;
			const __m128i src_vec = _mm_maskz_loadu_epi16(mask, &src[i]);
			const __m512d dst_vec = _mm512_cvtps_pd(_mm256_cvtpbh_ps((__m128bh) src_vec));
			_mm512_mask_storeu_pd(&dst[i], mask, dst_vec);
		}
		return;
#endif
		for (; i < length; ++i) {
			dst[i] = internal::bfloat16_to_double(src[i]);
		}
	}
} // end namespace cryptanalysislib
#endif
