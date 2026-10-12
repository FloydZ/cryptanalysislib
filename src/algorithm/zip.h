#ifndef CRYPTANALYSISLIB_ALGORITHM_BITINTERLEAVE_H
#define CRYPTANALYSISLIB_ALGORITHM_BITINTERLEAVE_H

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <type_traits>

#ifdef USE_AVX2
#include <immintrin.h>

// #if defined(__GNUC__) && !defined(__clang__) && !defined(__ICC)
// // this maps to multiple instructions in avx, but one in avx2 (?)
// static inline void
//         __attribute__((__always_inline__))
//         _mm256_storeu2_m128i(__m128i* const hiaddr,
//                              __m128i* const loaddr,
//                              const __m256i a) {
// 	_mm_storeu_si128(loaddr, _mm256_castsi256_si128(a));
// 	_mm_storeu_si128(hiaddr, _mm256_extracti128_si256(a, 1));
// }
// #endif

/// Computes Morton codes (Z-order curve values) for interleaving bits from two input arrays
///
/// Given two values like 0b1011 (11 in decimal) and 0b1100 (12 in decimal),
/// computes the interleaved value 0b11011010.
///
/// \param out[out]: Pointer to the output array where Morton codes will be stored
/// \param in0[in]: Pointer to the first input array
/// \param in1[in]: Pointer to the second input array
static inline void morton_vec8(uint64_t *out,
							   uint32_t *in0,
							   uint32_t *in1) {

	// nybble -> byte lookups
	__m256i m0 = _mm256_set_epi8(85, 84, 81, 80, 69, 68, 65, 64, 21, 20, 17,
								 16, 5, 4, 1, 0, 85, 84, 81, 80, 69, 68, 65,
								 64, 21, 20, 17, 16, 5, 4, 1, 0);

	__m256i m1 = _mm256_slli_epi64( m0, 1);

	__m256i nybble_mask = _mm256_set1_epi8(0x0F);

	__m256i word0 = _mm256_loadu_si256((__m256i *)in0);
	__m256i word1 = _mm256_loadu_si256((__m256i *)in1);

	__m256i even0 = _mm256_and_si256( word0, nybble_mask);  // 1 / .33 ?
	__m256i odd0  = _mm256_and_si256( _mm256_srli_epi64( word0, 4), nybble_mask);

	__m256i even1 = _mm256_and_si256( word1, nybble_mask);
	__m256i odd1  = _mm256_and_si256( _mm256_srli_epi64( word1, 4), nybble_mask);

	// lookup

	even0 = _mm256_shuffle_epi8( m0, even0 );  // 1 / 1
	odd0  = _mm256_shuffle_epi8( m0, odd0  );

	even1 = _mm256_shuffle_epi8( m1, even1 );
	odd1  = _mm256_shuffle_epi8( m1, odd1  );

	// combine

	__m256i even = _mm256_or_si256( even0, even1 );  // 1 / .33
	__m256i odd  = _mm256_or_si256( odd0, odd1 );

	// byte interleave
	__m256i los = _mm256_unpacklo_epi8( even, odd );  // 1 / 1
	__m256i his = _mm256_unpackhi_epi8( even, odd );

	// AC,BD => AB,CD
	// interleaved 128i stores to 8 x uint64_t
	_mm256_storeu2_m128i ((__m128i*) (out+4),(__m128i*) out, los);
	_mm256_storeu2_m128i ((__m128i*) (out+6),(__m128i*) (out+2), his);
}

namespace cryptanalysislib::internal {
	/// Interleaves the `n` 8-bit elements of `in1` and `in2` (AVX2,
	/// 32 elements of each input per iteration, scalar tail):
	/// out = [in1[0], in2[0], in1[1], in2[1], ...] (2n elements)
	/// \param out[out]: output array of 2n elements, must not overlap the inputs
	/// \param in1[in]: first input array of n elements
	/// \param in2[in]: second input array of n elements
	/// \param n[in]: number of elements in `in1` and `in2`
	static inline void zip_u8_avx2(uint8_t *__restrict__ out,
	                               const uint8_t *__restrict__ in1,
	                               const uint8_t *__restrict__ in2,
	                               const size_t n) noexcept {
		size_t i = 0;
		for (; (i + 32) <= n; i += 32) {
			const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
			const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
			const __m256i lo = _mm256_unpacklo_epi8(a, b);
			const __m256i hi = _mm256_unpackhi_epi8(a, b);
			// NOTE: unaligned stores, `out` points into an arbitrary array
			_mm256_storeu_si256((__m256i *)(out + 2 * i), _mm256_permute2x128_si256(lo, hi, 0x20));
			_mm256_storeu_si256((__m256i *)(out + 2 * i + 32), _mm256_permute2x128_si256(lo, hi, 0x31));
		}
		for (; i < n; ++i) {
			out[2 * i] = in1[i];
			out[2 * i + 1] = in2[i];
		}
	}

	/// Interleaves the `n` 16-bit elements of `in1` and `in2` (AVX2,
	/// 16 elements of each input per iteration, scalar tail):
	/// out = [in1[0], in2[0], in1[1], in2[1], ...] (2n elements)
	/// \param out[out]: output array of 2n elements, must not overlap the inputs
	/// \param in1[in]: first input array of n elements
	/// \param in2[in]: second input array of n elements
	/// \param n[in]: number of elements in `in1` and `in2`
	static inline void zip_u16_avx2(uint16_t *__restrict__ out,
	                                const uint16_t *__restrict__ in1,
	                                const uint16_t *__restrict__ in2,
	                                const size_t n) noexcept {
		size_t i = 0;
		for (; (i + 16) <= n; i += 16) {
			const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
			const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
			const __m256i lo = _mm256_unpacklo_epi16(a, b);
			const __m256i hi = _mm256_unpackhi_epi16(a, b);
			// NOTE: unaligned stores, `out` points into an arbitrary array
			_mm256_storeu_si256((__m256i *)(out + 2 * i), _mm256_permute2x128_si256(lo, hi, 0x20));
			_mm256_storeu_si256((__m256i *)(out + 2 * i + 16), _mm256_permute2x128_si256(lo, hi, 0x31));
		}
		for (; i < n; ++i) {
			out[2 * i] = in1[i];
			out[2 * i + 1] = in2[i];
		}
	}

	/// Interleaves the `n` 32-bit elements of `in1` and `in2` (AVX2,
	/// 8 elements of each input per iteration, scalar tail):
	/// out = [in1[0], in2[0], in1[1], in2[1], ...] (2n elements)
	/// \param out[out]: output array of 2n elements, must not overlap the inputs
	/// \param in1[in]: first input array of n elements
	/// \param in2[in]: second input array of n elements
	/// \param n[in]: number of elements in `in1` and `in2`
	static inline void zip_u32_avx2(uint32_t *__restrict__ out,
	                                const uint32_t *__restrict__ in1,
	                                const uint32_t *__restrict__ in2,
	                                const size_t n) noexcept {
		size_t i = 0;
		for (; (i + 8) <= n; i += 8) {
			const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
			const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
			const __m256i lo = _mm256_unpacklo_epi32(a, b);
			const __m256i hi = _mm256_unpackhi_epi32(a, b);
			// NOTE: unaligned stores, `out` points into an arbitrary array
			_mm256_storeu_si256((__m256i *)(out + 2 * i), _mm256_permute2x128_si256(lo, hi, 0x20));
			_mm256_storeu_si256((__m256i *)(out + 2 * i + 8), _mm256_permute2x128_si256(lo, hi, 0x31));
		}
		for (; i < n; ++i) {
			out[2 * i] = in1[i];
			out[2 * i + 1] = in2[i];
		}
	}

	/// Interleaves the `n` 64-bit elements of `in1` and `in2` (AVX2,
	/// 4 elements of each input per iteration, scalar tail):
	/// out = [in1[0], in2[0], in1[1], in2[1], ...] (2n elements)
	/// \param out[out]: output array of 2n elements, must not overlap the inputs
	/// \param in1[in]: first input array of n elements
	/// \param in2[in]: second input array of n elements
	/// \param n[in]: number of elements in `in1` and `in2`
	static inline void zip_u64_avx2(uint64_t *__restrict__ out,
	                                const uint64_t *__restrict__ in1,
	                                const uint64_t *__restrict__ in2,
	                                const size_t n) noexcept {
		size_t i = 0;
		for (; (i + 4) <= n; i += 4) {
			const __m256i a = _mm256_loadu_si256((const __m256i *)(in1 + i));
			const __m256i b = _mm256_loadu_si256((const __m256i *)(in2 + i));
			const __m256i lo = _mm256_unpacklo_epi64(a, b);
			const __m256i hi = _mm256_unpackhi_epi64(a, b);
			// NOTE: unaligned stores, `out` points into an arbitrary array
			_mm256_storeu_si256((__m256i *)(out + 2 * i), _mm256_permute2x128_si256(lo, hi, 0x20));
			_mm256_storeu_si256((__m256i *)(out + 2 * i + 4), _mm256_permute2x128_si256(lo, hi, 0x31));
		}
		for (; i < n; ++i) {
			out[2 * i] = in1[i];
			out[2 * i + 1] = in2[i];
		}
	}
} // end namespace cryptanalysislib::internal

#endif // USE_AVX2

#ifdef USE_AVX512F
#include <immintrin.h>

namespace cryptanalysislib::internal {
#ifdef USE_AVX512BW
	/// Interleaves the `n` 8-bit elements of `in1` and `in2` (AVX-512,
	/// 64 elements of each input per iteration, scalar tail):
	/// out = [in1[0], in2[0], in1[1], in2[1], ...] (2n elements)
	/// \param out[out]: output array of 2n elements, must not overlap the inputs
	/// \param in1[in]: first input array of n elements
	/// \param in2[in]: second input array of n elements
	/// \param n[in]: number of elements in `in1` and `in2`
	static inline void zip_u8_avx512(uint8_t *__restrict__ out,
	                                 const uint8_t *__restrict__ in1,
	                                 const uint8_t *__restrict__ in2,
	                                 const size_t n) noexcept {
		// `unpacklo/hi` work within the 128-bit lanes: the outputs are the
		// lanes lo.0 hi.0 lo.1 hi.1 and lo.2 hi.2 lo.3 hi.3
		const __m512i idx1 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
		const __m512i idx2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
		size_t i = 0;
		for (; (i + 64) <= n; i += 64) {
			const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
			const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
			const __m512i lo = _mm512_unpacklo_epi8(a, b);
			const __m512i hi = _mm512_unpackhi_epi8(a, b);
			_mm512_storeu_si512((void *)(out + 2 * i), _mm512_permutex2var_epi64(lo, idx1, hi));
			_mm512_storeu_si512((void *)(out + 2 * i + 64), _mm512_permutex2var_epi64(lo, idx2, hi));
		}
		for (; i < n; ++i) {
			out[2 * i] = in1[i];
			out[2 * i + 1] = in2[i];
		}
	}

	/// Interleaves the `n` 16-bit elements of `in1` and `in2` (AVX-512,
	/// 32 elements of each input per iteration, scalar tail):
	/// out = [in1[0], in2[0], in1[1], in2[1], ...] (2n elements)
	/// \param out[out]: output array of 2n elements, must not overlap the inputs
	/// \param in1[in]: first input array of n elements
	/// \param in2[in]: second input array of n elements
	/// \param n[in]: number of elements in `in1` and `in2`
	static inline void zip_u16_avx512(uint16_t *__restrict__ out,
	                                  const uint16_t *__restrict__ in1,
	                                  const uint16_t *__restrict__ in2,
	                                  const size_t n) noexcept {
		// `unpacklo/hi` work within the 128-bit lanes: the outputs are the
		// lanes lo.0 hi.0 lo.1 hi.1 and lo.2 hi.2 lo.3 hi.3
		const __m512i idx1 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
		const __m512i idx2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
		size_t i = 0;
		for (; (i + 32) <= n; i += 32) {
			const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
			const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
			const __m512i lo = _mm512_unpacklo_epi16(a, b);
			const __m512i hi = _mm512_unpackhi_epi16(a, b);
			_mm512_storeu_si512((void *)(out + 2 * i), _mm512_permutex2var_epi64(lo, idx1, hi));
			_mm512_storeu_si512((void *)(out + 2 * i + 32), _mm512_permutex2var_epi64(lo, idx2, hi));
		}
		for (; i < n; ++i) {
			out[2 * i] = in1[i];
			out[2 * i + 1] = in2[i];
		}
	}
#endif // USE_AVX512BW

	/// Interleaves the `n` 32-bit elements of `in1` and `in2` (AVX-512,
	/// 16 elements of each input per iteration, scalar tail):
	/// out = [in1[0], in2[0], in1[1], in2[1], ...] (2n elements)
	/// \param out[out]: output array of 2n elements, must not overlap the inputs
	/// \param in1[in]: first input array of n elements
	/// \param in2[in]: second input array of n elements
	/// \param n[in]: number of elements in `in1` and `in2`
	static inline void zip_u32_avx512(uint32_t *__restrict__ out,
	                                  const uint32_t *__restrict__ in1,
	                                  const uint32_t *__restrict__ in2,
	                                  const size_t n) noexcept {
		// `unpacklo/hi` work within the 128-bit lanes: the outputs are the
		// lanes lo.0 hi.0 lo.1 hi.1 and lo.2 hi.2 lo.3 hi.3
		const __m512i idx1 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
		const __m512i idx2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
		size_t i = 0;
		for (; (i + 16) <= n; i += 16) {
			const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
			const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
			const __m512i lo = _mm512_unpacklo_epi32(a, b);
			const __m512i hi = _mm512_unpackhi_epi32(a, b);
			_mm512_storeu_si512((void *)(out + 2 * i), _mm512_permutex2var_epi64(lo, idx1, hi));
			_mm512_storeu_si512((void *)(out + 2 * i + 16), _mm512_permutex2var_epi64(lo, idx2, hi));
		}
		for (; i < n; ++i) {
			out[2 * i] = in1[i];
			out[2 * i + 1] = in2[i];
		}
	}

	/// Interleaves the `n` 64-bit elements of `in1` and `in2` (AVX-512,
	/// 8 elements of each input per iteration, scalar tail):
	/// out = [in1[0], in2[0], in1[1], in2[1], ...] (2n elements)
	/// \param out[out]: output array of 2n elements, must not overlap the inputs
	/// \param in1[in]: first input array of n elements
	/// \param in2[in]: second input array of n elements
	/// \param n[in]: number of elements in `in1` and `in2`
	static inline void zip_u64_avx512(uint64_t *__restrict__ out,
	                                  const uint64_t *__restrict__ in1,
	                                  const uint64_t *__restrict__ in2,
	                                  const size_t n) noexcept {
		// `unpacklo/hi` work within the 128-bit lanes: the outputs are the
		// lanes lo.0 hi.0 lo.1 hi.1 and lo.2 hi.2 lo.3 hi.3
		const __m512i idx1 = _mm512_set_epi64(11, 10, 3, 2, 9, 8, 1, 0);
		const __m512i idx2 = _mm512_set_epi64(15, 14, 7, 6, 13, 12, 5, 4);
		size_t i = 0;
		for (; (i + 8) <= n; i += 8) {
			const __m512i a = _mm512_loadu_si512((const void *)(in1 + i));
			const __m512i b = _mm512_loadu_si512((const void *)(in2 + i));
			const __m512i lo = _mm512_unpacklo_epi64(a, b);
			const __m512i hi = _mm512_unpackhi_epi64(a, b);
			_mm512_storeu_si512((void *)(out + 2 * i), _mm512_permutex2var_epi64(lo, idx1, hi));
			_mm512_storeu_si512((void *)(out + 2 * i + 8), _mm512_permutex2var_epi64(lo, idx2, hi));
		}
		for (; i < n; ++i) {
			out[2 * i] = in1[i];
			out[2 * i + 1] = in2[i];
		}
	}
} // end namespace cryptanalysislib::internal
#endif // USE_AVX512F

#ifdef USE_ARM
#include <arm_neon.h>
#endif

/// Interleaves 8-bit elements from two arrays into a single array of 16-bit values
///
/// Combines each pair of 8-bit values from in1 and in2 into a 16-bit value.
/// Uses SIMD operations when possible, with scalar fallback for remaining elements.
///
/// \param out[out]: Pointer to output array to store interleaved values
/// \param in1[in]: Pointer to first input array
/// \param in2[in]: Pointer to second input array 
/// \param n[in]: Number of elements in `in1` and `in2`
static inline void zip_u8(uint16_t *__restrict__ out,
						  const uint8_t *__restrict__ in1,
						  const uint8_t *__restrict__ in2,
						  const size_t n) {
	size_t i = 0;
	// NOTE: little endian, the 16-bit value `in1[i] | in2[i] << 8` are the
	// 	two bytes in1[i], in2[i]
#if defined(USE_AVX512F) && defined(USE_AVX512BW)
	cryptanalysislib::internal::zip_u8_avx512((uint8_t *)out, in1, in2, n);
	return;
#elif defined(USE_AVX2)
	cryptanalysislib::internal::zip_u8_avx2((uint8_t *)out, in1, in2, n);
	return;
#elif defined(USE_ARM)
	// `vst2q_u8` stores the bytes interleaved: in1[0], in2[0], in1[1], ...
	for (; (i+16) <= n; i += 16) {
		const uint8x16x2_t t = {{vld1q_u8(in1 + i), vld1q_u8(in2 + i)}};
		vst2q_u8((uint8_t *)(out + i), t);
	}
#endif

	for (; i < n; i++) {
		const uint16_t t = (uint16_t)(in1[i]) | (((uint16_t)(in2[i])) << 8u);
		out[i] = t;
	}
}

namespace cryptanalysislib {
	/// Interleaves the elements of [first1, last1) and of the range starting
	/// at `first2` into the range starting at `d_first`:
	/// a0, b0, a1, b1, ... (2 * (last1 - first1) elements).
	/// NOTE: contiguous `uint8_t`/`uint16_t`/`uint32_t`/`uint64_t` ranges (input
	/// 	and output of the same type) use `internal::zip_uXX_avx512` or
	/// 	`internal::zip_uXX_avx2`
	/// \param first1[in]: beginning of the first input range
	/// \param last1[in]: end of the first input range
	/// \param first2[in]: beginning of the second input range (same length)
	/// \param d_first[out]: beginning of the output range, must not overlap
	/// 	the inputs
	/// \return iterator past the last written element
	template<class InputIt1,
	         class InputIt2,
	         class OutputIt>
#if __cplusplus > 201709L
	requires std::input_iterator<InputIt1> &&
	         std::input_iterator<InputIt2>
#endif
	constexpr OutputIt zip(InputIt1 first1,
	                       InputIt1 last1,
	                       InputIt2 first2,
	                       OutputIt d_first) noexcept {
#ifdef USE_AVX2
		using T = std::iter_value_t<InputIt1>;
		if constexpr (std::contiguous_iterator<InputIt1> &&
		              std::contiguous_iterator<InputIt2> &&
		              std::contiguous_iterator<OutputIt> &&
		              std::is_same_v<T, std::iter_value_t<InputIt2>> &&
		              std::is_same_v<T, std::iter_value_t<OutputIt>> &&
		              (std::is_same_v<T, uint8_t> || std::is_same_v<T, uint16_t> ||
		               std::is_same_v<T, uint32_t> || std::is_same_v<T, uint64_t>)) {
			if !consteval {
				const size_t n = static_cast<size_t>(last1 - first1);
				T *out = std::to_address(d_first);
				const T *a = std::to_address(first1), *b = std::to_address(first2);
				if constexpr (sizeof(T) == 1) {
#if defined(USE_AVX512F) && defined(USE_AVX512BW)
					internal::zip_u8_avx512(out, a, b, n);
#else
					internal::zip_u8_avx2(out, a, b, n);
#endif
				} else if constexpr (sizeof(T) == 2) {
#if defined(USE_AVX512F) && defined(USE_AVX512BW)
					internal::zip_u16_avx512(out, a, b, n);
#else
					internal::zip_u16_avx2(out, a, b, n);
#endif
				} else if constexpr (sizeof(T) == 4) {
#ifdef USE_AVX512F
					internal::zip_u32_avx512(out, a, b, n);
#else
					internal::zip_u32_avx2(out, a, b, n);
#endif
				} else {
#ifdef USE_AVX512F
					internal::zip_u64_avx512(out, a, b, n);
#else
					internal::zip_u64_avx2(out, a, b, n);
#endif
				}
				return d_first + 2 * n;
			}
		}
#endif
		for (; first1 != last1; ++first1, ++first2) {
			*d_first = *first1; ++d_first;
			*d_first = *first2; ++d_first;
		}
		return d_first;
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_BITINTERLEAVE_H
