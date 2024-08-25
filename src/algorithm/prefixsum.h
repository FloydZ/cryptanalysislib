#ifndef CRYPTANALYSISLIB_PREFIXSUM_H
#define CRYPTANALYSISLIB_PREFIXSUM_H

#ifdef USE_AVX2
#include <immintrin.h>

static inline void sse_prefixsum_u32(uint32_t *in) noexcept {
	__m128i x = _mm_loadu_si128((__m128i *) in);
	// x = 1, 2, 3, 4
	x = _mm_add_epi32(x, _mm_slli_si128(x, 4));
	// x = 1, 2, 3, 4
	//   + 0, 1, 2, 3
	//   = 1, 3, 5, 7
	x = _mm_add_epi32(x, _mm_slli_si128(x, 8));
	// x = 1, 3, 5, 7
	//   + 0, 0, 1, 3
	//   = 1, 3, 6, 10
	_mm_storeu_si128((__m128i *) in, x);
	// return x;
}

static inline void avx_prefix_prefixsum_u32(uint32_t *p) noexcept {
	__m256i x = _mm256_loadu_si256((__m256i *) p);
	x = _mm256_add_epi32(x, _mm256_slli_si256(x, 4));
	x = _mm256_add_epi32(x, _mm256_slli_si256(x, 8));
	_mm256_storeu_si256((__m256i *) p, x);
}

static inline __m128i sse_prefixsum_accumulate_u32(uint32_t *p,
                                                   const __m128i s) noexcept {
	__m128i d = (__m128i) _mm_broadcast_ss((float *) &p[3]);
	__m128i x = _mm_loadu_si128((__m128i *) p);
	x = _mm_add_epi32(s, x);
	_mm_storeu_si128((__m128i *) p, x);
	return _mm_add_epi32(s, d);
}

// This number was set experimental by FLoyd via the benchmark in
// `bench/algorithm/prefixsum`.
// The benchmark was conducted on a 12th Gen Intel(R) Core(TM) i5-1240P
constexpr size_t prefixsum_u32_block_size = 32;

/// specialized sub routines.
/// PrefixSum:
/// 	a[0] = a[0]
///	a[1] = a[0] + a[1]
/// \param a
/// \param s
/// \return
static __m128i avx2_local_prefixsum_u32(uint32_t *a,
                                        __m128i s) noexcept {
	for (uint32_t i = 0; i < prefixsum_u32_block_size; i += 8) {
		avx_prefix_prefixsum_u32(&a[i]);
	}

	for (uint32_t i = 0; i < prefixsum_u32_block_size; i += 4) {
		s = sse_prefixsum_accumulate_u32(&a[i], s);
	}

	return s;
}

static void avx2_prefixsum_u32(uint32_t *a,
                               const size_t n) noexcept {
	// simple version for small inputs
	if (n < prefixsum_u32_block_size) {
		for (uint32_t i = 1; i < n; i++) {
			a[i] += a[i - 1];
		}
		return;
	}

	__m128i s = _mm_setzero_si128();
	uint32_t i = 0;
	for (; i + prefixsum_u32_block_size <= n; i += prefixsum_u32_block_size) {
		s = avx2_local_prefixsum_u32(a + i, s);
	}

	// tail mngt.
	for (; i < n; i++) {
		a[i] += a[i - 1];
	}
}
#endif

namespace cryptanalysislib::algorithm {

	/// inplace prefix sum algorithm
	/// \tparam T
	/// \param data
	/// \param len
	/// \return
	template<typename T>
#if __cplusplus > 201709L
		requires std::is_arithmetic_v<T>
#endif
	constexpr void prefixsum(T *data,
	                      	 const size_t len) {
#ifdef USE_AVX2
		constexpr bool use_avx = true;
#else
		constexpr bool use_avx = false;
#endif

		if constexpr (std::is_same_v<T, uint32_t> && use_avx) {
			avx2_prefixsum_u32(data, len);
			return;
		}
		for (size_t i = 1; i < len; ++i) {
			data[i] += data[i-1];
		}
	}

	/// \tparam ForwardIt
	/// \param first
	/// \param last
	/// \return
	template<typename ForwardIt>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	void prefixsum(ForwardIt first,
	               ForwardIt last) {
		static_assert(std::is_arithmetic_v<typename ForwardIt::value_type>);
		const auto count = std::distance(first, last);
		prefixsum(&(*first), count);
	}
};
#endif//CRYPTANALYSISLIB_PREFIXSUM_H
