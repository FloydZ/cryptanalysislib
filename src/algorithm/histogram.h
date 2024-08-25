#ifndef CRYPTANALYSISLIB_ALGORITHM_HISTOGRAM_H
#define CRYPTANALYSISLIB_ALGORITHM_HISTOGRAM_H

#include "helper.h"
#include "memory/memory.h"

constexpr static uint32_t histogram_csize = 256;
#define HISTEND(_c_,_cn_,_cnt_) { uint32_t _i,_j;\
  memset(_cnt_, 0, 256*sizeof(_cnt_[0]));\
  for(_i=0; _i < 256; _i++)\
    for(_j=0; _j < _cn_;_j++) _cnt_[_i] += _c_[_j][_i];\
}

#define HISTEND8(_c_,_cnt_) HISTEND(_c_,8,_cnt_)
#define HISTEND4(_c_,_cnt_) HISTEND(_c_,4,_cnt_)

#ifdef USE_AVX512F
// this is way slower than the org one
constexpr static void avx512_histogram_u8_1x(uint32_t cnt[256],
									  		 const uint8_t *__restrict in,
									  		 const size_t inlen) noexcept {
	cryptanalysislib::template memset<uint32_t>(cnt, 0u, 256u);
	// const __m512i acc = _mm512_set1_epi32(1);

	uint32_t tmp1[16] __attribute__((aligned(64))) = {0};

	size_t i = 0;
	for (; (i+16) <= inlen; i+=16) {
		const __m128i t1 = _mm_loadu_si128((const __m128i_u *)(in + i));
		const __m512i t2 = _mm512_cvtepu8_epi64(t1);

		// NOTE: this was much slower
		// const __m512i d1 = _mm512_i32gather_epi32(t2, cnt, 4);
		// const __m512i d2 = _mm512_add_epi32(d1, acc);
		// _mm512_i32scatter_epi32(cnt, t2, d2, 4);

		_mm512_store_epi32(tmp1, t2);

		#pragma clang unroll
		for (uint32_t j = 0; j < 16u; ++j) {
			cnt[tmp1[j]] += 1u;
		}
	}

	// tailmng
	for (; i < inlen; ++i) {
		cnt[in[i]]++;
	}
}

#endif

/// NOTE: if an element occurs more than 2**32 times in the array
///		an overflow will happen, given `C = uint32_t`
/// NOTE: cnt needs to be 256 elements big
/// \param cnt output
/// \param in input
/// \param inlen nr of elements in the input.
template<typename T=uint8_t, typename C=uint32_t>
constexpr inline static void histogram_u8_1x(C cnt[256],
                     				 const T *__restrict in,
                     				 const size_t inlen) noexcept {
	const T *ip = in;
	// cryptanalysislib::template memset<uint32_t>(cnt, 0u, 256u);
	while(ip < in+inlen) {
		cnt[*ip++]++;
	}
}

template<typename T=uint8_t, typename C=uint32_t>
constexpr inline static void histogram_u8_4x(C cnt[256],
									 const T *__restrict in,
									 const size_t inlen) noexcept {
	C c[4][histogram_csize] __attribute__((aligned(64)))= {{0}};
	const T *ip = in;

	while(ip != in+(inlen&~(4-1))) c[0][*ip++]++, c[1][*ip++]++, c[2][*ip++]++, c[3][*ip++]++;
	while(ip != in+ inlen        ) c[0][*ip++]++;
	HISTEND4(c, cnt);
}

template<typename T=uint8_t, typename C=uint32_t>
constexpr inline static void histogram_u8_8x(C cnt[256],
									  const T *__restrict in,
									  const size_t inlen) noexcept {
	C c[8][histogram_csize] = {0},i;
	unsigned char *ip = in;

	while(ip != in+(inlen&~(8-1))) c[0][*ip++]++, c[1][*ip++]++, c[2][*ip++]++, c[3][*ip++]++, c[4][*ip++]++, c[5][*ip++]++, c[6][*ip++]++, c[7][*ip++]++;
	while(ip != in+ inlen        ) c[0][*ip++]++;
	HISTEND8(c, cnt);
}

namespace cryptanalysislib::algorithm {

	template<typename T=uint8_t, typename C=uint32_t>
	constexpr inline static void histogram(C *cnt,
											const T *__restrict in,
											const size_t inlen) noexcept {
		if constexpr (std::is_same_v<T, uint8_t>) {
			return histogram_u8_4x(cnt, in, inlen);
		}

		for (size_t i = 0; i < inlen; ++i) {
			cnt[in[i]] += 1u;
		}
	}
};
#endif
