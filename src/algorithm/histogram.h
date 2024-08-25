#ifndef CRYPTANALYSISLIB_ALGORITHM_HISTOGRAM_H
#define CRYPTANALYSISLIB_ALGORITHM_HISTOGRAM_H

#include "helper.h"
#include "memory/memory.h"

constexpr static uint32_t histogram_csize = 256;
#define HISTEND(_c_,_cn_,_cnt_) { int _i,_j;\
  memset(_cnt_, 0, 256*sizeof(_cnt_[0]));\
  for(_i=0; _i < 256; _i++)\
    for(_j=0; _j < _cn_;_j++) _cnt_[_i] += _c_[_j][_i];\
}

#define HISTEND8(_c_,_cnt_) HISTEND(_c_,8,_cnt_)
#define HISTEND4(_c_,_cnt_) HISTEND(_c_,4,_cnt_)


/// NOTE: if an element occurs more than 2**32 times in the array
///		an overflow will happen, given `C = uint32_t`
/// NOTE: cnt needs to be 256 elements big
/// \param cnt output
/// \param in input
/// \param inlen nr of elements in the input.
template<typename T=uint8_t, typename C=uint32_t>
constexpr static void histogram_u8_1x(C cnt[256],
                     				 const T *__restrict in,
                     				 const size_t inlen) noexcept {
#ifdef USE_AVX512
	if constexpr(std::is_same_v<T, uint8_t> &&
	        	 std::is_same_v<C, uint32_t>) {

	}
#endif
	const T *ip = in;
	cryptanalysislib::template memset<uint32_t>(cnt, 0u, 256u);
	while(ip < in+inlen) {
		cnt[*ip++]++;
	}
}

template<typename T=uint8_t, typename C=uint32_t>
constexpr static void histogram_u8_4x(C cnt[256],
									 const T *__restrict in,
									 const size_t inlen) noexcept {
	C c[4][histogram_csize] = {0},i;
	const T *ip = in;

	while(ip != in+(inlen&~(4-1))) c[0][*ip++]++, c[1][*ip++]++, c[2][*ip++]++, c[3][*ip++]++;
	while(ip != in+ inlen        ) c[0][*ip++]++;
	HISTEND4(c, cnt);
}

template<typename T=uint8_t, typename C=uint32_t>
constexpr static void histogram_u8_8x(C cnt[256],
									  const T *__restrict in,
									  const size_t inlen) noexcept {
	C c[8][histogram_csize] = {0},i;
	unsigned char *ip = in;

	while(ip != in+(inlen&~(8-1))) c[0][*ip++]++, c[1][*ip++]++, c[2][*ip++]++, c[3][*ip++]++, c[4][*ip++]++, c[5][*ip++]++, c[6][*ip++]++, c[7][*ip++]++;
	while(ip != in+ inlen        ) c[0][*ip++]++;
	HISTEND8(c, cnt);
}
#endif
