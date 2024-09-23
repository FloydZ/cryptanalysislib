#ifndef CRYPTANALYSISLIB_ALGORITHM_HISTOGRAM_H
#define CRYPTANALYSISLIB_ALGORITHM_HISTOGRAM_H

#include "helper.h"
#include "memory/memory.h"
#include <cmath>

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

/// org code from intel:
/// translation from 
/// https://github.com/WojciechMula/toys/blob/master/avx512-conflict-detection/histogram_intel_amd64.s
/// TODO take the rev code
static void avx512_histogram_u32z_v2(uint32_t C[256],
									const uint32_t *A,
									const size_t size){
	// TODO
	(void)C;
	(void)A;
	(void)size;
}

/// org source: https://github.com/WojciechMula/toys/pull/23jj
/// NOTE: `_mm512_set1_epi32` has a higher latency than 
/// 	  `_mm512_ternarylogic_epi32`
/// https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,selection:(endColumn:48,endLineNumber:9,positionColumn:48,positionLineNumber:9,selectionStartColumn:48,selectionStartLineNumber:9,startColumn:48,startLineNumber:9),source:'%23include+%3Cimmintrin.h%3E%0A%0A__m512i+set1()+%7B%0A++++return+_mm512_set1_epi32(1)%3B%0A%7D%0A%0A__m512i+set1_()+%7B%0A++++__m512i+a%3B%0A++++return+_mm512_ternarylogic_epi32(a,a,a,0xff)%3B%0A%7D'),l:'5',n:'1',o:'C%2B%2B+source+%231',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:clang1810,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,libs:!(),options:'-O3+-mavx512f',overrides:!(),selection:(endColumn:12,endLineNumber:9,positionColumn:12,positionLineNumber:9,selectionStartColumn:12,selectionStartLineNumber:9,startColumn:12,startLineNumber:9),source:1),l:'5',n:'0',o:'+x86-64+clang+18.1.0+(Editor+%231)',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4
static void avx512_histogram_u32_v3(uint32_t C[256],
									const uint8_t *A, 
									const size_t size) {
	const __m512i vid = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
	const __m512i one = _mm512_set1_epi32(1);

	for (uint32_t i = 0; i+16 <= size; i+=16) {
		const __m512i chunk = _mm512_loadu_epi32(A + i);
		__m512i offsets = _mm512_slli_epi32(chunk, 4);
		offsets = _mm512_add_epi32(offsets, vid);

		const __m512i oldv = _mm512_i32gather_epi32(offsets, C, 4);
		const __m512i newv = _mm512_add_epi32(oldv, one);
		_mm512_i32scatter_epi32(C, offsets, newv, 4);
	}

	for (uint32_t i = 0; i < 32; i++) {
		const uint32_t pos = i*8;
		uint32_t sum = 0;
		for (uint32_t j = 0; j < 8; j++) {
			sum += C[pos + j];
		}
		C[i] = sum;
	}
}

#endif


#ifdef USE_AVX2
/// NOTE: special histogram which the input data is 32bits
/// but only the lower 8 bits are used.
/// NOTE: buckets need to be 264 elements big.
/// \param C output buckets
/// \param A input
/// \param size number of elements
static void avx2_histogram_u32(uint32_t C[1024],
							   const uint32_t *A,
							   const size_t size) {
	const __m256i vid = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
	const __m256i one = _mm256_set1_epi32(1);
	uint32_t tmp1[8] __attribute__((aligned(64)));
	uint32_t tmp2[8] __attribute__((aligned(64)));

	for (uint32_t i = 0; i+8 <= size; i+=8) {
		const __m256i chunk = _mm256_loadu_si256((const __m256i *)(A + i));
		__m256i offsets = _mm256_slli_epi32(chunk, 3);
		offsets = _mm256_add_epi32(offsets, vid);
		_mm256_store_si256((__m256i *)tmp2, offsets);

		const __m256i oldv = _mm256_i32gather_epi32(C, offsets, 4);
		const __m256i newv = _mm256_add_epi32(oldv, one);
		
		// NOTE: there is no scatter instruction in avx2
		// _mm256_i32scatter_epi32(C, newv, 4);
		_mm256_store_si256((__m256i *)tmp1, newv);
		for (uint32_t j = 0; j < 8; j++) {
			C[tmp2[j]] = tmp1[j];
		}
	}
	// TODO tail mngt
	
	// TODO can be applied to an histogram algorithm
	for (uint32_t i = 0; i < 256; i++) {
		const uint32_t pos = i*8;
		uint32_t sum = 0;
		for (uint32_t j = 0; j < 8; j++) {
			sum += C[pos + j];
		}
		C[i] = sum;
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

	C c[8][histogram_csize] __attribute__((aligned(64)))= {{0}};
	const T *ip = in;

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


#undef HISTEND8
#undef HISTEND4
#undef HISTEND
#endif
