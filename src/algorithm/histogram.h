#ifndef CRYPTANALYSISLIB_ALGORITHM_HISTOGRAM_H
#define CRYPTANALYSISLIB_ALGORITHM_HISTOGRAM_H

#include <cmath>

#include "helper.h"
#include "memory/memory.h"
#include "algorithm/algorithm.h"
#include "alloc/alloc.h"
#include "simd/simd.h"

// TODO multiple parallel histograms, result in a speedup?

struct AlgorithmHistogramConfig : public AlgorithmConfig {
	constexpr static size_t min_size_per_thread = 1u << 14u;
};
constexpr static AlgorithmHistogramConfig algorithmHistogramConfig;

constexpr static uint32_t histogram_csize = 256;
#define HISTEND(_c_,_cn_,_cnt_) { uint32_t _i,_j;\
  memset(_cnt_, 0, 256*sizeof(_cnt_[0]));\
  for(_i=0; _i < 256; _i++)\
    for(_j=0; _j < _cn_;_j++) _cnt_[_i] += _c_[_j][_i];\
}

#define HISTEND8(_c_,_cnt_) HISTEND(_c_,8,_cnt_)
#define HISTEND4(_c_,_cnt_) HISTEND(_c_,4,_cnt_)

#ifdef USE_AVX512F
#include <immintrin.h>

// this is way slower than the org one
constexpr static void avx512_histogram_u8_1x(uint32_t cnt[256],
									  		 const uint8_t *__restrict in,
									  		 const size_t inlen) noexcept {
	// cryptanalysislib::template memset<uint32_t>(cnt, 0u, 256u);
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

/// org source: https://github.com/WojciechMula/toys/pull/23
/// NOTE: inputs are uint32_t: with values < 2**8
/// NOTE: `_mm512_set1_epi32` has a higher latency than 
/// 	  `_mm512_ternarylogic_epi32`
/// https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,selection:(endColumn:48,endLineNumber:9,positionColumn:48,positionLineNumber:9,selectionStartColumn:48,selectionStartLineNumber:9,startColumn:48,startLineNumber:9),source:'%23include+%3Cimmintrin.h%3E%0A%0A__m512i+set1()+%7B%0A++++return+_mm512_set1_epi32(1)%3B%0A%7D%0A%0A__m512i+set1_()+%7B%0A++++__m512i+a%3B%0A++++return+_mm512_ternarylogic_epi32(a,a,a,0xff)%3B%0A%7D'),l:'5',n:'1',o:'C%2B%2B+source+%231',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:clang1810,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,libs:!(),options:'-O3+-mavx512f',overrides:!(),selection:(endColumn:12,endLineNumber:9,positionColumn:12,positionLineNumber:9,selectionStartColumn:12,selectionStartLineNumber:9,startColumn:12,startLineNumber:9),source:1),l:'5',n:'0',o:'+x86-64+clang+18.1.0+(Editor+%231)',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4
static void avx512_histogram_u32_v3(uint32_t C[256],
									const uint32_t *A,
									const size_t size) noexcept {
	const __m512i vid = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
	const __m512i one = _mm512_set1_epi32(1);

	size_t i = 0;
	for (; i+16 <= size; i+=16) {
		const __m512i chunk = _mm512_loadu_epi32(A + i);
		__m512i offsets = _mm512_slli_epi32(chunk, 4);
		offsets = _mm512_add_epi32(offsets, vid);

		const __m512i oldv = _mm512_i32gather_epi32(offsets, C, 4);
		const __m512i newv = _mm512_add_epi32(oldv, one);
		_mm512_i32scatter_epi32(C, offsets, newv, 4);
	}

	for (uint32_t t = 0; t < 32; t++) {
		const uint32_t pos = t*8;
		uint32_t sum = 0;
		for (uint32_t j = 0; j < 8; j++) {
			sum += C[pos + j];
		}
		C[t] = sum;
	}

	// tailmng
	for (; i < size; ++i) {
		C[A[i]]++;
	}
}

/// using popcnt
/// NOTE: inputs are uint32_t: with values < 2**8
/// \param C
/// \param A
/// \param size
static void avx512_histogram_u32_v4(uint32_t C[256],
									const uint32_t *A,
									const size_t size) noexcept {
	const __m512i one = _mm512_set1_epi32(1u);
	for (uint32_t i = 0; i+16 <= size; i+=16) {
		const __m512i chunk = _mm512_loadu_epi32(A + i);
		const __m512i conflicts = _mm512_popcnt_epi32(_mm512_conflict_epi32(chunk));

		const __m512i oldv = _mm512_i32gather_epi32(chunk, C, 4);
		const __m512i newv = _mm512_add_epi32(_mm512_add_epi32(oldv, one), conflicts);
		_mm512_i32scatter_epi32(C, chunk, newv, 4);
	}
}



///
static inline 
void FA(__m512i& h, __m512i& l, __m512i a, __m512i b, __m512i c) {
    //__m512i tmp = _mm512_ternarylogic_epi32(c, b, a, 0x96);
    //h = _mm512_ternarylogic_epi32(c, b, a, 0xE8);    
    //l = tmp;

    l = _mm512_ternarylogic_epi32(c, b, a, 0x96);
    h = _mm512_ternarylogic_epi32(l, b, a, 0x8E);
}

static void consume_buffer_2(uint8_t* data, size_t N, uint16_t* hist16) {
    size_t tail = N & 63;
    if (tail) {
        // Round N up to a multiple of 64, padding the input with 0xff.
        __m512i* where = (__m512i*)(data + N - tail);
        _mm512_store_epi64(where, _mm512_or_epi64(_mm512_load_epi64(where), _mm512_movm_epi8(~0ull << tail)));
        N = N + 64 - tail;
    }
    N /= 64;

    // 64x 16-bit counters (32 lanes of u16)
    __m512i h0 = _mm512_setzero_si512();
    __m512i h1 = _mm512_setzero_si512();

    // 512x 3-bit counters (1 bit of each counter in each __m512i)
    __m512i w0_0 = _mm512_setzero_si512();
    __m512i w1_0 = _mm512_setzero_si512();
    __m512i w2_0 = _mm512_setzero_si512();

    __m512i tp = _mm512_setr_epi8(
        0, 8, 16, 24, 32, 40, 48, 56,
        1, 9, 17, 25, 33, 41, 49, 57,
        2, 10, 18, 26, 34, 42, 50, 58,
        3, 11, 19, 27, 35, 43, 51, 59,
        4, 12, 20, 28, 36, 44, 52, 60,
        5, 13, 21, 29, 37, 45, 53, 61,
        6, 14, 22, 30, 38, 46, 54, 62,
        7, 15, 23, 31, 39, 47, 55, 63);

    do {
        size_t M = N > 31 ? 31 : N;
        N -= M;
        __m512i w = _mm512_setzero_si512(); // 64x 8-bit counters (64 lanes of u8)
        do {
            // Each of these is 512x 1-bit counters.
            __m512i x0 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 0)));
            __m512i x1 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 8)));
            __m512i x2 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 16)));
            __m512i x3 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 24)));
            __m512i x4 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 32)));
            __m512i x5 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 40)));
            __m512i x6 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 48)));
            __m512i x7 = _mm512_sllv_epi64(_mm512_set1_epi64(1), _mm512_cvtepu8_epi64(_mm_loadu_si64(data + 56)));
            data += 64;

            // Add the 1-bit counters to the 3-bit counters.
            FA(x1, x2, x0, x1, x2);
            FA(x4, x5, x3, x4, x5);
            FA(x7, w0_0, x6, x7, w0_0);
            FA(x5, w0_0, x2, x5, w0_0);
            FA(x4, x7, x1, x4, x7);
            FA(x5, w1_0, x7, x5, w1_0);
            __m512i w3_0; // 4th bit of the 3-bit counters.
            FA(w3_0, w2_0, x5, x4, w2_0);

            // Change w3_0 from 512x 1-bit counters to 64x 8-bit counters.
            w3_0 = _mm512_permutexvar_epi8(tp, w3_0);
            w3_0 = _mm512_gf2p8affine_epi64_epi8(_mm512_set1_epi64(0x8040201008040201), w3_0, 0);
            w3_0 = _mm512_popcnt_epi8(w3_0);
            // Flush w3_0 into w.
            w = _mm512_add_epi8(w, w3_0);
        } while (--M);

        // Flush w into h.
        h0 = _mm512_add_epi16(h0, _mm512_and_epi64(w, _mm512_set1_epi16(0xFF)));
        h1 = _mm512_add_epi16(h1, _mm512_srli_epi16(w, 8));
    } while (N);

    // Change w0_0, w1_0, w2_0 from 512x 1-bit counters to 64x 8-bit counters.
    w0_0 = _mm512_permutexvar_epi8(tp, w0_0);
    w1_0 = _mm512_permutexvar_epi8(tp, w1_0);
    w2_0 = _mm512_permutexvar_epi8(tp, w2_0);
    w0_0 = _mm512_gf2p8affine_epi64_epi8(_mm512_set1_epi64(0x8040201008040201), w0_0, 0);
    w1_0 = _mm512_gf2p8affine_epi64_epi8(_mm512_set1_epi64(0x8040201008040201), w1_0, 0);
    w2_0 = _mm512_gf2p8affine_epi64_epi8(_mm512_set1_epi64(0x8040201008040201), w2_0, 0);
    w0_0 = _mm512_popcnt_epi8(w0_0);
    w1_0 = _mm512_popcnt_epi8(w1_0);
    w2_0 = _mm512_popcnt_epi8(w2_0);

    // h = (h << 3) + (w2_0 << 2) + (w1_0 << 1) + w0_0.
    __m512i w = _mm512_add_epi8(_mm512_add_epi8(w0_0, _mm512_add_epi8(w1_0, w1_0)), _mm512_slli_epi64(w2_0, 2));
    h0 = _mm512_add_epi16(_mm512_slli_epi16(h0, 3), _mm512_and_epi64(w, _mm512_set1_epi16(0xFF)));
    h1 = _mm512_add_epi16(_mm512_slli_epi16(h1, 3), _mm512_srli_epi16(w, 8));

    // Add h to hist16.
    _mm512_storeu_epi16(hist16, _mm512_add_epi16(h0, _mm512_loadu_epi16(hist16)));
    _mm512_storeu_epi16(hist16 + 32, _mm512_add_epi16(h1, _mm512_loadu_epi16(hist16 + 32)));
}

void hist256_2(uint8_t* ptr, size_t N, uint32_t* histogram) {
    // Scalar loop to align input pointer.
    if (N >= 64) {
        uint8_t* end = ptr + N;
        while ((uintptr_t)ptr & 63) {
            histogram[*ptr++] += 1;
        }
        N = end - ptr;
    }

    // Input bytes are binned into buffers; 0 for 0-63, 1 for 64-127, 2 for 128-191, 3 for 192-255.
    const size_t bufsize = 1024 * 16;
    //  = (uint8_t*)_aligned_malloc(bufsize * 4, 64);
    uint8_t  buffer0[bufsize*4] __attribute__((aligned(64)));
    uint8_t *buffer1 = buffer0 + bufsize;
    uint8_t *buffer2 = buffer1 + bufsize;
    uint8_t *buffer3 = buffer2 + bufsize;

    while (N >= 64) {
        // Consume up to 65472 (i.e. 2^^16 - 64) bytes, accumulating into 256x 16-bit counters.
        uint16_t hist16[256] = { 0 };
        size_t count0 = 0;
        size_t count1 = 0;
        size_t count2 = 0;
        size_t count3 = 0;
        size_t M = N >= 65472 ? 65472 : N & -64;
        N -= M;
        for (size_t i = 0; i < M; i += 64) {
            // Load 64 bytes, use high 2 bits of each to choose appropriate bin.
            __m512i data = _mm512_load_si512(ptr + i);
            __mmask64 bit7 = _mm512_movepi8_mask(data);
            __mmask64 bit6 = _mm512_movepi8_mask(_mm512_add_epi8(data, data));
            __mmask64 b00 = _knot_mask64(_kor_mask64(bit6, bit7));
            __mmask64 b01 = _kandn_mask64(bit7, bit6);
            __mmask64 b10 = _kandn_mask64(bit6, bit7);
            __mmask64 b11 = _kand_mask64(bit7, bit6);
            // Discard top two bits of each byte (they confuse _mm512_sllv_epi64), and perform a
            // rotation left by two bits on the bottom six bits (this gets undone by one when
            // 8-bit counters are promoted to 16-bit, and again by one when 16-bit counters are
            // promoted to 32-bit).
            data = _mm512_gf2p8affine_epi64_epi8(data, _mm512_set1_epi64(0x2010010204080000), 0);
            // Append into bins.
            _mm512_storeu_epi8(buffer0 + count0, _mm512_maskz_compress_epi8(b00, data));
            _mm512_storeu_epi8(buffer1 + count1, _mm512_maskz_compress_epi8(b01, data));
            _mm512_storeu_epi8(buffer2 + count2, _mm512_maskz_compress_epi8(b10, data));
            _mm512_storeu_epi8(buffer3 + count3, _mm512_maskz_compress_epi8(b11, data));
            count0 += _mm_popcnt_u64(b00);
            count1 += _mm_popcnt_u64(b01);
            count2 += _mm_popcnt_u64(b10);
            count3 += _mm_popcnt_u64(b11);

            // Empty any bins that might overflow on the next iteration.
            if (count0 >= bufsize - 64) consume_buffer_2(buffer0, count0, &hist16[0]), count0 = 0;
            if (count1 >= bufsize - 64) consume_buffer_2(buffer1, count1, &hist16[64]), count1 = 0;
            if (count2 >= bufsize - 64) consume_buffer_2(buffer2, count2, &hist16[128]), count2 = 0;
            if (count3 >= bufsize - 64) consume_buffer_2(buffer3, count3, &hist16[192]), count3 = 0;
        }
        ptr += M;

        // Empty the bins.
        if (count0) consume_buffer_2(buffer0, count0, &hist16[0]);
        if (count1) consume_buffer_2(buffer1, count1, &hist16[64]);
        if (count2) consume_buffer_2(buffer2, count2, &hist16[128]);
        if (count3) consume_buffer_2(buffer3, count3, &hist16[192]);

        // Flush the 16-bit counters to the 32-bit counters.
        for (size_t i = 0; i < 256; i += 64) {
            __m512i h0 = _mm512_loadu_epi16(hist16 + i);
            __m512i h1 = _mm512_loadu_epi16(hist16 + i + 32);
            __m512i w0 = _mm512_and_epi32(h0, _mm512_set1_epi32(0xFFFF));
            __m512i w1 = _mm512_srli_epi32(h0, 16);
            __m512i w2 = _mm512_and_epi32(h1, _mm512_set1_epi32(0xFFFF));
            __m512i w3 = _mm512_srli_epi32(h1, 16);
            _mm512_storeu_epi32(histogram + i, _mm512_add_epi32(w0, _mm512_loadu_epi32(histogram + i)));
            _mm512_storeu_epi32(histogram + i + 16, _mm512_add_epi32(w1, _mm512_loadu_epi32(histogram + i + 16)));
            _mm512_storeu_epi32(histogram + i + 32, _mm512_add_epi32(w2, _mm512_loadu_epi32(histogram + i + 32)));
            _mm512_storeu_epi32(histogram + i + 48, _mm512_add_epi32(w3, _mm512_loadu_epi32(histogram + i + 48)));
        }
    }

    // _aligned_free(buffer0);

    // Scalar loop to deal with any remaining input.
    while (N) {
        histogram[ptr[--N]] += 1;
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

	size_t i = 0;
	for (; i+8 <= size; i+=8) {
		const __m256i chunk = _mm256_loadu_si256((const __m256i *)(A + i));
		__m256i offsets = _mm256_slli_epi32(chunk, 3);
		offsets = _mm256_add_epi32(offsets, vid);
		_mm256_store_si256((__m256i *)tmp2, offsets);

		const __m256i oldv = _mm256_i32gather_epi32((const int *)C, offsets, 4);
		const __m256i newv = _mm256_add_epi32(oldv, one);
		
		// NOTE: there is no scatter instruction in avx2
		// _mm256_i32scatter_epi32(C, newv, 4);
		_mm256_store_si256((__m256i *)tmp1, newv);
		for (uint32_t j = 0; j < 8; j++) {
			C[tmp2[j]] = tmp1[j];
		}
	}

	for (uint32_t t = 0; t < 256; t++) {
		const uint32_t pos = t*8;
		uint32_t sum = 0;
		for (uint32_t j = 0; j < 8; j++) {
			sum += C[pos + j];
		}

		C[t] = sum;
	}

	// tailmng
	for (; i < size; ++i) {
		C[A[i]]++;
	}
}
#endif

/// NOTE: if an element occurs more than 2**32 times in the array
///		an overflow will happen, given `C = uint32_t`
/// NOTE: cnt needs to be 256 elements big
/// \param cnt output
/// \param in input
/// \param inlen nr of elements in the input.
template<typename T=uint8_t,
		 typename C=uint32_t>
constexpr inline static void histogram_u8_1x(C cnt[256],
                     				 const T *__restrict in,
                     				 const size_t inlen) noexcept {
	const T *ip = in;
	while(ip < in+inlen) {
		cnt[*ip++]++;
	}
}

/// \tparam T
/// \tparam C
/// \param cnt
/// \param in
/// \param inlen
template<typename T=uint8_t,
		 typename C=uint32_t>
constexpr inline static void histogram_u8_4x(C cnt[256],
									 const T *__restrict in,
									 const size_t inlen) noexcept {
	C c[4][histogram_csize] __attribute__((aligned(64)))= {{0}};
	const T *ip = in;

	while(ip != in+(inlen&~(4-1))) c[0][*ip++]++, c[1][*ip++]++, c[2][*ip++]++, c[3][*ip++]++;
	while(ip != in+ inlen        ) c[0][*ip++]++;
	HISTEND4(c, cnt);
}

/// \tparam T
/// \tparam C
/// \param cnt
/// \param in
/// \param inlen
template<typename T=uint8_t,
		 typename C=uint32_t>
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

	/// \tparam T
	/// \tparam C
	/// \param cnt
	/// \param in
	/// \param inlen
	template<typename T=uint8_t,
			 typename C=uint32_t,
			 const AlgorithmHistogramConfig &config=algorithmHistogramConfig>
	constexpr inline static void histogram(C *__restrict__ cnt,
											const T *__restrict in,
											const size_t inlen) noexcept {
		if constexpr (std::is_same_v<T, uint8_t>) {
			return histogram_u8_4x(cnt, in, inlen);
		}

		for (size_t i = 0; i < inlen; ++i) {
			cnt[in[i]] += 1u;
		}
	}

	///
	/// @tparam ExecPolicy 
	/// @tparam T 
	/// @tparam C 
	/// @tparam config 
	/// @param policy 
	/// @param cnt 
	/// @param in 
	/// @param size 
	template<class ExecPolicy,
		     typename T=uint8_t,
			 typename C=uint32_t,
			 const AlgorithmHistogramConfig &config=algorithmHistogramConfig,
			 typename Allocator=cryptanalysislib::alloc::alignment_allocator<C>>
	constexpr inline static void histogram(ExecPolicy && policy,
										   C *__restrict__ cnt,
										   const T *__restrict in,
										   const size_t size) noexcept {
        auto& task_pool = *policy.pool();
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::algorithm::histogram(cnt, in, size);
		}


        std::vector<std::future<void>> futures;
		const size_t chunks = size  / nthreads;

		// generic fallback implementation
		constexpr size_t k = sizeof(T)*8*sizeof(C);
		C *cnts = Allocator::allocate(nthreads*k);
		for (uint32_t i = 0; i < nthreads; i++) {
			futures.emplace_back(
			    task_pool.enqueue(
			       [i, cnts, in, chunks] () {
						const size_t l = i*chunks;
						// const size_t h = (i+1)*chunks;
			       	    histogram
			       			<T, C, config>
			       				(cnts + i*k,
			       	    		  in + l,
			       	    		  chunks);
			       }
			    )
			);
		}


		for (uint32_t i = 0; i < nthreads; i++) {
			futures[i].wait();
			for (size_t j = 0; j < k; j++) {
				cnt[j] += cnts[i*k + j];
			}
		}
	}
};


#undef HISTEND8
#undef HISTEND4
#undef HISTEND
#endif
