#ifndef CRYPTANALYSISLIB_SIMD_AVX512_H
#define CRYPTANALYSISLIB_SIMD_AVX512_H

#ifndef CRYPTANALYSISLIB_SIMD_H
#error "dont include this file directly. Use `#include <simd/simd.h>`"
#endif

#ifndef USE_AVX512F
#error "no avx512 enabled."
#endif

#include <cstdint>
#include <immintrin.h>

#include "helper.h"
#include "random.h"

#ifdef __clang__
/// what this is needed?
typedef char __v64qi_u __attribute__((__vector_size__(64), __may_alias__, __aligned__(1)));
#endif

struct uint16x32_t;
struct uint32x16_t;
struct uint64x8_t;

struct uint8x64_t {
	constexpr static uint32_t LIMBS = 64;
	using limb_type = uint8_t;

	constexpr uint8x64_t() noexcept = default;
	constexpr uint8x64_t(const uint16x32_t &b) noexcept;
	constexpr uint8x64_t(const uint32x16_t &b) noexcept;
	constexpr uint8x64_t(const uint64x8_t &b) noexcept;

	union {
		// compatibility with TxN_t
		uint8_t d[64];

		uint8_t v8[64];
		uint16_t v16[32];
		uint32_t v32[16];
		uint64_t v64[8];
		cryptanalysislib::_uint8x16_t v128[4];
		__m256i v256[2];
		__m512i v512;
	};

	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}
	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline uint8x64_t random() noexcept {
		uint8x64_t ret;
		for (size_t i = 0; i < 8; ++i) {
			ret.v64[i] = fastrandombytes_uint64();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	///
	[[nodiscard]] constexpr static inline uint8x64_t set (
			int __A, int __B, int __C, int __D,
			int __E, int __F, int __G, int __H,
			int __I, int __J, int __K, int __L,
			int __M, int __N, int __O, int __P) noexcept
	{
		uint8x64_t ret;
		ret.v512 = __extension__ (__m512i)(__v16si)
				{ __P, __O, __N, __M, __L, __K, __J, __I,
				  __H, __G, __F, __E, __D, __C, __B, __A };
		return ret;
	}

	///
	[[nodiscard]] constexpr static inline uint8x64_t setr (
	        int __A, int __B, int __C, int __D,
	        int __E, int __F, int __G, int __H,
	        int __I, int __J, int __K, int __L,
	        int __M, int __N, int __O, int __P) noexcept
	{
		uint8x64_t ret;
		ret.v512 = __extension__ (__m512i)(__v16si)
		{ __A, __B, __C, __D, __E, __F, __G, __H,
		  __I, __J, __K, __L, __M, __N, __O, __P};
		return ret;
	}

	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t set(
	        char __q63, char __q62, char __q61, char __q60,
	        char __q59, char __q58, char __q57, char __q56,
	        char __q55, char __q54, char __q53, char __q52,
	        char __q51, char __q50, char __q49, char __q48,
	        char __q47, char __q46, char __q45, char __q44,
	        char __q43, char __q42, char __q41, char __q40,
	        char __q39, char __q38, char __q37, char __q36,
	        char __q35, char __q34, char __q33, char __q32,
	        char __q31, char __q30, char __q29, char __q28,
	        char __q27, char __q26, char __q25, char __q24,
	        char __q23, char __q22, char __q21, char __q20,
	        char __q19, char __q18, char __q17, char __q16,
	        char __q15, char __q14, char __q13, char __q12,
	        char __q11, char __q10, char __q09, char __q08,
	        char __q07, char __q06, char __q05, char __q04,
	        char __q03, char __q02, char __q01, char __q00) noexcept {
		uint8x64_t out;
		out.v512 = __extension__(__m512i)(__v64qi){
		        __q00, __q01, __q02, __q03, __q04, __q05, __q06, __q07,
		        __q08, __q09, __q10, __q11, __q12, __q13, __q14, __q15,
		        __q16, __q17, __q18, __q19, __q20, __q21, __q22, __q23,
		        __q24, __q25, __q26, __q27, __q28, __q29, __q30, __q31,
		        __q32, __q33, __q34, __q35, __q36, __q37, __q38, __q39,
		        __q40, __q41, __q42, __q43, __q44, __q45, __q46, __q47,
		        __q48, __q49, __q50, __q51, __q52, __q53, __q54, __q55,
		        __q56, __q57, __q58, __q59, __q60, __q61, __q62, __q63};
		return out;
	}

	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t setr(
	        char __q63, char __q62, char __q61, char __q60,
	        char __q59, char __q58, char __q57, char __q56,
	        char __q55, char __q54, char __q53, char __q52,
	        char __q51, char __q50, char __q49, char __q48,
	        char __q47, char __q46, char __q45, char __q44,
	        char __q43, char __q42, char __q41, char __q40,
	        char __q39, char __q38, char __q37, char __q36,
	        char __q35, char __q34, char __q33, char __q32,
	        char __q31, char __q30, char __q29, char __q28,
	        char __q27, char __q26, char __q25, char __q24,
	        char __q23, char __q22, char __q21, char __q20,
	        char __q19, char __q18, char __q17, char __q16,
	        char __q15, char __q14, char __q13, char __q12,
	        char __q11, char __q10, char __q09, char __q08,
	        char __q07, char __q06, char __q05, char __q04,
	        char __q03, char __q02, char __q01, char __q00) noexcept {
		uint8x64_t out;
		out.v512 = __extension__(__m512i)(__v64qi){
		        __q63,
		        __q62,
		        __q61,
		        __q60,
		        __q59,
		        __q58,
		        __q57,
		        __q56,
		        __q55,
		        __q54,
		        __q53,
		        __q52,
		        __q51,
		        __q50,
		        __q49,
		        __q48,
		        __q47,
		        __q46,
		        __q45,
		        __q44,
		        __q43,
		        __q42,
		        __q41,
		        __q40,
		        __q39,
		        __q38,
		        __q37,
		        __q36,
		        __q35,
		        __q34,
		        __q33,
		        __q32,
		        __q31,
		        __q30,
		        __q29,
		        __q28,
		        __q27,
		        __q26,
		        __q25,
		        __q24,
		        __q23,
		        __q22,
		        __q21,
		        __q20,
		        __q19,
		        __q18,
		        __q17,
		        __q16,
		        __q15,
		        __q14,
		        __q13,
		        __q12,
		        __q11,
		        __q10,
		        __q09,
		        __q08,
		        __q07,
		        __q06,
		        __q05,
		        __q04,
		        __q03,
		        __q02,
		        __q01,
		        __q00,
		};
		return out;
	}

	[[nodiscard]] constexpr static inline uint8x64_t set1(char __A) noexcept {
		uint8x64_t out;
		out.v512 = __extension__(__m512i)(__v64qi){__A, __A, __A, __A, __A, __A, __A, __A,
		                                           __A, __A, __A, __A, __A, __A, __A, __A,
		                                           __A, __A, __A, __A, __A, __A, __A, __A,
		                                           __A, __A, __A, __A, __A, __A, __A, __A,
		                                           __A, __A, __A, __A, __A, __A, __A, __A,
		                                           __A, __A, __A, __A, __A, __A, __A, __A,
		                                           __A, __A, __A, __A, __A, __A, __A, __A,
		                                           __A, __A, __A, __A, __A, __A, __A, __A};

		return out;
	}


	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint8x64_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t aligned_load(const void *ptr) noexcept {
		const __m512i tmp = *(__m512i *) ptr;
		uint8x64_t out;
		out.v512 = tmp;
		return out;
	}


	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t unaligned_load(const void *ptr) noexcept {
		const __m512i tmp = (__m512i) (*(__v64qi_u *) ptr);
		uint8x64_t out;
		out.v512 = tmp;
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const uint8x64_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr, const uint8x64_t in) noexcept {
		auto *ptr512 = (__m512i *) ptr;
		*ptr512 = in.v512;
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint8x64_t in) noexcept {
		auto *ptr512 = (__m512i_u *) ptr;
		*(__m512i_u *) ptr512 = (__m512i_u) in.v512;
	}


	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t xor_(const uint8x64_t in1,
	                                                      const uint8x64_t in2) noexcept {
		uint8x64_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t and_(const uint8x64_t in1,
	                                                      const uint8x64_t in2) noexcept {
		uint8x64_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t or_(const uint8x64_t in1,
	                                                     const uint8x64_t in2) noexcept {
		uint8x64_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 | (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t andnot(const uint8x64_t in1,
	                                                        const uint8x64_t in2) noexcept {
		uint8x64_t out;
		out.v512 = (__m512i) (~(__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t not_(const uint8x64_t in1) noexcept {
		uint8x64_t out;
		const uint8x64_t minus_one = set1(-1);
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) minus_one.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t add(const uint8x64_t in1,
	                                                     const uint8x64_t in2) noexcept {
		uint8x64_t out;
		out.v512 = (__m512i) ((__v64qu) in1.v512 + (__v64qu) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t sub(const uint8x64_t in1,
	                                                     const uint8x64_t in2) noexcept {
		uint8x64_t out;
		out.v512 = (__m512i) ((__v64qu) in1.v512 - (__v64qu) in2.v512);
		return out;
	}

	/// 8 bit mul lo
	/// \param in1 first input
	/// \param in2
	/// \return in1*in2
	[[nodiscard]] constexpr static inline uint8x64_t mullo(const uint8x64_t in1,
	                                                       const uint8x64_t in2) noexcept {
		uint8x64_t out;
		(void) in1;
		(void) in2;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t mullo(const uint8x64_t in1,
	                                                       const uint8_t in2) noexcept {
		const uint8x64_t rs = uint8x64_t::set1(in2);
		return uint8x64_t::mullo(in1, rs);
	}


	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t popcnt(const uint8x64_t in1) noexcept {
		uint8x64_t ret;
#ifdef __clang__
		ret.v512 = (__m512i) __builtin_ia32_vpopcntb_512((__v64qi) in1.v512);
#else
  		ret.v512 = (__m512i) __builtin_ia32_vpopcountb_v64qi ((__v64qi)in1.v512);
#endif
		return ret;
	}

	/// Source:http://0x80.pl/notesen/2023-01-31-avx512-bsf.html
	/// needs`AVX512VPOPCNTDQ`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t lzcnt(const uint8x64_t in1) noexcept {
		uint8x64_t ret;
		constexpr uint8x64_t one = uint8x64_t::set1(1);
		ret = uint8x64_t::sub(in1, one);
		ret = uint8x64_t::and_(ret, uint8x64_t::not_(in1));
		ret = uint8x64_t::popcnt(ret);
		return ret;
	}

	/// needs `AVX512BW`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t min(const uint8x64_t in1,
	                                                     const uint8x64_t in2) noexcept {
		uint8x64_t ret;
#ifdef __clang__
		ret.v512 = (__m512i)__builtin_elementwise_min((__v8du)in1.v512, (__v8du)in2.v512);
  		//ret.v512 = (__m512i)__builtin_ia32_pminsb512((__v64qi)in1.v512, (__v64qi)in2.v512);
#else 
  		ret.v512 = (__m512i) __builtin_ia32_pminsb512_mask ((__v64qi)in1.v512,
						  (__v64qi)in2.v512,
						  (__v64qi)  __extension__ (__m512i)(__v8di){ 0, 0, 0, 0, 0, 0, 0, 0 },
						  (__mmask64) -1);
#endif
		return ret;
	}

	/// needs `AVX512BW`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t max(const uint8x64_t in1,
	                                                     const uint8x64_t in2) noexcept {
		uint8x64_t ret;
#ifdef __clang__
		ret.v512 = (__m512i)__builtin_elementwise_max((__v8du)in1.v512, (__v8du)in2.v512);
  		//ret.v512 = (__m512i)__builtin_ia32_pmaxsb512((__v64qi)in1.v512, (__v64qi)in2.v512);
#else
  		ret.v512 = (__m512i) __builtin_ia32_pmaxsb512_mask ((__v64qi)in1.v512,
						  (__v64qi)in2.v512,
						  (__v64qi) __extension__ (__m512i)(__v8di){ 0, 0, 0, 0, 0, 0, 0, 0 },
						  (__mmask64) -1);
#endif
		return ret;
	}

	/// SOURCE: https://github.com/WojciechMula/toys/blob/master/avx512-galois/transpose.cpp
	/// Each 64-bit word holds 8x8 bit matrix:
	/// LSB                   MSB
	/// [a0|b0|c0|d0|e0|f0|g0|h0] byte 0
	/// [a1|b1|c1|d1|e1|f1|g1|h1]
	/// [a2|b2|c2|d2|e2|f2|g2|h2]
	/// [a3|b3|c3|d3|e3|f3|g3|h3]
	/// [a4|b4|c4|d4|e4|f4|g4|h4]
	/// [a5|b5|c5|d5|e5|f5|g5|h5]
	/// [a6|b6|c6|d6|e6|f6|g6|h6]
	/// [a7|b7|c7|d7|e7|f7|g7|h7] byte 7
	/// Output is (note that bits from byte 7 are MSBs):
	/// LSB                   MSB
	/// [a7|a6|a5|a4|a3|a2|a1|a0] byte 0
	/// [b7|b6|b5|b4|b3|b2|b1|b0]
	/// [c7|c6|c5|c4|c3|c2|c1|c0]
	/// [d7|d6|d5|d4|d3|d2|d1|d0]
	/// [e7|e6|e5|e4|e3|e2|e1|e0]
	/// [f7|f6|f5|f4|f3|f2|f1|f0]
	/// [g7|g6|g5|g4|g3|g2|g1|g0]
	/// [h7|h6|h5|h4|h3|h2|h1|h0] byte 7
	[[nodiscard]] constexpr static inline uint8x64_t transpose(const uint8x64_t input) noexcept {
		uint8x64_t ret;
		const __m512i select = __extension__(__m512i)(__v8di){
		        static_cast<long long>(0x8040201008040201ull),
		        static_cast<long long>(0x8040201008040201ull),
		        static_cast<long long>(0x8040201008040201ull),
		        static_cast<long long>(0x8040201008040201ull),
		        static_cast<long long>(0x8040201008040201ull),
		        static_cast<long long>(0x8040201008040201ull),
		        static_cast<long long>(0x8040201008040201ull),
		        static_cast<long long>(0x8040201008040201ull),
		};
		ret.v512 = ((__m512i) __builtin_ia32_vgf2p8affineqb_v64qi((__v64qi) (__m512i) (select),
		                                                          (__v64qi) (__m512i) (input.v512), (char) (0x00)));
		return ret;
	}

	/// https://github.com/WojciechMula/toys/blob/master/avx512-galois/reverse.cpp
	/// \param input
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t reverse(const uint8x64_t input) noexcept {
		uint8x64_t ret;
		const long long __d = bit_shuffle_const(7, 6, 5, 4, 3, 2, 1, 0);
		const __m512i select = __extension__(__m512i)(__v8di){__d, __d, __d, __d, __d, __d, __d, __d};
		ret.v512 = ((__m512i) __builtin_ia32_vgf2p8affineqb_v64qi((__v64qi) (__m512i) (select),
		                                                          (__v64qi) (__m512i) (input.v512), (char) (0x00)));
		return ret;
	}

	///
	/// source:https://github.com/WojciechMula/toys/blob/master/avx512/avx512bw-rotate-by1.cpp
	/// needs `avx512bw`
	/// \param input
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t ror1(const uint8x64_t input) noexcept {
		uint8x64_t ret;
		// lanes order: 1, 2, 3, 0 => 0b00_11_10_01
#ifdef __clang__
		const __m512i permuted = (__m512i) __builtin_ia32_shuf_i32x4((__v16si) (__m512i) (input.v512),
		                                                             (__v16si) (__m512i) (input.v512), (int) (0x39));
		ret.v512 = ((__m512i) __builtin_ia32_palignr512((__v64qi) (__m512i) (permuted),
		                                                (__v64qi) (__m512i) (input.v512), (int) (1)));
#else

		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Winit-self"
		  __m512i Y = Y;
		#pragma GCC diagnostic pop

  		const __m512i permuted = ((__m512i)  __builtin_ia32_shuf_i32x4_mask ((__v16si)(__m512i)(input.v512),\
      								(__v16si)(__m512i)(input.v512), (int)(0x39),\
    								(__v16si)(__m512i)Y,\
    								(__mmask16)-1));

  		ret.v512 = ((__m512i) __builtin_ia32_palignr512 ((__v8di)(__m512i)(permuted),	\
											(__v8di)(__m512i)(input.v512),			    \
											(int)((1) * 8)));
#endif
		return ret;
	}

	/// needs `avx512bw`
	/// source:  http://0x80.pl/notesen/2021-02-02-all-bytes-in-reg-are-equal.html
	/// \param input
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const uint8x64_t input) noexcept {
#ifdef __clang__
		const __m128i lane0 = (__m128i) __builtin_shufflevector(input.v512, input.v512, 0, 1);
		const __m512i populated_0th_byte = _mm512_broadcastb_epi8(lane0);
		const __mmask16 mask = _mm512_cmp_epi32_mask((input.v512), (populated_0th_byte), _MM_CMPINT_EQ);
		return __builtin_ia32_kortestchi((__mmask16) mask, (__mmask16) mask);
#else 

		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Winit-self"
		  __m512i Y = Y;
		#pragma GCC diagnostic pop

		const __m128i lane0 = (__m128i) __builtin_shufflevector(input.v512, input.v512, 0, 1);
  		const __m512i populated_0th_byte = (__m512i) __builtin_ia32_pbroadcastb512_mask ((__v16qi)lane0,
						       					(__v64qi)Y,
						       					(__mmask64) -1);
		// const __m512i populated_0th_byte = (__m512i) __builtin_shufflevector((__v16qi) lane0, (__v16qi) lane0,
		//                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		//                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		//                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		//                                                                     0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  		const __mmask16 mask = (__mmask16)__builtin_ia32_cmpd512_mask((__v16si)input.v512,
						  			(__v16si)populated_0th_byte, _MM_CMPINT_EQ,
						  			(__mmask16) -1);
		return __builtin_ia32_kortestchi((__mmask16) mask, (__mmask16) mask);
#endif
	}

	/// source: https://github.com/WojciechMula/toys/tree/master/simd-basic/reverse-bytes
	/// \param input
	/// \return
	[[nodiscard]] constexpr static inline uint8x64_t reverse8(const uint8x64_t input) noexcept {
#if defined(USE_AVX512VBMI)
		const __m512i indices_byte = _mm512_set_epi64(
		        0x0001020304050607llu, 0x08090a0b0c0d0e0fllu,
		        0x1011121314151617llu, 0x18191a1b1c1d1e1fllu,
		        0x2021222324252627llu, 0x28292a2b2c2d2e2fllu,
		        0x3031323334353637llu, 0x38393a3b3c3d3e3fllu);

		uint8x64_t ret;
		ret.v512 = _mm512_permutexvar_epi8(indices_byte, input.v512);
		return ret;
#elif defined(USE_AVX512BW)
#ifdef __clang__
		// 1. reverse order of 128-bit lanes
		const __m512i indices = _mm512_setr_epi32(
		        12, 13, 14, 15,
		        8, 9, 10, 11,
		        4, 5, 6, 7,
		        0, 1, 2, 3);
		const __m512i swap_128 = _mm512_permutexvar_epi32(indices, input.v512);

		// 2. reverse order of bytes within 128-bit lanes
		const __m512i indices_byte = _mm512_set_epi64(
		        0x0001020304050607llu, 0x08090a0b0c0d0e0fllu,
		        0x0001020304050607llu, 0x08090a0b0c0d0e0fllu,
		        0x0001020304050607llu, 0x08090a0b0c0d0e0fllu,
		        0x0001020304050607llu, 0x08090a0b0c0d0e0fllu);

		uint8x64_t ret;
		ret.v512 = _mm512_shuffle_epi8(swap_128, indices_byte);
		return ret;
#else
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Winit-self"
		  __m512i Y = Y;
		#pragma GCC diagnostic pop

		// 1. reverse order of 128-bit lanes
		const __m512i indices = __extension__ (__m512i)(__v16si)
				{3,2,1,0,7,6,4,3,
				 11,10,9,8,15,14,13,12};
  		const __m512i swap_128  = (__m512i)__builtin_ia32_permvarsi512_mask((__v16si)indices,
						     (__v16si)input.v512,
						     (__v16si)Y, (__mmask16) -1);

		// 2. reverse order of bytes within 128-bit lanes
  		const __m512i indices_byte = __extension__ (__m512i) (__v8di)
		       {0x0001020304050607llu, 0x08090a0b0c0d0e0fllu,
		        0x0001020304050607llu, 0x08090a0b0c0d0e0fllu,
		        0x0001020304050607llu, 0x08090a0b0c0d0e0fllu,
		        0x0001020304050607llu, 0x08090a0b0c0d0e0fllu};

		uint8x64_t ret;
		ret.v512 = (__m512i) __builtin_ia32_pshufb512_mask ((__v64qi)swap_128,
						  (__v64qi)indices_byte,
						  (__v64qi)Y,
						  (__mmask64) -1);
		return ret;
#endif
#else
		uint8x64_t ret;
		// 1. reverse order of 32-bit words in register
		const __m512i indices = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		const __m512i swap_32 = _mm512_permutexvar_epi32(indices, v);

		// 2. reverse order of 16-bit words within 32-bit lanes
		// swap_32 = [ a | b | c | d ] x 16
		// swap_16 = [ c | d | a | b ] x 16
		const __m512i swap_16 = _mm512_rol_epi32(swap_32, 16);

		// 3. reverse bytes within 16-bit words

		// swap_16 = [ c | d | a | b ] x 16
		//      t0 = [ 0 | c | d | a ] x 16
		//      t1 = [ d | a | b | 0 ] x 16
		const __m512i t0 = _mm512_srli_epi32(swap_16, 8);
		const __m512i t1 = _mm512_slli_epi32(swap_16, 8);

		//   mask0 = [ 0 | ff| 0 | ff] x 16
		const __m512i mask0 = _mm512_set1_epi32(0x00ff00ff);

		//  result = (mask0 and t0) or (not mask0 and t1)
		//         = [ d | c | b | a]
		ret.v512 = _mm512_ternarylogic_epi32(mask0, t0, t1, 0xca);
		return ret;
#endif
	}


	/// finds the first limb in lane which is equal to in2
	/// SOURCE: http://0x80.pl/notesen/2023-02-06-avx512-find-first-byte-in-lane.html
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr inline static uint8x64_t find_first_byte_in_lane(const uint8x64_t in1,
	                                                                  const uint8_t in2) noexcept {
		uint8x64_t tmp1 = uint8x64_t::setr(
				0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,
				0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,
				0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,
				1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1
		);

		uint8x64_t tmp = uint8x64_t::set1(in2);
		constexpr uint8x64_t one = uint8x64_t::set1(1u);
		tmp = uint8x64_t::xor_(tmp, in1);
		uint8x64_t tmpp = uint8x64_t::xor_(uint8x64_t::min(one, tmp), one);
		tmp = uint8x64_t::sub(tmpp, tmp1);
		tmp = uint8x64_t::and_(tmp, uint8x64_t::not_(tmpp));
		return uint8x64_t::popcnt(tmp);
	}

	/// returns 
	/// if b == 0: return 0
   	/// if b < 0 : return -a 
    /// if b > 0 : return a
	[[nodiscard]] constexpr inline static uint8x64_t comp_sign(const uint8x64_t a,
			const uint8x64_t b) {
		uint8x64_t ret;
  		__m512i zero = _mm512_setzero_si512();
  		__mmask64 blt0 = _mm512_movepi8_mask(b.v512);
  		__mmask64 ble0 = _mm512_cmple_epi8_mask(b.v512, zero);
  		__m512i a_blt0 = _mm512_mask_mov_epi8(zero, blt0, a.v512);
		ret.v512 = _mm512_mask_sub_epi8(a.v512, ble0, zero, a_blt0);;
  		return ret;
	}
};

struct uint16x32_t {
	constexpr static uint32_t LIMBS = 32;
	using limb_type = uint16_t;

	constexpr uint16x32_t() noexcept = default;
	constexpr uint16x32_t(const uint8x64_t &b) noexcept;
	constexpr uint16x32_t(const uint32x16_t &b) noexcept;
	constexpr uint16x32_t(const uint64x8_t &b) noexcept;

	union {
		// compatibility with TxN_t
		uint16_t d[32];

		uint8_t v8[64];
		uint16_t v16[32];
		uint32_t v32[16];
		uint64_t v64[8];
		cryptanalysislib::_uint16x8_t v128[4];
		__m512i v256[2];
		__m512i v512;
	};

	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline uint8x64_t random() noexcept {
		uint8x64_t ret;
		for (size_t i = 0; i < 8; ++i) {
			ret.v64[i] = fastrandombytes_uint64();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t set(
	        char __q31, char __q30, char __q29, char __q28,
	        char __q27, char __q26, char __q25, char __q24,
	        char __q23, char __q22, char __q21, char __q20,
	        char __q19, char __q18, char __q17, char __q16,
	        char __q15, char __q14, char __q13, char __q12,
	        char __q11, char __q10, char __q09, char __q08,
	        char __q07, char __q06, char __q05, char __q04,
	        char __q03, char __q02, char __q01, char __q00) noexcept {
		uint16x32_t out;
		out.v512 = __extension__(__m512i)(__v64qi){
		        __q00, __q01, __q02, __q03, __q04, __q05, __q06, __q07,
		        __q08, __q09, __q10, __q11, __q12, __q13, __q14, __q15,
		        __q16, __q17, __q18, __q19, __q20, __q21, __q22, __q23,
		        __q24, __q25, __q26, __q27, __q28, __q29, __q30, __q31};
		return out;
	}

	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t setr(
	        char __q31, char __q30, char __q29, char __q28,
	        char __q27, char __q26, char __q25, char __q24,
	        char __q23, char __q22, char __q21, char __q20,
	        char __q19, char __q18, char __q17, char __q16,
	        char __q15, char __q14, char __q13, char __q12,
	        char __q11, char __q10, char __q09, char __q08,
	        char __q07, char __q06, char __q05, char __q04,
	        char __q03, char __q02, char __q01, char __q00) noexcept {
		uint16x32_t out;
		out.v512 = __extension__(__m512i)(__v64qi){
		        __q31, __q30, __q29, __q28, __q27, __q26, __q25, __q24,
		        __q23, __q22, __q21, __q20, __q19, __q18, __q17, __q16,
		        __q15, __q14, __q13, __q12, __q11, __q10, __q09, __q08,
		        __q07, __q06, __q05, __q04, __q03, __q02, __q01, __q00};
		return out;
	}

	[[nodiscard]] constexpr static inline uint16x32_t set1(char __A) noexcept {
		uint16x32_t out;
		out.v512 = __extension__(__m512i)(__v64qi){__A, __A, __A, __A, __A, __A, __A, __A,
		                                           __A, __A, __A, __A, __A, __A, __A, __A,
		                                           __A, __A, __A, __A, __A, __A, __A, __A,
		                                           __A, __A, __A, __A, __A, __A, __A, __A};

		return out;
	}


	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint16x32_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t aligned_load(const void *ptr) noexcept {
		const __m512i tmp = *(__m512i *) ptr;
		uint16x32_t out;
		out.v512 = tmp;
		return out;
	}


	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t unaligned_load(const void *ptr) noexcept {
		const __m512i tmp = (__m512i) (*(__v64qi_u *) ptr);
		uint16x32_t out;
		out.v512 = tmp;
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const uint16x32_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr, const uint16x32_t in) noexcept {
		auto *ptr512 = (__m512i *) ptr;
		*ptr512 = in.v512;
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint16x32_t in) noexcept {
		auto *ptr512 = (__m512i_u *) ptr;
		*(__m512i_u *) ptr512 = (__m512i_u) in.v512;
	}


	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t xor_(const uint16x32_t in1,
	                                                       const uint16x32_t in2) noexcept {
		uint16x32_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t and_(const uint16x32_t in1,
	                                                       const uint16x32_t in2) noexcept {
		uint16x32_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t or_(const uint16x32_t in1,
	                                                      const uint16x32_t in2) noexcept {
		uint16x32_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 | (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t andnot(const uint16x32_t in1,
	                                                         const uint16x32_t in2) noexcept {
		uint16x32_t out;
		out.v512 = (__m512i) (~(__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t not_(const uint16x32_t in1) noexcept {
		uint16x32_t out;
		const uint16x32_t minus_one = set1(-1);
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) minus_one.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t add(const uint16x32_t in1,
	                                                      const uint16x32_t in2) noexcept {
		uint16x32_t out;
		out.v512 = (__m512i) ((__v32hu) in1.v512 + (__v32hu) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t sub(const uint16x32_t in1,
	                                                      const uint16x32_t in2) noexcept {
		uint16x32_t out;
		out.v512 = (__m512i) ((__v32hu) in1.v512 - (__v32hu) in2.v512);
		return out;
	}

	/// 8 bit mul lo
	/// \param in1 first input
	/// \param in2
	/// \return in1*in2
	[[nodiscard]] constexpr static inline uint16x32_t mullo(const uint16x32_t in1,
	                                                        const uint16x32_t in2) noexcept {
		uint16x32_t out;
		(void) in1;
		(void) in2;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t mullo(const uint16x32_t in1,
	                                                        const uint8_t in2) noexcept {
		const uint16x32_t rs = uint16x32_t::set1(in2);
		return uint16x32_t::mullo(in1, rs);
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t popcnt(const uint16x32_t in1) noexcept {
		uint16x32_t ret;
#ifdef __clang__
		ret.v512 = (__m512i) __builtin_ia32_vpopcntw_512((__v32hi) in1.v512);
#else
  		ret.v512 = (__m512i) __builtin_ia32_vpopcountw_v32hi ((__v32hi)in1.v512);
#endif
		return ret;
	}

	/// Source:http://0x80.pl/notesen/2023-01-31-avx512-bsf.html
	/// needs`AVX512VPOPCNTDQ`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint16x32_t lzcnt(const uint16x32_t in1) noexcept {
		uint16x32_t ret;
		constexpr uint16x32_t one = uint16x32_t::set1(1);
		ret = uint16x32_t::sub(in1, one);
		ret = uint16x32_t::and_(ret, uint16x32_t::not_(in1));
		ret = uint16x32_t::popcnt(ret);
		return ret;
	}
};

struct uint32x16_t {
	constexpr static uint32_t LIMBS = 16;
	using limb_type = uint32_t;

	constexpr uint32x16_t() noexcept = default;
	constexpr uint32x16_t(const uint8x64_t &b) noexcept;
	constexpr uint32x16_t(const uint16x32_t &b) noexcept;
	constexpr uint32x16_t(const uint64x8_t &b) noexcept;

	union {
		// compatibility with TxN_t
		uint32_t d[16];

		uint8_t v8[64];
		uint16_t v16[32];
		uint32_t v32[16];
		uint64_t v64[8];
		cryptanalysislib::_uint32x4_t v128[4];
		__m512i v256[2];
		__m512i v512;
	};

	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline uint8x64_t random() noexcept {
		uint8x64_t ret;
		for (size_t i = 0; i < 8; ++i) {
			ret.v64[i] = fastrandombytes_uint64();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t set(
	        char __q15, char __q14, char __q13, char __q12,
	        char __q11, char __q10, char __q09, char __q08,
	        char __q07, char __q06, char __q05, char __q04,
	        char __q03, char __q02, char __q01, char __q00) noexcept {
		uint32x16_t out;
		out.v512 = __extension__(__m512i)(__v64qi){
		        __q00, __q01, __q02, __q03, __q04, __q05, __q06, __q07,
		        __q08, __q09, __q10, __q11, __q12, __q13, __q14, __q15};
		return out;
	}

	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t setr(
	        char __q15, char __q14, char __q13, char __q12,
	        char __q11, char __q10, char __q09, char __q08,
	        char __q07, char __q06, char __q05, char __q04,
	        char __q03, char __q02, char __q01, char __q00) noexcept {
		uint32x16_t out;
		out.v512 = __extension__(__m512i)(__v64qi){
		        __q15, __q14, __q13, __q12, __q11, __q10, __q09, __q08,
		        __q07, __q06, __q05, __q04, __q03, __q02, __q01, __q00};
		return out;
	}

	[[nodiscard]] constexpr static inline uint32x16_t set1(char __a) noexcept {
		uint32x16_t out;
		out.v512 = __extension__(__m512i)(__v64qi){__a, __a, __a, __a, __a, __a, __a, __a,
		                                           __a, __a, __a, __a, __a, __a, __a, __a};

		return out;
	}


	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint32x16_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t aligned_load(const void *ptr) noexcept {
		const __m512i tmp = *(__m512i *) ptr;
		uint32x16_t out;
		out.v512 = tmp;
		return out;
	}


	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t unaligned_load(const void *ptr) noexcept {
		const __m512i tmp = (__m512i) (*(__v64qi_u *) ptr);
		uint32x16_t out;
		out.v512 = tmp;
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const uint32x16_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr, const uint32x16_t in) noexcept {
		auto *ptr512 = (__m512i *) ptr;
		*ptr512 = in.v512;
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint32x16_t in) noexcept {
		auto *ptr512 = (__m512i_u *) ptr;
		*(__m512i_u *) ptr512 = (__m512i_u) in.v512;
	}


	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t xor_(const uint32x16_t in1,
	                                                       const uint32x16_t in2) noexcept {
		uint32x16_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t and_(const uint32x16_t in1,
	                                                       const uint32x16_t in2) noexcept {
		uint32x16_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t or_(const uint32x16_t in1,
	                                                      const uint32x16_t in2) noexcept {
		uint32x16_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 | (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t andnot(const uint32x16_t in1,
	                                                         const uint32x16_t in2) noexcept {
		uint32x16_t out;
		out.v512 = (__m512i) (~(__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t not_(const uint32x16_t in1) noexcept {
		uint32x16_t out;
		const uint32x16_t minus_one = set1(-1);
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) minus_one.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t add(const uint32x16_t in1,
	                                                      const uint32x16_t in2) noexcept {
		uint32x16_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 + (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t sub(const uint32x16_t in1,
	                                                      const uint32x16_t in2) noexcept {
		uint32x16_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 - (__v16su) in2.v512);
		return out;
	}

	/// 8 bit mul lo
	/// \param in1 first input
	/// \param in2
	/// \return in1*in2
	[[nodiscard]] constexpr static inline uint32x16_t mullo(const uint32x16_t in1,
	                                                        const uint32x16_t in2) noexcept {
		uint32x16_t out;
		(void) in1;
		(void) in2;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t mullo(const uint32x16_t in1,
	                                                        const uint32_t in2) noexcept {
		const uint32x16_t rs = uint32x16_t::set1(in2);
		return uint32x16_t::mullo(in1, rs);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t mullo(const uint32x16_t in1,
	                                                        const uint8_t in2) noexcept {
		const uint32x16_t rs = uint32x16_t::set1(in2);
		return uint32x16_t::mullo(in1, rs);
	}

	/// needs`AVX512VPOPCNTDQ`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t popcnt(const uint32x16_t in1) noexcept {
		uint32x16_t ret;
#ifdef __clang__
		ret.v512 = (__m512i) __builtin_ia32_vpopcntd_512((__v16si)in1.v512);
#else
  		ret.v512 = (__m512i) __builtin_ia32_vpopcountd_v16si ((__v16si)in1.v512);
#endif
		return ret;
	}

	/// TODO test
	/// Source:http://0x80.pl/notesen/2023-01-31-avx512-bsf.html
	/// needs`AVX512VPOPCNTDQ`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t cnt(const uint32x16_t in1) noexcept {
		uint32x16_t ret;
		const uint32x16_t one = uint32x16_t::set1(1u);
		ret = uint32x16_t::sub(in1, one);
		ret = uint32x16_t::and_(ret, uint32x16_t::not_(in1));
		ret = uint32x16_t::popcnt(ret);
		return ret;
	}

	/// needs `AVX512CD`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint32x16_t conflict(const uint32x16_t in1) noexcept {
		uint32x16_t ret;
#ifdef __clang__
		ret.v512 = (__m512i) __builtin_ia32_vpconflictsi_512((__v16si) in1.v512);
#else
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Winit-self"
		  __m512i Y = Y;
		#pragma GCC diagnostic pop

  		ret.v512 = (__m512i)__builtin_ia32_vpconflictsi_512_mask ((__v16si)in1.v512,
					       (__v16si)Y,
					       (__mmask16) -1);
#endif
		return ret;
	}

	/// needs `AVX512F`, wrapper around `_mm512_shuffle_i32x4`
	/// \param input
	/// \return
	template<const uint32_t imm>
	[[nodiscard]] constexpr static inline uint32x16_t shuffle_32x4(const uint32x16_t in1,
	                                                               const uint32x16_t in2) noexcept {
		uint32x16_t ret;
#ifdef __clang__
		ret.v512 = ((__m512i) __builtin_ia32_shuf_i32x4((__v16si) (__m512i) (in1.v512),
		                                                (__v16si) (__m512i) (in2.v512), (int) (imm)));
#else
		#pragma GCC diagnostic push
		#pragma GCC diagnostic ignored "-Winit-self"
		  __m512i Y = Y;
		#pragma GCC diagnostic pop

  		ret.v512 = (__m512i) __builtin_ia32_shuf_i32x4_mask ((__v16si)in1.v512,
						   (__v16si)in2.v512,
						   imm,
						   (__v16si)Y,
						   (__mmask16) -1);
#endif
		return ret;
	}
};

struct uint64x8_t {
	constexpr static uint32_t LIMBS = 8;
	using limb_type = uint64_t;

	constexpr uint64x8_t() noexcept = default;
	constexpr uint64x8_t(const uint8x64_t &b) noexcept;
	constexpr uint64x8_t(const uint32x16_t &b) noexcept;
	constexpr uint64x8_t(const uint16x32_t &b) noexcept;

	union {
		// compatibility with TxN_t
		uint64_t d[8];

		uint8_t v8[64];
		uint16_t v16[32];
		uint32_t v32[16];
		uint64_t v64[8];
		cryptanalysislib::_uint64x2_t v128[4];
		__m512i v256[2];
		__m512i v512;
	};

	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline uint8x64_t random() noexcept {
		uint8x64_t ret;
		for (size_t i = 0; i < 8; ++i) {
			ret.v64[i] = fastrandombytes_uint64();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t set(
	        char __q15, char __q14, char __q13, char __q12,
	        char __q11, char __q10, char __q09, char __q08,
	        char __q07, char __q06, char __q05, char __q04,
	        char __q03, char __q02, char __q01, char __q00) noexcept {
		uint64x8_t out;
		out.v512 = __extension__(__m512i)(__v64qi){
		        __q00, __q01, __q02, __q03, __q04, __q05, __q06, __q07,
		        __q08, __q09, __q10, __q11, __q12, __q13, __q14, __q15};
		return out;
	}

	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t setr(
	        char __q15, char __q14, char __q13, char __q12,
	        char __q11, char __q10, char __q09, char __q08,
	        char __q07, char __q06, char __q05, char __q04,
	        char __q03, char __q02, char __q01, char __q00) noexcept {
		uint64x8_t out;
		out.v512 = __extension__(__m512i)(__v64qi){
		        __q15, __q14, __q13, __q12, __q11, __q10, __q09, __q08,
		        __q07, __q06, __q05, __q04, __q03, __q02, __q01, __q00};
		return out;
	}

	[[nodiscard]] constexpr static inline uint64x8_t set1(char __a) noexcept {
		uint64x8_t out;
		out.v512 = __extension__(__m512i)(__v64qi){__a, __a, __a, __a, __a, __a, __a, __a,
		                                           __a, __a, __a, __a, __a, __a, __a, __a};

		return out;
	}


	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint64x8_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t aligned_load(const void *ptr) noexcept {
		const __m512i tmp = *(__m512i *) ptr;
		uint64x8_t out;
		out.v512 = tmp;
		return out;
	}


	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t unaligned_load(const void *ptr) noexcept {
		const __m512i tmp = (__m512i) (*(__v64qi_u *) ptr);
		uint64x8_t out;
		out.v512 = tmp;
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const uint64x8_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr, const uint64x8_t in) noexcept {
		auto *ptr512 = (__m512i *) ptr;
		*ptr512 = in.v512;
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint64x8_t in) noexcept {
		auto *ptr512 = (__m512i_u *) ptr;
		*(__m512i_u *) ptr512 = (__m512i_u) in.v512;
	}


	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t xor_(const uint64x8_t in1,
	                                                      const uint64x8_t in2) noexcept {
		uint64x8_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t and_(const uint64x8_t in1,
	                                                      const uint64x8_t in2) noexcept {
		uint64x8_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t or_(const uint64x8_t in1,
	                                                     const uint64x8_t in2) noexcept {
		uint64x8_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 | (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t andnot(const uint64x8_t in1,
	                                                        const uint64x8_t in2) noexcept {
		uint64x8_t out;
		out.v512 = (__m512i) (~(__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t not_(const uint64x8_t in1) noexcept {
		uint64x8_t out;
		const uint64x8_t minus_one = set1(-1);
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) minus_one.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t add(const uint64x8_t in1,
	                                                     const uint64x8_t in2) noexcept {
		uint64x8_t out;
		out.v512 = (__m512i) ((__v8du) in1.v512 + (__v8du) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t sub(const uint64x8_t in1,
	                                                     const uint64x8_t in2) noexcept {
		uint64x8_t out;
		out.v512 = (__m512i) ((__v8du) in1.v512 - (__v8du) in2.v512);
		return out;
	}

	/// 8 bit mul lo
	/// \param in1 first input
	/// \param in2
	/// \return in1*in2
	[[nodiscard]] constexpr static inline uint64x8_t mullo(const uint64x8_t in1,
	                                                       const uint64x8_t in2) noexcept {
		uint64x8_t out;
		(void) in1;
		(void) in2;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t mullo(const uint64x8_t in1,
	                                                       const uint8_t in2) noexcept {
		const uint64x8_t rs = uint64x8_t::set1(in2);
		return uint64x8_t::mullo(in1, rs);
	}


	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t popcnt(const uint64x8_t in1) noexcept {
		uint64x8_t ret;
#ifdef __clang__
		ret.v512 = (__m512i) __builtin_ia32_vpopcntq_512((__v8di) in1.v512);
#else
		ret.v512 = (__m512i) __builtin_ia32_vpopcountq_v8di(in1.v512);
#endif
		return ret;
	}

	/// Source:http://0x80.pl/notesen/2023-01-31-avx512-bsf.html
	/// needs `AVX512VPOPCNTDQ` + `AVX512VL`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t lzcnt(const uint64x8_t in1) noexcept {
		uint64x8_t ret;
		constexpr uint64x8_t one = uint64x8_t::set1(1);
		ret = uint64x8_t::sub(in1, one);
		ret = uint64x8_t::and_(ret, uint64x8_t::not_(in1));
		ret = uint64x8_t::popcnt(ret);
		return ret;
	}

	/// needs `AVX512CD`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t conflict(const uint64x8_t in1) noexcept {
		uint64x8_t ret;
#ifdef __clang__
		ret.v512 = (__m512i) __builtin_ia32_vpconflictdi_512((__v8di) in1.v512);
#else
		ret.v512 = (__m512i) __builtin_ia32_vpconflictdi_512_mask ((__v8di)in1.v512,
					       (__v8di) __extension__ (__m512i)(__v8di){ 0, 0, 0, 0, 0, 0, 0, 0 },
					       (__mmask8) -1);
#endif
		return ret;
	}

	/// TODO test and implement for uint32x16 and so on and implement hadd_epu16,...
	/// needs `AVX512BW`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t hadd_epu8(const uint64x8_t in1) noexcept {
		uint64x8_t ret;
		constexpr uint64x8_t zero = uint64x8_t::set1(0);
		ret.v512 = (__m512i) __builtin_ia32_psadbw512((__v64qi) in1.v512, (__v64qi) zero.v512);
		return ret;
	}

	/// TODO test, and implement for uint32x16 and so on
	/// Source: http://0x80.pl/notesen/2023-01-06-avx512-popcount-4bit.html
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x8_t histogram_epi4(const uint64x8_t in1,
	                                                                const uint8_t in2) noexcept {
		ASSERT(in2 < 16);
		uint64x8_t tmp = uint64x8_t::xor_(in1, uint64x8_t::set1(in2));
		tmp = uint64x8_t::sub(tmp, uint64x8_t::set1(1));
		tmp = uint64x8_t::popcnt(tmp);
		return tmp;
	}
};



///
uint8x64_t operator*(const uint8x64_t &lhs, const uint8x64_t &rhs) {
	return uint8x64_t::mullo(lhs, rhs);
}
inline uint8x64_t operator*(const uint8x64_t &lhs, const uint8_t &rhs) {
	return uint8x64_t::mullo(lhs, rhs);
}
inline uint8x64_t operator*(const uint8_t &lhs, const uint8x64_t &rhs) {
	return uint8x64_t::mullo(rhs, lhs);
}
inline uint8x64_t operator+(const uint8x64_t &lhs, const uint8x64_t &rhs) {
	return uint8x64_t::add(lhs, rhs);
}
inline uint8x64_t operator-(const uint8x64_t &lhs, const uint8x64_t &rhs) {
	return uint8x64_t::sub(lhs, rhs);
}
inline uint8x64_t operator&(const uint8x64_t &lhs, const uint8x64_t &rhs) {
	return uint8x64_t::and_(lhs, rhs);
}
inline uint8x64_t operator^(const uint8x64_t &lhs, const uint8x64_t &rhs) {
	return uint8x64_t::xor_(lhs, rhs);
}
inline uint8x64_t operator|(const uint8x64_t &lhs, const uint8x64_t &rhs) {
	return uint8x64_t::or_(lhs, rhs);
}
inline uint8x64_t operator~(const uint8x64_t &lhs) {
	return uint8x64_t::not_(lhs);
}
//inline uint8x64_t operator>> (const uint8x64_t& lhs, const uint32_t rhs) {
//	return uint8x64_t::srli(lhs, rhs);
//}
//inline uint8x64_t operator<< (const uint8x64_t& lhs, const uint32_t rhs) {
//	return uint8x64_t::slli(lhs, rhs);
//}
inline uint8x64_t operator^=(uint8x64_t &lhs, const uint8x64_t &rhs) {
	lhs = uint8x64_t::xor_(lhs, rhs);
	return lhs;
}
inline uint8x64_t operator&=(uint8x64_t &lhs, const uint8x64_t &rhs) {
	lhs = uint8x64_t::and_(lhs, rhs);
	return lhs;
}
inline uint8x64_t operator|=(uint8x64_t &lhs, const uint8x64_t &rhs) {
	lhs = uint8x64_t::or_(lhs, rhs);
	return lhs;
}


///
inline uint16x32_t operator*(const uint16x32_t &lhs, const uint16x32_t &rhs) {
	return uint16x32_t::mullo(lhs, rhs);
}
inline uint16x32_t operator*(const uint16x32_t &lhs, const uint8_t &rhs) {
	return uint16x32_t::mullo(lhs, rhs);
}
inline uint16x32_t operator*(const uint8_t &lhs, const uint16x32_t &rhs) {
	return uint16x32_t::mullo(rhs, lhs);
}
inline uint16x32_t operator+(const uint16x32_t &lhs, const uint16x32_t &rhs) {
	return uint16x32_t::add(lhs, rhs);
}
inline uint16x32_t operator-(const uint16x32_t &lhs, const uint16x32_t &rhs) {
	return uint16x32_t::sub(lhs, rhs);
}
inline uint16x32_t operator&(const uint16x32_t &lhs, const uint16x32_t &rhs) {
	return uint16x32_t::and_(lhs, rhs);
}
inline uint16x32_t operator^(const uint16x32_t &lhs, const uint16x32_t &rhs) {
	return uint16x32_t::xor_(lhs, rhs);
}
inline uint16x32_t operator|(const uint16x32_t &lhs, const uint16x32_t &rhs) {
	return uint16x32_t::or_(lhs, rhs);
}
inline uint16x32_t operator~(const uint16x32_t &lhs) {
	return uint16x32_t::not_(lhs);
}
//inline uint16x32_t operator>> (const uint16x32_t& lhs, const uint32_t rhs) {
//	return uint16x32_t::srli(lhs, rhs);
//}
//inline uint16x32_t operator<< (const uint16x32_t& lhs, const uint32_t rhs) {
//	return uint16x32_t::slli(lhs, rhs);
//}
inline uint16x32_t operator^=(uint16x32_t &lhs, const uint16x32_t &rhs) {
	lhs = uint16x32_t::xor_(lhs, rhs);
	return lhs;
}
inline uint16x32_t operator&=(uint16x32_t &lhs, const uint16x32_t &rhs) {
	lhs = uint16x32_t::and_(lhs, rhs);
	return lhs;
}
inline uint16x32_t operator|=(uint16x32_t &lhs, const uint16x32_t &rhs) {
	lhs = uint16x32_t::or_(lhs, rhs);
	return lhs;
}


///
inline uint32x16_t operator*(const uint32x16_t &lhs, const uint32x16_t &rhs) {
	return uint32x16_t::mullo(lhs, rhs);
}
inline uint32x16_t operator*(const uint32x16_t &lhs, const uint8_t &rhs) {
	return uint32x16_t::mullo(lhs, rhs);
}
inline uint32x16_t operator*(const uint8_t &lhs, const uint32x16_t &rhs) {
	return uint32x16_t::mullo(rhs, lhs);
}
inline uint32x16_t operator+(const uint32x16_t &lhs, const uint32x16_t &rhs) {
	return uint32x16_t::add(lhs, rhs);
}
inline uint32x16_t operator-(const uint32x16_t &lhs, const uint32x16_t &rhs) {
	return uint32x16_t::sub(lhs, rhs);
}
inline uint32x16_t operator&(const uint32x16_t &lhs, const uint32x16_t &rhs) {
	return uint32x16_t::and_(lhs, rhs);
}
inline uint32x16_t operator^(const uint32x16_t &lhs, const uint32x16_t &rhs) {
	return uint32x16_t::xor_(lhs, rhs);
}
inline uint32x16_t operator|(const uint32x16_t &lhs, const uint32x16_t &rhs) {
	return uint32x16_t::or_(lhs, rhs);
}
inline uint32x16_t operator~(const uint32x16_t &lhs) {
	return uint32x16_t::not_(lhs);
}
//inline uint32x16_t operator>> (const uint32x16_t& lhs, const uint32_t rhs) {
//	return uint32x16_t::srli(lhs, rhs);
//}
//inline uint32x16_t operator<< (const uint32x16_t& lhs, const uint32_t rhs) {
//	return uint32x16_t::slli(lhs, rhs);
//}
inline uint32x16_t operator^=(uint32x16_t &lhs, const uint32x16_t &rhs) {
	lhs = uint32x16_t::xor_(lhs, rhs);
	return lhs;
}
inline uint32x16_t operator&=(uint32x16_t &lhs, const uint32x16_t &rhs) {
	lhs = uint32x16_t::and_(lhs, rhs);
	return lhs;
}
inline uint32x16_t operator|=(uint32x16_t &lhs, const uint32x16_t &rhs) {
	lhs = uint32x16_t::or_(lhs, rhs);
	return lhs;
}


///
inline uint64x8_t operator*(const uint64x8_t &lhs, const uint64x8_t &rhs) {
	return uint64x8_t::mullo(lhs, rhs);
}
inline uint64x8_t operator*(const uint64x8_t &lhs, const uint8_t &rhs) {
	return uint64x8_t::mullo(lhs, rhs);
}
inline uint64x8_t operator*(const uint8_t &lhs, const uint64x8_t &rhs) {
	return uint64x8_t::mullo(rhs, lhs);
}
inline uint64x8_t operator+(const uint64x8_t &lhs, const uint64x8_t &rhs) {
	return uint64x8_t::add(lhs, rhs);
}
inline uint64x8_t operator-(const uint64x8_t &lhs, const uint64x8_t &rhs) {
	return uint64x8_t::sub(lhs, rhs);
}
inline uint64x8_t operator&(const uint64x8_t &lhs, const uint64x8_t &rhs) {
	return uint64x8_t::and_(lhs, rhs);
}
inline uint64x8_t operator^(const uint64x8_t &lhs, const uint64x8_t &rhs) {
	return uint64x8_t::xor_(lhs, rhs);
}
inline uint64x8_t operator|(const uint64x8_t &lhs, const uint64x8_t &rhs) {
	return uint64x8_t::or_(lhs, rhs);
}
inline uint64x8_t operator~(const uint64x8_t &lhs) {
	return uint64x8_t::not_(lhs);
}
//inline uint64x8_t operator>> (const uint64x8_t& lhs, const uint32_t rhs) {
//	return uint64x8_t::srli(lhs, rhs);
//}
//inline uint64x8_t operator<< (const uint64x8_t& lhs, const uint32_t rhs) {
//	return uint64x8_t::slli(lhs, rhs);
//}
inline uint64x8_t operator^=(uint64x8_t &lhs, const uint64x8_t &rhs) {
	lhs = uint64x8_t::xor_(lhs, rhs);
	return lhs;
}
inline uint64x8_t operator&=(uint64x8_t &lhs, const uint64x8_t &rhs) {
	lhs = uint64x8_t::and_(lhs, rhs);
	return lhs;
}
inline uint64x8_t operator|=(uint64x8_t &lhs, const uint64x8_t &rhs) {
	lhs = uint64x8_t::or_(lhs, rhs);
	return lhs;
}
#endif//CRYPTANALYSISLIB_AVX512_H
