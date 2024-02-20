#ifndef CRYPTANALYSISLIB_SIMD_AVX512_H
#define CRYPTANALYSISLIB_SIMD_AVX512_H

#ifndef CRYPTANALYSISLIB_SIMD_H
#error "dont include this file directly. Use `#include <simd/simd.h>`"
#endif

#ifndef USE_AVX512
#error "no avx512 enabled."
#endif

#include <cstdint>
#include <immintrin.h>

#include "random.h"

typedef char __v64qi_u __attribute__((__vector_size__(64), __may_alias__, __aligned__(1)));


struct uint8x64_t {
	union {
		uint8_t v8[64];
		uint16_t v16[32];
		uint32_t v32[16];
		uint64_t v64[8];
		// cryptanalysislib::_uint8x16_t v128[4];
		__m512i v256[2];
		__m512i v512;
	};

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
#ifndef __clang__
		out.v512 = (__m512i) __builtin_ia32_andnotsi256((__v4di) in1.v512, (__v4di) in2.v512);
#else
		out.v512 = (__m512i) (~(__v16su) in1.v512 & (__v16su) in2.v512);
#endif
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
};

struct uint16x32_t {
	union {
		uint8_t v8[64];
		uint16_t v16[32];
		uint32_t v32[16];
		uint64_t v64[8];
		// cryptanalysislib::_uint8x16_t v128[4];
		__m512i v256[2];
		__m512i v512;
	};

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
#ifndef __clang__
		out.v512 = (__m512i) __builtin_ia32_andnotsi256((__v4di) in1.v512, (__v4di) in2.v512);
#else
		out.v512 = (__m512i) (~(__v16su) in1.v512 & (__v16su) in2.v512);
#endif
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
};


struct uint32x16_t {
	union {
		uint8_t v8[64];
		uint16_t v16[32];
		uint32_t v32[16];
		uint64_t v64[8];
		// cryptanalysislib::_uint8x16_t v128[4];
		__m512i v256[2];
		__m512i v512;
	};

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
#ifndef __clang__
		out.v512 = (__m512i) __builtin_ia32_andnotsi256((__v4di) in1.v512, (__v4di) in2.v512);
#else
		out.v512 = (__m512i) (~(__v16su) in1.v512 & (__v16su) in2.v512);
#endif
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
	                                                        const uint8_t in2) noexcept {
		const uint32x16_t rs = uint32x16_t::set1(in2);
		return uint32x16_t::mullo(in1, rs);
	}
};


struct uint64x8_t {
	union {
		uint8_t v8[64];
		uint16_t v16[32];
		uint32_t v32[16];
		uint64_t v64[8];
		// cryptanalysislib::_uint8x16_t v128[4];
		__m512i v256[2];
		__m512i v512;
	};

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
#ifndef __clang__
		out.v512 = (__m512i) __builtin_ia32_andnotsi256((__v4di) in1.v512, (__v4di) in2.v512);
#else
		out.v512 = (__m512i) (~(__v16su) in1.v512 & (__v16su) in2.v512);
#endif
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
};

///
inline uint8x64_t operator*(const uint8x64_t &lhs, const uint8x64_t &rhs) {
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
