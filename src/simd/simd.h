#ifndef CRYPTANALYSISLIB_SIMD_H
#define CRYPTANALYSISLIB_SIMD_H

#include <cinttypes>
#include <cmath>
#include <cstdint>

#include "helper.h"
#include "popcount/popcount.h"
#include "print/print.h"
#include "random.h"

#define bit_shuffle_const(b0, b1, b2, b3, b4, b5, b6, b7) \
	((uint64_t(uint8_t(1 << b0)) << (7 * 8)) |            \
	 (uint64_t(uint8_t(1 << b1)) << (6 * 8)) |            \
	 (uint64_t(uint8_t(1 << b2)) << (5 * 8)) |            \
	 (uint64_t(uint8_t(1 << b3)) << (4 * 8)) |            \
	 (uint64_t(uint8_t(1 << b4)) << (3 * 8)) |            \
	 (uint64_t(uint8_t(1 << b5)) << (2 * 8)) |            \
	 (uint64_t(uint8_t(1 << b6)) << (1 * 8)) |            \
	 (uint64_t(uint8_t(1 << b7)) << (0 * 8)))

#if defined(USE_AVX2)

#include "simd/avx2.h"
#include "simd/float/avx2.h"
#if defined(USE_AVX512F)
#include "simd/avx512.h"
#endif

#elif defined(USE_ARM)
#include "simd/neon.h"
// TODO #include "simd/float/neon.h"
#include "simd/float/simd.h"
#elif defined(USE_RISCV)

#include "simd/riscv.h"

#else

namespace cryptanalysislib {
	struct _uint16x8_t;
	struct _uint32x4_t;
	struct _uint64x2_t;

	struct _uint8x16_t {
		constexpr static uint32_t LIMBS = 16;
		using limb_type = uint8_t;

		constexpr inline _uint8x16_t &operator=(const _uint16x8_t &b) noexcept;
		constexpr inline _uint8x16_t &operator=(const _uint32x4_t &b) noexcept;
		constexpr inline _uint8x16_t &operator=(const _uint64x2_t &b) noexcept;

		constexpr _uint8x16_t() noexcept {}
		constexpr _uint8x16_t(const _uint16x8_t &b) noexcept;
		constexpr _uint8x16_t(const _uint32x4_t &b) noexcept;
		constexpr _uint8x16_t(const _uint64x2_t &b) noexcept;

		union {
			// compatibility to `TxN_t`
			uint8_t d[16];

			uint8_t v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];
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
		static inline _uint8x16_t random() noexcept {
			_uint8x16_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = fastrandombytes_uint64();
			}

			return ret;
		}

		[[nodiscard]] constexpr static inline _uint8x16_t set(
		        uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_uint8x16_t ret;
			ret.v32[0] = d;
			ret.v32[1] = c;
			ret.v32[2] = b;
			ret.v32[3] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint8x16_t setr(
		        uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_uint8x16_t ret;
			ret.v32[0] = a;
			ret.v32[1] = b;
			ret.v32[2] = c;
			ret.v32[3] = d;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint8x16_t set(
		        uint8_t a, uint8_t b, uint8_t c, uint8_t d,
		        uint8_t e, uint8_t f, uint8_t g, uint8_t h,
		        uint8_t i, uint8_t j, uint8_t k, uint8_t l,
		        uint8_t m, uint8_t n, uint8_t o, uint8_t p) noexcept {
			_uint8x16_t ret;
			ret.v8[0] = p;
			ret.v8[1] = o;
			ret.v8[2] = n;
			ret.v8[3] = m;
			ret.v8[4] = l;
			ret.v8[5] = k;
			ret.v8[6] = j;
			ret.v8[7] = i;
			ret.v8[8] = h;
			ret.v8[9] = g;
			ret.v8[10] = f;
			ret.v8[11] = e;
			ret.v8[12] = d;
			ret.v8[13] = c;
			ret.v8[14] = b;
			ret.v8[15] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint8x16_t setr(
		        uint8_t a, uint8_t b, uint8_t c, uint8_t d,
		        uint8_t e, uint8_t f, uint8_t g, uint8_t h,
		        uint8_t i, uint8_t j, uint8_t k, uint8_t l,
		        uint8_t m, uint8_t n, uint8_t o, uint8_t p) noexcept {
			_uint8x16_t ret;
			ret.v8[0] = a;
			ret.v8[1] = b;
			ret.v8[2] = c;
			ret.v8[3] = d;
			ret.v8[4] = e;
			ret.v8[5] = f;
			ret.v8[6] = g;
			ret.v8[7] = h;
			ret.v8[8] = i;
			ret.v8[9] = j;
			ret.v8[10] = k;
			ret.v8[11] = l;
			ret.v8[12] = m;
			ret.v8[13] = n;
			ret.v8[14] = o;
			ret.v8[15] = p;
			return ret;
		}
	};

	struct _uint16x8_t {
		constexpr static uint32_t LIMBS = 8;
		using limb_type = uint16_t;

		constexpr inline _uint16x8_t &operator=(const _uint8x16_t &b) noexcept;
		constexpr inline _uint16x8_t &operator=(const _uint32x4_t &b) noexcept;
		constexpr inline _uint16x8_t &operator=(const _uint64x2_t &b) noexcept;

		constexpr _uint16x8_t() noexcept {}
		constexpr _uint16x8_t(const _uint8x16_t &b) noexcept;
		constexpr _uint16x8_t(const _uint32x4_t &b) noexcept;
		constexpr _uint16x8_t(const _uint64x2_t &b) noexcept;

		union {
			// compatibility to `TxN_t`
			uint16_t d[8];

			uint8_t v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];
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
		static inline _uint16x8_t random() noexcept {
			_uint16x8_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = fastrandombytes_uint64();
			}

			return ret;
		}

		[[nodiscard]] constexpr static inline _uint16x8_t set(
		        uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_uint16x8_t ret;
			ret.v32[0] = d;
			ret.v32[1] = c;
			ret.v32[2] = b;
			ret.v32[3] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint16x8_t setr(
		        uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_uint16x8_t ret;
			ret.v32[0] = a;
			ret.v32[1] = b;
			ret.v32[2] = c;
			ret.v32[3] = d;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint16x8_t set(
		        uint16_t a, uint16_t b, uint16_t c, uint16_t d,
		        uint16_t e, uint16_t f, uint16_t g, uint16_t h) noexcept {
			_uint16x8_t ret;
			ret.v16[0] = h;
			ret.v16[1] = g;
			ret.v16[2] = f;
			ret.v16[3] = e;
			ret.v16[4] = d;
			ret.v16[5] = c;
			ret.v16[6] = b;
			ret.v16[7] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint16x8_t setr(
		        uint16_t a, uint16_t b, uint16_t c, uint16_t d,
		        uint16_t e, uint16_t f, uint16_t g, uint16_t h) noexcept {
			_uint16x8_t ret;
			ret.v64[0] = a;
			ret.v64[1] = b;
			ret.v64[2] = c;
			ret.v64[3] = d;
			ret.v64[4] = e;
			ret.v64[5] = f;
			ret.v64[6] = g;
			ret.v64[7] = h;
			return ret;
		}
	};

	struct _uint32x4_t {
		constexpr static uint32_t LIMBS = 4;
		using limb_type = uint32_t;

		constexpr inline _uint32x4_t &operator=(const _uint8x16_t &b) noexcept;
		constexpr inline _uint32x4_t &operator=(const _uint16x8_t &b) noexcept;
		constexpr inline _uint32x4_t &operator=(const _uint64x2_t &b) noexcept;

		constexpr _uint32x4_t() noexcept {}
		constexpr _uint32x4_t(const _uint8x16_t &b) noexcept;
		constexpr _uint32x4_t(const _uint16x8_t &b) noexcept;
		constexpr _uint32x4_t(const _uint64x2_t &b) noexcept;

		union {
			// compatibility to `TxN_t`
			uint32_t d[4];

			uint8_t v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];
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
		static inline _uint32x4_t random() noexcept {
			_uint32x4_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = fastrandombytes_uint64();
			}

			return ret;
		}

		[[nodiscard]] constexpr static inline _uint32x4_t set(
		        uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_uint32x4_t ret;
			ret.v32[0] = d;
			ret.v32[1] = c;
			ret.v32[2] = b;
			ret.v32[3] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint32x4_t setr(
		        uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_uint32x4_t ret;
			ret.v32[0] = a;
			ret.v32[1] = b;
			ret.v32[2] = c;
			ret.v32[3] = d;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint32x4_t set(
		        uint64_t a, uint64_t b) noexcept {
			_uint32x4_t ret;
			ret.v64[0] = b;
			ret.v64[1] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint32x4_t setr(
		        uint64_t a, uint64_t b) noexcept {
			_uint32x4_t ret;
			ret.v64[0] = a;
			ret.v64[1] = b;
			return ret;
		}
	};

	struct _uint64x2_t {
		constexpr static uint32_t LIMBS = 2;
		using limb_type = uint64_t;

		constexpr inline _uint64x2_t &operator=(const _uint8x16_t &b) noexcept;
		constexpr inline _uint64x2_t &operator=(const _uint16x8_t &b) noexcept;
		constexpr inline _uint64x2_t &operator=(const _uint32x4_t &b) noexcept;

		constexpr _uint64x2_t() noexcept {}
		constexpr _uint64x2_t(const _uint8x16_t &b) noexcept;
		constexpr _uint64x2_t(const _uint16x8_t &b) noexcept;
		constexpr _uint64x2_t(const _uint32x4_t &b) noexcept;

		union {
			uint64_t d[2];

			uint8_t v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];
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
		static inline _uint64x2_t random() noexcept {
			_uint64x2_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = fastrandombytes_uint64();
			}

			return ret;
		}

		[[nodiscard]] constexpr static inline _uint64x2_t set(
		        uint64_t a, uint64_t b) noexcept {
			_uint64x2_t ret;
			ret.v64[0] = b;
			ret.v64[1] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _uint64x2_t setr(
		        uint64_t a, uint64_t b) noexcept {
			_uint64x2_t ret;
			ret.v64[0] = a;
			ret.v64[1] = b;
			return ret;
		}
	};
};// namespace cryptanalysislib

using namespace cryptanalysislib;

struct uint8x32_t {
	constexpr static uint32_t LIMBS = 32;
	using limb_type = uint8_t;

	union {
		uint8_t d[32];

		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
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
	static inline uint8x32_t random() noexcept {
		uint8x32_t ret;
		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = fastrandombytes_uint64();
		}

		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	///
	/// \param __q31
	/// \param __q30
	/// \param __q29
	/// \param __q28
	/// \param __q27
	/// \param __q26
	/// \param __q25
	/// \param __q24
	/// \param __q23
	/// \param __q22
	/// \param __q21
	/// \param __q20
	/// \param __q19
	/// \param __q18
	/// \param __q17
	/// \param __q16
	/// \param __q15
	/// \param __q14
	/// \param __q13
	/// \param __q12
	/// \param __q11
	/// \param __q10
	/// \param __q09
	/// \param __q08
	/// \param __q07
	/// \param __q06
	/// \param __q05
	/// \param __q04
	/// \param __q03
	/// \param __q02
	/// \param __q01
	/// \param __q00
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t setr(uint8_t __q31, uint8_t __q30, uint8_t __q29, uint8_t __q28,
	                                                      uint8_t __q27, uint8_t __q26, uint8_t __q25, uint8_t __q24,
	                                                      uint8_t __q23, uint8_t __q22, uint8_t __q21, uint8_t __q20,
	                                                      uint8_t __q19, uint8_t __q18, uint8_t __q17, uint8_t __q16,
	                                                      uint8_t __q15, uint8_t __q14, uint8_t __q13, uint8_t __q12,
	                                                      uint8_t __q11, uint8_t __q10, uint8_t __q09, uint8_t __q08,
	                                                      uint8_t __q07, uint8_t __q06, uint8_t __q05, uint8_t __q04,
	                                                      uint8_t __q03, uint8_t __q02, uint8_t __q01, uint8_t __q00) noexcept {
		uint8x32_t out;
		out.v8[0] = __q31;
		out.v8[1] = __q30;
		out.v8[2] = __q29;
		out.v8[3] = __q28;
		out.v8[4] = __q27;
		out.v8[5] = __q26;
		out.v8[6] = __q25;
		out.v8[7] = __q24;
		out.v8[8] = __q23;
		out.v8[9] = __q22;
		out.v8[10] = __q21;
		out.v8[11] = __q20;
		out.v8[12] = __q19;
		out.v8[13] = __q18;
		out.v8[14] = __q17;
		out.v8[15] = __q16;
		out.v8[16] = __q15;
		out.v8[17] = __q14;
		out.v8[18] = __q13;
		out.v8[19] = __q12;
		out.v8[20] = __q11;
		out.v8[21] = __q10;
		out.v8[22] = __q09;
		out.v8[23] = __q08;
		out.v8[24] = __q07;
		out.v8[25] = __q06;
		out.v8[26] = __q05;
		out.v8[27] = __q04;
		out.v8[28] = __q03;
		out.v8[29] = __q02;
		out.v8[30] = __q01;
		out.v8[31] = __q00;
		return out;
	}

	[[nodiscard]] constexpr static inline uint8x32_t set(uint8_t __q31, uint8_t __q30, uint8_t __q29, uint8_t __q28,
	                                                     uint8_t __q27, uint8_t __q26, uint8_t __q25, uint8_t __q24,
	                                                     uint8_t __q23, uint8_t __q22, uint8_t __q21, uint8_t __q20,
	                                                     uint8_t __q19, uint8_t __q18, uint8_t __q17, uint8_t __q16,
	                                                     uint8_t __q15, uint8_t __q14, uint8_t __q13, uint8_t __q12,
	                                                     uint8_t __q11, uint8_t __q10, uint8_t __q09, uint8_t __q08,
	                                                     uint8_t __q07, uint8_t __q06, uint8_t __q05, uint8_t __q04,
	                                                     uint8_t __q03, uint8_t __q02, uint8_t __q01, uint8_t __q00) noexcept {
		uint8x32_t out;
		out.v8[31] = __q31;
		out.v8[30] = __q30;
		out.v8[29] = __q29;
		out.v8[28] = __q28;
		out.v8[27] = __q27;
		out.v8[26] = __q26;
		out.v8[25] = __q25;
		out.v8[24] = __q24;
		out.v8[23] = __q23;
		out.v8[22] = __q22;
		out.v8[21] = __q21;
		out.v8[20] = __q20;
		out.v8[19] = __q19;
		out.v8[18] = __q18;
		out.v8[17] = __q17;
		out.v8[16] = __q16;
		out.v8[15] = __q15;
		out.v8[14] = __q14;
		out.v8[13] = __q13;
		out.v8[12] = __q12;
		out.v8[11] = __q11;
		out.v8[10] = __q10;
		out.v8[9] = __q09;
		out.v8[8] = __q08;
		out.v8[7] = __q07;
		out.v8[6] = __q06;
		out.v8[5] = __q05;
		out.v8[4] = __q04;
		out.v8[3] = __q03;
		out.v8[2] = __q02;
		out.v8[1] = __q01;
		out.v8[0] = __q00;
		return out;
	}

	/// sets all 32 8bit limbs to `a`
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t set1(const uint8_t a) noexcept {
		uint8x32_t out;
		out = uint8x32_t::set(a, a, a, a, a, a, a, a,
		                      a, a, a, a, a, a, a, a,
		                      a, a, a, a, a, a, a, a,
		                      a, a, a, a, a, a, a, a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	constexpr static inline uint8x32_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline uint8x32_t aligned_load(const void *ptr) noexcept {
		uint8x32_t out;
		const uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ptr64[i];
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	constexpr static inline uint8x32_t unaligned_load(const void *ptr) noexcept {
		uint8x32_t out;
		const uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ptr64[i];
		}
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const uint8x32_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint8x32_t in) noexcept {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint8x32_t in) noexcept {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t xor_(const uint8x32_t in1,
	                                                      const uint8x32_t in2) noexcept {
		uint8x32_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] ^ in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t and_(const uint8x32_t in1,
	                                                      const uint8x32_t in2) noexcept {
		uint8x32_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] & in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t or_(const uint8x32_t in1,
	                                                     const uint8x32_t in2) noexcept {
		uint8x32_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] | in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t andnot(const uint8x32_t in1,
	                                                        const uint8x32_t in2) noexcept {
		uint8x32_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ~(in1.v64[i] & in2.v64[i]);
		}
		return out;
	}

	///
	/// \param in1
	/// \return
	constexpr static inline uint8x32_t not_(const uint8x32_t in1) noexcept {
		uint8x32_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ~in1.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t add(
	        const uint8x32_t in1,
	        const uint8x32_t in2) noexcept {
		uint8x32_t out;
		for (uint32_t i = 0; i < 32; i++) {
			out.v8[i] = in1.v8[i] + in2.v8[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t sub(const uint8x32_t in1,
	                                                     const uint8x32_t in2) noexcept {
		uint8x32_t out;
		for (uint32_t i = 0; i < 32; i++) {
			out.v8[i] = in1.v8[i] - in2.v8[i];
		}
		return out;
	}

	/// 8 bit mul lo
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t mullo(const uint8x32_t in1,
	                                                       const uint8x32_t in2) noexcept {
		uint8x32_t out;
		for (uint32_t i = 0; i < 32; i++) {
			out.v8[i] = in1.v8[i] * in2.v8[i];
		}
		return out;
	}

	///
	[[nodiscard]] constexpr static inline uint8x32_t mullo(const uint8x32_t in1,
	                                                       const uint8_t in2) noexcept {
		uint8x32_t rs = uint8x32_t::set1(in2);
		return uint8x32_t::mullo(in1, rs);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t slli(const uint8x32_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		uint8x32_t out;
		for (uint32_t i = 0; i < 32; i++) {
			out.v8[i] = in1.v8[i] << in2;
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t srli(const uint8x32_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		uint8x32_t out;
		for (uint32_t i = 0; i < 32; i++) {
			out.v8[i] = in1.v8[i] >> in2;
		}
		return out;
	}

	[[nodiscard]] constexpr static inline uint8x32_t cmp_(const uint8x32_t in1,
	                                                      const uint8x32_t in2) noexcept {
		uint8x32_t ret;
		for (uint32_t i = 0; i < 32; i++) {
			ret.v8[i] = in1.v8[i] == in2.v8[i];
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline uint8x32_t gt_(const uint8x32_t in1,
	                                                     const uint8x32_t in2) noexcept {
		uint8x32_t ret;
		for (uint32_t i = 0; i < 32; i++) {
			ret.v8[i] = in1.v8[i] > in2.v8[i];
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline int cmp(const uint8x32_t in1,
	                                              const uint8x32_t in2) noexcept {
		int ret = 0;
		for (uint32_t i = 0; i < 32; i++) {
			ret ^= (in1.v8[i] == in2.v8[i]) << i;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline int gt(const uint8x32_t in1,
	                                             const uint8x32_t in2) {
		int ret = 0;
		for (uint32_t i = 0; i < 32; i++) {
			ret ^= (in1.v8[i] > in2.v8[i]) << i;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint8x32_t popcnt(const uint8x32_t in) {
		uint8x32_t ret;

		for (uint32_t i = 0; i < 32; i++) {
			ret.v8[i] = cryptanalysislib::popcount::popcount(in.v8[i]);
		}
		return ret;
	}

	[[nodiscard]] static inline uint32_t move(const uint8x32_t in1) {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < 32; i++) {
			ret ^= in1.v8[i] >> 7;
		}

		return ret;
	}
};

struct uint16x16_t {
	constexpr static uint32_t LIMBS = 16;
	using limb_type = uint16_t;

	union {
		uint16_t d[16];

		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
	};

	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		ASSERT(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline uint16x16_t random() noexcept {
		uint16x16_t ret;
		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = fastrandombytes_uint64();
		}

		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	[[nodiscard]] constexpr static inline uint16x16_t setr(
	        const uint16_t a0, const uint16_t a1, const uint16_t a2, const uint16_t a3,
	        const uint16_t a4, const uint16_t a5, const uint16_t a6, const uint16_t a7,
	        const uint16_t a8, const uint16_t a9, const uint16_t a10, const uint16_t a11,
	        const uint16_t a12, const uint16_t a13, const uint16_t a14, const uint16_t a15) {
		uint16x16_t out;
		out.v16[0] = a0;
		out.v16[1] = a1;
		out.v16[2] = a2;
		out.v16[3] = a3;
		out.v16[4] = a4;
		out.v16[5] = a5;
		out.v16[6] = a6;
		out.v16[7] = a7;
		out.v16[8] = a8;
		out.v16[9] = a9;
		out.v16[10] = a10;
		out.v16[11] = a11;
		out.v16[12] = a12;
		out.v16[13] = a13;
		out.v16[14] = a14;
		out.v16[15] = a15;
		return out;
	}

	[[nodiscard]] constexpr static inline uint16x16_t set(
	        const uint16_t a0, const uint16_t a1, const uint16_t a2, const uint16_t a3,
	        const uint16_t a4, const uint16_t a5, const uint16_t a6, const uint16_t a7,
	        const uint16_t a8, const uint16_t a9, const uint16_t a10, const uint16_t a11,
	        const uint16_t a12, const uint16_t a13, const uint16_t a14, const uint16_t a15) noexcept {
		return uint16x16_t::setr(a15, a14, a13, a12, a11, a10, a9, a8, a7, a6, a5, a4, a3, a2, a1, a0);
	}

	/// sets all 32 8bit limbs to `a`
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t set1(const uint16_t a) noexcept {
		uint16x16_t out;
		out = uint16x16_t::set(a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint16x16_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t aligned_load(const void *ptr) noexcept {
		uint16x16_t out;
		const uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ptr64[i];
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t unaligned_load(
	        const void *ptr) noexcept {
		uint16x16_t out;
		const uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ptr64[i];
		}
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const uint16x16_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint16x16_t in) noexcept {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint16x16_t in) noexcept {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t xor_(const uint16x16_t in1,
	                                                       const uint16x16_t in2) noexcept {
		uint16x16_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] ^ in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t and_(const uint16x16_t in1,
	                                                       const uint16x16_t in2) noexcept {
		uint16x16_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] & in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t or_(const uint16x16_t in1,
	                                                      const uint16x16_t in2) noexcept {
		uint16x16_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] | in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t andnot(const uint16x16_t in1,
	                                                         const uint16x16_t in2) noexcept {
		uint16x16_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ~(in1.v64[i] & in2.v64[i]);
		}
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t not_(const uint16x16_t in1) noexcept {
		uint16x16_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ~in1.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t add(
	        const uint16x16_t in1,
	        const uint16x16_t in2) noexcept {
		uint16x16_t out;
		for (uint32_t i = 0; i < 16; i++) {
			out.v16[i] = in1.v16[i] + in2.v16[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t sub(const uint16x16_t in1,
	                                                      const uint16x16_t in2) noexcept {
		uint16x16_t out;
		for (uint32_t i = 0; i < 16; i++) {
			out.v16[i] = in1.v16[i] - in2.v16[i];
		}
		return out;
	}

	/// 8 bit mul lo
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t mullo(const uint16x16_t in1,
	                                                        const uint16x16_t in2) noexcept {
		uint16x16_t out;
		for (uint32_t i = 0; i < 16; i++) {
			out.v16[i] = in1.v16[i] * in2.v16[i];
		}
		return out;
	}

	///
	[[nodiscard]] constexpr static inline uint16x16_t mullo(const uint16x16_t in1,
	                                                        const uint8_t in2) noexcept {
		uint16x16_t rs = uint16x16_t::set1(in2);
		return uint16x16_t::mullo(in1, rs);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t slli(const uint16x16_t in1,
	                                                       const uint8_t in2) noexcept {
		ASSERT(in2 <= 16);
		uint16x16_t out;
		for (uint32_t i = 0; i < 16; i++) {
			out.v16[i] = in1.v16[i] << in2;
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t srli(const uint16x16_t in1,
	                                                       const uint16_t in2) noexcept {
		ASSERT(in2 <= 8);
		uint16x16_t out;
		for (uint32_t i = 0; i < 16; i++) {
			out.v16[i] = in1.v16[i] >> in2;
		}
		return out;
	}

	[[nodiscard]] constexpr static inline uint16x16_t cmp_(
	        const uint16x16_t in1, const uint16x16_t in2) noexcept {
		uint16x16_t ret;
		for (uint32_t i = 0; i < 16; i++) {
			ret.v16[i] = in1.v16[i] == in2.v16[i];
		}

		return ret;
	}

	static inline uint16x16_t gt_(const uint16x16_t in1, const uint16x16_t in2) {
		uint16x16_t ret;
		for (uint32_t i = 0; i < 16; i++) {
			ret.v16[i] = in1.v16[i] == in2.v16[i];
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint32_t cmp(const uint16x16_t in1,
	                                                   const uint16x16_t in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < 16; i++) {
			ret ^= (in1.v16[i] == in2.v16[i]) << i;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint32_t gt(const uint16x16_t in1,
	                                                  const uint16x16_t in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < 16; i++) {
			ret ^= (in1.v16[i] > in2.v16[i]) << i;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint16x16_t popcnt(const uint16x16_t in) noexcept {
		uint16x16_t ret;

		for (uint32_t i = 0; i < 16; i++) {
			ret.v16[i] = cryptanalysislib::popcount::popcount(in.v16[i]);
		}
		return ret;
	}


	/// extracts the sign bit of each limb
	[[nodiscard]] static inline uint16_t move(const uint16x16_t in1) noexcept {
		uint16_t ret = 0;
		for (uint32_t i = 0; i < 16; i++) {
			ret ^= in1.v16[i] >> 15;
		}

		return ret;
	}
};

struct uint32x8_t {
	constexpr static uint32_t LIMBS = 8;
	using limb_type = uint32_t;

	union {
		uint32_t d[8];

		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
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
	[[nodiscard]] static inline uint32x8_t random() noexcept {
		uint32x8_t ret;
		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = fastrandombytes_uint64();
		}

		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	[[nodiscard]] constexpr static inline uint32x8_t setr(
	        const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
	        const uint32_t a4, const uint32_t a5, const uint32_t a6, const uint32_t a7) noexcept {
		uint32x8_t out;
		out.v32[0] = a0;
		out.v32[1] = a1;
		out.v32[2] = a2;
		out.v32[3] = a3;
		out.v32[4] = a4;
		out.v32[5] = a5;
		out.v32[6] = a6;
		out.v32[7] = a7;
		return out;
	}

	[[nodiscard]] constexpr static inline uint32x8_t set(
	        const uint32_t a0, const uint32_t a1, const uint32_t a2, const uint32_t a3,
	        const uint32_t a4, const uint32_t a5, const uint32_t a6, const uint32_t a7) noexcept {
		return uint32x8_t::setr(a7, a6, a5, a4, a3, a2, a1, a0);
	}

	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t set1(const uint32_t a) {
		uint32x8_t out;
		out = uint32x8_t::set(a, a, a, a, a, a, a, a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint32x8_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t aligned_load(const void *ptr) noexcept {
		uint32x8_t out;
		const uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ptr64[i];
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t unaligned_load(const void *ptr) noexcept {
		uint32x8_t out;
		const uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ptr64[i];
		}
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const uint32x8_t in) {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr, const uint32x8_t in) {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint32x8_t in) {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t xor_(const uint32x8_t in1,
	                                                      const uint32x8_t in2) noexcept {
		uint32x8_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] ^ in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t and_(const uint32x8_t in1,
	                                                      const uint32x8_t in2) noexcept {
		uint32x8_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] & in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t or_(const uint32x8_t in1,
	                                                     const uint32x8_t in2) noexcept {
		uint32x8_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] | in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t andnot(const uint32x8_t in1,
	                                                        const uint32x8_t in2) noexcept {
		uint32x8_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ~(in1.v64[i] & in2.v64[i]);
		}
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t not_(const uint32x8_t in1) noexcept {
		uint32x8_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ~in1.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t add(
	        const uint32x8_t in1,
	        const uint32x8_t in2) noexcept {
		uint32x8_t out;
		for (uint32_t i = 0; i < 8; i++) {
			out.v32[i] = in1.v32[i] + in2.v32[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t sub(const uint32x8_t in1,
	                                                     const uint32x8_t in2) noexcept {
		uint32x8_t out;
		for (uint32_t i = 0; i < 8; i++) {
			out.v32[i] = in1.v32[i] - in2.v32[i];
		}
		return out;
	}

	/// 8 bit mul lo
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t mullo(const uint32x8_t in1,
	                                                       const uint32x8_t in2) noexcept {
		uint32x8_t out;
		for (uint32_t i = 0; i < 8; i++) {
			out.v32[i] = in1.v32[i] * in2.v32[i];
		}
		return out;
	}

	///
	[[nodiscard]] constexpr static inline uint32x8_t mullo(const uint32x8_t in1,
	                                                       const uint8_t in2) noexcept {
		uint32x8_t rs = uint32x8_t::set1(in2);
		return uint32x8_t::mullo(in1, rs);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t slli(const uint32x8_t in1,
	                                                      const uint32_t in2) noexcept {
		ASSERT(in2 <= 32);
		uint32x8_t out;
		for (uint32_t i = 0; i < 8; i++) {
			out.v32[i] = in1.v32[i] << in2;
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t srli(const uint32x8_t in1,
	                                                      const uint16_t in2) noexcept {
		ASSERT(in2 <= 8);
		uint32x8_t out;
		for (uint32_t i = 0; i < 8; i++) {
			out.v32[i] = in1.v32[i] >> in2;
		}
		return out;
	}

	[[nodiscard]] constexpr static inline uint32x8_t cmp_(const uint32x8_t in1,
	                                                      const uint32x8_t in2) noexcept {
		uint32x8_t ret;
		for (uint32_t i = 0; i < 8; i++) {
			ret.v32[i] = in1.v32[i] == in2.v32[i];
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint32x8_t gt_(const uint32x8_t in1,
	                                                     const uint32x8_t in2) noexcept {
		uint32x8_t ret;
		for (uint32_t i = 0; i < 8; i++) {
			ret.v32[i] = in1.v32[i] > in2.v32[i];
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint32_t cmp(const uint32x8_t in1,
	                                                   const uint32x8_t in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < 8; i++) {
			ret ^= (in1.v32[i] == in2.v32[i]) << i;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint32_t gt(const uint32x8_t in1,
	                                                  const uint32x8_t in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < 8; i++) {
			ret ^= (in1.v32[i] > in2.v32[i]) << i;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint32x8_t popcnt(const uint32x8_t in) noexcept {
		uint32x8_t ret;

		for (uint32_t i = 0; i < 8; i++) {
			ret.v32[i] = cryptanalysislib::popcount::popcount(in.v32[i]);
		}
		return ret;
	}

	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline uint32x8_t gather(const void *ptr,
	                                                        const uint32x8_t data) noexcept {
		uint32x8_t ret;
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 8; i++) {
			ret.v32[i] = *(uint32_t *) (ptr8 + data.v32[i] * scale);
		}

		return ret;
	}

	///
	/// \param in
	/// \param perm
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t permute(const uint32x8_t in,
	                                                         const uint32x8_t perm) noexcept {
		uint32x8_t ret;
		for (uint32_t i = 0; i < 8; i++) {
			ret.v32[i] = in.v32[perm.v32[i] & 0x7];
		}
		return ret;
	}


	[[nodiscard]] static inline uint8_t move(const uint32x8_t in1) noexcept {
		uint8_t ret = 0;
		for (uint32_t i = 0; i < 8; i++) {
			ret ^= in1.v32[i] >> 31;
		}

		return ret;
	}


	/// input:
	/// 	mask: 0b010101010
	/// output: a permutation mask s.t, applied on in =  [ x0, x1, x2, x3, x4, x5, x6, x7 ],
	/// 			uint32x8_t::permute(in, permutation_mask) will result int
	///  	[x1, x3, x5, x7, 0, 0, 0, 0]
	[[nodiscard]] static inline uint32x8_t pack(uint32_t mask) noexcept {
		uint32x8_t ret = uint32x8_t::set1(0);
		for (uint32_t i = 0; (i < 8) && (mask != 0); i++ ) {
			const uint32_t pos = __builtin_ctz(mask);
			ret[i] = pos;
			mask ^= 1u << pos;
		}
		return ret;
	}


	[[nodiscard]] static inline uint32x8_t cvtepu8(const _uint8x16_t in) noexcept {
		uint32x8_t ret;
		for (uint32_t i = 0; i < 16; i++) {
			ret.d[i] = in.d[i];
		}

		return ret;
	}
};

struct uint64x4_t {
	constexpr static uint32_t LIMBS = 4;
	using limb_type = uint64_t;

	union {
		uint64_t d[4];

		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
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
	[[nodiscard]] static inline uint64x4_t random() noexcept {
		uint64x4_t ret;
		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = fastrandombytes_uint64();
		}

		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	[[nodiscard]] constexpr static inline uint64x4_t setr(
	        const uint64_t a0, const uint64_t a1,
	        const uint64_t a2, const uint64_t a3) noexcept {
		uint64x4_t out;
		out.v64[0] = a0;
		out.v64[1] = a1;
		out.v64[2] = a2;
		out.v64[3] = a3;
		return out;
	}

	[[nodiscard]] constexpr static inline uint64x4_t set(
	        const uint64_t a0, const uint64_t a1,
	        const uint64_t a2, const uint64_t a3) noexcept {
		return uint64x4_t::setr(a3, a2, a1, a0);
	}

	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t set1(const uint64_t a) noexcept {
		uint64x4_t out;
		out = uint64x4_t::set(a, a, a, a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline uint64x4_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t aligned_load(const void *ptr) noexcept {
		uint64x4_t out;
		const uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ptr64[i];
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t unaligned_load(const void *ptr) noexcept {
		uint64x4_t out;
		const uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ptr64[i];
		}
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const uint64x4_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr, const uint64x4_t in) noexcept {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint64x4_t in) noexcept {
		uint64_t *ptr64 = (uint64_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ptr64[i] = in.v64[i];
		}
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t xor_(const uint64x4_t in1,
	                                                      const uint64x4_t in2) noexcept {
		uint64x4_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] ^ in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t and_(const uint64x4_t in1,
	                                                      const uint64x4_t in2) noexcept {
		uint64x4_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] & in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t or_(const uint64x4_t in1,
	                                                     const uint64x4_t in2) noexcept {
		uint64x4_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] | in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t andnot(const uint64x4_t in1,
	                                                        const uint64x4_t in2) noexcept {
		uint64x4_t out{};
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ~(in1.v64[i] & in2.v64[i]);
		}
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t not_(const uint64x4_t in1) noexcept {
		uint64x4_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = ~in1.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t add(
	        const uint64x4_t in1,
	        const uint64x4_t in2) noexcept {
		uint64x4_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] + in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t sub(const uint64x4_t in1,
	                                                     const uint64x4_t in2) noexcept {
		uint64x4_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] - in2.v64[i];
		}
		return out;
	}

	/// 8 bit mul lo
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t mullo(const uint64x4_t in1,
	                                                       const uint64x4_t in2) noexcept {
		uint64x4_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] * in2.v64[i];
		}
		return out;
	}

	///
	[[nodiscard]] constexpr static inline uint64x4_t mullo(const uint64x4_t in1,
	                                                       const uint8_t in2) noexcept {
		uint64x4_t rs = uint64x4_t::set1(in2);
		return uint64x4_t::mullo(in1, rs);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t slli(const uint64x4_t in1,
	                                                      const uint64_t in2) noexcept {
		ASSERT(in2 <= 64);
		uint64x4_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] << in2;
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t srli(const uint64x4_t in1,
	                                                      const uint64_t in2) noexcept {
		ASSERT(in2 <= 64);
		uint64x4_t out;
		for (uint32_t i = 0; i < 4; i++) {
			out.v64[i] = in1.v64[i] >> in2;
		}
		return out;
	}

	[[nodiscard]] constexpr static inline uint64x4_t cmp_(const uint64x4_t in1,
	                                                      const uint64x4_t in2) noexcept {
		uint64x4_t ret;
		for (uint8_t i = 0; i < 4; i++) {
			ret.v64[i] = in1.v64[i] == in2.v64[i];
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline uint64x4_t gt_(const uint64x4_t in1,
	                                                     const uint64x4_t in2) noexcept {
		uint64x4_t ret;
		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = in1.v64[i] > in2.v64[i];
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint32_t cmp(const uint64x4_t in1,
	                                                   const uint64x4_t in2) noexcept {
		uint32_t ret = 0;
		for (uint8_t i = 0; i < 4; i++) {
			ret ^= (in1.v64[i] == in2.v64[i]) << i;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint32_t gt(const uint64x4_t in1,
	                                                  const uint64x4_t in2) noexcept {
		uint32_t ret = 0;
		for (uint32_t i = 0; i < 4; i++) {
			ret ^= (in1.v64[i] > in2.v64[i]) << i;
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint64x4_t popcnt(const uint64x4_t in) noexcept {
		uint64x4_t ret;

		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = cryptanalysislib::popcount::popcount(in.v64[i]);
		}
		return ret;
	}

	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline uint64x4_t gather(const void *ptr,
	                                                        const uint64x4_t data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);

		uint64x4_t ret;
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = *(uint64_t *) (ptr8 + data.v64[i] * scale);
		}

		return ret;
	}

	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline uint64x4_t gather(const void *ptr,
	                                                        const cryptanalysislib::_uint32x4_t data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		uint64x4_t ret;
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = *(uint64_t *) (ptr8 + data.v32[i] * scale);
		}
		return ret;
	}


	///
	/// \param in
	/// \param perm
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t permute(const uint64x4_t in,
	                                                         const uint64x4_t perm) noexcept {
		uint64x4_t ret;
		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = in.v64[perm.v64[i]];
		}
		return ret;
	}

	///
	/// \tparam in2
	/// \param in1
	/// \return
	template<const uint32_t in2>
	[[nodiscard]] constexpr static inline uint64x4_t permute(const uint64x4_t in1) noexcept {
		uint64x4_t ret;

		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = in1.v64[(in2 >> (2 * i)) & 0b11];
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline uint8_t move(const uint64x4_t in1) noexcept {
		uint8_t ret = 0;
		for (uint32_t i = 0; i < 4; i++) {
			ret ^= in1.v64[i] >> 63;
		}

		return ret;
	}
};


#include "simd/float/simd.h"
#endif// no SIMD unit available

///
constexpr inline uint8x32_t operator*(const uint8x32_t &lhs, const uint8x32_t &rhs) {
	return uint8x32_t::mullo(lhs, rhs);
}
constexpr inline uint8x32_t operator*(const uint8x32_t &lhs, const uint8_t &rhs) {
	return uint8x32_t::mullo(lhs, rhs);
}
constexpr inline uint8x32_t operator*(const uint8_t &lhs, const uint8x32_t &rhs) {
	return uint8x32_t::mullo(rhs, lhs);
}
constexpr inline uint8x32_t operator+(const uint8x32_t &lhs, const uint8x32_t &rhs) {
	return uint8x32_t::add(lhs, rhs);
}
constexpr inline uint8x32_t operator-(const uint8x32_t &lhs, const uint8x32_t &rhs) {
	return uint8x32_t::sub(lhs, rhs);
}
constexpr inline uint8x32_t operator&(const uint8x32_t &lhs, const uint8x32_t &rhs) {
	return uint8x32_t::and_(lhs, rhs);
}
constexpr inline uint8x32_t operator^(const uint8x32_t &lhs, const uint8x32_t &rhs) {
	return uint8x32_t::xor_(lhs, rhs);
}
constexpr inline uint8x32_t operator|(const uint8x32_t &lhs, const uint8x32_t &rhs) {
	return uint8x32_t::or_(lhs, rhs);
}
constexpr inline uint8x32_t operator~(const uint8x32_t &lhs) {
	return uint8x32_t::not_(lhs);
}
constexpr inline uint8x32_t operator>>(const uint8x32_t &lhs, const uint32_t rhs) {
	return uint8x32_t::srli(lhs, rhs);
}
constexpr inline uint8x32_t operator<<(const uint8x32_t &lhs, const uint32_t rhs) {
	return uint8x32_t::slli(lhs, rhs);
}
constexpr inline uint8x32_t operator^=(uint8x32_t &lhs, const uint8x32_t &rhs) {
	lhs = uint8x32_t::xor_(lhs, rhs);
	return lhs;
}
constexpr inline uint8x32_t operator&=(uint8x32_t &lhs, const uint8x32_t &rhs) {
	lhs = uint8x32_t::and_(lhs, rhs);
	return lhs;
}
constexpr inline uint8x32_t operator|=(uint8x32_t &lhs, const uint8x32_t &rhs) {
	lhs = uint8x32_t::or_(lhs, rhs);
	return lhs;
}


///
constexpr inline uint16x16_t operator*(const uint16x16_t &lhs, const uint16x16_t &rhs) {
	return uint16x16_t::mullo(lhs, rhs);
}
constexpr inline uint16x16_t operator*(const uint16x16_t &lhs, const uint8_t &rhs) {
	return uint16x16_t::mullo(lhs, rhs);
}
constexpr inline uint16x16_t operator*(const uint8_t &lhs, const uint16x16_t &rhs) {
	return uint16x16_t::mullo(rhs, lhs);
}
constexpr inline uint16x16_t operator+(const uint16x16_t &lhs, const uint16x16_t &rhs) {
	return uint16x16_t::add(lhs, rhs);
}
constexpr inline uint16x16_t operator-(const uint16x16_t &lhs, const uint16x16_t &rhs) {
	return uint16x16_t::sub(lhs, rhs);
}
constexpr inline uint16x16_t operator&(const uint16x16_t &lhs, const uint16x16_t &rhs) {
	return uint16x16_t::and_(lhs, rhs);
}
constexpr inline uint16x16_t operator^(const uint16x16_t &lhs, const uint16x16_t &rhs) {
	return uint16x16_t::xor_(lhs, rhs);
}
constexpr inline uint16x16_t operator|(const uint16x16_t &lhs, const uint16x16_t &rhs) {
	return uint16x16_t::or_(lhs, rhs);
}
constexpr inline uint16x16_t operator~(const uint16x16_t &lhs) {
	return uint16x16_t::not_(lhs);
}
constexpr inline uint16x16_t operator>>(const uint16x16_t &lhs, const uint32_t rhs) {
	return uint16x16_t::srli(lhs, rhs);
}
constexpr inline uint16x16_t operator<<(const uint16x16_t &lhs, const uint32_t rhs) {
	return uint16x16_t::slli(lhs, rhs);
}
constexpr inline uint16x16_t operator^=(uint16x16_t &lhs, const uint16x16_t &rhs) {
	lhs = uint16x16_t::xor_(lhs, rhs);
	return lhs;
}
constexpr inline uint16x16_t operator&=(uint16x16_t &lhs, const uint16x16_t &rhs) {
	lhs = uint16x16_t::and_(lhs, rhs);
	return lhs;
}
constexpr inline uint16x16_t operator|=(uint16x16_t &lhs, const uint16x16_t &rhs) {
	lhs = uint16x16_t::or_(lhs, rhs);
	return lhs;
}


///
constexpr inline uint32x8_t operator*(const uint32x8_t &lhs, const uint32x8_t &rhs) {
	return uint32x8_t::mullo(lhs, rhs);
}
constexpr inline uint32x8_t operator*(const uint32x8_t &lhs, const uint8_t &rhs) {
	return uint32x8_t::mullo(lhs, rhs);
}
constexpr inline uint32x8_t operator*(const uint8_t &lhs, const uint32x8_t &rhs) {
	return uint32x8_t::mullo(rhs, lhs);
}
constexpr inline uint32x8_t operator+(const uint32x8_t &lhs, const uint32x8_t &rhs) {
	return uint32x8_t::add(lhs, rhs);
}
constexpr inline uint32x8_t operator-(const uint32x8_t &lhs, const uint32x8_t &rhs) {
	return uint32x8_t::sub(lhs, rhs);
}
constexpr inline uint32x8_t operator&(const uint32x8_t &lhs, const uint32x8_t &rhs) {
	return uint32x8_t::and_(lhs, rhs);
}
constexpr inline uint32x8_t operator^(const uint32x8_t &lhs, const uint32x8_t &rhs) {
	return uint32x8_t::xor_(lhs, rhs);
}
constexpr inline uint32x8_t operator|(const uint32x8_t &lhs, const uint32x8_t &rhs) {
	return uint32x8_t::or_(lhs, rhs);
}
constexpr inline uint32x8_t operator~(const uint32x8_t &lhs) {
	return uint32x8_t::not_(lhs);
}
constexpr inline uint32x8_t operator>>(const uint32x8_t &lhs, const uint32_t rhs) {
	return uint32x8_t::srli(lhs, rhs);
}
constexpr inline uint32x8_t operator<<(const uint32x8_t &lhs, const uint32_t rhs) {
	return uint32x8_t::slli(lhs, rhs);
}
constexpr inline uint32x8_t operator^=(uint32x8_t &lhs, const uint32x8_t &rhs) {
	lhs = uint32x8_t::xor_(lhs, rhs);
	return lhs;
}
constexpr inline uint32x8_t operator&=(uint32x8_t &lhs, const uint32x8_t &rhs) {
	lhs = uint32x8_t::and_(lhs, rhs);
	return lhs;
}
constexpr inline uint32x8_t operator|=(uint32x8_t &lhs, const uint32x8_t &rhs) {
	lhs = uint32x8_t::or_(lhs, rhs);
	return lhs;
}


///
constexpr inline uint64x4_t operator*(const uint64x4_t &lhs, const uint64x4_t &rhs) {
	return uint64x4_t::mullo(lhs, rhs);
}
constexpr inline uint64x4_t operator*(const uint64x4_t &lhs, const uint64_t &rhs) {
	return uint64x4_t::mullo(lhs, rhs);
}
constexpr inline uint64x4_t operator*(const uint8_t &lhs, const uint64x4_t &rhs) {
	return uint64x4_t::mullo(rhs, lhs);
}
constexpr inline uint64x4_t operator+(const uint64x4_t &lhs, const uint64x4_t &rhs) {
	return uint64x4_t::add(lhs, rhs);
}
constexpr inline uint64x4_t operator-(const uint64x4_t &lhs, const uint64x4_t &rhs) {
	return uint64x4_t::sub(lhs, rhs);
}
constexpr inline uint64x4_t operator&(const uint64x4_t &lhs, const uint64x4_t &rhs) {
	return uint64x4_t::and_(lhs, rhs);
}
constexpr inline uint64x4_t operator^(const uint64x4_t &lhs, const uint64x4_t &rhs) {
	return uint64x4_t::xor_(lhs, rhs);
}
constexpr inline uint64x4_t operator|(const uint64x4_t &lhs, const uint64x4_t &rhs) {
	return uint64x4_t::or_(lhs, rhs);
}
constexpr inline uint64x4_t operator~(const uint64x4_t &lhs) {
	return uint64x4_t::not_(lhs);
}
constexpr inline uint64x4_t operator>>(const uint64x4_t &lhs, const uint32_t rhs) {
	return uint64x4_t::srli(lhs, rhs);
}
constexpr inline uint64x4_t operator<<(const uint64x4_t &lhs, const uint32_t rhs) {
	return uint64x4_t::slli(lhs, rhs);
}
constexpr inline uint64x4_t operator^=(uint64x4_t &lhs, const uint64x4_t &rhs) {
	lhs = uint64x4_t::xor_(lhs, rhs);
	return lhs;
}
constexpr inline uint64x4_t operator&=(uint64x4_t &lhs, const uint64x4_t &rhs) {
	lhs = uint64x4_t::and_(lhs, rhs);
	return lhs;
}
constexpr inline uint64x4_t operator|=(uint64x4_t &lhs, const uint64x4_t &rhs) {
	lhs = uint64x4_t::or_(lhs, rhs);
	return lhs;
}


/* 					 comparison									*/
constexpr inline int operator==(const uint8x32_t &a, const uint8x32_t &b) {
	return uint8x32_t::cmp(a, b);
}
constexpr inline int operator!=(const uint8x32_t &a, const uint8x32_t &b) {
	return 0xffffffff ^ uint8x32_t::cmp(a, b);
}
constexpr inline int operator<(const uint8x32_t &a, const uint8x32_t &b) {
	return uint8x32_t::gt(b, a);
}
constexpr inline int operator>(const uint8x32_t &a, const uint8x32_t &b) {
	return uint8x32_t::gt(a, b);
}


///
constexpr inline int operator==(const uint16x16_t &a, const uint16x16_t &b) {
	return (int) uint16x16_t::cmp(a, b);
}
constexpr inline int operator!=(const uint16x16_t &a, const uint16x16_t &b) {
	return 0xffff ^ uint16x16_t::cmp(a, b);
}
constexpr inline int operator<(const uint16x16_t &a, const uint16x16_t &b) {
	return (int) uint16x16_t::gt(b, a);
}
constexpr inline int operator>(const uint16x16_t &a, const uint16x16_t &b) {
	return (int) uint16x16_t::gt(a, b);
}


///
constexpr inline int operator==(const uint32x8_t &a, const uint32x8_t &b) {
	return (int) uint32x8_t::cmp(a, b);
}
constexpr inline int operator!=(const uint32x8_t &a, const uint32x8_t &b) {
	return 0xff ^ uint32x8_t::cmp(a, b);
}
constexpr inline int operator<(const uint32x8_t &a, const uint32x8_t &b) {
	return (int) uint32x8_t::gt(b, a);
}
constexpr inline int operator>(const uint32x8_t &a, const uint32x8_t &b) {
	return (int) uint32x8_t::gt(a, b);
}

constexpr inline int operator==(const uint64x4_t &a, const uint64x4_t &b) {
	return (int) uint64x4_t::cmp(a, b);
}
constexpr inline int operator!=(const uint64x4_t &a, const uint64x4_t &b) {
	return 0xf ^ uint64x4_t::cmp(a, b);
}
constexpr inline int operator<(const uint64x4_t &a, const uint64x4_t &b) {
	return (int) uint64x4_t::gt(b, a);
}
constexpr inline int operator>(const uint64x4_t &a, const uint64x4_t &b) {
	return (int) uint64x4_t::gt(a, b);
}


////////////////////////////////////////////////////////////


/// functions which are shared among all implementations.
constexpr inline void uint8x32_t::print(bool binary, bool hex) const {
	/// make sure that only one is defined
	ASSERT(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 32; i++) {
			print_binary(this->v8[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 32; i++) {
			printf("%hhx ", this->v8[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 32; i++) {
		printf("%u ", this->v8[i]);
	}
	printf("\n");
}

constexpr inline void uint16x16_t::print(bool binary, bool hex) const {
	/// make sure that only one is defined
	ASSERT(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 16; i++) {
			print_binary(this->v16[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 16; i++) {
			printf("%hx ", this->v16[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 16; i++) {
		printf("%u ", this->v16[i]);
	}
	printf("\n");
}

constexpr inline void uint32x8_t::print(bool binary, bool hex) const {
	/// make sure that only one is defined
	ASSERT(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 8; i++) {
			print_binary(this->v32[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 8; i++) {
			printf("%x ", this->v32[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 8; i++) {
		printf("%u ", this->v32[i]);
	}
	printf("\n");
}

constexpr inline void uint64x4_t::print(bool binary, bool hex) const {
	/// make sure that only one is defined
	ASSERT(binary + hex < 2);

	if (binary) {
		for (uint32_t i = 0; i < 4; i++) {
			print_binary(this->v64[i]);
		}

		return;
	}

	if (hex) {
		for (uint32_t i = 0; i < 4; i++) {
			printf("%" PRIu64 " ", this->v64[i]);
		}

		return;
	}

	for (uint32_t i = 0; i < 4; i++) {
		printf("%" PRIu64 " ", this->v64[i]);
	}
	printf("\n");
}

////////////////////////////////////////////////////////////////////////

//namespace cryptanalysislib {
//	constexpr inline _uint8x16_t &_uint8x16_t::operator=(const _uint16x8_t &b) noexcept {
//		_uint8x16_t ret;
//		for (uint32_t i = 0; i < 2; ++i) {
//			ret.v64[i] = b.v64[i];
//		}
//
//		return ret;
//	}
//	constexpr inline _uint8x16_t &_uint8x16_t::operator=(const _uint32x4_t &b) noexcept {
//		_uint8x16_t ret;
//		for (uint32_t i = 0; i < 2; ++i) {
//			ret.v64[i] = b.v64[i];
//		}
//
//		return ret;
//	}
//	constexpr inline _uint8x16_t &_uint8x16_t::operator=(const _uint64x2_t &b) noexcept {
//		_uint8x16_t ret;
//		for (uint32_t i = 0; i < 2; ++i) {
//			ret.v64[i] = b.v64[i];
//		}
//
//		return ret;
//	}
//
//	constexpr inline _uint16x8_t &_uint16x8_t::operator=(const _uint8x16_t &b) noexcept {
//		_uint16x8_t ret;
//		for (uint32_t i = 0; i < 2; ++i) {
//			ret.v64[i] = b.v64[i];
//		}
//
//		return ret;
//	}
//	constexpr inline _uint16x8_t &_uint16x8_t::operator=(const _uint32x4_t &b) noexcept {
//		_uint16x8_t ret;
//		for (uint32_t i = 0; i < 2; ++i) {
//			ret.v64[i] = b.v64[i];
//		}
//
//		return ret;
//	}
//	constexpr inline _uint16x8_t &_uint16x8_t::operator=(const _uint64x2_t &b) noexcept {
//		_uint16x8_t ret;
//		for (uint32_t i = 0; i < 2; ++i) {
//			ret.v64[i] = b.v64[i];
//		}
//
//		return ret;
//	}
//
//	constexpr inline _uint32x4_t &_uint32x4_t::operator=(const _uint16x8_t &b) noexcept {
//		_uint32x4_t ret;
//		for (uint32_t i = 0; i < 2; ++i) {
//			ret.v64[i] = b.v64[i];
//		}
//
//		return ret;
//	}
//	constexpr inline _uint32x4_t &_uint32x4_t::operator=(const _uint8x16_t &b) noexcept {
//		_uint32x4_t ret;
//		for (uint32_t i = 0; i < 2; ++i) {
//			ret.v64[i] = b.v64[i];
//		}
//
//		return ret;
//	}
//	constexpr inline _uint32x4_t &_uint32x4_t::operator=(const _uint64x2_t &b) noexcept {
//		_uint32x4_t ret;
//		for (uint32_t i = 0; i < 2; ++i) {
//			ret.v64[i] = b.v64[i];
//		}
//
//		return ret;
//	}
//
//	constexpr inline _uint64x2_t &_uint64x2_t::operator=(const _uint16x8_t &b) noexcept {
//		_uint64x2_t ret;
//		for (uint32_t i = 0; i < 2; ++i) {
//			ret.v64[i] = b.v64[i];
//		}
//
//		return ret;
//	}
//	constexpr inline _uint64x2_t &_uint64x2_t::operator=(const _uint32x4_t &b) noexcept {
//		_uint64x2_t ret;
//		for (uint32_t i = 0; i < 2; ++i) {
//			ret.v64[i] = b.v64[i];
//		}
//
//		return ret;
//	}
//	constexpr inline _uint64x2_t &_uint64x2_t::operator=(const _uint8x16_t &b) noexcept {
//		_uint64x2_t ret;
//		for (uint32_t i = 0; i < 2; ++i) {
//			ret.v64[i] = b.v64[i];
//		}
//
//		return ret;
//	}
//}

namespace cryptanalysislib {
	constexpr _uint8x16_t::_uint8x16_t(const _uint16x8_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
	constexpr _uint8x16_t::_uint8x16_t(const _uint32x4_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
	constexpr _uint8x16_t::_uint8x16_t(const _uint64x2_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}

	constexpr _uint16x8_t::_uint16x8_t(const _uint8x16_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
	constexpr _uint16x8_t::_uint16x8_t(const _uint32x4_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
	constexpr _uint16x8_t::_uint16x8_t(const _uint64x2_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}

	constexpr _uint32x4_t::_uint32x4_t(const _uint16x8_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
	constexpr _uint32x4_t::_uint32x4_t(const _uint8x16_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
	constexpr _uint32x4_t::_uint32x4_t(const _uint64x2_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}

	constexpr _uint64x2_t::_uint64x2_t(const _uint16x8_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
	constexpr _uint64x2_t::_uint64x2_t(const _uint32x4_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
	constexpr _uint64x2_t::_uint64x2_t(const _uint8x16_t &b) noexcept {
		for (uint32_t i = 0; i < 2; ++i) {
			v64[i] = b.v64[i];
		}
	}
}// namespace cryptanalysislib


void transpose8(unsigned char A[8], int m, int n,
                unsigned char B[8]) {
	unsigned x, y, t;

	// Load the array and pack it into x and y.

	x = (A[0] << 24) | (A[m] << 16) | (A[2 * m] << 8) | A[3 * m];
	y = (A[4 * m] << 24) | (A[5 * m] << 16) | (A[6 * m] << 8) | A[7 * m];

	t = (x ^ (x >> 7)) & 0x00AA00AA;
	x = x ^ t ^ (t << 7);
	t = (y ^ (y >> 7)) & 0x00AA00AA;
	y = y ^ t ^ (t << 7);

	t = (x ^ (x >> 14)) & 0x0000CCCC;
	x = x ^ t ^ (t << 14);
	t = (y ^ (y >> 14)) & 0x0000CCCC;
	y = y ^ t ^ (t << 14);

	t = (x & 0xF0F0F0F0) | ((y >> 4) & 0x0F0F0F0F);
	y = ((x << 4) & 0xF0F0F0F0) | (y & 0x0F0F0F0F);
	x = t;

	B[0] = x >> 24;
	B[n] = x >> 16;
	B[2 * n] = x >> 8;
	B[3 * n] = x;
	B[4 * n] = y >> 24;
	B[5 * n] = y >> 16;
	B[6 * n] = y >> 8;
	B[7 * n] = y;
}

/// input: in, a 64x64 matrix over GF(2)
/// output: out, transpose of in
void transpose_64x64(uint64_t *out, uint64_t *in) {
	const static uint64_t masks[6][2] = {
	        {0x5555555555555555, 0xAAAAAAAAAAAAAAAA},
	        {0x3333333333333333, 0xCCCCCCCCCCCCCCCC},
	        {0x0F0F0F0F0F0F0F0F, 0xF0F0F0F0F0F0F0F0},
	        {0x00FF00FF00FF00FF, 0xFF00FF00FF00FF00},
	        {0x0000FFFF0000FFFF, 0xFFFF0000FFFF0000},
	        {0x00000000FFFFFFFF, 0xFFFFFFFF00000000}};

	for (uint64_t i = 0; i < 64; i++) {
		out[i] = in[i];
	}

	for (int32_t d = 5; d >= 0; d--) {
		const uint32_t s = 1 << d;

		for (uint32_t i = 0; i < 64; i += s * 2) {
			for (uint32_t j = i; j < i + s; j++) {
				const uint64_t x = (out[j] & masks[d][0]) | ((out[j + s] & masks[d][0]) << s);
				const uint64_t y = ((out[j] & masks[d][1]) >> s) | (out[j + s] & masks[d][1]);
				out[j + 0] = x;
				out[j + s] = y;
			}
		}
	}
}


// inplace
inline void transpose64(uint64_t a[64]) noexcept {
	for (uint64_t j = 32, m = 0x00000000FFFFFFFF; j; j >>= 1, m ^= m << j) {
		for (uint64_t k = 0; k < 64; k = ((k | j) + 1) & ~j) {
			uint64_t t = (a[k] ^ (a[k | j] >> j)) & m;
			a[k] ^= t;
			a[k | j] ^= (t << j);
		}
	}
}

// TODO
//struct b256x64_T

#include "simd/bits/bits.h"
#include "simd/generic.h"
#endif//CRYPTANALYSISLIB_SIMD_H
