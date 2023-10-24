#ifndef CRYPTANALYSISLIB_SIMD_NEON_H
#define CRYPTANALYSISLIB_SIMD_NEON_H

#include <_types/_uint32_t.h>
#include <_types/_uint64_t.h>
#include <arm_neon.h>
#include <cstdint>

#include "helper.h"
#include "random.h"

// TODO clang++ not working
#ifndef __clang__
#include <arm_bf16.h>
#include <arm_fp16.h>
#include <stdint.h>
typedef __Uint8x8_t  __uint8x8_t;
typedef __Uint8x16_t __uint8x16_t;
typedef __Uint16x4_t __uint16x4_t;
typedef __Uint16x8_t __uint16x8_t;
typedef __Uint32x2_t __uint32x2_t;
typedef __Uint32x4_t __uint32x4_t;
typedef __Uint64x1_t __uint64x1_t;
typedef __Uint64x2_t __uint64x2_t;
#else
typedef __attribute__((neon_vector_type(8))) uint8_t  __uint8x8_t;
typedef __attribute__((neon_vector_type(16))) uint8_t __uint8x16_t;
typedef __attribute__((neon_vector_type(4))) uint16_t __uint16x4_t;
typedef __attribute__((neon_vector_type(8))) uint16_t __uint16x8_t;
typedef __attribute__((neon_vector_type(2))) uint32_t __uint32x2_t;
typedef __attribute__((neon_vector_type(4))) uint32_t __uint32x4_t;
typedef __attribute__((neon_vector_type(1))) uint64_t __uint64x1_t;
typedef __attribute__((neon_vector_type(2))) uint64_t __uint64x2_t;
#endif

/// taken from: https://github.com/DLTcollab/sse2neon/blob/de2817727c72fc2f4ce9f54e2db6e40ce0548414/sse2neon.h#L4540
/// helper function, which collects the sign bits of each 8 bit limb
inline uint32_t _mm_movemask_epi8(const uint8x16_t input) {
    // Use increasingly wide shifts+adds to collect the sign bits together.
    // Since the widening shifts would be rather confusing to follow in little
    // endian, everything will be illustrated in big endian order instead. This
    // has a different result - the bits would actually be reversed on a big
    // endian machine.

    // Starting input (only half the elements are shown):
    // 89 ff 1d c0 00 10 99 33
    // uint8x16_t input = vreinterpretq_u8_m128i(a);

    // Shift out everything but the sign bits with an unsigned shift right.
    //
    // Bytes of the vector::
    // 89 ff 1d c0 00 10 99 33
    // \  \  \  \  \  \  \  \    high_bits = (uint16x4_t)(input >> 7)
    //  |  |  |  |  |  |  |  |
    // 01 01 00 01 00 00 01 00
    //
    // Bits of first important lane(s):
    // 10001001 (89)
    // \______
    //        |
    // 00000001 (01)
    uint16x8_t high_bits = vreinterpretq_u16_u8(vshrq_n_u8(input, 7));

    // Merge the even lanes together with a 16-bit unsigned shift right + add.
    // 'xx' represents garbage data which will be ignored in the final result.
    // In the important bytes, the add functions like a binary OR.
    //
    // 01 01 00 01 00 00 01 00
    //  \_ |  \_ |  \_ |  \_ |   paired16 = (uint32x4_t)(input + (input >> 7))
    //    \|    \|    \|    \|
    // xx 03 xx 01 xx 00 xx 02
    //
    // 00000001 00000001 (01 01)
    //        \_______ |
    //                \|
    // xxxxxxxx xxxxxx11 (xx 03)
    uint32x4_t paired16 =
        vreinterpretq_u32_u16(vsraq_n_u16(high_bits, high_bits, 7));

    // Repeat with a wider 32-bit shift + add.
    // xx 03 xx 01 xx 00 xx 02
    //     \____ |     \____ |  paired32 = (uint64x1_t)(paired16 + (paired16 >>
    //     14))
    //          \|          \|
    // xx xx xx 0d xx xx xx 02
    //
    // 00000011 00000001 (03 01)
    //        \\_____ ||
    //         '----.\||
    // xxxxxxxx xxxx1101 (xx 0d)
    uint64x2_t paired32 =
        vreinterpretq_u64_u32(vsraq_n_u32(paired16, paired16, 14));

    // Last, an even wider 64-bit shift + add to get our result in the low 8 bit
    // lanes. xx xx xx 0d xx xx xx 02
    //            \_________ |   paired64 = (uint8x8_t)(paired32 + (paired32 >>
    //            28))
    //                      \|
    // xx xx xx xx xx xx xx d2
    //
    // 00001101 00000010 (0d 02)
    //     \   \___ |  |
    //      '---.  \|  |
    // xxxxxxxx 11010010 (xx d2)
    uint8x16_t paired64 =
        vreinterpretq_u8_u64(vsraq_n_u64(paired32, paired32, 28));

    // Extract the low 8 bits from each 64-bit lane with 2 8-bit extracts.
    // xx xx xx xx xx xx xx d2
    //                      ||  return paired64[0]
    //                      d2
    // Note: Little endian would return the correct value 4b (01001011) instead.
    return vgetq_lane_u8(paired64, 0) | ((int) vgetq_lane_u8(paired64, 8) << 8);
}

inline uint32_t _mm_movemask_epi16(const uint16x8_t input) {
	constexpr int16_t shift[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint16x8_t tmp = vshrq_n_u16(input, 15);
    return vaddvq_u16(vshlq_u16(tmp, vld1q_s16(shift)));
}
	
// Set each bit of mask dst based on the most significant bit of the
// corresponding packed single-precision (32-bit) floating-point element in a.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movemask_ps
inline uint32_t _mm_movemask_epi32(const uint32x4_t input) {
	constexpr int32_t shift[4] = {0, 1, 2, 3};
    uint32x4_t tmp = vshrq_n_u32(input, 31);
    return vaddvq_u32(vshlq_u32(tmp, vld1q_s32(shift)));
}

// Set each bit of mask dst based on the most significant bit of the
// corresponding packed double-precision (64-bit) floating-point element in a.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movemask_pd
inline uint32_t _mm_movemask_epi64(const uint64x2_t input){
    uint64x2_t high_bits = vshrq_n_u64(input, 63);
    return (uint32_t) (vgetq_lane_u64(high_bits, 0) | (vgetq_lane_u64(high_bits, 1) << 1));
}

struct uint8x32_t {
	union {
		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
		__uint8x16_t v128[2];
	};


	constexpr uint8x32_t() noexcept {}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	///
	/// \return
	static inline uint8x32_t random() noexcept {
		uint8x32_t ret;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 4; ++i) {
			ret.v64[i] = fastrandombytes_uint64();
		}
		return ret;
	}
	
	[[nodiscard]] constexpr static inline uint8x32_t set(char __q31, char __q30, char __q29, char __q28,
	                                                     char __q27, char __q26, char __q25, char __q24,
	                                                     char __q23, char __q22, char __q21, char __q20,
	                                                     char __q19, char __q18, char __q17, char __q16,
	                                                     char __q15, char __q14, char __q13, char __q12,
	                                                     char __q11, char __q10, char __q09, char __q08,
	                                                     char __q07, char __q06, char __q05, char __q04,
	                                                     char __q03, char __q02, char __q01, char __q00) noexcept {
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
		out.v8[ 9] = __q09;
		out.v8[ 8] = __q08;
		out.v8[ 7] = __q07;
		out.v8[ 6] = __q06;
		out.v8[ 5] = __q05;
		out.v8[ 4] = __q04;
		out.v8[ 3] = __q03;
		out.v8[ 2] = __q02;
		out.v8[ 1] = __q01;
		out.v8[ 0] = __q00;
		return out;
	}

	[[nodiscard]] constexpr static inline uint8x32_t setr(char __q31, char __q30, char __q29, char __q28,
	                                                      char __q27, char __q26, char __q25, char __q24,
	                                                      char __q23, char __q22, char __q21, char __q20,
	                                                      char __q19, char __q18, char __q17, char __q16,
	                                                      char __q15, char __q14, char __q13, char __q12,
	                                                      char __q11, char __q10, char __q09, char __q08,
	                                                      char __q07, char __q06, char __q05, char __q04,
	                                                      char __q03, char __q02, char __q01, char __q00) noexcept {
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
		auto *ptr128 = (poly128_t *) ptr;
		uint8x32_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (__uint8x16_t) vldrq_p128(ptr128);
#else
			out.v128[i] = (__uint8x16_t) __builtin_neon_vldrq_p128(ptr128);
#endif
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	constexpr static inline uint8x32_t unaligned_load(const void *ptr) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		uint8x32_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (__uint8x16_t) vldrq_p128(ptr128);
#else
			out.v128[i] = (__uint8x16_t) __builtin_neon_vldrq_p128(ptr128);
#endif
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
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#endif
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint8x32_t in) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#endif
		}
	}

	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM1ykrgAyeAyYAHI%2BAEaYxCCSABykAA6oCoRODB7evv6BaRmOAqHhUSyx8Um2mPbFDEIETMQEOT5%2BATV1WY3NBKWRMXEJyQpNLW15nWN9A%2BWVIwCUtqhexMjsHOYAzGHI3lgA1CbbbmP4qAB0CCfYJhoAgjt7B5jHp%2Be0eNHXt/dPZl2DH2XiOJzcjjYv22d0ez2Br3ebmaLAA%2BuEBNDYQCgSCwadkOcwgQ/nDHgQAJ4pTBYKiHVGopgEAjEb5eAiYBlQDEMVEAN0wDhIqMp1IgiUWksOXmJiRFh3pqJlgkSqjlJO2VnJVJpmDpDKZLLZHK5EB5/MFRGIIp1EC4ADZJYtpbL5QzlQRVQ6RSctQ9Rbr9YzmazouzOajuZgBBahdaAxBJE6XYJvQRFR6HapJD7Nf8A7TFYbQ%2BHTeaBXGbWKJVLM/a3UriVn1b78zrCwaQ8aI1GYxWrVXMBAzMmPdszA2x2ZVBONX6C3qi12wybI2bo7z%2B8KE0na8Tx5P99Oc3O29SO8GjSue%2Bu%2B5bt7auKPifaTxmX5JVFxc/P24vO1epZruW97xraI57oIr6HlBn6zq2ZJPI8YzEF4Dgpp6qgHumJgAOx%2BvcACcXgMFkxz4f8RFER66oKnyiTHAArBY44mIxAAiCFIYR1FNvW6Z8g6bEWEJHFcVRGjEUe8p8qxzGHIkbGcXmZI8ZJHrQQJr7CYckhKeJklEe6spfvxhyCWYinMWY%2BkqdxeHKQRkloAwYyYKoKTEBhqrYRAzoMKg7kbCkOEUbhjn/AqCqRVFwnCQF%2BAKKIxDoPpHGHDMjjIIcYRfOE3lYbOhyqMKEAuWMBXYTlDCBJRhn1WpjUSU1DXNeV6Y0YV8phCO5FOY1nVVasp6IVFY3HICnlMMALBMNKDDEJ4tB1Y1/BeRAU7de87GHBovo5Uihw2ZqE1WJYeDOnh/XNQA9Ddw2XBZVkWHgSnbYqYZ0I4vJMM0yAINB0SiKoACOgn2iDeBKl4MMQGEXCPVwlnCa9HGkNVZiI8jzGo%2BxiwGS1hEPU9KNvScO3w1jz240x2AY1TpNiXZzUOStRHEJgBBrAwhzDQTrNwuFBkPMShyzWEfl9TFCqDUVTCkNEpCmMzjxjWthwbdJ6Z4O9e0nTr4KHKx%2BvWNYl0Uar40Kkwj3U2T2w7cQhjoH5XFWwqPz0YzO3k0dABUTsMC7%2BN2WNAsPGN2W%2B7LIogCAJXWhA8uHNEId%2BmNHNc8QPPII9ckibZBHhRwyy0JwjG8H4HBaKQqCcG4puWBlqzrG8Ow8KQBCaCXywANYgIxGj6JwkiV93tecLwCggEPXfVyXpBwLASBoCwKR0HE5CUKv6/0PEKL0QAtPshjAA6GhD1gfJ4BsABqeCYAA7gA8tSVcdzQtAcsQ08QNE4/RDCM0CknAO6AOYMQCkz9ojaEtKA3gq82CCGfgwWgID56kCwGGYAyJaC0GntwXgWBZpGHEBg/AHMHB4AFAQmu7lBThngeQQQtRx5fGiE7SBHgsDjyNCweBywqAGGAAoe%2BT9X6MCYfwQQIgxDsCkDIQQigVDqAwboQIBgjAoEbjYdh09IDLFQCFLIBDD4sCoCkLw5MeSHEPs/bYvBUACmIKyLA%2Bi/JdEtFkFwQdJh%2BECCEMIgwKjDAKOkTIAg/F6EKBEhgcwhjxECHYLxAhegTE8O0PQySqGpPGP0IJ8xQm2DyVEpJeT4khMScsBQLcNgSFLuXMeGC64cEOAfRIh85onyMIcB0lwND9I1rgQgJAJrbCfLwOeWhJSkH7oPYeHBR6kCrjXFpU8Z6d27jMsuHAzBNNWZPTZ88ZnOIyM4SQQA%3D
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t xor_(const uint8x32_t in1,
	                                                      const uint8x32_t in2) noexcept {
		uint8x32_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] ^ in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t and_(const uint8x32_t in1,
	                                                      const uint8x32_t in2) noexcept {
		uint8x32_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] & in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t or_(const uint8x32_t in1,
	                                                     const uint8x32_t in2) {
		uint8x32_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] | in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t andnot(const uint8x32_t in1,
	                                                        const uint8x32_t in2) noexcept {
		uint8x32_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 4; ++i) {
			out.v64[i] = ~(in1.v64[i] & in2.v64[i]);
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t not_(const uint8x32_t in1) noexcept {
		uint8x32_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 4; ++i) {
			out.v64[i] = ~in1.v64[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t add(const uint8x32_t in1,
	                                                     const uint8x32_t in2) noexcept {
		uint8x32_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			/// TODO not correct, carry and sruff
			out.v128[i] = in1.v128[i] + in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t sub(const uint8x32_t in1,
	                                                     const uint8x32_t in2) noexcept {
		uint8x32_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] - in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t mullo(const uint8x32_t in1,
	                                                       const uint8x32_t in2) noexcept {
		uint8x32_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] * in2.v128[i];
		}
		return out;
	}

	static inline uint8x32_t mullo(const uint8x32_t in1,
	                               const uint8_t in2) {
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

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			out.v128[i] = vshlq_n_u8(in1.v128[i], in2);
#else
			const uint8x32_t tmp = uint8x32_t::set1(in2);
			out.v128[i] = __builtin_neon_vshlq_v(in1.v128[i], (__uint8x16_t) tmp.v128[0], 48u);
#endif
		}

		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t slri(const uint8x32_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		uint8x32_t out;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			//out.v128[i] = vshrq_n_u8(in1.v128[i], in2);
  			out.v128[i] = (uint8x16_t) __builtin_neon_vshrq_n_v((__uint8x16_t)in1.v128[i], in2, 48);
#else
			out.v128[i] = __builtin_neon_vshrq_n_u8(in1.v128[i], in2);
#endif
		}

		return out;
	}

	[[nodiscard]] constexpr static inline uint32_t gt(const uint8x32_t in1,
												 	  const uint8x32_t in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const __uint8x16_t tmp = vcgtq_u8(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi8(tmp) << i*16;
#else
			const __uint8x16_t tmp = __builtin_neon_vcgtq_u8(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi8(tmp) << i*16;
#endif
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline int cmp(const uint8x32_t in1,
												  const uint8x32_t in2) noexcept {
		uint32_t ret = 0;
		
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const __uint8x16_t tmp = vceqq_u8(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi8(tmp) << i*16;
#else
			const __uint8x16_t tmp = __builtin_neon_vceqq_u8(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi8(tmp) << i*16;
#endif
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline uint8x32_t popcnt(const uint8x32_t in) noexcept {
		uint8x32_t out;
		
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			out.v128[i] = vcntq_u8(in.v128[i]);
#else
			out.v128[i] = __builtin_neon_vcntq_u8(in.v128[i]);
#endif
		}

		return out;
	}
};

struct uint16x16_t {
	union {
		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
		__uint16x8_t v128[2];
	};

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	///
	/// \return
	static inline uint16x16_t random() noexcept {
		uint16x16_t ret;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 4; ++i) {
			ret.v64[i] = fastrandombytes_uint64();
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline uint16x16_t set(uint16_t __q31, uint16_t __q30, uint16_t __q29, uint16_t __q28,
	                                                      uint16_t __q27, uint16_t __q26, uint16_t __q25, uint16_t __q24,
	                                                      uint16_t __q23, uint16_t __q22, uint16_t __q21, uint16_t __q20,
	                                                      uint16_t __q19, uint16_t __q18, uint16_t __q17, uint16_t __q16) noexcept {
		uint16x16_t out;
		out.v16[ 0] = __q31;
		out.v16[ 1] = __q30;
		out.v16[ 2] = __q29;
		out.v16[ 3] = __q28;
		out.v16[ 4] = __q27;
		out.v16[ 5] = __q26;
		out.v16[ 6] = __q25;
		out.v16[ 7] = __q24;
		out.v16[ 8] = __q23;
		out.v16[ 9] = __q22;
		out.v16[10] = __q21;
		out.v16[11] = __q20;
		out.v16[12] = __q19;
		out.v16[13] = __q18;
		out.v16[14] = __q17;
		out.v16[15] = __q16;
		return out;
	}
	
	[[nodiscard]] constexpr static inline uint16x16_t setr(uint16_t __q31, uint16_t __q30, uint16_t __q29, uint16_t __q28,
	                                                       uint16_t __q27, uint16_t __q26, uint16_t __q25, uint16_t __q24,
	                                                       uint16_t __q23, uint16_t __q22, uint16_t __q21, uint16_t __q20,
	                                                       uint16_t __q19, uint16_t __q18, uint16_t __q17, uint16_t __q16) noexcept {
		uint16x16_t out;
		out.v16[15] = __q31;
		out.v16[14] = __q30;
		out.v16[13] = __q29;
		out.v16[12] = __q28;
		out.v16[11] = __q27;
		out.v16[10] = __q26;
		out.v16[ 9] = __q25;
		out.v16[ 8] = __q24;
		out.v16[ 7] = __q23;
		out.v16[ 6] = __q22;
		out.v16[ 5] = __q21;
		out.v16[ 4] = __q20;
		out.v16[ 3] = __q19;
		out.v16[ 2] = __q18;
		out.v16[ 1] = __q17;
		out.v16[ 0] = __q16;
		return out;
	}
	/// sets all 32 8bit limbs to `a`
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t set1(const uint8_t a) noexcept {
		uint16x16_t out;
		out = uint16x16_t::set(a, a, a, a, a, a, a, a,
		                       a, a, a, a, a, a, a, a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	constexpr static inline uint16x16_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline uint16x16_t aligned_load(const void *ptr) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		uint16x16_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (__uint16x8_t) vldrq_p128(ptr128);
#else
			out.v128[i] = (__uint16x8_t) __builtin_neon_vldrq_p128(ptr128);
#endif
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	constexpr static inline uint16x16_t unaligned_load(const void *ptr) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		uint16x16_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (__uint16x8_t) vldrq_p128(ptr128);
#else
			out.v128[i] = (__uint16x8_t) __builtin_neon_vldrq_p128(ptr128);
#endif
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
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#endif
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint16x16_t in) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#endif
		}
	}

	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM1ykrgAyeAyYAHI%2BAEaYxCCSABykAA6oCoRODB7evv6BaRmOAqHhUSyx8Um2mPbFDEIETMQEOT5%2BATV1WY3NBKWRMXEJyQpNLW15nWN9A%2BWVIwCUtqhexMjsHOYAzGHI3lgA1CbbbmP4qAB0CCfYJhoAgjt7B5jHp%2Be0eNHXt/dPZl2DH2XiOJzcjjYv22d0ez2Br3ebmaLAA%2BuEBNDYQCgSCwadkOcwgQ/nDHgQAJ4pTBYKiHVGopgEAjEb5eAiYBlQDEMVEAN0wDhIqMp1IgiUWksOXmJiRFh3pqJlgkSqjlJO2VnJVJpmDpDKZLLZHK5EB5/MFRGIIp1EC4ADZJYtpbL5QzlQRVQ6RSctQ9Rbr9YzmazouzOajuZgBBahdaAxBJE6XYJvQRFR6HapJD7Nf8A7TFYbQ%2BHTeaBXGbWKJVLM/a3UriVn1b78zrCwaQ8aI1GYxWrVXMBAzMmPdszA2x2ZVBONX6C3qi12wybI2bo7z%2B8KE0na8Tx5P99Oc3O29SO8GjSue%2Bu%2B5bt7auKPifaTxmX5JVFxc/P24vO1epZruW97xraI57oIr6HlBn6zq2ZJPI8YzEF4Dgpp6qgHumJgAOx%2BvcACcXgMFkxz4f8RFER66oKnyiTHAArBY44mIxAAiCFIYR1FNvW6Z8g6bEWEJHFcVRGjEUe8p8qxzGHIkbGcXmZI8ZJHrQQJr7CYckhKeJklEe6spfvxhyCWYinMWY%2BkqdxeHKQRkloAwYyYKoKTEBhqrYRAzoMKg7kbCkOEUbhjn/AqCqRVFwnCQF%2BAKKIxDoPpHGHDMjjIIcYRfOE3lYbOhyqMKEAuWMBXYTlDCBJRhn1WpjUSU1DXNeV6Y0YV8phCO5FOY1nVVasp6IVFY3HICnlMMALBMNKDDEJ4tB1Y1/BeRAU7de87GHBovo5Uihw2ZqE1WJYeDOnh/XNQA9Ddw2XBZVkWHgSnbYqYZ0I4vJMM0yAINB0SiKoACOgn2iDeBKl4MMQGEXCPVwlnCa9HGkNVZiI8jzGo%2BxiwGS1hEPU9KNvScO3w1jz240x2AY1TpNiXZzUOStRHEJgBBrAwhzDQTrNwuFBkPMShyzWEfl9TFCqDUVTCkNEpCmMzjxjWthwbdJ6Z4O9e0nTr4KHKx%2BvWNYl0Uar40Kkwj3U2T2w7cQhjoH5XFWwqPz0YzO3k0dABUTsMC7%2BN2WNAsPGN2W%2B7LIogCAJXWhA8uHNEId%2BmNHNc8QPPII9ckibZBHhRwyy0JwjG8H4HBaKQqCcG4puWBlqzrG8Ow8KQBCaCXywANYgIxGj6JwkiV93tecLwCggEPXfVyXpBwLASBoCwKR0HE5CUKv6/0PEKL0QAtPshjAA6GhD1gfJ4BsABqeCYAA7gA8tSVcdzQtAcsQ08QNE4/RDCM0CknAO6AOYMQCkz9ojaEtKA3gq82CCGfgwWgID56kCwGGYAyJaC0GntwXgWBZpGHEBg/AHMHB4AFAQmu7lBThngeQQQtRx5fGiE7SBHgsDjyNCweBywqAGGAAoe%2BT9X6MCYfwQQIgxDsCkDIQQigVDqAwboQIBgjAoEbjYdh09IDLFQCFLIBDD4sCoCkLw5MeSHEPs/bYvBUACmIKyLA%2Bi/JdEtFkFwQdJh%2BECCEMIgwKjDAKOkTIAg/F6EKBEhgcwhjxECHYLxAhegTE8O0PQySqGpPGP0IJ8xQm2DyVEpJeT4khMScsBQLcNgSFLuXMeGC64cEOAfRIh85onyMIcB0lwND9I1rgQgJAJrbCfLwOeWhJSkH7oPYeHBR6kCrjXFpU8Z6d27jMsuHAzBNNWZPTZ88ZnOIyM4SQQA%3D
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t xor_(const uint16x16_t in1,
	                                                       const uint16x16_t in2) noexcept {
		uint16x16_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] ^ in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t and_(const uint16x16_t in1,
	                                                       const uint16x16_t in2) noexcept {
		uint16x16_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] & in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t or_(const uint16x16_t in1,
	                                                      const uint16x16_t in2) noexcept {
		uint16x16_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] | in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t andnot(const uint16x16_t in1,
	                                                         const uint16x16_t in2) noexcept {
		uint16x16_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 4; ++i) {
			out.v64[i] = ~(in1.v64[i] & in2.v64[i]);
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t not_(const uint16x16_t in1) noexcept {
		uint16x16_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 4; ++i) {
			out.v64[i] = ~in1.v64[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t add(const uint16x16_t in1,
	                                                      const uint16x16_t in2) noexcept {
		uint16x16_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = vaddq_u16(in1.v128[i], in2.v128[i]);
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t sub(const uint16x16_t in1,
	                                                      const uint16x16_t in2) noexcept {
		uint16x16_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] - in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t mullo(const uint16x16_t in1,
	                                                        const uint16x16_t in2) noexcept {
		uint16x16_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] * in2.v128[i];
		}
		return out;
	}

	static inline uint16x16_t mullo(const uint16x16_t in1,
	                                const uint8_t in2) {
		uint16x16_t rs = uint16x16_t::set1(in2);
		return uint16x16_t::mullo(in1, rs);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t slli(const uint16x16_t in1,
	                                                       const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		uint16x16_t out;


		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			out.v128[i] = vshlq_n_u16(in1.v128[i], in2);
#else
			out.v128[i] = __builtin_neon_vshlq_n_16(in1.v128[i], in2);
#endif
		}

		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t slri(const uint16x16_t in1,
	                                                       const uint16_t in2) noexcept {
		ASSERT(in2 <= 16);
		uint16x16_t out;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			out.v128[i] = vshrq_n_u16(in1.v128[i], in2);
#else
			out.v128[i] = __builtin_neon_vshrq_n_u16(in1.v128[i], in2);
#endif
		}

		return out;
	}


	static inline int gt(const uint16x16_t in1, const uint16x16_t in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			const __uint16x8_t tmp = vcgtq_u16(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi16(tmp) << i*8;
#else
			const __uint16x8_t tmp = __builtin_neon_vcgtq_16(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi16(tmp) << i*8;
#endif
		}

		return ret;
	}

	static inline int cmp(const uint16x16_t in1, const uint16x16_t in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			const __uint16x8_t tmp = vceqq_u16(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi16(tmp) << i*8;
#else
			const __uint16x8_t tmp = __builtin_neon_vcgeq_u16(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi16(tmp) << i*8;
#endif
		}

		return ret;
	}

	static inline uint16x16_t popcnt(const uint16x16_t in) noexcept {
		uint16x16_t out;
		__uint16x8_t mask = vdupq_n_u16(0xff);
		
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			const __uint16x8_t tmp = (__uint16x8_t)vcntq_u8((__uint8x16_t)in.v128[i]);
			out.v128[i] = vaddq_u16(vshrq_n_u16(tmp, 8), vandq_u16(tmp, mask));

#else
			const __uint16x8_t tmp = (__uint16x8_t)__builtin_neon_vcntq_u8((__uint8x16_t)in.v128[i]);
			out.v128[i] = vaddq_u16(vshrq_n_u16(tmp, 8), vandq_u16(tmp, mask));
#endif
		}

		return out;
	}
};

struct uint32x8_t {
	union {
		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
		__uint32x4_t v128[2];
	};

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	///
	/// \return
	static inline uint32x8_t random() noexcept {
		uint32x8_t ret;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 4; ++i) {
			ret.v64[i] = fastrandombytes_uint64();
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline uint32x8_t set(uint32_t __q31, uint32_t __q30, uint32_t __q29, uint32_t __q28,
	                                                     uint32_t __q27, uint32_t __q26, uint32_t __q25, uint32_t __q24) noexcept {
		uint32x8_t out;
		out.v32[0] = __q31;
		out.v32[1] = __q30;
		out.v32[2] = __q29;
		out.v32[3] = __q28;
		out.v32[4] = __q27;
		out.v32[5] = __q26;
		out.v32[6] = __q25;
		out.v32[7] = __q24;
		return out;
	}
	
	[[nodiscard]] constexpr static inline uint32x8_t setr(uint32_t __q31, uint32_t __q30, uint32_t __q29, uint32_t __q28,
	                                                      uint32_t __q27, uint32_t __q26, uint32_t __q25, uint32_t __q24) noexcept {
		uint32x8_t out;
		out.v32[7] = __q31;
		out.v32[6] = __q30;
		out.v32[5] = __q29;
		out.v32[4] = __q28;
		out.v32[3] = __q27;
		out.v32[2] = __q26;
		out.v32[1] = __q25;
		out.v32[0] = __q24;
		return out;
	}

	/// sets all 32 8bit limbs to `a`
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t set1(const uint8_t a) noexcept {
		uint32x8_t out;
		out = uint32x8_t::set(a, a, a, a, a, a, a, a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	constexpr static inline uint32x8_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline uint32x8_t aligned_load(const void *ptr) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		uint32x8_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (__uint32x4_t) vldrq_p128(ptr128);
#else
			out.v128[i] = (__uint32x4_t) __builtin_neon_vldrq_p128(ptr128);
#endif
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	constexpr static inline uint32x8_t unaligned_load(const void *ptr) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		uint32x8_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (__uint32x4_t) vldrq_p128(ptr128);
#else
			out.v128[i] = (__uint32x4_t) __builtin_neon_vldrq_p128(ptr128);
#endif
		}
		return out;
	}


	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const uint32x8_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint32x8_t in) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#endif
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint32x8_t in) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#endif
		}
	}

	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM1ykrgAyeAyYAHI%2BAEaYxCCSABykAA6oCoRODB7evv6BaRmOAqHhUSyx8Um2mPbFDEIETMQEOT5%2BATV1WY3NBKWRMXEJyQpNLW15nWN9A%2BWVIwCUtqhexMjsHOYAzGHI3lgA1CbbbmP4qAB0CCfYJhoAgjt7B5jHp%2Be0eNHXt/dPZl2DH2XiOJzcjjYv22d0ez2Br3ebmaLAA%2BuEBNDYQCgSCwadkOcwgQ/nDHgQAJ4pTBYKiHVGopgEAjEb5eAiYBlQDEMVEAN0wDhIqMp1IgiUWksOXmJiRFh3pqJlgkSqjlJO2VnJVJpmDpDKZLLZHK5EB5/MFRGIIp1EC4ADZJYtpbL5QzlQRVQ6RSctQ9Rbr9YzmazouzOajuZgBBahdaAxBJE6XYJvQRFR6HapJD7Nf8A7TFYbQ%2BHTeaBXGbWKJVLM/a3UriVn1b78zrCwaQ8aI1GYxWrVXMBAzMmPdszA2x2ZVBONX6C3qi12wybI2bo7z%2B8KE0na8Tx5P99Oc3O29SO8GjSue%2Bu%2B5bt7auKPifaTxmX5JVFxc/P24vO1epZruW97xraI57oIr6HlBn6zq2ZJPI8YzEF4Dgpp6qgHumJgAOx%2BvcACcXgMFkxz4f8RFER66oKnyiTHAArBY44mIxAAiCFIYR1FNvW6Z8g6bEWEJHFcVRGjEUe8p8qxzGHIkbGcXmZI8ZJHrQQJr7CYckhKeJklEe6spfvxhyCWYinMWY%2BkqdxeHKQRkloAwYyYKoKTEBhqrYRAzoMKg7kbCkOEUbhjn/AqCqRVFwnCQF%2BAKKIxDoPpHGHDMjjIIcYRfOE3lYbOhyqMKEAuWMBXYTlDCBJRhn1WpjUSU1DXNeV6Y0YV8phCO5FOY1nVVasp6IVFY3HICnlMMALBMNKDDEJ4tB1Y1/BeRAU7de87GHBovo5Uihw2ZqE1WJYeDOnh/XNQA9Ddw2XBZVkWHgSnbYqYZ0I4vJMM0yAINB0SiKoACOgn2iDeBKl4MMQGEXCPVwlnCa9HGkNVZiI8jzGo%2BxiwGS1hEPU9KNvScO3w1jz240x2AY1TpNiXZzUOStRHEJgBBrAwhzDQTrNwuFBkPMShyzWEfl9TFCqDUVTCkNEpCmMzjxjWthwbdJ6Z4O9e0nTr4KHKx%2BvWNYl0Uar40Kkwj3U2T2w7cQhjoH5XFWwqPz0YzO3k0dABUTsMC7%2BN2WNAsPGN2W%2B7LIogCAJXWhA8uHNEId%2BmNHNc8QPPII9ckibZBHhRwyy0JwjG8H4HBaKQqCcG4puWBlqzrG8Ow8KQBCaCXywANYgIxGj6JwkiV93tecLwCggEPXfVyXpBwLASBoCwKR0HE5CUKv6/0PEKL0QAtPshjAA6GhD1gfJ4BsABqeCYAA7gA8tSVcdzQtAcsQ08QNE4/RDCM0CknAO6AOYMQCkz9ojaEtKA3gq82CCGfgwWgID56kCwGGYAyJaC0GntwXgWBZpGHEBg/AHMHB4AFAQmu7lBThngeQQQtRx5fGiE7SBHgsDjyNCweBywqAGGAAoe%2BT9X6MCYfwQQIgxDsCkDIQQigVDqAwboQIBgjAoEbjYdh09IDLFQCFLIBDD4sCoCkLw5MeSHEPs/bYvBUACmIKyLA%2Bi/JdEtFkFwQdJh%2BECCEMIgwKjDAKOkTIAg/F6EKBEhgcwhjxECHYLxAhegTE8O0PQySqGpPGP0IJ8xQm2DyVEpJeT4khMScsBQLcNgSFLuXMeGC64cEOAfRIh85onyMIcB0lwND9I1rgQgJAJrbCfLwOeWhJSkH7oPYeHBR6kCrjXFpU8Z6d27jMsuHAzBNNWZPTZ88ZnOIyM4SQQA%3D
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t xor_(const uint32x8_t in1,
	                                                      const uint32x8_t in2) noexcept {
		uint32x8_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] ^ in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t and_(const uint32x8_t in1,
	                                                      const uint32x8_t in2) noexcept {
		uint32x8_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] & in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t or_(const uint32x8_t in1,
	                                                     const uint32x8_t in2) noexcept {
		uint32x8_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] | in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t andnot(const uint32x8_t in1,
	                                                        const uint32x8_t in2) noexcept {
		uint32x8_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 4; ++i) {
			out.v64[i] = ~(in1.v64[i] & in2.v64[i]);
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t not_(const uint32x8_t in1) noexcept {
		uint32x8_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 4; ++i) {
			out.v64[i] = ~in1.v64[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t add(const uint32x8_t in1,
	                                                     const uint32x8_t in2) noexcept {
		uint32x8_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			/// TODO not correct, carry and sruff
			out.v128[i] = in1.v128[i] + in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t sub(const uint32x8_t in1,
	                                                     const uint32x8_t in2) noexcept {
		uint32x8_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] - in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t mullo(const uint32x8_t in1,
	                                                       const uint32x8_t in2) noexcept {
		uint32x8_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] * in2.v128[i];
		}
		return out;
	}

	static inline uint32x8_t mullo(const uint32x8_t in1,
	                               const uint8_t in2) {
		uint32x8_t rs = uint32x8_t::set1(in2);
		return uint32x8_t::mullo(in1, rs);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t slli(const uint32x8_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 32);
		uint32x8_t out;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			out.v128[i] = vshlq_n_u32(in1.v128[i], in2);
#else
			out.v128[i] = __builtin_neon_vshlq_n_u32(in1.v128[i], in2);
#endif
		}

		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t slri(const uint32x8_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 32);
		uint32x8_t out;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			out.v128[i] = vshrq_n_u32(in1.v128[i], in2);
#else
			out.v128[i] = __builtin_neon_vshrq_n_u32(in1.v128[i], in2);
#endif
		}

		return out;
	}


	static inline int gt(const uint32x8_t in1, const uint32x8_t in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			const __uint32x4_t tmp = vcgtq_u32(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi32(tmp) << i*4;
#else
			const __uint32x4_t tmp = __builtin_neon_vcgtq_u32(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi32(tmp) << i*4;
#endif
		}

		return ret;
	}

	static inline int cmp(const uint32x8_t in1, const uint32x8_t in2) noexcept {
		uint32_t ret = 0;
		
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			const __uint32x4_t tmp = vceqq_u32(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi32(tmp) << i*4;
#else
			const __uint32x4_t tmp = __builtin_neon_vceqq_u32(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi32(tmp) << i*4;
#endif
		}

		return ret;
	}

	static inline uint32x8_t popcnt(const uint32x8_t in) {
		uint32x8_t out;
		__uint16x8_t mask = vdupq_n_u16(0xff);
		
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			const __uint16x8_t tmp1 = (__uint16x8_t)vcntq_u8((__uint8x16_t)in.v128[i]);
			const __uint16x8_t tmp2 = vaddq_u16(vshrq_n_u16(tmp1, 8), vandq_u16(tmp1, mask));
			out.v128[i] = vaddq_u32(vshrq_n_u32((__uint32x4_t)tmp2, 16), (__uint32x4_t)tmp2);
#else

			const __uint16x8_t tmp1 = (__uint16x8_t)__builtin_neon_vcntq_u8((__uint8x16_t)in.v128[i]);
			const __uint16x8_t tmp2 = __builtin_neon_vaddq_u16(__builtin_neon_vshrq_n_u16(tmp1, 8), __builtin_neon_vandq_u16(tmp1, mask));
			out.v128[i] = __builtin_neon_vaddq_u32(__builtin_neon_vshrq_n_u32((__uint32x4_t)tmp2, 16), (__uint32x4_t)tmp2);
#endif
		}

		return out;

	}
};

struct uint64x4_t {
	union {
		uint8_t v8[32];
		uint16_t v16[16];
		uint32_t v32[8];
		uint64_t v64[4];
		__uint64x2_t v128[2];
	};

	///
	/// \return
	inline uint64x4_t random() {
		uint64x4_t ret;
		for (uint32_t i = 0; i < 4; ++i) {
			ret.v64[i] = fastrandombytes_uint64();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	[[nodiscard]] constexpr static inline uint64x4_t set(const uint64_t i0,
	                                                     const uint64_t i1,
	                                                     const uint64_t i2,
	                                                     const uint64_t i3) noexcept {
		uint64x4_t ret;
		ret.v64[0] = i0;
		ret.v64[1] = i1;
		ret.v64[2] = i2;
		ret.v64[3] = i3;
		return ret;
	}

	[[nodiscard]] constexpr static inline uint64x4_t setr(const uint64_t i0,
	                                                      const uint64_t i1,
	                                                      const uint64_t i2,
	                                                      const uint64_t i3) noexcept {
		return uint64x4_t::set(i3, i2, i1, i0);
	}

	///
	/// \param a
	/// \return
	static inline uint64x4_t set1(const uint64_t a) {
		return uint64x4_t::set(a, a, a, a);
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned>
	static inline uint64x4_t load(const void *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	static inline uint64x4_t aligned_load(const void *ptr) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		uint64x4_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (__uint64x2_t) vldrq_p128(ptr128);
#else
			out.v128[i] = (__uint64x2_t) __builtin_neon_vldrq_p128(ptr128);
#endif
		}
		return out;
	}

	///
	/// \param ptr
	/// \return
	static inline uint64x4_t unaligned_load(const void *ptr) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		uint64x4_t out;
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (__uint64x2_t) vldrq_p128(ptr128);
#else
			out.v128[i] = (__uint64x2_t) __builtin_neon_vldrq_p128(ptr128);
#endif
		}
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned>
	static inline void store(void *ptr, const uint64x4_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, 
									 const uint64x4_t in) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#endif
		}
	}

	///
	/// \param ptr
	/// \param in
	static inline void unaligned_store(void *ptr,
									   const uint64x4_t in) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t) in.v128[i]);
#endif
		}
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t xor_(const uint64x4_t in1,
	                       const uint64x4_t in2) noexcept {
		uint64x4_t out;
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] ^ in2.v128[i];
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
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] & in2.v128[i];
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
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] | in2.v128[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t andnot(const uint64x4_t in1,
	                         const uint64x4_t in2) {
		uint64x4_t out;
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = ~(in1.v128[i] & in2.v128[i]);
		}
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t not_(const uint64x4_t in1) noexcept {
		uint64x4_t out;
		for (uint32_t i = 0; i < 4; ++i) {
			out.v64[i] = ~in1.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t add(const uint64x4_t in1,
	                      const uint64x4_t in2) noexcept {
		uint64x4_t out;
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = vaddq_u64(in1.v128[i], in2.v128[i]);
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
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = vsubq_u64(in1.v128[i], in2.v128[i]);
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t mullo(const uint64x4_t in1,
	                        const uint64x4_t in2) noexcept {
		uint64x4_t out;
		for (uint32_t i = 0; i < 4; ++i) {
			out.v64[i] = in1.v64[i] * in2.v64[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t mullo(const uint64x4_t in1,
	                        const uint64_t in2) noexcept {
		uint64x4_t out;
		for (uint32_t i = 0; i < 4; ++i) {
			out.v64[i] = in1.v64[i] * in2;
		}
		return out;
	}

	[[nodiscard]] constexpr static inline uint64x4_t slli(const uint64x4_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 64);
		uint64x4_t out;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			out.v128[i] = vshlq_n_u64(in1.v128[i], in2);
#else
			out.v128[i] = __builtin_neon_vshlq_n_u64(in1.v128[i], in2);
#endif
		}

		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64x4_t slri(const uint64x4_t in1,
	                                                      const uint8_t in2) noexcept {
		ASSERT(in2 <= 8);
		uint64x4_t out;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			out.v128[i] = vshrq_n_u64(in1.v128[i], in2);
#else
			out.v128[i] = __builtin_neon_vshrq_n_u64(in1.v128[i], in2);
#endif
		}

		return out;
	}
	
	///
	/// \param in1
	/// \param in2
	/// \return
	static inline uint64x4_t permute(const uint64x4_t in1,
									 const uint32_t in2) noexcept {
		uint64x4_t ret;
		ASSERT(0);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	static inline int gt(const uint64x4_t in1, 
						 const uint64x4_t in2) noexcept {
		int ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			const __uint64x2_t tmp = vcgtq_u64(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi64(tmp) << i*2;
#else
			const __uint64x2_t tmp = __builtin_neon_vcgtq_u64(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi64(tmp) << i*2;
#endif
		}
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	static inline int cmp(const uint64x4_t in1, const uint64x4_t in2) noexcept {
		int ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			const __uint64x2_t tmp = vceqq_u64(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi64(tmp) << i*4;
#else
			const __uint64x2_t tmp = __builtin_neon_vceqq_u64(in1.v128[i], in2.v128[i]);
    		ret ^= _mm_movemask_epi64(tmp) << i*4;
#endif
		}
		return ret;
	}

	///
	/// \param in
	/// \return
	static inline uint64x4_t popcnt(const uint64x4_t in) noexcept {
		uint64x4_t ret;
		__uint16x8_t mask = vdupq_n_u16(0xff);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			const __uint16x8_t tmp1 = (__uint16x8_t)vcntq_u8((__uint8x16_t)in.v128[i]);
			const __uint16x8_t tmp2 = vaddq_u16(vshrq_n_u16(tmp1, 8), vandq_u16(tmp1, mask));
			const __uint32x4_t tmp3 = vaddq_u32(vshrq_n_u32((__uint32x4_t)tmp2, 16), (__uint32x4_t)tmp2);
			ret.v128[i] = vaddq_u64(vshrq_n_u64((__uint64x2_t)tmp3, 32), (__uint64x2_t)tmp3);
#else
			// TODO
#endif
		}
		return ret;
	}
};

#endif
