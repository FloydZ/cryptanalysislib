#ifndef CRYPTANALYSISLIB_SIMD_NEON_H
#define CRYPTANALYSISLIB_SIMD_NEON_H

#include <arm_neon.h>
#include <cstdint>

#ifndef __clang__
#include <arm_fp16.h>
#include <arm_bf16.h>
#include <stdint.h>
typedef __Uint8x8_t   __uint8x8_t;
typedef __Uint8x16_t  __uint8x16_t;
typedef __Uint16x4_t  __uint16x4_t;
typedef __Uint16x8_t  __uint16x8_t;
typedef __Uint32x2_t  __uint32x2_t;
typedef __Uint32x4_t  __uint32x4_t;
typedef __Uint64x1_t  __uint64x1_t;
typedef __Uint64x2_t  __uint64x2_t;
#else
typedef __attribute__((neon_vector_type(8))) uint8_t  __uint8x8_t;
typedef __attribute__((neon_vector_type(16))) uint8_t __uint8x16_t;
typedef __attribute__((neon_vector_type(16)))  int8_t __int8x16_t;
typedef __attribute__((neon_vector_type(4))) uint16_t __uint16x4_t;
typedef __attribute__((neon_vector_type(8))) uint16_t __uint16x8_t;
typedef __attribute__((neon_vector_type(2))) uint32_t __uint32x2_t;
typedef __attribute__((neon_vector_type(4))) uint32_t __uint32x4_t;
typedef __attribute__((neon_vector_type(1))) uint64_t __uint64x1_t;
typedef __attribute__((neon_vector_type(2))) uint64_t __uint64x2_t;
#endif


struct uint8x32_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__uint8x16_t v128[2];
	};


	constexpr uint8x32_t() noexcept = default;

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary=false, bool hex=false) const;

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
														 char __q03, char __q02, char __q01, char __q00){
		uint8x32_t out;
		out.v8[ 0] = __q31;
		out.v8[ 1] = __q30;
		out.v8[ 2] = __q29;
		out.v8[ 3] = __q28;
		out.v8[ 4] = __q27;
		out.v8[ 5] = __q26;
		out.v8[ 6] = __q25;
		out.v8[ 7] = __q24;
		out.v8[ 8] = __q23;
		out.v8[ 9] = __q22;
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
	[[nodiscard]] constexpr static inline uint8x32_t set1(const uint8_t a) {
		uint8x32_t out;
		out =  uint8x32_t::set(a, a, a, a, a, a, a, a,
							   a, a, a, a, a, a, a, a,
							   a, a, a, a, a, a, a, a,
							   a, a, a, a, a, a, a, a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned=false>
	constexpr static inline uint8x32_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline uint8x32_t aligned_load(const void *ptr) {
		auto *ptr128 = (poly128_t *)ptr;
		uint8x32_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifdef __GNUC__
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
	constexpr static inline uint8x32_t unaligned_load(const void *ptr) {
		auto *ptr128 = (poly128_t *)ptr;
		uint8x32_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifdef __GNUC__
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
	template<const bool aligned=false>
	constexpr static inline void store(void *ptr, const uint8x32_t in) {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint8x32_t in) {
		auto *ptr128 = (poly128_t *)ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#endif
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint8x32_t in) {
		auto *ptr128 = (poly128_t *)ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#endif
		}
	}

	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM1ykrgAyeAyYAHI%2BAEaYxCCSABykAA6oCoRODB7evv6BaRmOAqHhUSyx8Um2mPbFDEIETMQEOT5%2BATV1WY3NBKWRMXEJyQpNLW15nWN9A%2BWVIwCUtqhexMjsHOYAzGHI3lgA1CbbbmP4qAB0CCfYJhoAgjt7B5jHp%2Be0eNHXt/dPZl2DH2XiOJzcjjYv22d0ez2Br3ebmaLAA%2BuEBNDYQCgSCwadkOcwgQ/nDHgQAJ4pTBYKiHVGopgEAjEb5eAiYBlQDEMVEAN0wDhIqMp1IgiUWksOXmJiRFh3pqJlgkSqjlJO2VnJVJpmDpDKZLLZHK5EB5/MFRGIIp1EC4ADZJYtpbL5QzlQRVQ6RSctQ9Rbr9YzmazouzOajuZgBBahdaAxBJE6XYJvQRFR6HapJD7Nf8A7TFYbQ%2BHTeaBXGbWKJVLM/a3UriVn1b78zrCwaQ8aI1GYxWrVXMBAzMmPdszA2x2ZVBONX6C3qi12wybI2bo7z%2B8KE0na8Tx5P99Oc3O29SO8GjSue%2Bu%2B5bt7auKPifaTxmX5JVFxc/P24vO1epZruW97xraI57oIr6HlBn6zq2ZJPI8YzEF4Dgpp6qgHumJgAOx%2BvcACcXgMFkxz4f8RFER66oKnyiTHAArBY44mIxAAiCFIYR1FNvW6Z8g6bEWEJHFcVRGjEUe8p8qxzGHIkbGcXmZI8ZJHrQQJr7CYckhKeJklEe6spfvxhyCWYinMWY%2BkqdxeHKQRkloAwYyYKoKTEBhqrYRAzoMKg7kbCkOEUbhjn/AqCqRVFwnCQF%2BAKKIxDoPpHGHDMjjIIcYRfOE3lYbOhyqMKEAuWMBXYTlDCBJRhn1WpjUSU1DXNeV6Y0YV8phCO5FOY1nVVasp6IVFY3HICnlMMALBMNKDDEJ4tB1Y1/BeRAU7de87GHBovo5Uihw2ZqE1WJYeDOnh/XNQA9Ddw2XBZVkWHgSnbYqYZ0I4vJMM0yAINB0SiKoACOgn2iDeBKl4MMQGEXCPVwlnCa9HGkNVZiI8jzGo%2BxiwGS1hEPU9KNvScO3w1jz240x2AY1TpNiXZzUOStRHEJgBBrAwhzDQTrNwuFBkPMShyzWEfl9TFCqDUVTCkNEpCmMzjxjWthwbdJ6Z4O9e0nTr4KHKx%2BvWNYl0Uar40Kkwj3U2T2w7cQhjoH5XFWwqPz0YzO3k0dABUTsMC7%2BN2WNAsPGN2W%2B7LIogCAJXWhA8uHNEId%2BmNHNc8QPPII9ckibZBHhRwyy0JwjG8H4HBaKQqCcG4puWBlqzrG8Ow8KQBCaCXywANYgIxGj6JwkiV93tecLwCggEPXfVyXpBwLASBoCwKR0HE5CUKv6/0PEKL0QAtPshjAA6GhD1gfJ4BsABqeCYAA7gA8tSVcdzQtAcsQ08QNE4/RDCM0CknAO6AOYMQCkz9ojaEtKA3gq82CCGfgwWgID56kCwGGYAyJaC0GntwXgWBZpGHEBg/AHMHB4AFAQmu7lBThngeQQQtRx5fGiE7SBHgsDjyNCweBywqAGGAAoe%2BT9X6MCYfwQQIgxDsCkDIQQigVDqAwboQIBgjAoEbjYdh09IDLFQCFLIBDD4sCoCkLw5MeSHEPs/bYvBUACmIKyLA%2Bi/JdEtFkFwQdJh%2BECCEMIgwKjDAKOkTIAg/F6EKBEhgcwhjxECHYLxAhegTE8O0PQySqGpPGP0IJ8xQm2DyVEpJeT4khMScsBQLcNgSFLuXMeGC64cEOAfRIh85onyMIcB0lwND9I1rgQgJAJrbCfLwOeWhJSkH7oPYeHBR6kCrjXFpU8Z6d27jMsuHAzBNNWZPTZ88ZnOIyM4SQQA%3D
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t xor_(const uint8x32_t in1,
														  const uint8x32_t in2) {
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
														  const uint8x32_t in2) {
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
														   const uint8x32_t in2) {
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
	[[nodiscard]] constexpr static inline uint8x32_t not_(const uint8x32_t in1) {
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
	                                                     const uint8x32_t in2) {
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
	                                                     const uint8x32_t in2) {
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
														   const uint8x32_t in2) {
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
														  const uint8_t in2) {
		ASSERT(in2 <= 8);
		uint8x32_t out;
		uint8x32_t tmp = uint8x32_t::set1(in2);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			//TODO out.v128[i] = vshlq_v(in1.v128[i], (__int8x16_t)tmp.v128[0], 48u);
#else
			out.v128[i] = __builtin_neon_vshlq_v(in1.v128[i], (__int8x16_t)tmp.v128[0], 48u);
#endif
		}

		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8x32_t slri(const uint8x32_t in1,
														  const uint8_t in2) {
		ASSERT(in2 <= 8);
		uint8x32_t out;
		uint8x32_t tmp = uint8x32_t::set1(-in2);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			//TODO out.v128[i] = vshlq_v(in1.v128[i], (__int8x16_t)tmp.v128[0], 48u);
#else
			out.v128[i] = __builtin_neon_vshlq_v(in1.v128[i], (__int8x16_t)tmp.v128[0], 48u);
#endif
		}

		return out;
	}

	static inline int gt(const uint8x32_t in1, const uint8x32_t in2) {

	}

	static inline int cmp(const uint8x32_t in1, const uint8x32_t in2) {

	}

	static inline uint8x32_t popcnt(const uint8x32_t in) {

	}

};

struct uint16x16_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__uint16x8_t v128[2];
	};
	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary=false, bool hex=false) const;

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

	[[nodiscard]] constexpr static inline uint16x16_t set(char __q31, char __q30, char __q29, char __q28,
	                                                     char __q27, char __q26, char __q25, char __q24,
	                                                     char __q23, char __q22, char __q21, char __q20,
	                                                     char __q19, char __q18, char __q17, char __q16,
	                                                     char __q15, char __q14, char __q13, char __q12,
	                                                     char __q11, char __q10, char __q09, char __q08,
	                                                     char __q07, char __q06, char __q05, char __q04,
	                                                     char __q03, char __q02, char __q01, char __q00){
		uint16x16_t out;
		out.v8[ 0] = __q31;
		out.v8[ 1] = __q30;
		out.v8[ 2] = __q29;
		out.v8[ 3] = __q28;
		out.v8[ 4] = __q27;
		out.v8[ 5] = __q26;
		out.v8[ 6] = __q25;
		out.v8[ 7] = __q24;
		out.v8[ 8] = __q23;
		out.v8[ 9] = __q22;
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
	[[nodiscard]] constexpr static inline uint16x16_t set1(const uint8_t a) {
		uint16x16_t out;
		out =  uint16x16_t::set(a, a, a, a, a, a, a, a,
		                      a, a, a, a, a, a, a, a,
		                      a, a, a, a, a, a, a, a,
		                      a, a, a, a, a, a, a, a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned=false>
	constexpr static inline uint16x16_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline uint16x16_t aligned_load(const void *ptr) {
		auto *ptr128 = (poly128_t *)ptr;
		uint16x16_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifdef __GNUC__
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
	constexpr static inline uint16x16_t unaligned_load(const void *ptr) {
		auto *ptr128 = (poly128_t *)ptr;
		uint16x16_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifdef __GNUC__
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
	template<const bool aligned=false>
	constexpr static inline void store(void *ptr, const uint16x16_t in) {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint16x16_t in) {
		auto *ptr128 = (poly128_t *)ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#endif
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint16x16_t in) {
		auto *ptr128 = (poly128_t *)ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#endif
		}
	}

	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM1ykrgAyeAyYAHI%2BAEaYxCCSABykAA6oCoRODB7evv6BaRmOAqHhUSyx8Um2mPbFDEIETMQEOT5%2BATV1WY3NBKWRMXEJyQpNLW15nWN9A%2BWVIwCUtqhexMjsHOYAzGHI3lgA1CbbbmP4qAB0CCfYJhoAgjt7B5jHp%2Be0eNHXt/dPZl2DH2XiOJzcjjYv22d0ez2Br3ebmaLAA%2BuEBNDYQCgSCwadkOcwgQ/nDHgQAJ4pTBYKiHVGopgEAjEb5eAiYBlQDEMVEAN0wDhIqMp1IgiUWksOXmJiRFh3pqJlgkSqjlJO2VnJVJpmDpDKZLLZHK5EB5/MFRGIIp1EC4ADZJYtpbL5QzlQRVQ6RSctQ9Rbr9YzmazouzOajuZgBBahdaAxBJE6XYJvQRFR6HapJD7Nf8A7TFYbQ%2BHTeaBXGbWKJVLM/a3UriVn1b78zrCwaQ8aI1GYxWrVXMBAzMmPdszA2x2ZVBONX6C3qi12wybI2bo7z%2B8KE0na8Tx5P99Oc3O29SO8GjSue%2Bu%2B5bt7auKPifaTxmX5JVFxc/P24vO1epZruW97xraI57oIr6HlBn6zq2ZJPI8YzEF4Dgpp6qgHumJgAOx%2BvcACcXgMFkxz4f8RFER66oKnyiTHAArBY44mIxAAiCFIYR1FNvW6Z8g6bEWEJHFcVRGjEUe8p8qxzGHIkbGcXmZI8ZJHrQQJr7CYckhKeJklEe6spfvxhyCWYinMWY%2BkqdxeHKQRkloAwYyYKoKTEBhqrYRAzoMKg7kbCkOEUbhjn/AqCqRVFwnCQF%2BAKKIxDoPpHGHDMjjIIcYRfOE3lYbOhyqMKEAuWMBXYTlDCBJRhn1WpjUSU1DXNeV6Y0YV8phCO5FOY1nVVasp6IVFY3HICnlMMALBMNKDDEJ4tB1Y1/BeRAU7de87GHBovo5Uihw2ZqE1WJYeDOnh/XNQA9Ddw2XBZVkWHgSnbYqYZ0I4vJMM0yAINB0SiKoACOgn2iDeBKl4MMQGEXCPVwlnCa9HGkNVZiI8jzGo%2BxiwGS1hEPU9KNvScO3w1jz240x2AY1TpNiXZzUOStRHEJgBBrAwhzDQTrNwuFBkPMShyzWEfl9TFCqDUVTCkNEpCmMzjxjWthwbdJ6Z4O9e0nTr4KHKx%2BvWNYl0Uar40Kkwj3U2T2w7cQhjoH5XFWwqPz0YzO3k0dABUTsMC7%2BN2WNAsPGN2W%2B7LIogCAJXWhA8uHNEId%2BmNHNc8QPPII9ckibZBHhRwyy0JwjG8H4HBaKQqCcG4puWBlqzrG8Ow8KQBCaCXywANYgIxGj6JwkiV93tecLwCggEPXfVyXpBwLASBoCwKR0HE5CUKv6/0PEKL0QAtPshjAA6GhD1gfJ4BsABqeCYAA7gA8tSVcdzQtAcsQ08QNE4/RDCM0CknAO6AOYMQCkz9ojaEtKA3gq82CCGfgwWgID56kCwGGYAyJaC0GntwXgWBZpGHEBg/AHMHB4AFAQmu7lBThngeQQQtRx5fGiE7SBHgsDjyNCweBywqAGGAAoe%2BT9X6MCYfwQQIgxDsCkDIQQigVDqAwboQIBgjAoEbjYdh09IDLFQCFLIBDD4sCoCkLw5MeSHEPs/bYvBUACmIKyLA%2Bi/JdEtFkFwQdJh%2BECCEMIgwKjDAKOkTIAg/F6EKBEhgcwhjxECHYLxAhegTE8O0PQySqGpPGP0IJ8xQm2DyVEpJeT4khMScsBQLcNgSFLuXMeGC64cEOAfRIh85onyMIcB0lwND9I1rgQgJAJrbCfLwOeWhJSkH7oPYeHBR6kCrjXFpU8Z6d27jMsuHAzBNNWZPTZ88ZnOIyM4SQQA%3D
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t xor_(const uint16x16_t in1,
	                                                      const uint16x16_t in2) {
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
	                                                      const uint16x16_t in2) {
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
	                                                     const uint16x16_t in2) {
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
	                                                        const uint16x16_t in2) {
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
	[[nodiscard]] constexpr static inline uint16x16_t not_(const uint16x16_t in1) {
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
	                                                     const uint16x16_t in2) {
		uint16x16_t out;
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
	[[nodiscard]] constexpr static inline uint16x16_t sub(const uint16x16_t in1,
	                                                     const uint16x16_t in2) {
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
	                                                       const uint16x16_t in2) {
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
	                                                      const uint8_t in2) {
		ASSERT(in2 <= 8);
		uint16x16_t out;
		uint16x16_t tmp = uint16x16_t::set1(in2);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			//TODO out.v128[i] = vshlq_v(in1.v128[i], (__int8x16_t)tmp.v128[0], 48u);
#else
			out.v128[i] = __builtin_neon_vshlq_v(in1.v128[i], (__int8x16_t)tmp.v128[0], 48u);
#endif
		}

		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16x16_t slri(const uint16x16_t in1,
	                                                      const uint8_t in2) {
		ASSERT(in2 <= 8);
		uint16x16_t out;
		uint16x16_t tmp = uint16x16_t::set1(-in2);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			//TODO out.v128[i] = vshlq_v(in1.v128[i], (__int8x16_t)tmp.v128[0], 48u);
#else
			out.v128[i] = __builtin_neon_vshlq_v(in1.v128[i], (__int8x16_t) tmp.v128[0], 48u);
#endif
		}

		return out;
	}


	static inline int gt(const uint16x16_t in1, const uint16x16_t in2) {
		return 0; // TODO
	}

	static inline int cmp(const uint16x16_t in1, const uint16x16_t in2) {
		return 0; // TODO

	}

	static inline uint16x16_t popcnt(const uint16x16_t in) {
		return in; // TODO
	}

};

struct uint32x8_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__uint32x4_t v128[2];
	};
	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary=false, bool hex=false) const;

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

	[[nodiscard]] constexpr static inline uint32x8_t set(char __q31, char __q30, char __q29, char __q28,
		                                                 char __q27, char __q26, char __q25, char __q24,
		                                                 char __q23, char __q22, char __q21, char __q20,
		                                                 char __q19, char __q18, char __q17, char __q16,
		                                                 char __q15, char __q14, char __q13, char __q12,
		                                                 char __q11, char __q10, char __q09, char __q08,
		                                                 char __q07, char __q06, char __q05, char __q04,
		                                                 char __q03, char __q02, char __q01, char __q00){
		uint32x8_t out;
		out.v8[ 0] = __q31;
		out.v8[ 1] = __q30;
		out.v8[ 2] = __q29;
		out.v8[ 3] = __q28;
		out.v8[ 4] = __q27;
		out.v8[ 5] = __q26;
		out.v8[ 6] = __q25;
		out.v8[ 7] = __q24;
		out.v8[ 8] = __q23;
		out.v8[ 9] = __q22;
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
	[[nodiscard]] constexpr static inline uint32x8_t set1(const uint8_t a) {
		uint32x8_t out;
		out =  uint32x8_t::set(a, a, a, a, a, a, a, a,
			                  a, a, a, a, a, a, a, a,
			                  a, a, a, a, a, a, a, a,
			                  a, a, a, a, a, a, a, a);
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned=false>
	constexpr static inline uint32x8_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline uint32x8_t aligned_load(const void *ptr) {
		auto *ptr128 = (poly128_t *)ptr;
		uint32x8_t out;
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
	constexpr static inline uint32x8_t unaligned_load(const void *ptr) {
		auto *ptr128 = (poly128_t *)ptr;
		uint32x8_t out;
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
	template<const bool aligned=false>
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
	static inline void aligned_store(void *ptr, const uint32x8_t in) {
		auto *ptr128 = (poly128_t *)ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#endif
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const uint32x8_t in) {
		auto *ptr128 = (poly128_t *)ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#endif
		}
	}

	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM1ykrgAyeAyYAHI%2BAEaYxCCSABykAA6oCoRODB7evv6BaRmOAqHhUSyx8Um2mPbFDEIETMQEOT5%2BATV1WY3NBKWRMXEJyQpNLW15nWN9A%2BWVIwCUtqhexMjsHOYAzGHI3lgA1CbbbmP4qAB0CCfYJhoAgjt7B5jHp%2Be0eNHXt/dPZl2DH2XiOJzcjjYv22d0ez2Br3ebmaLAA%2BuEBNDYQCgSCwadkOcwgQ/nDHgQAJ4pTBYKiHVGopgEAjEb5eAiYBlQDEMVEAN0wDhIqMp1IgiUWksOXmJiRFh3pqJlgkSqjlJO2VnJVJpmDpDKZLLZHK5EB5/MFRGIIp1EC4ADZJYtpbL5QzlQRVQ6RSctQ9Rbr9YzmazouzOajuZgBBahdaAxBJE6XYJvQRFR6HapJD7Nf8A7TFYbQ%2BHTeaBXGbWKJVLM/a3UriVn1b78zrCwaQ8aI1GYxWrVXMBAzMmPdszA2x2ZVBONX6C3qi12wybI2bo7z%2B8KE0na8Tx5P99Oc3O29SO8GjSue%2Bu%2B5bt7auKPifaTxmX5JVFxc/P24vO1epZruW97xraI57oIr6HlBn6zq2ZJPI8YzEF4Dgpp6qgHumJgAOx%2BvcACcXgMFkxz4f8RFER66oKnyiTHAArBY44mIxAAiCFIYR1FNvW6Z8g6bEWEJHFcVRGjEUe8p8qxzGHIkbGcXmZI8ZJHrQQJr7CYckhKeJklEe6spfvxhyCWYinMWY%2BkqdxeHKQRkloAwYyYKoKTEBhqrYRAzoMKg7kbCkOEUbhjn/AqCqRVFwnCQF%2BAKKIxDoPpHGHDMjjIIcYRfOE3lYbOhyqMKEAuWMBXYTlDCBJRhn1WpjUSU1DXNeV6Y0YV8phCO5FOY1nVVasp6IVFY3HICnlMMALBMNKDDEJ4tB1Y1/BeRAU7de87GHBovo5Uihw2ZqE1WJYeDOnh/XNQA9Ddw2XBZVkWHgSnbYqYZ0I4vJMM0yAINB0SiKoACOgn2iDeBKl4MMQGEXCPVwlnCa9HGkNVZiI8jzGo%2BxiwGS1hEPU9KNvScO3w1jz240x2AY1TpNiXZzUOStRHEJgBBrAwhzDQTrNwuFBkPMShyzWEfl9TFCqDUVTCkNEpCmMzjxjWthwbdJ6Z4O9e0nTr4KHKx%2BvWNYl0Uar40Kkwj3U2T2w7cQhjoH5XFWwqPz0YzO3k0dABUTsMC7%2BN2WNAsPGN2W%2B7LIogCAJXWhA8uHNEId%2BmNHNc8QPPII9ckibZBHhRwyy0JwjG8H4HBaKQqCcG4puWBlqzrG8Ow8KQBCaCXywANYgIxGj6JwkiV93tecLwCggEPXfVyXpBwLASBoCwKR0HE5CUKv6/0PEKL0QAtPshjAA6GhD1gfJ4BsABqeCYAA7gA8tSVcdzQtAcsQ08QNE4/RDCM0CknAO6AOYMQCkz9ojaEtKA3gq82CCGfgwWgID56kCwGGYAyJaC0GntwXgWBZpGHEBg/AHMHB4AFAQmu7lBThngeQQQtRx5fGiE7SBHgsDjyNCweBywqAGGAAoe%2BT9X6MCYfwQQIgxDsCkDIQQigVDqAwboQIBgjAoEbjYdh09IDLFQCFLIBDD4sCoCkLw5MeSHEPs/bYvBUACmIKyLA%2Bi/JdEtFkFwQdJh%2BECCEMIgwKjDAKOkTIAg/F6EKBEhgcwhjxECHYLxAhegTE8O0PQySqGpPGP0IJ8xQm2DyVEpJeT4khMScsBQLcNgSFLuXMeGC64cEOAfRIh85onyMIcB0lwND9I1rgQgJAJrbCfLwOeWhJSkH7oPYeHBR6kCrjXFpU8Z6d27jMsuHAzBNNWZPTZ88ZnOIyM4SQQA%3D
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t xor_(const uint32x8_t in1,
		                                                  const uint32x8_t in2) {
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
		                                                  const uint32x8_t in2) {
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
		                                                 const uint32x8_t in2) {
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
		                                                    const uint32x8_t in2) {
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
	[[nodiscard]] constexpr static inline uint32x8_t not_(const uint32x8_t in1) {
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
		                                                 const uint32x8_t in2) {
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
		                                                 const uint32x8_t in2) {
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
		                                                   const uint32x8_t in2) {
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
		                                                  const uint8_t in2) {
		ASSERT(in2 <= 8);
		uint32x8_t out;
		uint32x8_t tmp = uint32x8_t::set1(in2);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			//TODO out.v128[i] = vshlq_v(in1.v128[i], (__int8x16_t)tmp.v128[0], 48u);
#else
			out.v128[i] = __builtin_neon_vshlq_v(in1.v128[i], (__int8x16_t)tmp.v128[0], 48u);
#endif
		}

		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32x8_t slri(const uint32x8_t in1,
		                                                  const uint8_t in2) {
		ASSERT(in2 <= 8);
		uint32x8_t out;
		uint32x8_t tmp = uint32x8_t::set1(-in2);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			//TODO out.v128[i] = vshlq_v(in1.v128[i], (__int8x16_t)tmp.v128[0], 48u);
#else
			out.v128[i] = __builtin_neon_vshlq_v(in1.v128[i], (__int8x16_t) tmp.v128[0], 48u);
#endif
		}

		return out;
	}


	static inline int gt(const uint32x8_t in1, const uint32x8_t in2) {
		return 0; /// TODO
	}

	static inline int cmp(const uint32x8_t in1, const uint32x8_t in2) {
		return 0; /// TODO
	}

	static inline uint32x8_t popcnt(const uint32x8_t in) {
		uint32x8_t ret;
		/// TODO
		return ret;
	}
};

struct uint64x4_t {
	union {
		uint8_t  v8 [32];
		uint16_t v16[16];
		uint32_t v32[ 8];
		uint64_t v64[ 4];
		__uint64x2_t v128[2];
	};

	///
	/// \return
	inline uint64x4_t random(){
		uint64x4_t ret;
		for (uint32_t i = 0; i < 4; ++i) {
			ret.v64[i] = fastrandombytes_uint64();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary=false, bool hex=false) const;

	[[nodiscard]] constexpr static inline uint64x4_t set(const int64_t i0,
	                                                     const int64_t i1,
														 const int64_t i2,
														 const int64_t i3){
		uint64x4_t ret;
		ret.v64[0] = i0;
		ret.v64[1] = i1;
		ret.v64[2] = i2;
		ret.v64[3] = i3;
		return ret;
	}

	[[nodiscard]] constexpr static inline uint64x4_t setr(const int64_t i0,
														  const int64_t i1,
														  const int64_t i2,
														  const int64_t i3){
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
	static inline uint64x4_t load(const void *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	static inline uint64x4_t aligned_load(const void *ptr) {
		auto *ptr128 = (poly128_t *)ptr;
		uint64x4_t out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifdef __GNUC__
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
	static inline uint64x4_t unaligned_load(const void *ptr) {
		auto *ptr128 = (poly128_t *)ptr;
		uint64x4_t out;
		for (uint32_t i = 0; i < 2u; ++i) {
#ifdef __GNUC__
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
	template<const bool aligned>
	static inline void store(void *ptr, const uint64x4_t in) {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr, const uint64x4_t in) {
		auto *ptr128 = (poly128_t *)ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#endif
		}

	}

	///
	/// \param ptr
	/// \param in
	static inline void unaligned_store(void *ptr, const uint64x4_t in) {
		auto *ptr128 = (poly128_t *)ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128, (poly128_t)in.v128[i]);
#endif
		}
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t xor_(const uint64x4_t in1,
						   const uint64x4_t in2) {
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
	inline uint64x4_t and_(const uint64x4_t in1,
						   const uint64x4_t in2) {
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
	inline uint64x4_t or_(const uint64x4_t in1,
						  const uint64x4_t in2) {
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
	inline uint64x4_t andnot(const uint64x4_t in1,
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
	inline uint64x4_t not_(const uint64x4_t in1) {
		uint64x4_t out;
		/// TODO
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t add(const uint64x4_t in1,
						  const uint64x4_t in2) {
		uint64x4_t out;
		/// TODO
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t sub(const uint64x4_t in1,
						  const uint64x4_t in2) {
		uint64x4_t out;
		/// TODO
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	inline uint64x4_t mullo(const uint64x4_t in1,
							const uint64x4_t in2) {
		ASSERT(false);
		uint64x4_t out;
		return out;
	}

	static inline uint64x4_t permute(const uint64x4_t in1, const uint32_t in2) {
		uint64x4_t ret;
		return ret;
	}
	static inline int gt(const uint64x4_t in1, const uint64x4_t in2) {
		int ret = 0;
		return ret;
	}

	static inline int cmp(const uint64x4_t in1, const uint64x4_t in2) {
		int ret = 0;
		return ret;
	}

	static inline uint64x4_t popcnt(const uint64x4_t in) {
		uint64x4_t ret;
		return ret;
	}
};

#endif
