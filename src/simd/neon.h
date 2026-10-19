#ifndef CRYPTANALYSISLIB_SIMD_NEON_H
#define CRYPTANALYSISLIB_SIMD_NEON_H

#include <arm_neon.h>
#include <cstdint>

#include "helper.h"
#include "random.h"



/// taken from: https://github.com/DLTcollab/sse2neon/blob/de2817727c72fc2f4ce9f54e2db6e40ce0548414/sse2neon.h#L4540
/// helper function, which collects the sign bits of each 8 bit limbs
constexpr inline uint32_t _mm_movemask_epi8(const uint8x16_t input) noexcept {
#ifdef __clang__
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
#else
	uint16x8_t high_bits = (uint16x8_t) __builtin_aarch64_lshrv16qi_uus((int8x16_t) input, 7);
	uint32x4_t paired16 = (uint32x4_t) __builtin_aarch64_ssra_nv8hi(high_bits, high_bits, 7);
	uint64x2_t paired32 = (uint64x2_t) __builtin_aarch64_usra_nv4si_uuus(paired16, paired16, 14);
	uint8x16_t paired64 = (uint8x16_t) __builtin_aarch64_usra_nv2di_uuus(paired32, paired32, 28);
	return paired64[0] | paired64[8] << 8;
#endif
}

constexpr inline uint32_t _mm_movemask_epi16(const uint16x8_t input) noexcept {
	constexpr int16_t shift[8] = {0, 1, 2, 3, 4, 5, 6, 7};
#ifdef __clang__
	uint16x8_t tmp = vshrq_n_u16(input, 15);
	return vaddvq_u16(vshlq_u16(tmp, vld1q_s16(shift)));
#else

	uint16x8_t tmp = __builtin_aarch64_lshrv8hi_uus(input, 15);
	return __builtin_aarch64_reduc_plus_scal_v8hi_uu(__builtin_aarch64_ushlv8hi_uus(tmp, __builtin_aarch64_ld1v8hi(shift)));
#endif
}

// Set each bit of mask dst based on the most significant bit of the
// corresponding packed single-precision (32-bit) floating-point element in a.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movemask_ps
constexpr inline uint32_t _mm_movemask_epi32(const uint32x4_t input) noexcept {
	constexpr int32_t shift[4] = {0, 1, 2, 3};
#ifdef __clang__
	uint32x4_t tmp = vshrq_n_u32(input, 31);
	return vaddvq_u32(vshlq_u32(tmp, vld1q_s32(shift)));
#else
	uint32x4_t tmp = __builtin_aarch64_lshrv4si_uus(input, 31);
	return __builtin_aarch64_reduc_plus_scal_v4si_uu(__builtin_aarch64_ushlv4si_uus(tmp, __builtin_aarch64_ld1v4si(shift)));
#endif
}

// Set each bit of mask dst based on the most significant bit of the
// corresponding packed double-precision (64-bit) floating-point element in a.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_movemask_pd
constexpr inline uint32_t _mm_movemask_epi64(const uint64x2_t input) noexcept {
#ifdef __clang__
	uint64x2_t high_bits = vshrq_n_u64(input, 63);
	return (uint32_t) (vgetq_lane_u64(high_bits, 0) | (vgetq_lane_u64(high_bits, 1) << 1));
#else
	uint64x2_t high_bits = __builtin_aarch64_lshrv2di_uus(input, 63);
	return (uint32_t) ((high_bits[0]) | (high_bits[1] << 1));
#endif
}

namespace cryptanalysislib {
    template<const bool __unsigned=true>
	struct _Xint8x16_t;
    template<const bool __unsigned=true>
	struct _Xint16x8_t;
    template<const bool __unsigned=true>
	struct _Xint32x4_t;
    template<const bool __unsigned=true>
	struct _Xint64x2_t;


    template<const bool __unsigned>
	struct _Xint8x16_t {
		constexpr static uint32_t LIMBS = 16;
		using limb_type = std::conditional_t<__unsigned, uint8_t, int8_t>;
	    using S = _Xint8x16_t;

		constexpr inline _Xint8x16_t operator=(const _Xint16x8_t<> &b) noexcept;
		constexpr inline _Xint8x16_t operator=(const _Xint32x4_t<> &b) noexcept;
		constexpr inline _Xint8x16_t operator=(const _Xint64x2_t<> &b) noexcept;

		constexpr _Xint8x16_t() noexcept = default;
		constexpr _Xint8x16_t(const _Xint16x8_t<> &b) noexcept;
		constexpr _Xint8x16_t(const _Xint32x4_t<> &b) noexcept;
		constexpr _Xint8x16_t(const _Xint64x2_t<> &b) noexcept;

		union {
			// compatibility to `TxN_t`
			limb_type d[16];

			uint8_t v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];
			uint8x16_t v128;
		};

        [[nodiscard]] constexpr inline static size_t size() noexcept {
            return LIMBS;
        }

	    [[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
            return __unsigned;
        }

		[[nodiscard]] constexpr inline limb_type& operator[](const uint32_t i) noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		///
		/// \return
		static inline _Xint8x16_t random() noexcept {
			_Xint8x16_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = rng();
			}

			return ret;
		}

		/// \param binary[in]:
		/// \param hex[in]:
		constexpr inline void print(bool binary = false,
		                            bool hex = false) const;

		[[nodiscard]] constexpr static inline _Xint8x16_t set1(const limb_type i) noexcept {
			_Xint8x16_t ret;
			for (uint32_t j = 0; j < 16u; ++j) {
				ret.v8[j] = i;
			}
			return ret;
		}

		/// set the `i`-th bit in each limb
		/// \param pos[in]: bit position to set
		constexpr inline void set_bit(const uint32_t pos) noexcept {
			v128 = vdupq_n_u8(uint8_t(1u << pos));
		}

		[[nodiscard]] constexpr static inline _Xint8x16_t set(
				uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_Xint8x16_t ret;
			ret.v32[0] = d;
			ret.v32[1] = c;
			ret.v32[2] = b;
			ret.v32[3] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint8x16_t setr(
				uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_Xint8x16_t ret;
			ret.v32[0] = a;
			ret.v32[1] = b;
			ret.v32[2] = c;
			ret.v32[3] = d;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint8x16_t set(
				const limb_type a, const limb_type b, const limb_type c, const limb_type d,
				const limb_type e, const limb_type f, const limb_type g, const limb_type h,
				const limb_type i, const limb_type j, const limb_type k, const limb_type l,
				const limb_type m, const limb_type n, const limb_type o, const limb_type p
		) noexcept {
			_Xint8x16_t ret;
			ret.v8[ 0] = p;
			ret.v8[ 1] = o;
			ret.v8[ 2] = n;
			ret.v8[ 3] = m;
			ret.v8[ 4] = l;
			ret.v8[ 5] = k;
			ret.v8[ 6] = j;
			ret.v8[ 7] = i;
			ret.v8[ 8] = h;
			ret.v8[ 9] = g;
			ret.v8[10] = f;
			ret.v8[11] = e;
			ret.v8[12] = d;
			ret.v8[13] = c;
			ret.v8[14] = b;
			ret.v8[15] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint8x16_t setr(
				const limb_type a, const limb_type b, const limb_type c, const limb_type d,
				const limb_type e, const limb_type f, const limb_type g, const limb_type h,
				const limb_type i, const limb_type j, const limb_type k, const limb_type l,
				const limb_type m, const limb_type n, const limb_type o, const limb_type p
		) noexcept {
			_Xint8x16_t ret;
			ret.v8[ 0] = a;
			ret.v8[ 1] = b;
			ret.v8[ 2] = c;
			ret.v8[ 3] = d;
			ret.v8[ 4] = e;
			ret.v8[ 5] = f;
			ret.v8[ 6] = g;
			ret.v8[ 7] = h;
			ret.v8[ 8] = i;
			ret.v8[ 9] = j;
			ret.v8[10] = k;
			ret.v8[11] = l;
			ret.v8[12] = m;
			ret.v8[13] = n;
			ret.v8[14] = o;
			ret.v8[15] = p;
			return ret;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline _Xint8x16_t load(const void *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint8x16_t aligned_load(const void *ptr) noexcept {
			_Xint8x16_t out;
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128 = vld1q_u8((const uint8_t *) ptr);
			return out;
		}

		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint8x16_t unaligned_load(const void *ptr) noexcept {
			_Xint8x16_t out;
			// byte-wise load: no alignment requirement
			out.v128 = vld1q_u8((const uint8_t *)ptr);
			return out;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(void *ptr,
										   const _Xint8x16_t in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(void *ptr,
												   const _Xint8x16_t in) noexcept {
			auto *ptr128 = (uint8x16_t *) ptr;
			*ptr128 = in.v128;
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr,
													 const _Xint8x16_t in) noexcept {
			// byte-wise store: no alignment requirement
			vst1q_u8((uint8_t *)ptr, in.v128);
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 ^ in2
	    [[nodiscard]] constexpr static inline S xor_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
        	if constexpr (__unsigned) {
        		out.v128 = veorq_u8(in1.v128, in2.v128);
        	} else {
        		out.v128 = veorq_s8(in1.v128, in2.v128);
        	}
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 & in2
	    [[nodiscard]] constexpr static inline S and_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
        	if constexpr (__unsigned) {
        		out.v128 = vandq_u8(in1.v128, in2.v128);
        	} else {
        		out.v128 = vandq_s8(in1.v128, in2.v128);
        	}
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 | in2
	    [[nodiscard]] constexpr static inline S or_(const S in1,
	                                                const S in2) noexcept {
	    	S out;
        	if constexpr (__unsigned) {
        		out.v128 = vorrq_u8(in1.v128, in2.v128);
        	} else {
        		out.v128 = vorrq_u8(in1.v128, in2.v128);
        	}
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return (~in1) & in2
	    [[nodiscard]] constexpr static inline S andnot(const S in1,
	                                                   const S in2) noexcept {
	    	S out;
        	if constexpr (__unsigned) {
        		out.v128 = vandq_u8(vmvnq_u8(in1.v128), in2.v128);
        	} else {
        		out.v128 = vandq_u8(vmvnq_u8(in1.v128), in2.v128);
        	}
			return out;
	    }

	    /// \param in1[in]: vector element
	    /// \return ~in1
	    [[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
	    	S out;
        	out.v128 = ~in1.v128;
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 + in2
	    [[nodiscard]] constexpr static inline S add(const S in1,
	                                                const S in2) noexcept {
	    	S out;
        	if constexpr (__unsigned) {
        		out.v128 = vaddq_u8(in1.v128, in2.v128);
        	} else {
        		out.v128 = vaddq_s8(in1.v128, in2.v128);
        	}
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 - in2
	    [[nodiscard]] constexpr static inline S sub(const S in1,
	                                                const S in2) noexcept {
	    	S out;
        	if constexpr (__unsigned) {
        		out.v128 = vsubq_u8(in1.v128, in2.v128);
        	} else {
        		out.v128 = vsubq_s8(in1.v128, in2.v128);
        	}
	    	return out;
	    }

	    /// 8 bit mul lo
	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1*in2
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const S in2) noexcept {
		    S out;
        	if constexpr (__unsigned) {
        		out.v128 = vmulq_u8(in1.v128, in2.v128);
        	} else {
        		out.v128 = vmulq_s8(in1.v128, in2.v128);
        	}
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const limb_type in2) noexcept {
        	S out;
        	// NOTE: there is no `vmulq_n_u8`/`vmulq_n_s8`, so broadcast `in2`;
        	// 	the lower 8 bits of the product do not depend on the sign
        	out.v128 = vmulq_u8(in1.v128, vdupq_n_u8((uint8_t)in2));
        	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]:
	    /// \return in1 / in2, lane-wise
	    [[nodiscard]] constexpr static inline S div(const S in1,
	                                                const limb_type in2) noexcept {
            if constexpr (__unsigned) {
                // NOTE: division by the invariant `in2`, see `internal::udiv_invariant`
                using VT = cryptanalysislib::internal::vec_t<limb_type, LIMBS>;
                S out;
                out.v128 = (uint8x16_t) cryptanalysislib::internal::udiv_invariant<limb_type, LIMBS>((VT) in1.v128, in2);
                return out;
            }

            S out;
            for (uint32_t i = 0; i < LIMBS; i++) {
                out[i] = in1[i] / in2;
            }
            return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 << in2
	    [[nodiscard]] constexpr static inline S slli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
	    	S out, t = S::set1(in2);
        	out.v128 = vshlq_u8(in1.v128, t.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
        	S out, t = S::set1(-in2);
        	out.v128 = vshlq_u8(in1.v128, t.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]:
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S ror(const S in1,
	                                                 const limb_type in2) noexcept {

	    	const uint8_t s = in2 % 8u;
	    	S out, t1 = S::set1(-s), t2 = S::set1(8u - s);
			out.v128 = vshlq_u8(in1.v128, t1.v128) ^ (vshlq_u8(in1.v128, t2.v128));
	    	return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]:
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S rol(const S in1,
	                                                 const uint8_t in2) noexcept {
	    	const uint8_t s = in2 % 8u;
	    	// left by s, and right by 8-s (negative shift count)
	    	S out, t1 = S::set1(s), t2 = S::set1(s - 8u);
	    	out.v128 = vshlq_u8(in1.v128, t1.v128) ^ vshlq_u8(in1.v128, t2.v128);
	    	return out;
	    }

		/// \param in1
		/// \param in2
		/// \return in1 > in2 uncompressed
		[[nodiscard]] constexpr static inline S gt_(const S in1,
													const S in2) noexcept {
	    	S out;
        	if constexpr (__unsigned) {
				out.v128 = vcgtq_u8(in1.v128, in2.v128);
        	} else {
        		out.v128 = vcgtq_s8(in1.v128, in2.v128);
        	}
        	return out;
		}

		/// \param in1
		/// \param in2
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t gt(const _Xint8x16_t in1,
													 	  const _Xint8x16_t in2) noexcept {
        	if constexpr (__unsigned) {
				return _mm_movemask_epi8(vcgtq_u8(in1.v128, in2.v128));
        	} else {
				return _mm_movemask_epi8(vcgtq_s8(in1.v128, in2.v128));
        	}
		}

		/// \param in1
		/// \param in2
		/// \return in1 > in2 uncompressed
		[[nodiscard]] constexpr static inline S ge_(const S in1,
													const S in2) noexcept {
	    	S out;
        	if constexpr (__unsigned) {
				out.v128 = vcgeq_u8(in1.v128, in2.v128);
        	} else {
        		out.v128 = vcgeq_s8(in1.v128, in2.v128);
        	}
        	return out;
		}

		/// \param in1
		/// \param in2
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t ge(const _Xint8x16_t in1,
													 	  const _Xint8x16_t in2) noexcept {
        	if constexpr (__unsigned) {
				return _mm_movemask_epi8(vcgeq_u8(in1.v128, in2.v128));
        	} else {
				return _mm_movemask_epi8(vcgeq_s8(in1.v128, in2.v128));
        	}
		}

		/// \param in1
		/// \param in2
		/// \return in1 < in2 uncompressed
		[[nodiscard]] constexpr static inline S lt_(const S in1,
													const S in2) noexcept {
	    	S out;
        	if constexpr (__unsigned) {
				out.v128 = vcltq_u8(in1.v128, in2.v128);
        	} else {
        		out.v128 = vcltq_s8(in1.v128, in2.v128);
        	}
        	return out;
		}

		/// NOTE: signed comparison
		/// \param in1
		/// \param in2
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t lt(const _Xint8x16_t in1,
													      const _Xint8x16_t in2) noexcept {
        	if constexpr (__unsigned) {
				return _mm_movemask_epi8(vcltq_u8(in1.v128, in2.v128));
        	} else {
				return _mm_movemask_epi8(vcltq_s8(in1.v128, in2.v128));
        	}
		}

		/// \param in1
		/// \param in2
		/// \return in1 < in2 uncompressed
		[[nodiscard]] constexpr static inline S le_(const S in1,
													const S in2) noexcept {
	    	S out;
        	if constexpr (__unsigned) {
				out.v128 = vcleq_u8(in1.v128, in2.v128);
        	} else {
        		out.v128 = vcleq_s8(in1.v128, in2.v128);
        	}
        	return out;
		}

		/// NOTE: signed comparison
		/// \param in1
		/// \param in2
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t le(const _Xint8x16_t in1,
													      const _Xint8x16_t in2) noexcept {
        	if constexpr (__unsigned) {
				return _mm_movemask_epi8(vcleq_u8(in1.v128, in2.v128));
        	} else {
				return _mm_movemask_epi8(vcleq_s8(in1.v128, in2.v128));
        	}
		}

		/// \param in1
		/// \param in2
		/// \return in1 == in2 ucompressed
		[[nodiscard]] constexpr static inline S eq_(const S in1,
													const S in2) noexcept {
	    	S out;
        	if constexpr (__unsigned) {
				out.v128 = vceqq_u8(in1.v128, in2.v128);
        	} else {
        		out.v128 = vceqq_s8(in1.v128, in2.v128);
        	}
        	return out;
		}

		///
		/// \param in1
		/// \param in2
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t eq(const _Xint8x16_t in1,
		                                                  const _Xint8x16_t in2) noexcept {
        	if constexpr (__unsigned) {
				return _mm_movemask_epi8(vceqq_u8(in1.v128, in2.v128));
        	} else {
				return _mm_movemask_epi8(vceqq_s8(in1.v128, in2.v128));
        	}
		}

        /// \param in1
	    /// \param in2
	    /// \return {-1, 0, 1}
	    [[nodiscard]] constexpr static inline S cmp_(const S in1,
	    											 const S in2) noexcept {
	    	S ret;
        	// lanes that differ (like the 256-bit types: lt ^ gt); equality does not depend on the sign
			ret.v128 = vmvnq_u8(vceqq_u8(in1.v128, in2.v128));
	    	return ret;
	    }

	    /// \param in1
	    /// \param in2
	    /// \return
	    [[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
	    											       const S in2) noexcept {
            S ret = S::cmp_(in1, in2);
	    	return _mm_movemask_epi8(ret.v128);
	    }

	    /// \param in[in]: vector element
		/// \return [popcnt(in[0]), ..., popcnt(in[7])]
	    [[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
	    	S ret;
        	if constexpr (__unsigned) {
        		ret.v128 = vcntq_u8(in.v128);
        	} else {
        		ret.v128 = vcntq_s8(in.v128);
        	}
	    	return ret;
	    }

	    [[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
        	// compare every lane with lane 0; all lanes equal <=> min of the comparison is 0xFF
        	const uint8x16_t b = vdupq_laneq_u8(in.v128, 0);
        	return vminvq_u8(vceqq_u8(in.v128, b)) == 0xFFu;
        }

        // just shuffle the 16 u8 elements
	    [[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
	    	S ret;
        	// reverse the bytes within each 64-bit half, then swap the halves
        	const uint8x16_t r = vrev64q_u8(in.v128);
        	ret.v128 = vextq_u8(r, r, 8);
	    	return ret;
        }

	    /// kmoves the msb into each bit
	    [[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
        	return _mm_movemask_epi8(in.v128);
	    }

        /// \tparam scale[in]:
        /// \param ptr[in]:
        /// \param data[in]:
        /// \return
	    template<const uint32_t scale = 1>
	    [[nodiscard]] constexpr static inline S gather(const limb_type *ptr,
	    											   const S data) noexcept {
	    	static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
	    	S ret;

	    	const uint8_t *ptr8 = (uint8_t *) ptr;
	    	for (uint32_t i = 0; i < S::LIMBS; i++) {
	    		ret.d[i] = ptr8[data.d[i] * scale];
	    	}
	    	return ret;
	    }

        /// \tparam scale[in]:
        /// \param ptr[in]:
        /// \param offset[in]:
        /// \param data[in]:
        /// \return
	    template<const uint32_t scale = 1>
	    constexpr static inline void scatter(const limb_type *ptr,
	    									 const S offset,
	    									 const S data) noexcept {
	    	static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
	    	uint8_t *ptr8 = (uint8_t *) ptr;
	    	for (uint32_t i = 0; i < S::LIMBS; i++) {
	    		*(ptr8 + offset.d[i] * scale) = data.d[i];
	    	}
	    }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [min(a[0], b[0]), ..., min(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S min(const S a,
                                                    const S b) noexcept {
            S c;
        	if constexpr (__unsigned) {
        		c.v128 = vminq_u8(a.v128, b.v128);
        	} else {
        		c.v128 = vminq_s8(a.v128, b.v128);
        	}
            return c;
        }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [max(a[0], b[0]), ..., max(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S max(const S a,
                                                    const S b) noexcept {
            S c;
        	if constexpr (__unsigned) {
        		c.v128 = vmaxq_u8(a.v128, b.v128);
        	} else {
        		c.v128 = vmaxq_s8(a.v128, b.v128);
        	}
            return c;
        }
	};

    template<const bool __unsigned>
	struct _Xint16x8_t {
		constexpr static uint32_t LIMBS = 8;
		using limb_type = std::conditional_t<__unsigned, uint16_t, int16_t>;
		using S = _Xint16x8_t;

        constexpr inline _Xint16x8_t operator=(const _Xint8x16_t<> &b) noexcept;
		constexpr inline _Xint16x8_t operator=(const _Xint32x4_t<> &b) noexcept;
		constexpr inline _Xint16x8_t operator=(const _Xint64x2_t<> &b) noexcept;

		constexpr _Xint16x8_t() noexcept = default;
		constexpr _Xint16x8_t(const _Xint8x16_t<> &b) noexcept;
		constexpr _Xint16x8_t(const _Xint32x4_t<> &b) noexcept;
		constexpr _Xint16x8_t(const _Xint64x2_t<> &b) noexcept;

		union {
			// compatibility to `TxN_t`
			limb_type d[8];

			uint8_t v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];
			uint16x8_t v128;
		};

		[[nodiscard]] constexpr inline limb_type& operator[](const uint32_t i) noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		///
		/// \return
		static inline _Xint16x8_t random() noexcept {
			_Xint16x8_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = rng();
			}

			return ret;
		}

		/// \param binary[in]:
		/// \param hex[in]:
		constexpr inline void print(bool binary = false,
		                            bool hex = false) const;

		[[nodiscard]] constexpr static inline _Xint16x8_t set1(const uint16_t a) noexcept {
			_Xint16x8_t ret;
			for (uint32_t i = 0; i < 8; ++i) {
				ret.v16[i] = a;
			}
			return ret;
		}

		/// set the `i`-th bit in each limb
		/// \param pos[in]: bit position to set
		constexpr inline void set_bit(const uint32_t pos) noexcept {
			v128 = vdupq_n_u16(uint16_t(1u << pos));
		}

		[[nodiscard]] constexpr static inline _Xint16x8_t set(
				uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_Xint16x8_t ret;
			ret.v32[0] = d;
			ret.v32[1] = c;
			ret.v32[2] = b;
			ret.v32[3] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint16x8_t setr(
				uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_Xint16x8_t ret;
			ret.v32[0] = a;
			ret.v32[1] = b;
			ret.v32[2] = c;
			ret.v32[3] = d;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint16x8_t set(
				uint16_t a, uint16_t b, uint16_t c, uint16_t d,
				uint16_t e, uint16_t f, uint16_t g, uint16_t h) noexcept {
			_Xint16x8_t ret;
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

		[[nodiscard]] constexpr static inline _Xint16x8_t setr(
				uint16_t a, uint16_t b, uint16_t c, uint16_t d,
				uint16_t e, uint16_t f, uint16_t g, uint16_t h) noexcept {
			_Xint16x8_t ret;
			ret.v16[0] = a;
			ret.v16[1] = b;
			ret.v16[2] = c;
			ret.v16[3] = d;
			ret.v16[4] = e;
			ret.v16[5] = f;
			ret.v16[6] = g;
			ret.v16[7] = h;
			return ret;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline _Xint16x8_t load(const void *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint16x8_t aligned_load(const void *ptr) noexcept {
			_Xint16x8_t out;
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128 = vreinterpretq_u16_u8(vld1q_u8((const uint8_t *) ptr));
			return out;
		}


		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint16x8_t unaligned_load(const void *ptr) noexcept {
			_Xint16x8_t out;
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128 = vreinterpretq_u16_u8(vld1q_u8((const uint8_t *) ptr));
			return out;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(void *ptr, const _Xint16x8_t in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(void *ptr, const _Xint16x8_t in) noexcept {
			// byte-wise store: alias-safe for any `ptr`
			vst1q_u8((uint8_t *)ptr, vreinterpretq_u8_u16(in.v128));
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr, const _Xint16x8_t in) noexcept {
			// byte-wise store: no alignment requirement
			vst1q_u8((uint8_t *)ptr, vreinterpretq_u8_u16(in.v128));
		}

		[[nodiscard]] constexpr inline static size_t size() noexcept {
			return LIMBS;
		}

		[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
			return __unsigned;
		}

		/// \return in1 ^ in2
		[[nodiscard]] constexpr static inline S xor_(const S in1,
		                                             const S in2) noexcept {
			S out;
			out.v128 = veorq_u16(in1.v128, in2.v128);
			return out;
		}

		/// \return in1 & in2
		[[nodiscard]] constexpr static inline S and_(const S in1,
		                                             const S in2) noexcept {
			S out;
			out.v128 = vandq_u16(in1.v128, in2.v128);
			return out;
		}

		/// \return in1 | in2
		[[nodiscard]] constexpr static inline S or_(const S in1,
		                                            const S in2) noexcept {
			S out;
			out.v128 = vorrq_u16(in1.v128, in2.v128);
			return out;
		}

		/// \return (~in1) & in2
		[[nodiscard]] constexpr static inline S andnot(const S in1,
		                                               const S in2) noexcept {
			S out;
			out.v128 = vbicq_u16(in2.v128, in1.v128);
			return out;
		}

		/// \return ~in1
		[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
			S out;
			out.v128 = ~in1.v128;
			return out;
		}

		/// \return in1 + in2 (lane-wise, wrap around)
		[[nodiscard]] constexpr static inline S add(const S in1,
		                                            const S in2) noexcept {
			S out;
			out.v128 = vaddq_u16(in1.v128, in2.v128);
			return out;
		}

		/// \return in1 - in2 (lane-wise, wrap around)
		[[nodiscard]] constexpr static inline S sub(const S in1,
		                                            const S in2) noexcept {
			S out;
			out.v128 = vsubq_u16(in1.v128, in2.v128);
			return out;
		}

		/// \return lower 16 bits of in1 * in2 (lane-wise)
		[[nodiscard]] constexpr static inline S mullo(const S in1,
		                                              const S in2) noexcept {
			S out;
			out.v128 = vmulq_u16(in1.v128, in2.v128);
			return out;
		}

		/// \return lower 16 bits of in1 * in2 (lane-wise)
		[[nodiscard]] constexpr static inline S mullo(const S in1,
		                                              const limb_type in2) noexcept {
			return S::mullo(in1, S::set1(in2));
		}

		/// \return in1 / in2 (lane-wise)
		[[nodiscard]] constexpr static inline S div(const S in1,
		                                            const limb_type in2) noexcept {
			if constexpr (__unsigned) {
				// NOTE: division by the invariant `in2`, see `internal::udiv_invariant`
				using VT = cryptanalysislib::internal::vec_t<limb_type, LIMBS>;
				S out;
				out.v128 = (uint16x8_t) cryptanalysislib::internal::udiv_invariant<limb_type, LIMBS>((VT) in1.v128, in2);
				return out;
			}

			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out.d[i] = in1.d[i] / in2;
			}
			return out;
		}

		/// \return in1 << in2 (lane-wise, 0 if in2 == 16)
		[[nodiscard]] constexpr static inline S slli(const S in1,
		                                             const uint8_t in2) noexcept {
			assert(in2 <= 16);
			S out;
			out.v128 = vshlq_u16(in1.v128, vdupq_n_s16((int16_t) in2));
			return out;
		}

		/// \return in1 >> in2 (logical, lane-wise, 0 if in2 == 16)
		[[nodiscard]] constexpr static inline S srli(const S in1,
		                                             const uint8_t in2) noexcept {
			assert(in2 <= 16);
			S out;
			out.v128 = vshlq_u16(in1.v128, vdupq_n_s16(-(int16_t) in2));
			return out;
		}

		/// \return in1 rotated right by in2 (lane-wise)
		[[nodiscard]] constexpr static inline S ror(const S in1,
		                                            const uint8_t in2) noexcept {
			const uint32_t s = in2 % 16u;
			if (s == 0) {
				return in1;
			}
			return S::or_(S::srli(in1, s), S::slli(in1, 16u - s));
		}

		/// \return in1 rotated left by in2 (lane-wise)
		[[nodiscard]] constexpr static inline S rol(const S in1,
		                                            const uint8_t in2) noexcept {
			const uint32_t s = in2 % 16u;
			if (s == 0) {
				return in1;
			}
			return S::or_(S::slli(in1, s), S::srli(in1, 16u - s));
		}

		/// lane-wise comparisons; signed lanes are compared as signed values
		[[nodiscard]] constexpr static inline uint16x8_t cmpgt128(const uint16x8_t a, const uint16x8_t b) noexcept {
			if constexpr (__unsigned) {
				return vcgtq_u16(a, b);
			} else {
				return vcgtq_s16(vreinterpretq_s16_u16(a), vreinterpretq_s16_u16(b));
			}
		}
		[[nodiscard]] constexpr static inline uint16x8_t cmpge128(const uint16x8_t a, const uint16x8_t b) noexcept {
			if constexpr (__unsigned) {
				return vcgeq_u16(a, b);
			} else {
				return vcgeq_s16(vreinterpretq_s16_u16(a), vreinterpretq_s16_u16(b));
			}
		}

		/// \return in1 > in2 uncompressed
		[[nodiscard]] constexpr static inline S gt_(const S in1, const S in2) noexcept {
			S out; out.v128 = cmpgt128(in1.v128, in2.v128); return out;
		}
		/// \return in1 >= in2 uncompressed
		[[nodiscard]] constexpr static inline S ge_(const S in1, const S in2) noexcept {
			S out; out.v128 = cmpge128(in1.v128, in2.v128); return out;
		}
		/// \return in1 < in2 uncompressed
		[[nodiscard]] constexpr static inline S lt_(const S in1, const S in2) noexcept {
			S out; out.v128 = cmpgt128(in2.v128, in1.v128); return out;
		}
		/// \return in1 <= in2 uncompressed
		[[nodiscard]] constexpr static inline S le_(const S in1, const S in2) noexcept {
			S out; out.v128 = cmpge128(in2.v128, in1.v128); return out;
		}
		/// \return in1 == in2 uncompressed
		[[nodiscard]] constexpr static inline S eq_(const S in1, const S in2) noexcept {
			S out; out.v128 = vceqq_u16(in1.v128, in2.v128); return out;
		}
		/// \return in1 != in2 uncompressed (-1 if the limbs differ, 0 otherwise)
		[[nodiscard]] constexpr static inline S cmp_(const S in1, const S in2) noexcept {
			S out; out.v128 = ~vceqq_u16(in1.v128, in2.v128); return out;
		}

		/// \return in1 > in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t gt(const S in1, const S in2) noexcept {
			return _mm_movemask_epi16(cmpgt128(in1.v128, in2.v128));
		}
		/// \return in1 >= in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t ge(const S in1, const S in2) noexcept {
			return _mm_movemask_epi16(cmpge128(in1.v128, in2.v128));
		}
		/// \return in1 < in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t lt(const S in1, const S in2) noexcept {
			return _mm_movemask_epi16(cmpgt128(in2.v128, in1.v128));
		}
		/// \return in1 <= in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t le(const S in1, const S in2) noexcept {
			return _mm_movemask_epi16(cmpge128(in2.v128, in1.v128));
		}
		/// \return in1 == in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t eq(const S in1, const S in2) noexcept {
			return _mm_movemask_epi16(vceqq_u16(in1.v128, in2.v128));
		}
		/// \return in1 != in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t cmp(const S in1, const S in2) noexcept {
			return _mm_movemask_epi16(S::cmp_(in1, in2).v128);
		}

		/// \return the most significant bit of every lane
		[[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
			return _mm_movemask_epi16(in.v128);
		}

		/// \return [popcnt(in[0]), ..., popcnt(in[7])]
		[[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
			S out;
			// pairwise widening adds of the byte counts
			const uint8x16_t c8 = vcntq_u8(vreinterpretq_u8_u16(in.v128));
			out.v128 = vpaddlq_u8(c8);
			return out;
		}

		/// \return true if all lanes of `in` are equal
		[[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
			for (uint32_t i = 1; i < LIMBS; ++i) {
				if (in.d[0] != in.d[i]) {
					return false;
				}
			}
			return true;
		}

		/// \return the lanes in reversed order
		[[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out.d[LIMBS - 1 - i] = in.d[i];
			}
			return out;
		}

		/// \return lane-wise minimum
		[[nodiscard]] constexpr static inline S min(const S a, const S b) noexcept {
			S out;
			out.v128 = vbslq_u16(cmpgt128(a.v128, b.v128), b.v128, a.v128);
			return out;
		}

		/// \return lane-wise maximum
		[[nodiscard]] constexpr static inline S max(const S a, const S b) noexcept {
			S out;
			out.v128 = vbslq_u16(cmpgt128(a.v128, b.v128), a.v128, b.v128);
			return out;
		}

		/// NOTE: NEON has no gather instruction, so this is a scalar loop
		/// \return ret[i] = limb at byte offset data[i] * scale of `ptr`
		template<const uint32_t scale = 1>
		[[nodiscard]] constexpr static inline S gather(const void *ptr,
		                                               const S data) noexcept {
			static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
			S ret;
			const uint8_t *ptr8 = (const uint8_t *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				// NOTE: byte offsets, so the address may not be aligned
				__builtin_memcpy(&ret.d[i], ptr8 + data.d[i] * scale, sizeof(limb_type));
			}
			return ret;
		}

		/// writes data[i] to the byte offset offset[i] * scale of `ptr`
		template<const uint32_t scale = 1>
		constexpr static inline void scatter(void *ptr,
		                                     const S offset,
		                                     const S data) noexcept {
			static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
			uint8_t *ptr8 = (uint8_t *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				__builtin_memcpy(ptr8 + offset.d[i] * scale, &data.d[i], sizeof(limb_type));
			}
		}
	};

    template<const bool __unsigned>
	struct _Xint32x4_t {
		constexpr static uint32_t LIMBS = 4;
		using limb_type = std::conditional_t<__unsigned, uint32_t, int32_t>;
	    using S = _Xint32x4_t;

		constexpr inline _Xint32x4_t operator=(const _Xint8x16_t<> &b) noexcept;
		constexpr inline _Xint32x4_t operator=(const _Xint16x8_t<> &b) noexcept;
		constexpr inline _Xint32x4_t operator=(const _Xint64x2_t<> &b) noexcept;

		constexpr _Xint32x4_t() noexcept = default;
		constexpr _Xint32x4_t(const _Xint8x16_t<> &b) noexcept;
		constexpr _Xint32x4_t(const _Xint16x8_t<> &b) noexcept;
		constexpr _Xint32x4_t(const _Xint64x2_t<> &b) noexcept;

		union {
			// compatibility to `TxN_t`
			limb_type d[4];

			uint8_t v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];
			uint32x4_t v128;
		};

		[[nodiscard]] constexpr inline limb_type& operator[](const uint32_t i) noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		///
		/// \return
		static inline _Xint32x4_t random() noexcept {
			_Xint32x4_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = rng();
			}

			return ret;
		}

		/// \param binary[in]:
		/// \param hex[in]:
		constexpr inline void print(bool binary = false,
		                            bool hex = false) const;

		[[nodiscard]] constexpr static inline _Xint32x4_t set1(const uint32_t a) noexcept {
			_Xint32x4_t ret;
			for (uint32_t i = 0; i < 4; ++i) {
				ret.v32[i] = a;
			}
			return ret;
		}

		/// set the `i`-th bit in each limb
		/// \param pos[in]: bit position to set
		constexpr inline void set_bit(const uint32_t pos) noexcept {
			v128 = vdupq_n_u32(uint32_t(1u) << pos);
		}

		[[nodiscard]] constexpr static inline _Xint32x4_t set(
				uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_Xint32x4_t ret;
			ret.v32[0] = d;
			ret.v32[1] = c;
			ret.v32[2] = b;
			ret.v32[3] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint32x4_t setr(
				uint32_t a, uint32_t b, uint32_t c, uint32_t d) noexcept {
			_Xint32x4_t ret;
			ret.v32[0] = a;
			ret.v32[1] = b;
			ret.v32[2] = c;
			ret.v32[3] = d;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint32x4_t set(
				uint64_t a, uint64_t b) noexcept {
			_Xint32x4_t ret;
			ret.v64[0] = b;
			ret.v64[1] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint32x4_t setr(
				uint64_t a, uint64_t b) noexcept {
			_Xint32x4_t ret;
			ret.v64[0] = a;
			ret.v64[1] = b;
			return ret;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline _Xint32x4_t load(const void *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint32x4_t aligned_load(const void *ptr) noexcept {
			_Xint32x4_t out;
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128 = vreinterpretq_u32_u8(vld1q_u8((const uint8_t *) ptr));
			return out;
		}


		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint32x4_t unaligned_load(const void *ptr) noexcept {
			_Xint32x4_t out;
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128 = vreinterpretq_u32_u8(vld1q_u8((const uint8_t *) ptr));
			return out;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(void *ptr, const _Xint32x4_t in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(void *ptr, const _Xint32x4_t in) noexcept {
			// byte-wise store: alias-safe for any `ptr`
			vst1q_u8((uint8_t *)ptr, vreinterpretq_u8_u32(in.v128));
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr, const _Xint32x4_t in) noexcept {
			// byte-wise store: no alignment requirement
			vst1q_u8((uint8_t *)ptr, vreinterpretq_u8_u32(in.v128));
		}

		[[nodiscard]] constexpr inline static size_t size() noexcept {
			return LIMBS;
		}

		[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
			return __unsigned;
		}

		/// \return in1 ^ in2
		[[nodiscard]] constexpr static inline S xor_(const S in1,
		                                             const S in2) noexcept {
			S out;
			out.v128 = veorq_u32(in1.v128, in2.v128);
			return out;
		}

		/// \return in1 & in2
		[[nodiscard]] constexpr static inline S and_(const S in1,
		                                             const S in2) noexcept {
			S out;
			out.v128 = vandq_u32(in1.v128, in2.v128);
			return out;
		}

		/// \return in1 | in2
		[[nodiscard]] constexpr static inline S or_(const S in1,
		                                            const S in2) noexcept {
			S out;
			out.v128 = vorrq_u32(in1.v128, in2.v128);
			return out;
		}

		/// \return (~in1) & in2
		[[nodiscard]] constexpr static inline S andnot(const S in1,
		                                               const S in2) noexcept {
			S out;
			out.v128 = vbicq_u32(in2.v128, in1.v128);
			return out;
		}

		/// \return ~in1
		[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
			S out;
			out.v128 = ~in1.v128;
			return out;
		}

		/// \return in1 + in2 (lane-wise, wrap around)
		[[nodiscard]] constexpr static inline S add(const S in1,
		                                            const S in2) noexcept {
			S out;
			out.v128 = vaddq_u32(in1.v128, in2.v128);
			return out;
		}

		/// \return in1 - in2 (lane-wise, wrap around)
		[[nodiscard]] constexpr static inline S sub(const S in1,
		                                            const S in2) noexcept {
			S out;
			out.v128 = vsubq_u32(in1.v128, in2.v128);
			return out;
		}

		/// \return lower 32 bits of in1 * in2 (lane-wise)
		[[nodiscard]] constexpr static inline S mullo(const S in1,
		                                              const S in2) noexcept {
			S out;
			out.v128 = vmulq_u32(in1.v128, in2.v128);
			return out;
		}

		/// \return lower 32 bits of in1 * in2 (lane-wise)
		[[nodiscard]] constexpr static inline S mullo(const S in1,
		                                              const limb_type in2) noexcept {
			return S::mullo(in1, S::set1(in2));
		}

		/// \return in1 / in2 (lane-wise)
		[[nodiscard]] constexpr static inline S div(const S in1,
		                                            const limb_type in2) noexcept {
			if constexpr (__unsigned) {
				// NOTE: division by the invariant `in2`, see `internal::udiv_invariant`
				using VT = cryptanalysislib::internal::vec_t<limb_type, LIMBS>;
				S out;
				out.v128 = (uint32x4_t) cryptanalysislib::internal::udiv_invariant<limb_type, LIMBS>((VT) in1.v128, in2);
				return out;
			}

			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out.d[i] = in1.d[i] / in2;
			}
			return out;
		}

		/// \return in1 << in2 (lane-wise, 0 if in2 == 32)
		[[nodiscard]] constexpr static inline S slli(const S in1,
		                                             const uint8_t in2) noexcept {
			assert(in2 <= 32);
			S out;
			out.v128 = vshlq_u32(in1.v128, vdupq_n_s32((int32_t) in2));
			return out;
		}

		/// \return in1 >> in2 (logical, lane-wise, 0 if in2 == 32)
		[[nodiscard]] constexpr static inline S srli(const S in1,
		                                             const uint8_t in2) noexcept {
			assert(in2 <= 32);
			S out;
			out.v128 = vshlq_u32(in1.v128, vdupq_n_s32(-(int32_t) in2));
			return out;
		}

		/// \return in1 rotated right by in2 (lane-wise)
		[[nodiscard]] constexpr static inline S ror(const S in1,
		                                            const uint8_t in2) noexcept {
			const uint32_t s = in2 % 32u;
			if (s == 0) {
				return in1;
			}
			return S::or_(S::srli(in1, s), S::slli(in1, 32u - s));
		}

		/// \return in1 rotated left by in2 (lane-wise)
		[[nodiscard]] constexpr static inline S rol(const S in1,
		                                            const uint8_t in2) noexcept {
			const uint32_t s = in2 % 32u;
			if (s == 0) {
				return in1;
			}
			return S::or_(S::slli(in1, s), S::srli(in1, 32u - s));
		}

		/// lane-wise comparisons; signed lanes are compared as signed values
		[[nodiscard]] constexpr static inline uint32x4_t cmpgt128(const uint32x4_t a, const uint32x4_t b) noexcept {
			if constexpr (__unsigned) {
				return vcgtq_u32(a, b);
			} else {
				return vcgtq_s32(vreinterpretq_s32_u32(a), vreinterpretq_s32_u32(b));
			}
		}
		[[nodiscard]] constexpr static inline uint32x4_t cmpge128(const uint32x4_t a, const uint32x4_t b) noexcept {
			if constexpr (__unsigned) {
				return vcgeq_u32(a, b);
			} else {
				return vcgeq_s32(vreinterpretq_s32_u32(a), vreinterpretq_s32_u32(b));
			}
		}

		/// \return in1 > in2 uncompressed
		[[nodiscard]] constexpr static inline S gt_(const S in1, const S in2) noexcept {
			S out; out.v128 = cmpgt128(in1.v128, in2.v128); return out;
		}
		/// \return in1 >= in2 uncompressed
		[[nodiscard]] constexpr static inline S ge_(const S in1, const S in2) noexcept {
			S out; out.v128 = cmpge128(in1.v128, in2.v128); return out;
		}
		/// \return in1 < in2 uncompressed
		[[nodiscard]] constexpr static inline S lt_(const S in1, const S in2) noexcept {
			S out; out.v128 = cmpgt128(in2.v128, in1.v128); return out;
		}
		/// \return in1 <= in2 uncompressed
		[[nodiscard]] constexpr static inline S le_(const S in1, const S in2) noexcept {
			S out; out.v128 = cmpge128(in2.v128, in1.v128); return out;
		}
		/// \return in1 == in2 uncompressed
		[[nodiscard]] constexpr static inline S eq_(const S in1, const S in2) noexcept {
			S out; out.v128 = vceqq_u32(in1.v128, in2.v128); return out;
		}
		/// \return in1 != in2 uncompressed (-1 if the limbs differ, 0 otherwise)
		[[nodiscard]] constexpr static inline S cmp_(const S in1, const S in2) noexcept {
			S out; out.v128 = ~vceqq_u32(in1.v128, in2.v128); return out;
		}

		/// \return in1 > in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t gt(const S in1, const S in2) noexcept {
			return _mm_movemask_epi32(cmpgt128(in1.v128, in2.v128));
		}
		/// \return in1 >= in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t ge(const S in1, const S in2) noexcept {
			return _mm_movemask_epi32(cmpge128(in1.v128, in2.v128));
		}
		/// \return in1 < in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t lt(const S in1, const S in2) noexcept {
			return _mm_movemask_epi32(cmpgt128(in2.v128, in1.v128));
		}
		/// \return in1 <= in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t le(const S in1, const S in2) noexcept {
			return _mm_movemask_epi32(cmpge128(in2.v128, in1.v128));
		}
		/// \return in1 == in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t eq(const S in1, const S in2) noexcept {
			return _mm_movemask_epi32(vceqq_u32(in1.v128, in2.v128));
		}
		/// \return in1 != in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t cmp(const S in1, const S in2) noexcept {
			return _mm_movemask_epi32(S::cmp_(in1, in2).v128);
		}

		/// \return the most significant bit of every lane
		[[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
			return _mm_movemask_epi32(in.v128);
		}

		/// \return [popcnt(in[0]), ..., popcnt(in[3])]
		[[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
			S out;
			// pairwise widening adds of the byte counts
			const uint8x16_t c8 = vcntq_u8(vreinterpretq_u8_u32(in.v128));
			out.v128 = vpaddlq_u16(vpaddlq_u8(c8));
			return out;
		}

		/// \return true if all lanes of `in` are equal
		[[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
			for (uint32_t i = 1; i < LIMBS; ++i) {
				if (in.d[0] != in.d[i]) {
					return false;
				}
			}
			return true;
		}

		/// \return the lanes in reversed order
		[[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out.d[LIMBS - 1 - i] = in.d[i];
			}
			return out;
		}

		/// \return lane-wise minimum
		[[nodiscard]] constexpr static inline S min(const S a, const S b) noexcept {
			S out;
			out.v128 = vbslq_u32(cmpgt128(a.v128, b.v128), b.v128, a.v128);
			return out;
		}

		/// \return lane-wise maximum
		[[nodiscard]] constexpr static inline S max(const S a, const S b) noexcept {
			S out;
			out.v128 = vbslq_u32(cmpgt128(a.v128, b.v128), a.v128, b.v128);
			return out;
		}

		/// NOTE: NEON has no gather instruction, so this is a scalar loop
		/// \return ret[i] = limb at byte offset data[i] * scale of `ptr`
		template<const uint32_t scale = 1>
		[[nodiscard]] constexpr static inline S gather(const void *ptr,
		                                               const S data) noexcept {
			static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
			S ret;
			const uint8_t *ptr8 = (const uint8_t *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				// NOTE: byte offsets, so the address may not be aligned
				__builtin_memcpy(&ret.d[i], ptr8 + data.d[i] * scale, sizeof(limb_type));
			}
			return ret;
		}

		/// writes data[i] to the byte offset offset[i] * scale of `ptr`
		template<const uint32_t scale = 1>
		constexpr static inline void scatter(void *ptr,
		                                     const S offset,
		                                     const S data) noexcept {
			static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
			uint8_t *ptr8 = (uint8_t *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				__builtin_memcpy(ptr8 + offset.d[i] * scale, &data.d[i], sizeof(limb_type));
			}
		}
	};

    template<const bool __unsigned>
	struct _Xint64x2_t {
		constexpr static uint32_t LIMBS = 2;
		using limb_type = std::conditional_t<__unsigned, uint64_t, int64_t>;
		using S = _Xint64x2_t;

		constexpr inline _Xint64x2_t operator=(const _Xint8x16_t<> &b) noexcept;
		constexpr inline _Xint64x2_t operator=(const _Xint16x8_t<> &b) noexcept;
		constexpr inline _Xint64x2_t operator=(const _Xint32x4_t<> &b) noexcept;

		constexpr _Xint64x2_t() noexcept = default;
		constexpr _Xint64x2_t(const _Xint8x16_t<> &b) noexcept;
		constexpr _Xint64x2_t(const _Xint16x8_t<> &b) noexcept;
		constexpr _Xint64x2_t(const _Xint32x4_t<> &b) noexcept;

		union {
			limb_type d[2];

			uint8_t v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];
			uint64x2_t v128;
		};

		[[nodiscard]] constexpr inline limb_type& operator[](const uint32_t i) noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		///
		/// \return
		static inline _Xint64x2_t random() noexcept {
			_Xint64x2_t ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = rng();
			}

			return ret;
		}

		/// \param binary[in]:
		/// \param hex[in]:
		constexpr inline void print(bool binary = false,
		                            bool hex = false) const;

		[[nodiscard]] constexpr static inline _Xint64x2_t set1(const uint64_t a) noexcept {
			_Xint64x2_t ret;
			for (uint32_t i = 0; i < 2; ++i) {
				ret.v64[i] = a;
			}
			return ret;
		}

		/// set the `i`-th bit in each limb
		/// \param pos[in]: bit position to set
		constexpr inline void set_bit(const uint32_t pos) noexcept {
			v128 = vdupq_n_u64(uint64_t(1u) << pos);
		}

		[[nodiscard]] constexpr static inline _Xint64x2_t set(
				uint64_t a, uint64_t b) noexcept{
			_Xint64x2_t ret;
			ret.v64[0] = b;
			ret.v64[1] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint64x2_t setr(
				uint64_t a, uint64_t b) noexcept {
			_Xint64x2_t ret;
			ret.v64[0] = a;
			ret.v64[1] = b;
			return ret;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline _Xint64x2_t load(const void *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint64x2_t aligned_load(const void *ptr) noexcept {
			_Xint64x2_t out;
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128 = vreinterpretq_u64_u8(vld1q_u8((const uint8_t *) ptr));
			return out;
		}


		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint64x2_t unaligned_load(const void *ptr) noexcept {
			_Xint64x2_t out;
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128 = vreinterpretq_u64_u8(vld1q_u8((const uint8_t *) ptr));
			return out;
		}

		///
		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(void *ptr, const _Xint64x2_t in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(void *ptr,
												   const _Xint64x2_t in) noexcept {
			// byte-wise store: alias-safe for any `ptr`
			vst1q_u8((uint8_t *)ptr, vreinterpretq_u8_u64(in.v128));
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr,
													 const _Xint64x2_t in) noexcept {
			// byte-wise store: no alignment requirement
			vst1q_u8((uint8_t *)ptr, vreinterpretq_u8_u64(in.v128));
		}

		[[nodiscard]] constexpr inline static size_t size() noexcept {
			return LIMBS;
		}

		[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
			return __unsigned;
		}

		/// \return in1 ^ in2
		[[nodiscard]] constexpr static inline S xor_(const S in1,
		                                             const S in2) noexcept {
			S out;
			out.v128 = veorq_u64(in1.v128, in2.v128);
			return out;
		}

		/// \return in1 & in2
		[[nodiscard]] constexpr static inline S and_(const S in1,
		                                             const S in2) noexcept {
			S out;
			out.v128 = vandq_u64(in1.v128, in2.v128);
			return out;
		}

		/// \return in1 | in2
		[[nodiscard]] constexpr static inline S or_(const S in1,
		                                            const S in2) noexcept {
			S out;
			out.v128 = vorrq_u64(in1.v128, in2.v128);
			return out;
		}

		/// \return (~in1) & in2
		[[nodiscard]] constexpr static inline S andnot(const S in1,
		                                               const S in2) noexcept {
			S out;
			out.v128 = vbicq_u64(in2.v128, in1.v128);
			return out;
		}

		/// \return ~in1
		[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
			S out;
			out.v128 = ~in1.v128;
			return out;
		}

		/// \return in1 + in2 (lane-wise, wrap around)
		[[nodiscard]] constexpr static inline S add(const S in1,
		                                            const S in2) noexcept {
			S out;
			out.v128 = vaddq_u64(in1.v128, in2.v128);
			return out;
		}

		/// \return in1 - in2 (lane-wise, wrap around)
		[[nodiscard]] constexpr static inline S sub(const S in1,
		                                            const S in2) noexcept {
			S out;
			out.v128 = vsubq_u64(in1.v128, in2.v128);
			return out;
		}

		/// \return lower 64 bits of in1 * in2 (lane-wise)
		[[nodiscard]] constexpr static inline S mullo(const S in1,
		                                              const S in2) noexcept {
			S out;
			// NOTE: NEON has no 64 bit lane multiplication
			for (uint32_t i = 0; i < LIMBS; i++) {
				out.v64[i] = in1.v64[i] * in2.v64[i];
			}
			return out;
		}

		/// \return lower 64 bits of in1 * in2 (lane-wise)
		[[nodiscard]] constexpr static inline S mullo(const S in1,
		                                              const limb_type in2) noexcept {
			return S::mullo(in1, S::set1(in2));
		}

		/// \return in1 / in2 (lane-wise)
		[[nodiscard]] constexpr static inline S div(const S in1,
		                                            const limb_type in2) noexcept {
			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out.d[i] = in1.d[i] / in2;
			}
			return out;
		}

		/// \return in1 << in2 (lane-wise, 0 if in2 == 64)
		[[nodiscard]] constexpr static inline S slli(const S in1,
		                                             const uint8_t in2) noexcept {
			assert(in2 <= 64);
			S out;
			out.v128 = vshlq_u64(in1.v128, vdupq_n_s64((int64_t) in2));
			return out;
		}

		/// \return in1 >> in2 (logical, lane-wise, 0 if in2 == 64)
		[[nodiscard]] constexpr static inline S srli(const S in1,
		                                             const uint8_t in2) noexcept {
			assert(in2 <= 64);
			S out;
			out.v128 = vshlq_u64(in1.v128, vdupq_n_s64(-(int64_t) in2));
			return out;
		}

		/// \return in1 rotated right by in2 (lane-wise)
		[[nodiscard]] constexpr static inline S ror(const S in1,
		                                            const uint8_t in2) noexcept {
			const uint32_t s = in2 % 64u;
			if (s == 0) {
				return in1;
			}
			return S::or_(S::srli(in1, s), S::slli(in1, 64u - s));
		}

		/// \return in1 rotated left by in2 (lane-wise)
		[[nodiscard]] constexpr static inline S rol(const S in1,
		                                            const uint8_t in2) noexcept {
			const uint32_t s = in2 % 64u;
			if (s == 0) {
				return in1;
			}
			return S::or_(S::slli(in1, s), S::srli(in1, 64u - s));
		}

		/// lane-wise comparisons; signed lanes are compared as signed values
		[[nodiscard]] constexpr static inline uint64x2_t cmpgt128(const uint64x2_t a, const uint64x2_t b) noexcept {
			if constexpr (__unsigned) {
				return vcgtq_u64(a, b);
			} else {
				return vcgtq_s64(vreinterpretq_s64_u64(a), vreinterpretq_s64_u64(b));
			}
		}
		[[nodiscard]] constexpr static inline uint64x2_t cmpge128(const uint64x2_t a, const uint64x2_t b) noexcept {
			if constexpr (__unsigned) {
				return vcgeq_u64(a, b);
			} else {
				return vcgeq_s64(vreinterpretq_s64_u64(a), vreinterpretq_s64_u64(b));
			}
		}

		/// \return in1 > in2 uncompressed
		[[nodiscard]] constexpr static inline S gt_(const S in1, const S in2) noexcept {
			S out; out.v128 = cmpgt128(in1.v128, in2.v128); return out;
		}
		/// \return in1 >= in2 uncompressed
		[[nodiscard]] constexpr static inline S ge_(const S in1, const S in2) noexcept {
			S out; out.v128 = cmpge128(in1.v128, in2.v128); return out;
		}
		/// \return in1 < in2 uncompressed
		[[nodiscard]] constexpr static inline S lt_(const S in1, const S in2) noexcept {
			S out; out.v128 = cmpgt128(in2.v128, in1.v128); return out;
		}
		/// \return in1 <= in2 uncompressed
		[[nodiscard]] constexpr static inline S le_(const S in1, const S in2) noexcept {
			S out; out.v128 = cmpge128(in2.v128, in1.v128); return out;
		}
		/// \return in1 == in2 uncompressed
		[[nodiscard]] constexpr static inline S eq_(const S in1, const S in2) noexcept {
			S out; out.v128 = vceqq_u64(in1.v128, in2.v128); return out;
		}
		/// \return in1 != in2 uncompressed (-1 if the limbs differ, 0 otherwise)
		[[nodiscard]] constexpr static inline S cmp_(const S in1, const S in2) noexcept {
			S out; out.v128 = ~vceqq_u64(in1.v128, in2.v128); return out;
		}

		/// \return in1 > in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t gt(const S in1, const S in2) noexcept {
			return _mm_movemask_epi64(cmpgt128(in1.v128, in2.v128));
		}
		/// \return in1 >= in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t ge(const S in1, const S in2) noexcept {
			return _mm_movemask_epi64(cmpge128(in1.v128, in2.v128));
		}
		/// \return in1 < in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t lt(const S in1, const S in2) noexcept {
			return _mm_movemask_epi64(cmpgt128(in2.v128, in1.v128));
		}
		/// \return in1 <= in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t le(const S in1, const S in2) noexcept {
			return _mm_movemask_epi64(cmpge128(in2.v128, in1.v128));
		}
		/// \return in1 == in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t eq(const S in1, const S in2) noexcept {
			return _mm_movemask_epi64(vceqq_u64(in1.v128, in2.v128));
		}
		/// \return in1 != in2 compressed (one bit per lane)
		[[nodiscard]] constexpr static inline uint32_t cmp(const S in1, const S in2) noexcept {
			return _mm_movemask_epi64(S::cmp_(in1, in2).v128);
		}

		/// \return the most significant bit of every lane
		[[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
			return _mm_movemask_epi64(in.v128);
		}

		/// \return [popcnt(in[0]), ..., popcnt(in[1])]
		[[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
			S out;
			// pairwise widening adds of the byte counts
			const uint8x16_t c8 = vcntq_u8(vreinterpretq_u8_u64(in.v128));
			out.v128 = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(c8)));
			return out;
		}

		/// \return true if all lanes of `in` are equal
		[[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
			for (uint32_t i = 1; i < LIMBS; ++i) {
				if (in.d[0] != in.d[i]) {
					return false;
				}
			}
			return true;
		}

		/// \return the lanes in reversed order
		[[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
			S out;
			for (uint32_t i = 0; i < LIMBS; i++) {
				out.d[LIMBS - 1 - i] = in.d[i];
			}
			return out;
		}

		/// \return lane-wise minimum
		[[nodiscard]] constexpr static inline S min(const S a, const S b) noexcept {
			S out;
			out.v128 = vbslq_u64(cmpgt128(a.v128, b.v128), b.v128, a.v128);
			return out;
		}

		/// \return lane-wise maximum
		[[nodiscard]] constexpr static inline S max(const S a, const S b) noexcept {
			S out;
			out.v128 = vbslq_u64(cmpgt128(a.v128, b.v128), a.v128, b.v128);
			return out;
		}

		/// NOTE: NEON has no gather instruction, so this is a scalar loop
		/// \return ret[i] = limb at byte offset data[i] * scale of `ptr`
		template<const uint32_t scale = 1>
		[[nodiscard]] constexpr static inline S gather(const void *ptr,
		                                               const S data) noexcept {
			static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
			S ret;
			const uint8_t *ptr8 = (const uint8_t *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				// NOTE: byte offsets, so the address may not be aligned
				__builtin_memcpy(&ret.d[i], ptr8 + data.d[i] * scale, sizeof(limb_type));
			}
			return ret;
		}

		/// writes data[i] to the byte offset offset[i] * scale of `ptr`
		template<const uint32_t scale = 1>
		constexpr static inline void scatter(void *ptr,
		                                     const S offset,
		                                     const S data) noexcept {
			static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
			uint8_t *ptr8 = (uint8_t *) ptr;
			for (uint32_t i = 0; i < LIMBS; i++) {
				__builtin_memcpy(ptr8 + offset.d[i] * scale, &data.d[i], sizeof(limb_type));
			}
		}
	};


    using _uint8x16_t = _Xint8x16_t<true>;
    using  _int8x16_t = _Xint8x16_t<false>;

    using _uint16x8_t = _Xint16x8_t<true>;
    using  _int16x8_t = _Xint16x8_t<false>;

    using _uint32x4_t = _Xint32x4_t<true>;
    using  _int32x4_t = _Xint32x4_t<false>
	;
    using _uint64x2_t = _Xint64x2_t<true>;
    using  _int64x2_t = _Xint64x2_t<false>;

};// namespace cryptanalysislib


constexpr static uint8x16_t u8tom128(const uint8_t t[16]) noexcept {
	uint8x16_t tmp = {t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7],t[8],t[9],t[10],t[11],t[12],t[13],t[14],t[15]};
	return tmp;
}
constexpr static uint16x8_t u16tom128(const uint16_t t[8]) noexcept {
	uint16x8_t tmp = {t[0],t[1],t[2],t[3],t[4],t[5],t[6],t[7]};
	return tmp;
}
constexpr static uint32x4_t u32tom128(const uint32_t t[4]) noexcept {
	uint32x4_t tmp = {t[0],t[1],t[2],t[3]};
	return tmp;
}
constexpr static uint64x2_t u64tom128(const uint64_t t[2]) noexcept {
	uint64x2_t tmp = {t[0],t[1]};
	return tmp;
}

// Shuffle packed 8-bit integers in a according to shuffle control mask in the
// corresponding 8-bit element of b, and store the results in dst.
// https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html#text=_mm_shuffle_epi8
inline uint8x16_t shuffle_epi8(uint8x16_t a, uint8x16_t b) {
	int8x16_t tbl = a;
	uint8x16_t idx = b;
	uint8x16_t idx_masked =  vandq_u8(idx, vdupq_n_u8(0x8F));  // avoid using meaningless bits
	return vqtbl1q_s8(tbl, idx_masked);
}

// implementation of `_mm_shuffle_epi16`
inline uint16x8_t shuffle_epi16(const uint16x8_t a,
								const uint16x8_t b) {
    const uint16x8_t tmp = b*2;
    const uint16x8_t s  = tmp ^ vshlq_n_u16(tmp, 8);
    const uint16x8_t ss = vaddq_u16(s, vdupq_n_u16(0x100));
    return shuffle_epi8(a, ss);
}

///
template<const bool __unsigned=true>
struct Xint8x32_t {
	constexpr static uint32_t LIMBS = 32;
	using limb_type = std::conditional_t<__unsigned, uint8_t, int8_t>;
    using S = Xint8x32_t<__unsigned>;

    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		// compatibility with txn_t
		T8   d [32];

		T8  v8 [32];
		T16 v16[16];
		T32 v32[ 8];
		T64 v64[ 4];
		uint8x16_t v128[2];
	};

	/// \return number of limbs
	[[nodiscard]] constexpr inline static size_t size() noexcept {
		return LIMBS;
	}

	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
		return __unsigned;
	}

	[[nodiscard]] constexpr inline limb_type& operator[](const uint32_t i) noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	constexpr Xint8x32_t() noexcept = default;

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	///
	/// \return
	static inline S random() noexcept {
		S ret;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 4; ++i) {
			ret.v64[i] = rng();
		}
		return ret;
	}

	/// NOTE: in constexpr mode this ensure that the v128 union elements are
	/// the active one
	[[nodiscard]] constexpr static inline S set(char __q31, char __q30, char __q29, char __q28,
	                                            char __q27, char __q26, char __q25, char __q24,
	                                            char __q23, char __q22, char __q21, char __q20,
	                                            char __q19, char __q18, char __q17, char __q16,
	                                            char __q15, char __q14, char __q13, char __q12,
	                                            char __q11, char __q10, char __q09, char __q08,
	                                            char __q07, char __q06, char __q05, char __q04,
	                                            char __q03, char __q02, char __q01, char __q00) noexcept {
		S out;
		out.d[31] = __q31;
		out.d[30] = __q30;
		out.d[29] = __q29;
		out.d[28] = __q28;
		out.d[27] = __q27;
		out.d[26] = __q26;
		out.d[25] = __q25;
		out.d[24] = __q24;
		out.d[23] = __q23;
		out.d[22] = __q22;
		out.d[21] = __q21;
		out.d[20] = __q20;
		out.d[19] = __q19;
		out.d[18] = __q18;
		out.d[17] = __q17;
		out.d[16] = __q16;
		out.d[15] = __q15;
		out.d[14] = __q14;
		out.d[13] = __q13;
		out.d[12] = __q12;
		out.d[11] = __q11;
		out.d[10] = __q10;
		out.d[ 9] = __q09;
		out.d[ 8] = __q08;
		out.d[ 7] = __q07;
		out.d[ 6] = __q06;
		out.d[ 5] = __q05;
		out.d[ 4] = __q04;
		out.d[ 3] = __q03;
		out.d[ 2] = __q02;
		out.d[ 1] = __q01;
		out.d[ 0] = __q00;

		if (std::is_constant_evaluated()) {
			S out1;
			out1.v128[0] = u8tom128((const uint8_t *)(out.d +  0));
			out1.v128[1] = u8tom128((const uint8_t *)(out.d + 16));
			return out1;
		}

		return out;
	}

	/// NOTE: in constexpr mode this ensure that the v128 union elements are
	/// the active one
	[[nodiscard]] constexpr static inline S setr(char __q31, char __q30, char __q29, char __q28,
	                                             char __q27, char __q26, char __q25, char __q24,
	                                             char __q23, char __q22, char __q21, char __q20,
	                                             char __q19, char __q18, char __q17, char __q16,
	                                             char __q15, char __q14, char __q13, char __q12,
	                                             char __q11, char __q10, char __q09, char __q08,
	                                             char __q07, char __q06, char __q05, char __q04,
	                                             char __q03, char __q02, char __q01, char __q00) noexcept {
		S out;
		out.d[ 0] = __q31;
		out.d[ 1] = __q30;
		out.d[ 2] = __q29;
		out.d[ 3] = __q28;
		out.d[ 4] = __q27;
		out.d[ 5] = __q26;
		out.d[ 6] = __q25;
		out.d[ 7] = __q24;
		out.d[ 8] = __q23;
		out.d[ 9] = __q22;
		out.d[10] = __q21;
		out.d[11] = __q20;
		out.d[12] = __q19;
		out.d[13] = __q18;
		out.d[14] = __q17;
		out.d[15] = __q16;
		out.d[16] = __q15;
		out.d[17] = __q14;
		out.d[18] = __q13;
		out.d[19] = __q12;
		out.d[20] = __q11;
		out.d[21] = __q10;
		out.d[22] = __q09;
		out.d[23] = __q08;
		out.d[24] = __q07;
		out.d[25] = __q06;
		out.d[26] = __q05;
		out.d[27] = __q04;
		out.d[28] = __q03;
		out.d[29] = __q02;
		out.d[30] = __q01;
		out.d[31] = __q00;

		if (std::is_constant_evaluated()) {
			S out1;
			out1.v128[0] = u8tom128((const uint8_t *)(out.d +  0));
			out1.v128[1] = u8tom128((const uint8_t *)(out.d + 16));
			return out1;
		}

		return out;
	}

	/// sets all 32 8bit limbs to `a`
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline S set1(const uint8_t a) noexcept {
		S out1 = S::set(a, a, a, a, a, a, a, a,
		                a, a, a, a, a, a, a, a,
		                a, a, a, a, a, a, a, a,
		                a, a, a, a, a, a, a, a);
		return out1;
	}

	/// set the `i`-th bit in each limb
	/// \param pos[in]: bit position to set
	constexpr inline void set_bit(const uint32_t pos) noexcept {
		v128[0] = vdupq_n_u8(uint8_t(1u << pos));
		v128[1] = v128[0];
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	constexpr static inline S load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u8tom128((const uint8_t *)(ptr +  0));
			out.v128[1] = u8tom128((const uint8_t *)(ptr + 16));
			return out;
		}

		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128[i] = vld1q_u8((const uint8_t *) ptr + 16u * i);
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u8tom128((const uint8_t *)(ptr +  0));
			out.v128[1] = u8tom128((const uint8_t *)(ptr + 16));
			return out;
		}

		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128[i] = vld1q_u8((const uint8_t *) ptr + 16u * i);
		}
		return out;
	}

	/// masked load: lane i is loaded if bit i of `m` is set, otherwise 0.
	/// Masked out lanes are not accessed (no fault), as in the AVX-512 version.
	/// \param ptr
	/// \param m lane mask
	/// \return
	[[nodiscard]] static inline S maskz_unaligned_load(const limb_type *ptr,
	                                                   const Mask<LIMBS> &m) noexcept {
		// NOTE: NEON has no masked load and a full load could fault on the
		// 	masked out lanes
		const uint32_t k = uint32_t(MaskType<LIMBS>(m));
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ((k >> i) & 1u) ? ptr[i] : limb_type(0);
		}
		return out;
	}

	/// same as `maskz_unaligned_load` (no alignment needed)
	[[nodiscard]] static inline S maskz_aligned_load(const limb_type *ptr,
	                                                 const Mask<LIMBS> &m) noexcept {
		return maskz_unaligned_load(ptr, m);
	}


	/// NOTE: can never be constexpr
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr,
									  const S in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	/// NOTE: can never be constexpr
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr,
											   const S in) noexcept {
		auto *ptr128 = (uint8x16_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			ptr128[i] = in.v128[i];
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr,
												 const S in) noexcept {
		auto *ptr8 = (uint8_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			// byte-wise store: no alignment requirement
			vst1q_u8(ptr8 + 16u * i, in.v128[i]);
		}
	}

	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM1ykrgAyeAyYAHI%2BAEaYxCCSABykAA6oCoRODB7evv6BaRmOAqHhUSyx8Um2mPbFDEIETMQEOT5%2BATV1WY3NBKWRMXEJyQpNLW15nWN9A%2BWVIwCUtqhexMjsHOYAzGHI3lgA1CbbbmP4qAB0CCfYJhoAgjt7B5jHp%2Be0eNHXt/dPZl2DH2XiOJzcjjYv22d0ez2Br3ebmaLAA%2BuEBNDYQCgSCwadkOcwgQ/nDHgQAJ4pTBYKiHVGopgEAjEb5eAiYBlQDEMVEAN0wDhIqMp1IgiUWksOXmJiRFh3pqJlgkSqjlJO2VnJVJpmDpDKZLLZHK5EB5/MFRGIIp1EC4ADZJYtpbL5QzlQRVQ6RSctQ9Rbr9YzmazouzOajuZgBBahdaAxBJE6XYJvQRFR6HapJD7Nf8A7TFYbQ%2BHTeaBXGbWKJVLM/a3UriVn1b78zrCwaQ8aI1GYxWrVXMBAzMmPdszA2x2ZVBONX6C3qi12wybI2bo7z%2B8KE0na8Tx5P99Oc3O29SO8GjSue%2Bu%2B5bt7auKPifaTxmX5JVFxc/P24vO1epZruW97xraI57oIr6HlBn6zq2ZJPI8YzEF4Dgpp6qgHumJgAOx%2BvcACcXgMFkxz4f8RFER66oKnyiTHAArBY44mIxAAiCFIYR1FNvW6Z8g6bEWEJHFcVRGjEUe8p8qxzGHIkbGcXmZI8ZJHrQQJr7CYckhKeJklEe6spfvxhyCWYinMWY%2BkqdxeHKQRkloAwYyYKoKTEBhqrYRAzoMKg7kbCkOEUbhjn/AqCqRVFwnCQF%2BAKKIxDoPpHGHDMjjIIcYRfOE3lYbOhyqMKEAuWMBXYTlDCBJRhn1WpjUSU1DXNeV6Y0YV8phCO5FOY1nVVasp6IVFY3HICnlMMALBMNKDDEJ4tB1Y1/BeRAU7de87GHBovo5Uihw2ZqE1WJYeDOnh/XNQA9Ddw2XBZVkWHgSnbYqYZ0I4vJMM0yAINB0SiKoACOgn2iDeBKl4MMQGEXCPVwlnCa9HGkNVZiI8jzGo%2BxiwGS1hEPU9KNvScO3w1jz240x2AY1TpNiXZzUOStRHEJgBBrAwhzDQTrNwuFBkPMShyzWEfl9TFCqDUVTCkNEpCmMzjxjWthwbdJ6Z4O9e0nTr4KHKx%2BvWNYl0Uar40Kkwj3U2T2w7cQhjoH5XFWwqPz0YzO3k0dABUTsMC7%2BN2WNAsPGN2W%2B7LIogCAJXWhA8uHNEId%2BmNHNc8QPPII9ckibZBHhRwyy0JwjG8H4HBaKQqCcG4puWBlqzrG8Ow8KQBCaCXywANYgIxGj6JwkiV93tecLwCggEPXfVyXpBwLASBoCwKR0HE5CUKv6/0PEKL0QAtPshjAA6GhD1gfJ4BsABqeCYAA7gA8tSVcdzQtAcsQ08QNE4/RDCM0CknAO6AOYMQCkz9ojaEtKA3gq82CCGfgwWgID56kCwGGYAyJaC0GntwXgWBZpGHEBg/AHMHB4AFAQmu7lBThngeQQQtRx5fGiE7SBHgsDjyNCweBywqAGGAAoe%2BT9X6MCYfwQQIgxDsCkDIQQigVDqAwboQIBgjAoEbjYdh09IDLFQCFLIBDD4sCoCkLw5MeSHEPs/bYvBUACmIKyLA%2Bi/JdEtFkFwQdJh%2BECCEMIgwKjDAKOkTIAg/F6EKBEhgcwhjxECHYLxAhegTE8O0PQySqGpPGP0IJ8xQm2DyVEpJeT4khMScsBQLcNgSFLuXMeGC64cEOAfRIh85onyMIcB0lwND9I1rgQgJAJrbCfLwOeWhJSkH7oPYeHBR6kCrjXFpU8Z6d27jMsuHAzBNNWZPTZ88ZnOIyM4SQQA%3D
	/// \param in1
	/// \param in2
	/// \return in1 ^ in2
	[[nodiscard]] constexpr static inline S xor_(const S in1,
	                                             const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] ^ in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return in1 & in2
	[[nodiscard]] constexpr static inline S and_(const S in1,
	                                             const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] & in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S or_(const S in1,
	                                            const S in2) {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] | in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S andnot(const S in1,
	                                               const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = ~in1.v128[i] & in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = ~in1.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S add(const S in1,
	                                            const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			// NOTE: lane-wise add with wrap-around (as the AVX2 version), the
			// 	lanes are independent, so no carry needs to be propagated
			out.v128[i] = in1.v128[i] + in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S sub(const S in1,
	                                            const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] - in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] * in2.v128[i];
		}
		return out;
	}

	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const limb_type in2) noexcept {
		S rs = S::set1(in2);
		return S::mullo(in1, rs);
	}

	/// NOTE: assumes that in constexpr mode, that v128 is the active union member
	/// \param in1
	/// \param in2
	/// \return in1.v8[i] >> in2
	[[nodiscard]] constexpr static inline S slli(const S in1,
	                                             const uint8_t in2) noexcept {
		assert(in2 <= 8);
		S out;
		if (std::is_constant_evaluated()) {
			const uint8_t tmp = ~((1u<<in2) - 1);
			uint8x16_t t = {tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp};

			for (uint32_t i = 0; i < 2; i++) {
				out.v128[i] = in1.v128[i] << in2;
				out.v128[i] &= t;
			}

			return out;
		}

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			const S tmp = S::set1(in2);
#ifdef __clang__
			out.v128[i] = __builtin_neon_vshlq_v(in1.v128[i], (uint8x16_t) tmp.v128[0], 48u);
#else
			out.v128[i] = vshlq_u8(in1.v128[i], (uint8x16_t) tmp.v128[0]);
#endif
		}

		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const uint8_t in2) noexcept {
		assert(in2 <= 8);
		S out;
		if (std::is_constant_evaluated()) {

			for (uint32_t i = 0; i < 2; i++) {
				// logical shift of the unsigned lanes; a shift by the full width yields 0
				out.v128[i] = in2 >= 8u ? (in1.v128[i] ^ in1.v128[i]) : (in1.v128[i] >> in2);
			}

			return out;
		}

		cryptanalysislib::_Xint8x16_t<__unsigned> helper = cryptanalysislib::_Xint8x16_t<__unsigned>::set1(-in2);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = vshlq_u8(in1.v128[i], helper.v128);
		}

		return out;
	}

	/// \param in1[in]: vector element
	/// \param in2[in]:
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S ror(const S in1,
	                                             const uint8_t in2) noexcept {
		constexpr uint32_t bits = sizeof(limb_type) * 8u;
		const uint32_t s = in2 % bits;
		if (s == 0) {
			return in1;
		}
		return S::or_(S::srli(in1, s), S::slli(in1, bits - s));
	}

	/// \param in1[in]: vector element
	/// \param in2[in]:
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S rol(const S in1,
	                                             const uint8_t in2) noexcept {
		constexpr uint32_t bits = sizeof(limb_type) * 8u;
		const uint32_t s = in2 % bits;
		if (s == 0) {
			return in1;
		}
		return S::or_(S::slli(in1, s), S::srli(in1, bits - s));
	}

	/// transposes the 8x8 bit matrix in each 64 bit word, as the AVX-512
	/// version: bit i of output byte j is bit j of input byte (7 - i)
	/// \param input
	/// \return
	[[nodiscard]] constexpr static inline S transpose(const S input) noexcept {
		// reverse the bytes of each 64 bit word, then transpose with the
		// delta swaps of Hacker's Delight `transpose8`
		const uint64x2_t m1 = vdupq_n_u64(0x00AA00AA00AA00AAull);
		const uint64x2_t m2 = vdupq_n_u64(0x0000CCCC0000CCCCull);
		const uint64x2_t m3 = vdupq_n_u64(0x00000000F0F0F0F0ull);
		S ret;
		for (uint32_t h = 0; h < 2; h++) {
			uint64x2_t x = vreinterpretq_u64_u8(vrev64q_u8(input.v128[h]));
			uint64x2_t t;
			t = vandq_u64(veorq_u64(x, vshrq_n_u64(x, 7)), m1);
			x = veorq_u64(x, veorq_u64(t, vshlq_n_u64(t, 7)));
			t = vandq_u64(veorq_u64(x, vshrq_n_u64(x, 14)), m2);
			x = veorq_u64(x, veorq_u64(t, vshlq_n_u64(t, 14)));
			t = vandq_u64(veorq_u64(x, vshrq_n_u64(x, 28)), m3);
			x = veorq_u64(x, veorq_u64(t, vshlq_n_u64(t, 28)));
			ret.v128[h] = vreinterpretq_u8_u64(x);
		}
		return ret;
	}

	/// as `_mm256_sign_epi8`
	/// \return per lane: 0 if b == 0, -a if b < 0, a if b > 0
	[[nodiscard]] constexpr static inline S comp_sign(const S a,
	                                                  const S b) noexcept {
		S ret;
		for (uint32_t h = 0; h < 2; h++) {
			const int8x16_t sa = vreinterpretq_s8_u8(a.v128[h]);
			const int8x16_t sb = vreinterpretq_s8_u8(b.v128[h]);
			const uint8x16_t neg = vcltq_s8(sb, vdupq_n_s8(0));
			const uint8x16_t r = vbslq_u8(neg, vreinterpretq_u8_s8(vnegq_s8(sa)), a.v128[h]);
			ret.v128[h] = vandq_u8(r, vtstq_u8(b.v128[h], b.v128[h]));
		}
		return ret;
	}

	/// lane-wise comparisons of the (unsigned) registers;
	/// signed lanes are compared as signed values if `!__unsigned`
	[[nodiscard]] constexpr static inline uint8x16_t cmpgt128(const uint8x16_t a, const uint8x16_t b) noexcept {
		if constexpr (__unsigned) {
			return vcgtq_u8(a, b);
		} else {
			return vcgtq_s8(vreinterpretq_s8_u8(a), vreinterpretq_s8_u8(b));
		}
	}
	[[nodiscard]] constexpr static inline uint8x16_t cmpge128(const uint8x16_t a, const uint8x16_t b) noexcept {
		if constexpr (__unsigned) {
			return vcgeq_u8(a, b);
		} else {
			return vcgeq_s8(vreinterpretq_s8_u8(a), vreinterpretq_s8_u8(b));
		}
	}
	[[nodiscard]] constexpr static inline uint8x16_t cmplt128(const uint8x16_t a, const uint8x16_t b) noexcept { return cmpgt128(b, a); }
	[[nodiscard]] constexpr static inline uint8x16_t cmple128(const uint8x16_t a, const uint8x16_t b) noexcept { return cmpge128(b, a); }

	[[nodiscard]] constexpr static inline uint32_t gt(const S in1,
	                                                  const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint8x16_t tmp = cmpgt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi8(tmp) << i * 16;
#else
			const uint8x16_t tmp = cmpgt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi8(tmp) << i * 16;
#endif
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline S gt_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmpgt128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmpgt128(in1.v128[i], in2.v128[i]);
#endif
		}
		return ret;
	}
	[[nodiscard]] constexpr static inline uint32_t ge(const S in1,
	                                                  const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint8x16_t tmp = cmpge128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi8(tmp) << i * 16;
#else
			const uint8x16_t tmp = cmpge128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi8(tmp) << i * 16;
#endif
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline S ge_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmpge128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmpge128(in1.v128[i], in2.v128[i]);
#endif
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline uint32_t lt(const S in1,
	                                                  const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint8x16_t tmp = cmplt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi8(tmp) << i * 16;
#else
			const uint8x16_t tmp = cmplt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi8(tmp) << i * 16;
#endif
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline S lt_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmplt128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmplt128(in1.v128[i], in2.v128[i]);
#endif
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline uint32_t le(const S in1,
	                                                  const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint8x16_t tmp = cmple128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi8(tmp) << i * 16;
#else
			const uint8x16_t tmp = cmple128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi8(tmp) << i * 16;
#endif
		}

		return ret;
	}

	[[nodiscard]] constexpr static inline S le_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmple128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmple128(in1.v128[i], in2.v128[i]);
#endif
		}
		return ret;
	}


	[[nodiscard]] constexpr static inline uint32_t eq(const S in1,
	                                             const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint8x16_t tmp = vceqq_u8(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi8(tmp) << i * 16;
#else
			const uint8x16_t tmp = in1.v128[i] == in2.v128[i];
			ret ^= _mm_movemask_epi8(tmp) << i * 16;
#endif
		}

		return ret;
	}


    /// \param in1
	/// \param in2
	/// \return {-1, 0, 1}
	[[nodiscard]] constexpr static inline S eq_(const S in1,
												const S in2) noexcept {
		S ret;
    	if constexpr (__unsigned) {
			ret.v128[0] = vceqq_u8(in1.v128[0], in2.v128[0]);
			ret.v128[1] = vceqq_u8(in1.v128[1], in2.v128[1]);
    	} else {
			ret.v128[0] = vceqq_s8(in1.v128[0], in2.v128[0]);
			ret.v128[1] = vceqq_s8(in1.v128[1], in2.v128[1]);
    	}
		return ret;
	}

    /// \param in1
	/// \param in2
	/// \return {-1, 0, 1}
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
												 const S in2) noexcept {
		S ret;
    	if constexpr (__unsigned) {
			ret.v128[0]  = vcltq_u8(in1.v128[0], in2.v128[0]);
			ret.v128[0] ^= vcgtq_u8(in1.v128[0], in2.v128[0]);
			ret.v128[1]  = vcltq_u8(in1.v128[1], in2.v128[1]);
			ret.v128[1] ^= vcgtq_u8(in1.v128[1], in2.v128[1]);
    	} else {
			ret.v128[0]  = vcltq_s8(in1.v128[0], in2.v128[0]);
			ret.v128[0] ^= vcgtq_s8(in1.v128[0], in2.v128[0]);
			ret.v128[1]  = vcltq_s8(in1.v128[1], in2.v128[1]);
			ret.v128[1] ^= vcgtq_s8(in1.v128[1], in2.v128[1]);
    	}
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
												       const S in2) noexcept {
        S ret = S::cmp_(in1, in2);
		return _mm_movemask_epi8(ret.v128[0]) ^ (_mm_movemask_epi8(ret.v128[1]) << 16u);
	}

    /// \param in[in]:
	[[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
		S out;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			out.v128[i] = vcntq_u8(in.v128[i]);
#else
			out.v128[i] = __builtin_aarch64_popcountv16qi(in.v128[i]);
#endif
		}

		return out;
	}

	/// \param in[in]:
	/// \return the number of leading zeros of each 8 bit lane (8 for 0)
	[[nodiscard]] constexpr static inline S lzcnt(const S in) noexcept {
		S out;
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = vclzq_u8(in.v128[i]);
		}
		return out;
	}

	/// tzcnt(x) = popcnt((x - 1) & ~x)
	/// \param in[in]:
	/// \return the number of trailing zeros of each 8 bit lane (8 for 0)
	[[nodiscard]] constexpr static inline S tzcnt(const S in) noexcept {
		const S t = S::and_(S::sub(in, S::set1(1)), S::not_(in));
		return S::popcnt(t);
	}

	[[nodiscard]] constexpr static inline uint32_t move(const S in1) noexcept {
		uint32_t ret = 0;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			ret ^= _mm_movemask_epi8(in1.v128[i]) << i * 16;
		}
		return ret;
	}

	/// gather permutation: ret[i] = in[perm[i] % 32]
	/// \param in
	/// \param perm
	/// \return
	[[nodiscard]] constexpr static inline S permute(const S in,
	                                                const S perm) noexcept {
		const uint8x16x2_t table = {{in.v128[0], in.v128[1]}};
		const uint8x16_t m = vdupq_n_u8(31u);
		S ret;
		ret.v128[0] = vqtbl2q_u8(table, vandq_u8(perm.v128[0], m));
		ret.v128[1] = vqtbl2q_u8(table, vandq_u8(perm.v128[1], m));
		return ret;
	}

	/// \param in
	/// \return true if all lanes of `in` are equal
	[[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
		for (uint32_t i = 1; i < LIMBS; ++i) {
			if (in.d[0] != in.d[i]) {
				return false;
			}
		}

		return true;
	}

	[[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[LIMBS - 1 - i] = in.d[i];
		}

		return out;
	}

	/// NOTE: NEON has no gather instruction (only SVE has), so this is a
	/// 	scalar loop.
	/// \tparam scale
	/// \param ptr base pointer
	/// \param data offsets, in units of `scale` bytes
	/// \return ret[i] = limb at byte offset data[i] * scale of `ptr`
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
	                                               const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		S ret;
		const uint8_t *ptr8 = (const uint8_t *) ptr;
		for (uint32_t i = 0; i < LIMBS; i++) {
			__builtin_memcpy(&ret.d[i], ptr8 + size_t(uint8_t(data.d[i])) * scale, sizeof(limb_type));
		}
		return ret;
	}

	/// writes data[i] to the byte offset `offset[i] * scale` of `ptr`
	/// \tparam scale
	/// \param ptr base pointer
	/// \param offset offsets, in units of `scale` bytes
	/// \param data
	template<const uint32_t scale = 1>
	constexpr static inline void scatter(void *ptr,
	                                     const S offset,
	                                     const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < LIMBS; i++) {
			__builtin_memcpy(ptr8 + size_t(uint8_t(offset.d[i])) * scale, &data.d[i], sizeof(limb_type));
		}
	}

    /// lane-wise minimum
    /// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < 2; i++) {
			if constexpr (__unsigned) {
				c.v128[i] = vminq_u8(a.v128[i], b.v128[i]);
			} else {
				c.v128[i] = vreinterpretq_u8_s8(vminq_s8(vreinterpretq_s8_u8(a.v128[i]), vreinterpretq_s8_u8(b.v128[i])));
			}
		}

        return c;
    }

	/// as `_mm512_test_epi8_mask`
	/// \param in1[in]: register to test
	/// \param bit_pos[in]: bit position to test
	/// \return bit i is set, if bit `bit_pos` of `in1[i]` is set
	[[nodiscard]] constexpr static inline Mask<LIMBS> test(const S in1,
	                                                       const uint32_t bit_pos) noexcept {
		const uint8x16_t b = vdupq_n_u8(uint8_t(1u << bit_pos));
		S t;
		t.v128[0] = vtstq_u8(in1.v128[0], b);
		t.v128[1] = vtstq_u8(in1.v128[1], b);
		return Mask<LIMBS>(MaskType<LIMBS>(S::move(t)));
	}

	/// lane-wise maximum
	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < 2; i++) {
			if constexpr (__unsigned) {
				c.v128[i] = vmaxq_u8(a.v128[i], b.v128[i]);
			} else {
				c.v128[i] = vreinterpretq_u8_s8(vmaxq_s8(vreinterpretq_s8_u8(a.v128[i]), vreinterpretq_s8_u8(b.v128[i])));
			}
		}

        return c;
    }

	/// \param in1[in]:
	/// \return the minimum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_min(const S in1) noexcept {
		if constexpr (__unsigned) {
			return vminvq_u8(vminq_u8(in1.v128[0], in1.v128[1]));
		} else {
			return vminvq_s8(vminq_s8(vreinterpretq_s8_u8(in1.v128[0]), vreinterpretq_s8_u8(in1.v128[1])));
		}
	}

	/// \param in1[in]:
	/// \return the maximum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_max(const S in1) noexcept {
		if constexpr (__unsigned) {
			return vmaxvq_u8(vmaxq_u8(in1.v128[0], in1.v128[1]));
		} else {
			return vmaxvq_s8(vmaxq_s8(vreinterpretq_s8_u8(in1.v128[0]), vreinterpretq_s8_u8(in1.v128[1])));
		}
	}

	/// \param in1[in]: vector element
	/// \param in2[in]: divisor, must not be zero
	/// \return in1 / in2, lane-wise
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
		assert(in2 != 0);
		if constexpr (__unsigned) {
			// NOTE: division by the invariant `in2`, see `internal::udiv_invariant`
			using VT = cryptanalysislib::internal::vec_t<limb_type, 128u / 8>;
			S out;
			for (uint32_t i = 0; i < 2; i++) {
				out.v128[i] = (uint8x16_t) cryptanalysislib::internal::udiv_invariant<limb_type, 128u / 8>((VT) in1.v128[i], in2);
			}
			return out;
		}

		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = in1.d[i] / in2;
		}
		return out;
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are zero, as `_mm512_maskz_compress_epi*`
	/// \param in1[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const Mask<LIMBS> &m) noexcept {
		return compress(in1, S::set1(0), m);
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are taken from `in2`, as `_mm512_mask_compress_epi*`
	/// \param in1[in]:
	/// \param in2[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const S in2,
	                                                 const Mask<LIMBS> &m) noexcept {
		// NOTE: NEON has no compress instruction (only SVE has `compact`)
		const uint64_t k = uint64_t(MaskType<LIMBS>(m));
		S ret = in2;
		uint32_t j = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			if ((k >> i) & 1u) { ret.d[j++] = in1.d[i]; }
		}
		return ret;
	}

	/// writes the lanes of `in1` selected by `m` contiguously to `ptr`,
	/// as `_mm512_mask_compressstoreu_epi*`
	/// \param ptr[out]:
	/// \param in1[in]:
	/// \param m[in]:
	constexpr static inline void compress(limb_type *ptr,
	                                      const S in1,
	                                      const Mask<LIMBS> &m) noexcept {
		const uint64_t k = uint64_t(MaskType<LIMBS>(m));
		uint32_t j = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			if ((k >> i) & 1u) { ptr[j++] = in1.d[i]; }
		}
	}
};

///
using uint8x32_t = Xint8x32_t<true>;
using  int8x32_t = Xint8x32_t<false>;

template<const bool __unsigned=true>
struct Xint16x16_t {
	constexpr static uint32_t LIMBS = 16;
	using limb_type = std::conditional_t<__unsigned, uint16_t, int16_t>;
	using S = Xint16x16_t;

    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		// compatibility with txn_t
		T16 d  [16];

		T8  v8 [32];
		T16 v16[16];
		T32 v32[ 8];
		T64 v64[ 4];
		uint16x8_t v128[2];
	};

	/// \return number of limbs
	[[nodiscard]] constexpr inline static size_t size() noexcept {
		return LIMBS;
	}

	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
		return __unsigned;
	}

	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const {
		assert(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) {
		assert(i < LIMBS);
		return d[i];
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false,
                                bool hex = false) const;

	///
	/// \return
	static inline S random() noexcept {
		S ret;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 4; ++i) {
			ret.v64[i] = rng();
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline S setr(uint16_t __q31, uint16_t __q30, uint16_t __q29, uint16_t __q28,
	                                            uint16_t __q27, uint16_t __q26, uint16_t __q25, uint16_t __q24,
	                                            uint16_t __q23, uint16_t __q22, uint16_t __q21, uint16_t __q20,
	                                            uint16_t __q19, uint16_t __q18, uint16_t __q17, uint16_t __q16) noexcept {
		S out;
		out.d[ 0] = __q31;
		out.d[ 1] = __q30;
		out.d[ 2] = __q29;
		out.d[ 3] = __q28;
		out.d[ 4] = __q27;
		out.d[ 5] = __q26;
		out.d[ 6] = __q25;
		out.d[ 7] = __q24;
		out.d[ 8] = __q23;
		out.d[ 9] = __q22;
		out.d[10] = __q21;
		out.d[11] = __q20;
		out.d[12] = __q19;
		out.d[13] = __q18;
		out.d[14] = __q17;
		out.d[15] = __q16;
		if (std::is_constant_evaluated()) {
			S out1;
			out1.v128[0] = u16tom128((const uint16_t *)(out.d + 0));
			out1.v128[1] = u16tom128((const uint16_t *)(out.d + 8));
			return out1;
		}
		return out;
	}

	[[nodiscard]] constexpr static inline S set(uint16_t __q31, uint16_t __q30, uint16_t __q29, uint16_t __q28,
	                                             uint16_t __q27, uint16_t __q26, uint16_t __q25, uint16_t __q24,
	                                             uint16_t __q23, uint16_t __q22, uint16_t __q21, uint16_t __q20,
	                                             uint16_t __q19, uint16_t __q18, uint16_t __q17, uint16_t __q16) noexcept {
		S out;
		out.d[15] = __q31;
		out.d[14] = __q30;
		out.d[13] = __q29;
		out.d[12] = __q28;
		out.d[11] = __q27;
		out.d[10] = __q26;
		out.d[ 9] = __q25;
		out.d[ 8] = __q24;
		out.d[ 7] = __q23;
		out.d[ 6] = __q22;
		out.d[ 5] = __q21;
		out.d[ 4] = __q20;
		out.d[ 3] = __q19;
		out.d[ 2] = __q18;
		out.d[ 1] = __q17;
		out.d[ 0] = __q16;
		if (std::is_constant_evaluated()) {
			S out1;
			out1.v128[0] = u16tom128((const uint16_t *)(out.d + 0));
			out1.v128[1] = u16tom128((const uint16_t *)(out.d + 8));
			return out1;
		}
		return out;
	}
	/// sets all 32 8bit limbs to `a`
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline S set1(const uint16_t a) noexcept {
		S out;
		out = S::set(a, a, a, a, a, a, a, a,
		                       a, a, a, a, a, a, a, a);
		return out;
	}

	/// set the `i`-th bit in each limb
	/// \param pos[in]: bit position to set
	constexpr inline void set_bit(const uint32_t pos) noexcept {
		v128[0] = vdupq_n_u16(uint16_t(1u << pos));
		v128[1] = v128[0];
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	constexpr static inline S load(const limb_type *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u16tom128((const uint16_t *)(ptr + 0));
			out.v128[1] = u16tom128((const uint16_t *)(ptr + 8));
			return out;
		}

		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128[i] = vreinterpretq_u16_u8(vld1q_u8((const uint8_t *) ptr + 16u * i));
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u16tom128((const uint16_t *)(ptr + 0));
			out.v128[1] = u16tom128((const uint16_t *)(ptr + 8));
			return out;
		}

		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128[i] = vreinterpretq_u16_u8(vld1q_u8((const uint8_t *) ptr + 16u * i));
		}
		return out;
	}

	/// masked load: lane i is loaded if bit i of `m` is set, otherwise 0.
	/// Masked out lanes are not accessed (no fault), as in the AVX-512 version.
	/// \param ptr
	/// \param m lane mask
	/// \return
	[[nodiscard]] static inline S maskz_unaligned_load(const limb_type *ptr,
	                                                   const Mask<LIMBS> &m) noexcept {
		// NOTE: NEON has no masked load and a full load could fault on the
		// 	masked out lanes
		const uint32_t k = uint32_t(MaskType<LIMBS>(m));
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ((k >> i) & 1u) ? ptr[i] : limb_type(0);
		}
		return out;
	}

	/// same as `maskz_unaligned_load` (no alignment needed)
	[[nodiscard]] static inline S maskz_aligned_load(const limb_type *ptr,
	                                                 const Mask<LIMBS> &m) noexcept {
		return maskz_unaligned_load(ptr, m);
	}


	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr,
                                       const S in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr,
                                               const S in) noexcept {
		auto *ptr8 = (uint8_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			// byte-wise store: alias-safe for any `ptr`, no alignment requirement
			vst1q_u8(ptr8 + 16u * i, vreinterpretq_u8_u16(in.v128[i]));
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const S in) noexcept {
		auto *ptr8 = (uint8_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			// byte-wise store: alias-safe for any `ptr`, no alignment requirement
			vst1q_u8(ptr8 + 16u * i, vreinterpretq_u8_u16(in.v128[i]));
		}
	}

	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM1ykrgAyeAyYAHI%2BAEaYxCCSABykAA6oCoRODB7evv6BaRmOAqHhUSyx8Um2mPbFDEIETMQEOT5%2BATV1WY3NBKWRMXEJyQpNLW15nWN9A%2BWVIwCUtqhexMjsHOYAzGHI3lgA1CbbbmP4qAB0CCfYJhoAgjt7B5jHp%2Be0eNHXt/dPZl2DH2XiOJzcjjYv22d0ez2Br3ebmaLAA%2BuEBNDYQCgSCwadkOcwgQ/nDHgQAJ4pTBYKiHVGopgEAjEb5eAiYBlQDEMVEAN0wDhIqMp1IgiUWksOXmJiRFh3pqJlgkSqjlJO2VnJVJpmDpDKZLLZHK5EB5/MFRGIIp1EC4ADZJYtpbL5QzlQRVQ6RSctQ9Rbr9YzmazouzOajuZgBBahdaAxBJE6XYJvQRFR6HapJD7Nf8A7TFYbQ%2BHTeaBXGbWKJVLM/a3UriVn1b78zrCwaQ8aI1GYxWrVXMBAzMmPdszA2x2ZVBONX6C3qi12wybI2bo7z%2B8KE0na8Tx5P99Oc3O29SO8GjSue%2Bu%2B5bt7auKPifaTxmX5JVFxc/P24vO1epZruW97xraI57oIr6HlBn6zq2ZJPI8YzEF4Dgpp6qgHumJgAOx%2BvcACcXgMFkxz4f8RFER66oKnyiTHAArBY44mIxAAiCFIYR1FNvW6Z8g6bEWEJHFcVRGjEUe8p8qxzGHIkbGcXmZI8ZJHrQQJr7CYckhKeJklEe6spfvxhyCWYinMWY%2BkqdxeHKQRkloAwYyYKoKTEBhqrYRAzoMKg7kbCkOEUbhjn/AqCqRVFwnCQF%2BAKKIxDoPpHGHDMjjIIcYRfOE3lYbOhyqMKEAuWMBXYTlDCBJRhn1WpjUSU1DXNeV6Y0YV8phCO5FOY1nVVasp6IVFY3HICnlMMALBMNKDDEJ4tB1Y1/BeRAU7de87GHBovo5Uihw2ZqE1WJYeDOnh/XNQA9Ddw2XBZVkWHgSnbYqYZ0I4vJMM0yAINB0SiKoACOgn2iDeBKl4MMQGEXCPVwlnCa9HGkNVZiI8jzGo%2BxiwGS1hEPU9KNvScO3w1jz240x2AY1TpNiXZzUOStRHEJgBBrAwhzDQTrNwuFBkPMShyzWEfl9TFCqDUVTCkNEpCmMzjxjWthwbdJ6Z4O9e0nTr4KHKx%2BvWNYl0Uar40Kkwj3U2T2w7cQhjoH5XFWwqPz0YzO3k0dABUTsMC7%2BN2WNAsPGN2W%2B7LIogCAJXWhA8uHNEId%2BmNHNc8QPPII9ckibZBHhRwyy0JwjG8H4HBaKQqCcG4puWBlqzrG8Ow8KQBCaCXywANYgIxGj6JwkiV93tecLwCggEPXfVyXpBwLASBoCwKR0HE5CUKv6/0PEKL0QAtPshjAA6GhD1gfJ4BsABqeCYAA7gA8tSVcdzQtAcsQ08QNE4/RDCM0CknAO6AOYMQCkz9ojaEtKA3gq82CCGfgwWgID56kCwGGYAyJaC0GntwXgWBZpGHEBg/AHMHB4AFAQmu7lBThngeQQQtRx5fGiE7SBHgsDjyNCweBywqAGGAAoe%2BT9X6MCYfwQQIgxDsCkDIQQigVDqAwboQIBgjAoEbjYdh09IDLFQCFLIBDD4sCoCkLw5MeSHEPs/bYvBUACmIKyLA%2Bi/JdEtFkFwQdJh%2BECCEMIgwKjDAKOkTIAg/F6EKBEhgcwhjxECHYLxAhegTE8O0PQySqGpPGP0IJ8xQm2DyVEpJeT4khMScsBQLcNgSFLuXMeGC64cEOAfRIh85onyMIcB0lwND9I1rgQgJAJrbCfLwOeWhJSkH7oPYeHBR6kCrjXFpU8Z6d27jMsuHAzBNNWZPTZ88ZnOIyM4SQQA%3D
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S xor_(const S in1,
	                                             const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] ^ in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S and_(const S in1,
	                                             const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] & in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S or_(const S in1,
	                                            const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] | in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S andnot(const S in1,
	                                               const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = ~in1.v128[i] & in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = ~in1.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S add(const S in1,
	                                            const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = vaddq_u16(in1.v128[i], in2.v128[i]);
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S sub(const S in1,
	                                            const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] - in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] * in2.v128[i];
		}
		return out;
	}

	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const limb_type in2) noexcept {
		S rs = S::set1(in2);
		return S::mullo(in1, rs);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S slli(const S in1,
	                                             const uint8_t in2) noexcept {
		assert(in2 <= 16);
		S out;
		if (std::is_constant_evaluated()) {
			const uint16_t tmp = ~((1u<<in2) - 1);
			uint16x8_t t = {tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp};

			for (uint32_t i = 0; i < 2; i++) {
				out.v128[i] = in1.v128[i] << in2;
				out.v128[i] &= t;
			}

			return out;
		}

		cryptanalysislib::_Xint16x8_t<__unsigned> helper = cryptanalysislib::_Xint16x8_t<__unsigned>::set1(in2);
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = vshlq_u16(in1.v128[i], helper.v128);
		}

		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const uint16_t in2) noexcept {
		assert(in2 <= 16);
		S out;
		if (std::is_constant_evaluated()) {

			for (uint32_t i = 0; i < 2; i++) {
				// logical shift of the unsigned lanes; a shift by the full width yields 0
				out.v128[i] = in2 >= 16u ? (in1.v128[i] ^ in1.v128[i]) : (in1.v128[i] >> in2);
			}

			return out;
		}

		cryptanalysislib::_uint16x8_t helper = cryptanalysislib::_uint16x8_t::set1(-in2);
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = vshlq_u16(in1.v128[i], helper.v128);
		}

		return out;
	}

	/// \param in1[in]: vector element
	/// \param in2[in]:
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S ror(const S in1,
	                                             const uint8_t in2) noexcept {
		constexpr uint32_t bits = sizeof(limb_type) * 8u;
		const uint32_t s = in2 % bits;
		if (s == 0) {
			return in1;
		}
		return S::or_(S::srli(in1, s), S::slli(in1, bits - s));
	}

	/// \param in1[in]: vector element
	/// \param in2[in]:
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S rol(const S in1,
	                                             const uint8_t in2) noexcept {
		constexpr uint32_t bits = sizeof(limb_type) * 8u;
		const uint32_t s = in2 % bits;
		if (s == 0) {
			return in1;
		}
		return S::or_(S::slli(in1, s), S::srli(in1, bits - s));
	}

	/// lane-wise comparisons of the (unsigned) registers;
	/// signed lanes are compared as signed values if `!__unsigned`
	[[nodiscard]] constexpr static inline uint16x8_t cmpgt128(const uint16x8_t a, const uint16x8_t b) noexcept {
		if constexpr (__unsigned) {
			return vcgtq_u16(a, b);
		} else {
			return vcgtq_s16(vreinterpretq_s16_u16(a), vreinterpretq_s16_u16(b));
		}
	}
	[[nodiscard]] constexpr static inline uint16x8_t cmpge128(const uint16x8_t a, const uint16x8_t b) noexcept {
		if constexpr (__unsigned) {
			return vcgeq_u16(a, b);
		} else {
			return vcgeq_s16(vreinterpretq_s16_u16(a), vreinterpretq_s16_u16(b));
		}
	}
	[[nodiscard]] constexpr static inline uint16x8_t cmplt128(const uint16x8_t a, const uint16x8_t b) noexcept { return cmpgt128(b, a); }
	[[nodiscard]] constexpr static inline uint16x8_t cmple128(const uint16x8_t a, const uint16x8_t b) noexcept { return cmpge128(b, a); }

	constexpr static inline int gt(const S in1,
								   const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint16x8_t tmp = cmpgt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi16(tmp) << i * 8;
#else
			const uint16x8_t tmp = cmpgt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi16(tmp) << i * 8;
#endif
		}

		return ret;
	}

	constexpr static inline S gt_(const S in1,
								  const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmpgt128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmpgt128(in1.v128[i], in2.v128[i]);
#endif
		}

		return ret;
    }

	constexpr static inline int ge(const S in1,
								   const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint16x8_t tmp = cmpge128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi16(tmp) << i * 8;
#else
			const uint16x8_t tmp = cmpge128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi16(tmp) << i * 8;
#endif
		}

		return ret;
	}

	constexpr static inline S ge_(const S in1,
								  const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmpge128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmpge128(in1.v128[i], in2.v128[i]);
#endif
		}

		return ret;
	}

	constexpr static inline int lt(const S in1,
								   const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint16x8_t tmp = cmplt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi16(tmp) << i * 8;
#else
			const uint16x8_t tmp = cmplt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi16(tmp) << i * 8;
#endif
		}

		return ret;
	}

	constexpr static inline S lt_(const S in1,
							      const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmplt128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmplt128(in1.v128[i], in2.v128[i]);
#endif
		}

		return ret;
	}

	constexpr static inline int le(const S in1,
								   const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint16x8_t tmp = cmple128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi16(tmp) << i * 8;
#else
			const uint16x8_t tmp = cmple128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi16(tmp) << i * 8;
#endif
		}

		return ret;
	}

	constexpr static inline S le_(const S in1,
							      const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmple128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmple128(in1.v128[i], in2.v128[i]);
#endif
		}

		return ret;
	}

	constexpr static inline int eq(const S in1,
								   const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint16x8_t tmp = vceqq_u16(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi16(tmp) << i * 8;
#else
			const uint16x8_t tmp = in1.v128[i] == in2.v128[i];
			ret ^= _mm_movemask_epi16(tmp) << i * 8;
#endif
		}

		return ret;
	}

	constexpr static inline S eq_(const S in1,
								   const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = vceqq_u16(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = in1.v128[i] == in2.v128[i];
#endif
		}

		return ret;
	}

    /// \param in1
	/// \param in2
	/// \return {-1, 0, 1}
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
												 const S in2) noexcept {
		S ret;
    	if constexpr (__unsigned) {
			ret.v128[0]  = vcltq_u16(in1.v128[0], in2.v128[0]);
			ret.v128[0] ^= vcgtq_u16(in1.v128[0], in2.v128[0]);
			ret.v128[1]  = vcltq_u16(in1.v128[1], in2.v128[1]);
			ret.v128[1] ^= vcgtq_u16(in1.v128[1], in2.v128[1]);
    	} else {
			ret.v128[0]  = vcltq_s16(in1.v128[0], in2.v128[0]);
			ret.v128[0] ^= vcgtq_s16(in1.v128[0], in2.v128[0]);
			ret.v128[1]  = vcltq_s16(in1.v128[1], in2.v128[1]);
			ret.v128[1] ^= vcgtq_s16(in1.v128[1], in2.v128[1]);
    	}
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
												       const S in2) noexcept {
        S ret = S::cmp_(in1, in2);
		return _mm_movemask_epi16(ret.v128[0]) ^ (_mm_movemask_epi16(ret.v128[1]) << 8u);
	}

	constexpr static inline S popcnt(const S in) noexcept {
		S out;

		cryptanalysislib::_Xint16x8_t<__unsigned> mask = cryptanalysislib::_Xint16x8_t<__unsigned>::set1(0xff);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint16x8_t tmp = (uint16x8_t) vcntq_u8((uint8x16_t) in.v128[i]);
			out.v128[i] = vaddq_u16(vshrq_n_u16(tmp, 8), vandq_u16(tmp, mask.v128));

#else
			const uint16x8_t tmp = (uint16x8_t) __builtin_aarch64_popcountv16qi((uint8x16_t) in.v128[i]);
			out.v128[i] = vshrq_n_u16(tmp, 8) + vandq_u16(tmp, mask.v128);
#endif
		}

		return out;
	}

	/// \param in
	/// \return the number of leading zeros of each 16 bit lane (16 for 0)
	[[nodiscard]] constexpr static inline S lzcnt(const S in) noexcept {
		S out;
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = vclzq_u16(in.v128[i]);
		}
		return out;
	}

	/// tzcnt(x) = popcnt((x - 1) & ~x)
	/// \param in
	/// \return the number of trailing zeros of each 16 bit lane (16 for 0)
	[[nodiscard]] constexpr static inline S tzcnt(const S in) noexcept {
		const S t = S::and_(S::sub(in, S::set1(1)), S::not_(in));
		return S::popcnt(t);
	}

	/// same as `lzcnt`, same name as in the AVX2 backend
	[[nodiscard]] constexpr static inline S clz(const S in) noexcept { return lzcnt(in); }

	/// same as `tzcnt`, same name as in the AVX2 backend
	[[nodiscard]] constexpr static inline S ctz(const S in) noexcept { return tzcnt(in); }

	/// \param in
	/// \return true if all lanes of `in` are equal
	[[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
		for (uint32_t i = 1; i < LIMBS; ++i) {
			if (in.v16[0] != in.v16[i]) {
				return false;
			}
		}

		return true;
	}

	/// NOTE: NEON has no gather instruction (only SVE has), so this is a
	/// 	scalar loop.
	/// \tparam scale
	/// \param ptr base pointer
	/// \param data offsets, in units of `scale` bytes
	/// \return ret[i] = limb at byte offset data[i] * scale of `ptr`
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
	                                               const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		S ret;
		const uint8_t *ptr8 = (const uint8_t *) ptr;
		for (uint32_t i = 0; i < LIMBS; i++) {
			__builtin_memcpy(&ret.d[i], ptr8 + size_t(uint16_t(data.d[i])) * scale, sizeof(limb_type));
		}
		return ret;
	}

	/// writes data[i] to the byte offset `offset[i] * scale` of `ptr`
	/// \tparam scale
	/// \param ptr base pointer
	/// \param offset offsets, in units of `scale` bytes
	/// \param data
	template<const uint32_t scale = 1>
	constexpr static inline void scatter(void *ptr,
	                                     const S offset,
	                                     const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < LIMBS; i++) {
			__builtin_memcpy(ptr8 + size_t(uint16_t(offset.d[i])) * scale, &data.d[i], sizeof(limb_type));
		}
	}

	[[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.v16[LIMBS - 1 - i] = in.v16[i];
		}

		return out;
	}

	/// NOTE: scatter semantics `ret[perm[i]] = in[i]` (as the AVX2 and
	/// 	AVX-512 16 bit versions)
	/// \param in
	/// \param perm a permutation of [0, LIMBS)
	/// \return ret[perm[i]] = in[i]
	[[nodiscard]] constexpr static inline S permute(const S in,
	                                                const S perm) noexcept {
		// NEON has no scatter: select `in[i]` into the lanes equal to `perm[i]`
		const uint16x8_t idx0 = {0, 1, 2, 3, 4, 5, 6, 7};
		const uint16x8_t idx1 = {8, 9, 10, 11, 12, 13, 14, 15};
		uint16x8_t r0 = vdupq_n_u16(0), r1 = vdupq_n_u16(0);
		for (uint32_t i = 0; i < LIMBS; i++) {
			const uint16x8_t p = vdupq_n_u16(uint16_t(perm.d[i]));
			const uint16x8_t v = vdupq_n_u16(uint16_t(in.d[i]));
			r0 = vbslq_u16(vceqq_u16(idx0, p), v, r0);
			r1 = vbslq_u16(vceqq_u16(idx1, p), v, r1);
		}
		S ret;
		ret.v128[0] = r0;
		ret.v128[1] = r1;
		return ret;
	}

	constexpr static inline uint16_t move(const S in1) noexcept {
		uint32_t ret = 0;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			ret ^= _mm_movemask_epi16(in1.v128[i]) << i * 8;
		}
		return ret;
	}

	/// same semantics as `vpconflict` (AVX512CD):
	/// lane i is the bitmask of all lanes j < i with in1[j] == in1[i].
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S conflict(const S in1) noexcept {
		// compare all lanes with the broadcasted lane j, keep the lanes > j
		const uint16x8_t idx0 = {0, 1, 2, 3, 4, 5, 6, 7};
		const uint16x8_t idx1 = {8, 9, 10, 11, 12, 13, 14, 15};
		uint16x8_t r0 = vdupq_n_u16(0), r1 = vdupq_n_u16(0);
		for (uint32_t j = 0; j < LIMBS - 1u; ++j) {
			const uint16x8_t b = vdupq_n_u16(uint16_t(in1.d[j]));
			const uint16x8_t jj = vdupq_n_u16(uint16_t(j));
			const uint16x8_t bit = vdupq_n_u16(uint16_t(1u << j));
			r0 = vorrq_u16(r0, vandq_u16(vandq_u16(vceqq_u16(in1.v128[0], b), vcgtq_u16(idx0, jj)), bit));
			r1 = vorrq_u16(r1, vandq_u16(vandq_u16(vceqq_u16(in1.v128[1], b), vcgtq_u16(idx1, jj)), bit));
		}
		S ret;
		ret.v128[0] = r0;
		ret.v128[1] = r1;
		return ret;
	}

    /// lane-wise minimum
    /// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < 2; i++) {
			if constexpr (__unsigned) {
				c.v128[i] = vminq_u16(a.v128[i], b.v128[i]);
			} else {
				c.v128[i] = vreinterpretq_u16_s16(vminq_s16(vreinterpretq_s16_u16(a.v128[i]), vreinterpretq_s16_u16(b.v128[i])));
			}
		}

        return c;
    }

	/// as `_mm512_test_epi16_mask`
	/// \param in1[in]: register to test
	/// \param bit_pos[in]: bit position to test
	/// \return bit i is set, if bit `bit_pos` of `in1[i]` is set
	[[nodiscard]] constexpr static inline Mask<LIMBS> test(const S in1,
	                                                       const uint32_t bit_pos) noexcept {
		const uint16x8_t b = vdupq_n_u16(uint16_t(1u << bit_pos));
		S t;
		t.v128[0] = vtstq_u16(in1.v128[0], b);
		t.v128[1] = vtstq_u16(in1.v128[1], b);
		return Mask<LIMBS>(MaskType<LIMBS>(S::move(t)));
	}

	/// lane-wise maximum
	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < 2; i++) {
			if constexpr (__unsigned) {
				c.v128[i] = vmaxq_u16(a.v128[i], b.v128[i]);
			} else {
				c.v128[i] = vreinterpretq_u16_s16(vmaxq_s16(vreinterpretq_s16_u16(a.v128[i]), vreinterpretq_s16_u16(b.v128[i])));
			}
		}

        return c;
    }

	/// \param in1[in]:
	/// \return the minimum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_min(const S in1) noexcept {
		if constexpr (__unsigned) {
			return vminvq_u16(vminq_u16(in1.v128[0], in1.v128[1]));
		} else {
			return vminvq_s16(vminq_s16(vreinterpretq_s16_u16(in1.v128[0]), vreinterpretq_s16_u16(in1.v128[1])));
		}
	}

	/// \param in1[in]:
	/// \return the maximum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_max(const S in1) noexcept {
		if constexpr (__unsigned) {
			return vmaxvq_u16(vmaxq_u16(in1.v128[0], in1.v128[1]));
		} else {
			return vmaxvq_s16(vmaxq_s16(vreinterpretq_s16_u16(in1.v128[0]), vreinterpretq_s16_u16(in1.v128[1])));
		}
	}

	/// \param in1[in]: vector element
	/// \param in2[in]: divisor, must not be zero
	/// \return in1 / in2, lane-wise
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
		assert(in2 != 0);
		if constexpr (__unsigned) {
			// NOTE: division by the invariant `in2`, see `internal::udiv_invariant`
			using VT = cryptanalysislib::internal::vec_t<limb_type, 128u / 16>;
			S out;
			for (uint32_t i = 0; i < 2; i++) {
				out.v128[i] = (uint16x8_t) cryptanalysislib::internal::udiv_invariant<limb_type, 128u / 16>((VT) in1.v128[i], in2);
			}
			return out;
		}

		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = in1.d[i] / in2;
		}
		return out;
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are zero, as `_mm512_maskz_compress_epi*`
	/// \param in1[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const Mask<LIMBS> &m) noexcept {
		return compress(in1, S::set1(0), m);
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are taken from `in2`, as `_mm512_mask_compress_epi*`
	/// \param in1[in]:
	/// \param in2[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const S in2,
	                                                 const Mask<LIMBS> &m) noexcept {
		// NOTE: NEON has no compress instruction (only SVE has `compact`)
		const uint64_t k = uint64_t(MaskType<LIMBS>(m));
		S ret = in2;
		uint32_t j = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			if ((k >> i) & 1u) { ret.d[j++] = in1.d[i]; }
		}
		return ret;
	}

	/// writes the lanes of `in1` selected by `m` contiguously to `ptr`,
	/// as `_mm512_mask_compressstoreu_epi*`
	/// \param ptr[out]:
	/// \param in1[in]:
	/// \param m[in]:
	constexpr static inline void compress(limb_type *ptr,
	                                      const S in1,
	                                      const Mask<LIMBS> &m) noexcept {
		const uint64_t k = uint64_t(MaskType<LIMBS>(m));
		uint32_t j = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			if ((k >> i) & 1u) { ptr[j++] = in1.d[i]; }
		}
	}
};

///
using uint16x16_t = Xint16x16_t<true>;
using  int16x16_t = Xint16x16_t<false>;

template<const bool __unsigned=true>
struct Xint32x8_t {
	constexpr static uint32_t LIMBS = 8;
	using limb_type = std::conditional_t<__unsigned, uint32_t, int32_t>;
	using S = Xint32x8_t;

    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		// compatibility with txn_t
        T32 d  [ 8];

		T8  v8 [32];
		T16 v16[16];
		T32 v32[ 8];
		T64 v64[ 4];
		uint32x4_t v128[2];
	};

	/// \return number of limbs
	[[nodiscard]] constexpr inline static size_t size() noexcept {
		return LIMBS;
	}

	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
		return __unsigned;
	}

	[[nodiscard]] constexpr inline limb_type& operator[](const uint32_t i) noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false,
                                bool hex = false) const;

	///
	/// \return
	static inline S random() noexcept {
		S ret;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 4; ++i) {
			ret.v64[i] = rng();
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline S setr(uint32_t __q31, uint32_t __q30, uint32_t __q29, uint32_t __q28,
	                                            uint32_t __q27, uint32_t __q26, uint32_t __q25, uint32_t __q24) noexcept {
		S out;
		out.d[0] = __q31;
		out.d[1] = __q30;
		out.d[2] = __q29;
		out.d[3] = __q28;
		out.d[4] = __q27;
		out.d[5] = __q26;
		out.d[6] = __q25;
		out.d[7] = __q24;
		if (std::is_constant_evaluated()) {
			S out1;
			out1.v128[0] = u32tom128((const uint32_t *)(out.d + 0));
			out1.v128[1] = u32tom128((const uint32_t *)(out.d + 4));
			return out1;
		}
		return out;
	}

	[[nodiscard]] constexpr static inline S set(uint32_t __q31, uint32_t __q30, uint32_t __q29, uint32_t __q28,
	                                             uint32_t __q27, uint32_t __q26, uint32_t __q25, uint32_t __q24) noexcept {
		S out;
		out.d[7] = __q31;
		out.d[6] = __q30;
		out.d[5] = __q29;
		out.d[4] = __q28;
		out.d[3] = __q27;
		out.d[2] = __q26;
		out.d[1] = __q25;
		out.d[0] = __q24;
		if (std::is_constant_evaluated()) {
			S out1;
			out1.v128[0] = u32tom128((const uint32_t *)(out.d + 0));
			out1.v128[1] = u32tom128((const uint32_t *)(out.d + 4));
			return out1;
		}
		return out;
	}

	/// sets all 32 8bit limbs to `a`
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline S set1(const uint32_t a) noexcept {
		S out;
		out = S::set(a, a, a, a, a, a, a, a);
		return out;
	}

	/// set the `i`-th bit in each limb
	/// \param pos[in]: bit position to set
	constexpr inline void set_bit(const uint32_t pos) noexcept {
		v128[0] = vdupq_n_u32(uint32_t(1u) << pos);
		v128[1] = v128[0];
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	constexpr static inline S load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u32tom128((const uint32_t *)(ptr + 0));
			out.v128[1] = u32tom128((const uint32_t *)(ptr + 4));
			return out;
		}

		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128[i] = vreinterpretq_u32_u8(vld1q_u8((const uint8_t *) ptr + 16u * i));
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u32tom128((const uint32_t *)(ptr + 0));
			out.v128[1] = u32tom128((const uint32_t *)(ptr + 4));
			return out;
		}

		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128[i] = vreinterpretq_u32_u8(vld1q_u8((const uint8_t *) ptr + 16u * i));
		}
		return out;
	}

	/// masked load: lane i is loaded if bit i of `m` is set, otherwise 0.
	/// Masked out lanes are not accessed (no fault), as in the AVX-512 version.
	/// \param ptr
	/// \param m lane mask
	/// \return
	[[nodiscard]] static inline S maskz_unaligned_load(const limb_type *ptr,
	                                                   const Mask<LIMBS> &m) noexcept {
		// NOTE: NEON has no masked load and a full load could fault on the
		// 	masked out lanes
		const uint32_t k = uint32_t(MaskType<LIMBS>(m));
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ((k >> i) & 1u) ? ptr[i] : limb_type(0);
		}
		return out;
	}

	/// same as `maskz_unaligned_load` (no alignment needed)
	[[nodiscard]] static inline S maskz_aligned_load(const limb_type *ptr,
	                                                 const Mask<LIMBS> &m) noexcept {
		return maskz_unaligned_load(ptr, m);
	}


	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr,
                                       const S in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	static inline void aligned_store(void *ptr,
                                     const S in) noexcept {
		auto *ptr8 = (uint8_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			// byte-wise store: alias-safe for any `ptr`, no alignment requirement
			vst1q_u8(ptr8 + 16u * i, vreinterpretq_u8_u32(in.v128[i]));
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr,
                                                 const S in) noexcept {
		auto *ptr8 = (uint8_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			// byte-wise store: alias-safe for any `ptr`, no alignment requirement
			vst1q_u8(ptr8 + 16u * i, vreinterpretq_u8_u32(in.v128[i]));
		}
	}

	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM1ykrgAyeAyYAHI%2BAEaYxCCSABykAA6oCoRODB7evv6BaRmOAqHhUSyx8Um2mPbFDEIETMQEOT5%2BATV1WY3NBKWRMXEJyQpNLW15nWN9A%2BWVIwCUtqhexMjsHOYAzGHI3lgA1CbbbmP4qAB0CCfYJhoAgjt7B5jHp%2Be0eNHXt/dPZl2DH2XiOJzcjjYv22d0ez2Br3ebmaLAA%2BuEBNDYQCgSCwadkOcwgQ/nDHgQAJ4pTBYKiHVGopgEAjEb5eAiYBlQDEMVEAN0wDhIqMp1IgiUWksOXmJiRFh3pqJlgkSqjlJO2VnJVJpmDpDKZLLZHK5EB5/MFRGIIp1EC4ADZJYtpbL5QzlQRVQ6RSctQ9Rbr9YzmazouzOajuZgBBahdaAxBJE6XYJvQRFR6HapJD7Nf8A7TFYbQ%2BHTeaBXGbWKJVLM/a3UriVn1b78zrCwaQ8aI1GYxWrVXMBAzMmPdszA2x2ZVBONX6C3qi12wybI2bo7z%2B8KE0na8Tx5P99Oc3O29SO8GjSue%2Bu%2B5bt7auKPifaTxmX5JVFxc/P24vO1epZruW97xraI57oIr6HlBn6zq2ZJPI8YzEF4Dgpp6qgHumJgAOx%2BvcACcXgMFkxz4f8RFER66oKnyiTHAArBY44mIxAAiCFIYR1FNvW6Z8g6bEWEJHFcVRGjEUe8p8qxzGHIkbGcXmZI8ZJHrQQJr7CYckhKeJklEe6spfvxhyCWYinMWY%2BkqdxeHKQRkloAwYyYKoKTEBhqrYRAzoMKg7kbCkOEUbhjn/AqCqRVFwnCQF%2BAKKIxDoPpHGHDMjjIIcYRfOE3lYbOhyqMKEAuWMBXYTlDCBJRhn1WpjUSU1DXNeV6Y0YV8phCO5FOY1nVVasp6IVFY3HICnlMMALBMNKDDEJ4tB1Y1/BeRAU7de87GHBovo5Uihw2ZqE1WJYeDOnh/XNQA9Ddw2XBZVkWHgSnbYqYZ0I4vJMM0yAINB0SiKoACOgn2iDeBKl4MMQGEXCPVwlnCa9HGkNVZiI8jzGo%2BxiwGS1hEPU9KNvScO3w1jz240x2AY1TpNiXZzUOStRHEJgBBrAwhzDQTrNwuFBkPMShyzWEfl9TFCqDUVTCkNEpCmMzjxjWthwbdJ6Z4O9e0nTr4KHKx%2BvWNYl0Uar40Kkwj3U2T2w7cQhjoH5XFWwqPz0YzO3k0dABUTsMC7%2BN2WNAsPGN2W%2B7LIogCAJXWhA8uHNEId%2BmNHNc8QPPII9ckibZBHhRwyy0JwjG8H4HBaKQqCcG4puWBlqzrG8Ow8KQBCaCXywANYgIxGj6JwkiV93tecLwCggEPXfVyXpBwLASBoCwKR0HE5CUKv6/0PEKL0QAtPshjAA6GhD1gfJ4BsABqeCYAA7gA8tSVcdzQtAcsQ08QNE4/RDCM0CknAO6AOYMQCkz9ojaEtKA3gq82CCGfgwWgID56kCwGGYAyJaC0GntwXgWBZpGHEBg/AHMHB4AFAQmu7lBThngeQQQtRx5fGiE7SBHgsDjyNCweBywqAGGAAoe%2BT9X6MCYfwQQIgxDsCkDIQQigVDqAwboQIBgjAoEbjYdh09IDLFQCFLIBDD4sCoCkLw5MeSHEPs/bYvBUACmIKyLA%2Bi/JdEtFkFwQdJh%2BECCEMIgwKjDAKOkTIAg/F6EKBEhgcwhjxECHYLxAhegTE8O0PQySqGpPGP0IJ8xQm2DyVEpJeT4khMScsBQLcNgSFLuXMeGC64cEOAfRIh85onyMIcB0lwND9I1rgQgJAJrbCfLwOeWhJSkH7oPYeHBR6kCrjXFpU8Z6d27jMsuHAzBNNWZPTZ88ZnOIyM4SQQA%3D
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S xor_(const S in1,
	                                             const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] ^ in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S and_(const S in1,
	                                             const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] & in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S or_(const S in1,
	                                            const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] | in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S andnot(const S in1,
	                                               const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = ~in1.v128[i] & in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = ~in1.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S add(const S in1,
	                                            const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			// NOTE: lane-wise add with wrap-around (as the AVX2 version), the
			// 	lanes are independent, so no carry needs to be propagated
			out.v128[i] = in1.v128[i] + in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S sub(const S in1,
	                                            const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] - in2.v128[i];
		}
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] * in2.v128[i];
		}
		return out;
	}

	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const limb_type in2) noexcept {
		S rs = S::set1(in2);
		return S::mullo(in1, rs);
	}

	/// \return [in1[0]*in2[0], ..., in1[7]*in2[7]] (lower 32 bits, same as `mullo`)
	[[nodiscard]] constexpr static inline S mul(const S in1,
	                                            const S in2) noexcept {
		return S::mullo(in1, in2);
	}

	/// \return upper 32 bits of the 64 bit products in1[i]*in2[i]
	[[nodiscard]] constexpr static inline S mulhi(const S in1,
	                                              const S in2) noexcept {
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			if constexpr (__unsigned) {
				const uint64x2_t lo = vmull_u32(vget_low_u32(in1.v128[i]), vget_low_u32(in2.v128[i]));
				const uint64x2_t hi = vmull_high_u32(in1.v128[i], in2.v128[i]);
				out.v128[i] = vcombine_u32(vshrn_n_u64(lo, 32), vshrn_n_u64(hi, 32));
			} else {
				const int32x4_t a = vreinterpretq_s32_u32(in1.v128[i]);
				const int32x4_t b = vreinterpretq_s32_u32(in2.v128[i]);
				const int64x2_t lo = vmull_s32(vget_low_s32(a), vget_low_s32(b));
				const int64x2_t hi = vmull_high_s32(a, b);
				out.v128[i] = vreinterpretq_u32_s32(vcombine_s32(vshrn_n_s64(lo, 32), vshrn_n_s64(hi, 32)));
			}
		}
		return out;
	}

	/// horizontal add of the 4 unsigned bytes of each 32 bit lane
	/// \param in1
	/// \return sum of the bytes, per 32 bit lane
	[[nodiscard]] constexpr static inline S hadd_epu8(const S in1) noexcept {
		S ret;
		for (uint32_t i = 0; i < 2; i++) {
			ret.v128[i] = vpaddlq_u16(vpaddlq_u8(vreinterpretq_u8_u32(in1.v128[i])));
		}
		return ret;
	}

	/// horizontal add of the 2 unsigned 16 bit values of each 32 bit lane
	/// \param in1
	/// \return sum, per 32 bit lane
	[[nodiscard]] constexpr static inline S hadd_epu16(const S in1) noexcept {
		S ret;
		for (uint32_t i = 0; i < 2; i++) {
			ret.v128[i] = vpaddlq_u16(vreinterpretq_u16_u32(in1.v128[i]));
		}
		return ret;
	}

	/// number of 4 bit values (nibbles) equal to `in2` in each 32 bit lane
	/// \param in1
	/// \param in2 nibble to count, < 16
	/// \return per 32 bit lane: #{i : nibble i of the lane == in2}, in [0, 8]
	[[nodiscard]] constexpr static inline S histogram_epi4(const S in1,
	                                                       const uint8_t in2) noexcept {
		assert(in2 < 16);
		const uint32x4_t c = vdupq_n_u32(0x11111111u * in2);
		const uint32x4_t m = vdupq_n_u32(0x11111111u);
		S ret;
		for (uint32_t i = 0; i < 2; i++) {
			// bit 0 of a nibble is set, iff the nibble != in2
			const uint32x4_t z = veorq_u32(in1.v128[i], c);
			const uint32x4_t nz = vandq_u32(vorrq_u32(vorrq_u32(z, vshrq_n_u32(z, 1)),
			                                          vorrq_u32(vshrq_n_u32(z, 2), vshrq_n_u32(z, 3))), m);
			const uint32x4_t cnt = vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u32(nz))));
			ret.v128[i] = vsubq_u32(vdupq_n_u32(8), cnt);
		}
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S slli(const S in1,
	                                             const uint8_t in2) noexcept {
		assert(in2 <= 32);
		S out;
		if (std::is_constant_evaluated()) {
			const uint32_t tmp = ~((1u<<in2) - 1u);
			uint32x4_t t = {tmp,tmp,tmp,tmp};

			for (uint32_t i = 0; i < 2; i++) {
				out.v128[i] = in1.v128[i] << in2;
				out.v128[i] &= t;
			}

			return out;
		}
		cryptanalysislib::_Xint32x4_t<__unsigned> helper = cryptanalysislib::_Xint32x4_t<__unsigned>::set1(in2);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = vshlq_u32(in1.v128[i], helper.v128);
		}

		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const uint8_t in2) noexcept {
		assert(in2 <= 32);
		S out;
		if (std::is_constant_evaluated()) {

			for (uint32_t i = 0; i < 2; i++) {
				// logical shift of the unsigned lanes; a shift by the full width yields 0
				out.v128[i] = in2 >= 32u ? (in1.v128[i] ^ in1.v128[i]) : (in1.v128[i] >> in2);
			}

			return out;
		}
		cryptanalysislib::_Xint32x4_t<__unsigned> helper = cryptanalysislib::_Xint32x4_t<__unsigned>::set1(-in2);
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = vshlq_u32(in1.v128[i], helper.v128);
		}

		return out;
	}

	/// \param in1[in]: vector element
	/// \param in2[in]:
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S ror(const S in1,
	                                             const uint8_t in2) noexcept {
		constexpr uint32_t bits = sizeof(limb_type) * 8u;
		const uint32_t s = in2 % bits;
		if (s == 0) {
			return in1;
		}
		return S::or_(S::srli(in1, s), S::slli(in1, bits - s));
	}

	/// \param in1[in]: vector element
	/// \param in2[in]:
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S rol(const S in1,
	                                             const uint8_t in2) noexcept {
		constexpr uint32_t bits = sizeof(limb_type) * 8u;
		const uint32_t s = in2 % bits;
		if (s == 0) {
			return in1;
		}
		return S::or_(S::slli(in1, s), S::srli(in1, bits - s));
	}

	/// lane-wise comparisons of the (unsigned) registers;
	/// signed lanes are compared as signed values if `!__unsigned`
	[[nodiscard]] constexpr static inline uint32x4_t cmpgt128(const uint32x4_t a, const uint32x4_t b) noexcept {
		if constexpr (__unsigned) {
			return vcgtq_u32(a, b);
		} else {
			return vcgtq_s32(vreinterpretq_s32_u32(a), vreinterpretq_s32_u32(b));
		}
	}
	[[nodiscard]] constexpr static inline uint32x4_t cmpge128(const uint32x4_t a, const uint32x4_t b) noexcept {
		if constexpr (__unsigned) {
			return vcgeq_u32(a, b);
		} else {
			return vcgeq_s32(vreinterpretq_s32_u32(a), vreinterpretq_s32_u32(b));
		}
	}
	[[nodiscard]] constexpr static inline uint32x4_t cmplt128(const uint32x4_t a, const uint32x4_t b) noexcept { return cmpgt128(b, a); }
	[[nodiscard]] constexpr static inline uint32x4_t cmple128(const uint32x4_t a, const uint32x4_t b) noexcept { return cmpge128(b, a); }

	[[nodiscard]] constexpr static inline uint32x4_t min128(const uint32x4_t a, const uint32x4_t b) noexcept {
		if constexpr (__unsigned) {
			return vminq_u32(a, b);
		} else {
			return vreinterpretq_u32_s32(vminq_s32(vreinterpretq_s32_u32(a), vreinterpretq_s32_u32(b)));
		}
	}
	[[nodiscard]] constexpr static inline uint32x4_t max128(const uint32x4_t a, const uint32x4_t b) noexcept {
		if constexpr (__unsigned) {
			return vmaxq_u32(a, b);
		} else {
			return vreinterpretq_u32_s32(vmaxq_s32(vreinterpretq_s32_u32(a), vreinterpretq_s32_u32(b)));
		}
	}

	constexpr static inline int gt(const S in1,
                                   const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint32x4_t tmp = cmpgt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi32(tmp) << i * 4;
#else
			const uint32x4_t tmp = cmpgt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi32(tmp) << i * 4;
#endif
		}

		return ret;
	}

	constexpr static inline S gt_(const S in1, const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmpgt128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmpgt128(in1.v128[i], in2.v128[i]);
#endif
		}

		return ret;
	}

	constexpr static inline int ge(const S in1,
                                   const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint32x4_t tmp = cmpge128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi32(tmp) << i * 4;
#else
			const uint32x4_t tmp = cmpge128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi32(tmp) << i * 4;
#endif
		}

		return ret;
	}

	constexpr static inline S ge_(const S in1, const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmpge128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmpge128(in1.v128[i], in2.v128[i]);
#endif
		}

		return ret;
	}

	constexpr static inline int lt(const S in1,
                                   const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint32x4_t tmp = cmplt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi32(tmp) << i * 4;
#else
			const uint32x4_t tmp = cmplt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi32(tmp) << i * 4;
#endif
		}

		return ret;
	}

	constexpr static inline S lt_(const S in1,
                                  const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmplt128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmplt128(in1.v128[i], in2.v128[i]);
#endif
		}

		return ret;
	}

	constexpr static inline int le(const S in1,
                                   const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint32x4_t tmp = cmple128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi32(tmp) << i * 4;
#else
			const uint32x4_t tmp = cmple128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi32(tmp) << i * 4;
#endif
		}

		return ret;
	}

	constexpr static inline S le_(const S in1,
                                  const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmple128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmple128(in1.v128[i], in2.v128[i]);
#endif
		}

		return ret;
	}

	constexpr static inline int eq(const S in1,
                                    const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint32x4_t tmp = vceqq_u32(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi32(tmp) << i * 4;
#else
			const uint32x4_t tmp = in1.v128[i] == in2.v128[i];
			ret ^= _mm_movemask_epi32(tmp) << i * 4;
#endif
		}

		return ret;
	}

	constexpr static inline S eq_(const S in1,
                                   const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = vceqq_u32(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = in1.v128[i] == in2.v128[i];
#endif
		}

		return ret;
	}

    /// \param in1
	/// \param in2
	/// \return {-1, 0, 1}
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
												 const S in2) noexcept {
		S ret;
    	if constexpr (__unsigned) {
			ret.v128[0]  = vcltq_u32(in1.v128[0], in2.v128[0]);
			ret.v128[0] ^= vcgtq_u32(in1.v128[0], in2.v128[0]);
			ret.v128[1]  = vcltq_u32(in1.v128[1], in2.v128[1]);
			ret.v128[1] ^= vcgtq_u32(in1.v128[1], in2.v128[1]);
    	} else {
			ret.v128[0]  = vcltq_s32(in1.v128[0], in2.v128[0]);
			ret.v128[0] ^= vcgtq_s32(in1.v128[0], in2.v128[0]);
			ret.v128[1]  = vcltq_s32(in1.v128[1], in2.v128[1]);
			ret.v128[1] ^= vcgtq_s32(in1.v128[1], in2.v128[1]);
    	}
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
												       const S in2) noexcept {
        S ret = S::cmp_(in1, in2);
		return _mm_movemask_epi32(ret.v128[0]) ^ (_mm_movemask_epi32(ret.v128[1]) << 4u);
	}

	[[nodiscard]] constexpr static inline uint16_t move(const S in1) noexcept {
		uint16_t ret = 0;
		for (uint32_t i = 0; i < 2; i++) {
			ret ^= _mm_movemask_epi32(in1.v128[i]) << i * 4;
		}

		return ret;
	}

	/// same semantics as `vpconflictd` (AVX512CD):
	/// lane i is the bitmask of all lanes j < i with in1[j] == in1[i].
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S conflict(const S in1) noexcept {
		// compare all lanes with the broadcasted lane j, keep the lanes > j
		const uint32x4_t idx0 = {0, 1, 2, 3};
		const uint32x4_t idx1 = {4, 5, 6, 7};
		uint32x4_t r0 = vdupq_n_u32(0), r1 = vdupq_n_u32(0);
		for (uint32_t j = 0; j < LIMBS - 1u; ++j) {
			const uint32x4_t b = vdupq_n_u32(in1.d[j]);
			const uint32x4_t jj = vdupq_n_u32(j);
			const uint32x4_t bit = vdupq_n_u32(1u << j);
			r0 = vorrq_u32(r0, vandq_u32(vandq_u32(vceqq_u32(in1.v128[0], b), vcgtq_u32(idx0, jj)), bit));
			r1 = vorrq_u32(r1, vandq_u32(vandq_u32(vceqq_u32(in1.v128[1], b), vcgtq_u32(idx1, jj)), bit));
		}
		S ret;
		ret.v128[0] = r0;
		ret.v128[1] = r1;
		return ret;
	}

	// NOTE: NEON has no gather instruction (only SVE has), so this is a
	// 	scalar loop.
	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	// NOTE: `scale` defaults to 1 (byte offsets) like the AVX2 gathers
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
												   const S data) {
		S ret;
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 8; i++) {
			// NOTE: byte offsets, so the address may not be aligned
			__builtin_memcpy(&ret.d[i], ptr8 + (data.d[i] * scale), sizeof(uint32_t));
		}

		return ret;
	}

	/// writes data[i] to the byte offset `offset[i] * scale` of `ptr`
	/// \tparam scale
	/// \param ptr
	/// \param offset
	/// \param data
	template<const uint32_t scale = 1>
	constexpr static inline void scatter(void *ptr,
	                                     const S offset,
	                                     const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 8; i++) {
			__builtin_memcpy(ptr8 + (offset.v32[i] * scale), &data.v32[i], sizeof(uint32_t));
		}
	}

	/// gather permutation: ret[i] = in[perm[i] % 8]
	/// \param in
	/// \param perm
	/// \return
	[[nodiscard]] constexpr static inline S permute(const S in,
												    const S perm) {
		S ret;
		for (uint32_t i = 0; i < 8; i++) {
			ret.v32[i] = in.v32[perm.v32[i] & 0x7];
		}
		return ret;
	}

	/// same as the AVX2 version: permutation mask for `permute`, which packs
	/// the lanes selected by `mask` to the front, e.g. 0b10101010 -> [1, 3, 5, 7, 0, 0, 0, 0]
	/// \param mask
	/// \return
	[[nodiscard]] constexpr static inline S pack(const uint32_t mask) noexcept {
		S ret{};
		uint32_t k = 0;
		for (uint32_t i = 0; i < 8; i++) {
			if ((mask >> i) & 1u) {
				ret.v32[k++] = i;
			}
		}
		return ret;
	}

	/// same as the AVX2 version: zero extends the lower 8 bytes of `in` to
	/// 32 bits
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S cvtepu8(const cryptanalysislib::_uint8x16_t in) noexcept {
		const uint16x8_t w = vmovl_u8(vget_low_u8(in.v128));
		S ret;
		ret.v128[0] = vmovl_u16(vget_low_u16(w));
		ret.v128[1] = vmovl_u16(vget_high_u16(w));
		return ret;
	}

	constexpr static inline S popcnt(const S in) {
		S out;
		const cryptanalysislib::_uint16x8_t mask = cryptanalysislib::_uint16x8_t::set1(0xff);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint16x8_t tmp1 = (uint16x8_t) vcntq_u8((uint8x16_t) in.v128[i]);
			const uint16x8_t tmp2 = vaddq_u16(vshrq_n_u16(tmp1, 8), vandq_u16(tmp1, mask.v128));
			// fold the two 16-bit counts; mask the low one so the high one is not counted twice
			out.v128[i] = vaddq_u32(vshrq_n_u32((uint32x4_t) tmp2, 16), vandq_u32((uint32x4_t) tmp2, vdupq_n_u32(0xffffu)));
#else

			const uint16x8_t tmp1 = (uint16x8_t) __builtin_aarch64_popcountv16qi((uint8x16_t) in.v128[i]);
			const uint16x8_t tmp2 = __builtin_aarch64_lshrv8hi_uus(tmp1, 8) + (tmp1 & mask.v128);
			out.v128[i] = __builtin_aarch64_lshrv4si_uus((uint32x4_t) tmp2, 16) + ((uint32x4_t) tmp2 & 0xffffu);
#endif
		}

		return out;
	}

	/// ctz(x) = popcnt((x - 1) & ~x)
	/// \param in
	/// \return the number of trailing zeros of each 32 bit lane (32 for 0)
	[[nodiscard]] constexpr static inline S ctz(const S in) noexcept {
		const S t = S::and_(S::sub(in, S::set1(1)), S::not_(in));
		return S::popcnt(t);
	}
	/// \param in
	/// \return true if all lanes of `in` are equal
	[[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
		for (uint32_t i = 1; i < LIMBS; ++i) {
			if (in.d[0] != in.d[i]) {
				return false;
			}
		}

		return true;
	}

	[[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[LIMBS - 1 - i] = in.d[i];
		}

		return out;
	}

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
		c.v128[0] = min128(a.v128[0], b.v128[0]);
		c.v128[1] = min128(a.v128[1], b.v128[1]);
        return c;
    }

	/// as `_mm512_test_epi32_mask`
	/// \param in1[in]: register to test
	/// \param bit_pos[in]: bit position to test
	/// \return bit i is set, if bit `bit_pos` of `in1[i]` is set
	[[nodiscard]] constexpr static inline Mask<LIMBS> test(const S in1,
	                                                       const uint32_t bit_pos) noexcept {
		const uint32x4_t b = vdupq_n_u32(uint32_t(1u) << bit_pos);
		S t;
		t.v128[0] = vtstq_u32(in1.v128[0], b);
		t.v128[1] = vtstq_u32(in1.v128[1], b);
		return Mask<LIMBS>(MaskType<LIMBS>(S::move(t)));
	}

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
		c.v128[0] = max128(a.v128[0], b.v128[0]);
		c.v128[1] = max128(a.v128[1], b.v128[1]);
        return c;
    }

	/// \param in1[in]:
	/// \return the minimum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_min(const S in1) noexcept {
		if constexpr (__unsigned) {
			return vminvq_u32(vminq_u32(in1.v128[0], in1.v128[1]));
		} else {
			return vminvq_s32(vminq_s32(vreinterpretq_s32_u32(in1.v128[0]), vreinterpretq_s32_u32(in1.v128[1])));
		}
	}

	/// \param in1[in]:
	/// \return the maximum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_max(const S in1) noexcept {
		if constexpr (__unsigned) {
			return vmaxvq_u32(vmaxq_u32(in1.v128[0], in1.v128[1]));
		} else {
			return vmaxvq_s32(vmaxq_s32(vreinterpretq_s32_u32(in1.v128[0]), vreinterpretq_s32_u32(in1.v128[1])));
		}
	}

	/// \param in1[in]: vector element
	/// \param in2[in]: divisor, must not be zero
	/// \return in1 / in2, lane-wise
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
		assert(in2 != 0);
		if constexpr (__unsigned) {
			// NOTE: division by the invariant `in2`, see `internal::udiv_invariant`
			using VT = cryptanalysislib::internal::vec_t<limb_type, 128u / 32>;
			S out;
			for (uint32_t i = 0; i < 2; i++) {
				out.v128[i] = (uint32x4_t) cryptanalysislib::internal::udiv_invariant<limb_type, 128u / 32>((VT) in1.v128[i], in2);
			}
			return out;
		}

		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = in1.d[i] / in2;
		}
		return out;
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are zero, as `_mm512_maskz_compress_epi*`
	/// \param in1[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const Mask<LIMBS> &m) noexcept {
		return compress(in1, S::set1(0), m);
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are taken from `in2`, as `_mm512_mask_compress_epi*`
	/// \param in1[in]:
	/// \param in2[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const S in2,
	                                                 const Mask<LIMBS> &m) noexcept {
		// NOTE: NEON has no compress instruction (only SVE has `compact`)
		const uint64_t k = uint64_t(MaskType<LIMBS>(m));
		S ret = in2;
		uint32_t j = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			if ((k >> i) & 1u) { ret.d[j++] = in1.d[i]; }
		}
		return ret;
	}

	/// writes the lanes of `in1` selected by `m` contiguously to `ptr`,
	/// as `_mm512_mask_compressstoreu_epi*`
	/// \param ptr[out]:
	/// \param in1[in]:
	/// \param m[in]:
	constexpr static inline void compress(limb_type *ptr,
	                                      const S in1,
	                                      const Mask<LIMBS> &m) noexcept {
		const uint64_t k = uint64_t(MaskType<LIMBS>(m));
		uint32_t j = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			if ((k >> i) & 1u) { ptr[j++] = in1.d[i]; }
		}
	}
};

///
using uint32x8_t = Xint32x8_t<true>;
using  int32x8_t = Xint32x8_t<false>;

template<const bool __unsigned=true>
struct Xint64x4_t {
	constexpr static uint32_t LIMBS = 4;
	using limb_type = std::conditional_t<__unsigned, uint64_t, int64_t>;

    using S = Xint64x4_t;

    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		// compatibility with txn_t
		T64  d [ 4];

		T8  v8 [32];
		T16 v16[16];
		T32 v32[ 8];
		T64 v64[ 4];
		uint64x2_t v128[2];
	};

	/// \return number of limbs
	[[nodiscard]] constexpr inline static size_t size() noexcept {
		return LIMBS;
	}

	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
		return __unsigned;
	}

	[[nodiscard]] constexpr inline limb_type& operator[](const uint32_t i) noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	static inline S random() {
		S ret;
		for (uint32_t i = 0; i < 4; ++i) {
			ret.v64[i] = rng();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	[[nodiscard]] constexpr static inline S set(const uint64_t i0,
	                                                     const uint64_t i1,
	                                                     const uint64_t i2,
	                                                     const uint64_t i3) noexcept {
		S ret;
		// first argument -> highest lane (like _mm256_set_epi64x)
		ret.d[3] = i0;
		ret.d[2] = i1;
		ret.d[1] = i2;
		ret.d[0] = i3;
		if (std::is_constant_evaluated()) {
			S out1;
			out1.v128[0] = u64tom128((const uint64_t *)(ret.d + 0));
			out1.v128[1] = u64tom128((const uint64_t *)(ret.d + 2));
			return out1;
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline S setr(const uint64_t i0,
	                                                      const uint64_t i1,
	                                                      const uint64_t i2,
	                                                      const uint64_t i3) noexcept {
		return S::set(i3, i2, i1, i0);
	}

	///
	/// \param a
	/// \return
	constexpr static inline S set1(const uint64_t a) noexcept {
		return S::set(a, a, a, a);
	}

	/// set the `i`-th bit in each limb
	/// \param pos[in]: bit position to set
	constexpr inline void set_bit(const uint32_t pos) noexcept {
		v128[0] = vdupq_n_u64(uint64_t(1u) << pos);
		v128[1] = v128[0];
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = true>
	constexpr static inline S load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u64tom128((const uint64_t *)(ptr + 0));
			out.v128[1] = u64tom128((const uint64_t *)(ptr + 2));
			return out;
		}

		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128[i] = vreinterpretq_u64_u8(vld1q_u8((const uint8_t *) ptr + 16u * i));
		}
		return out;
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u64tom128((const uint64_t *)(ptr + 0));
			out.v128[1] = u64tom128((const uint64_t *)(ptr + 2));
			return out;
		}

		S out;
		for (uint32_t i = 0; i < 2u; ++i) {
			// byte-wise load: alias-safe for any `ptr`, no alignment requirement
			out.v128[i] = vreinterpretq_u64_u8(vld1q_u8((const uint8_t *) ptr + 16u * i));
		}
		return out;
	}

	/// masked load: lane i is loaded if bit i of `m` is set, otherwise 0.
	/// Masked out lanes are not accessed (no fault), as in the AVX-512 version.
	/// \param ptr
	/// \param m lane mask
	/// \return
	[[nodiscard]] static inline S maskz_unaligned_load(const limb_type *ptr,
	                                                   const Mask<LIMBS> &m) noexcept {
		// NOTE: NEON has no masked load and a full load could fault on the
		// 	masked out lanes
		const uint32_t k = uint32_t(MaskType<LIMBS>(m));
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ((k >> i) & 1u) ? ptr[i] : limb_type(0);
		}
		return out;
	}

	/// same as `maskz_unaligned_load` (no alignment needed)
	[[nodiscard]] static inline S maskz_aligned_load(const limb_type *ptr,
	                                                 const Mask<LIMBS> &m) noexcept {
		return maskz_unaligned_load(ptr, m);
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const S in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr,
	                                           const S in) noexcept {
		auto *ptr8 = (uint8_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			// byte-wise store: alias-safe for any `ptr`, no alignment requirement
			vst1q_u8(ptr8 + 16u * i, vreinterpretq_u8_u64(in.v128[i]));
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr,
	                                             const S in) noexcept {
		auto *ptr8 = (uint8_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			// byte-wise store: alias-safe for any `ptr`, no alignment requirement
			vst1q_u8(ptr8 + 16u * i, vreinterpretq_u8_u64(in.v128[i]));
		}
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S xor_(const S in1,
	                                             const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] ^ in2.v128[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S and_(const S in1,
	                                             const S in2) noexcept {
		S out;
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
	[[nodiscard]] constexpr static inline S or_(const S in1,
	                                            const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] | in2.v128[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S andnot(const S in1,
	                                               const S in2) {
		S out;
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = ~in1.v128[i] & in2.v128[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
		S out;
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = ~in1.v128[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S add(const S in1,
	                                            const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] + in2.v128[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S sub(const S in1,
	                                                     const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] - in2.v128[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const S in2) noexcept {
		S out;
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = in1.v128[i] * in2.v128[i];
		}
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const uint64_t in2) noexcept {
		S out;
		for (uint32_t i = 0; i < 4; ++i) {
			out.v64[i] = in1.v64[i] * in2;
		}
		return out;
	}

	/// horizontal add of the 8 unsigned bytes of each 64 bit lane
	/// \param in1
	/// \return sum of the bytes, per 64 bit lane
	[[nodiscard]] constexpr static inline S hadd_epu8(const S in1) noexcept {
		S ret;
		for (uint32_t i = 0; i < 2; i++) {
			ret.v128[i] = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vreinterpretq_u8_u64(in1.v128[i]))));
		}
		return ret;
	}

	/// horizontal add of the 4 unsigned 16 bit values of each 64 bit lane
	/// \param in1
	/// \return sum, per 64 bit lane
	[[nodiscard]] constexpr static inline S hadd_epu16(const S in1) noexcept {
		S ret;
		for (uint32_t i = 0; i < 2; i++) {
			ret.v128[i] = vpaddlq_u32(vpaddlq_u16(vreinterpretq_u16_u64(in1.v128[i])));
		}
		return ret;
	}

	/// horizontal add of the 2 unsigned 32 bit values of each 64 bit lane
	/// \param in1
	/// \return sum, per 64 bit lane
	[[nodiscard]] constexpr static inline S hadd_epu32(const S in1) noexcept {
		S ret;
		for (uint32_t i = 0; i < 2; i++) {
			ret.v128[i] = vpaddlq_u32(vreinterpretq_u32_u64(in1.v128[i]));
		}
		return ret;
	}

	/// number of 4 bit values (nibbles) equal to `in2` in each 64 bit lane
	/// \param in1
	/// \param in2 nibble to count, < 16
	/// \return per 64 bit lane: #{i : nibble i of the lane == in2}, in [0, 16]
	[[nodiscard]] constexpr static inline S histogram_epi4(const S in1,
	                                                       const uint8_t in2) noexcept {
		assert(in2 < 16);
		const uint64x2_t c = vdupq_n_u64(0x1111111111111111ull * in2);
		const uint64x2_t m = vdupq_n_u64(0x1111111111111111ull);
		S ret;
		for (uint32_t i = 0; i < 2; i++) {
			// bit 0 of a nibble is set, iff the nibble != in2
			const uint64x2_t z = veorq_u64(in1.v128[i], c);
			const uint64x2_t nz = vandq_u64(vorrq_u64(vorrq_u64(z, vshrq_n_u64(z, 1)),
			                                          vorrq_u64(vshrq_n_u64(z, 2), vshrq_n_u64(z, 3))), m);
			const uint64x2_t cnt = vpaddlq_u32(vpaddlq_u16(vpaddlq_u8(vcntq_u8(vreinterpretq_u8_u64(nz)))));
			ret.v128[i] = vsubq_u64(vdupq_n_u64(16), cnt);
		}
		return ret;
	}

	[[nodiscard]] constexpr static inline S slli(const S in1,
	                                             const uint8_t in2) noexcept {
		assert(in2 <= 64);
		S out;
		if (std::is_constant_evaluated()) {
			const uint64_t tmp = ~((1ull<<in2) - 1ull);
			uint64x2_t t = {tmp,tmp};

			for (uint32_t i = 0; i < 2; i++) {
				out.v128[i] = in1.v128[i] << in2;
				out.v128[i] &= t;
			}

			return out;
		}

		const cryptanalysislib::_uint64x2_t helper = cryptanalysislib::_uint64x2_t::set1(in2);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = vshlq_u64(in1.v128[i], helper.v128);
		}

		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const uint8_t in2) noexcept {
		assert(in2 <= 64);
		S out;
		if (std::is_constant_evaluated()) {

			for (uint32_t i = 0; i < 2; i++) {
				// logical shift of the unsigned lanes; a shift by the full width yields 0
				out.v128[i] = in2 >= 64u ? (in1.v128[i] ^ in1.v128[i]) : (in1.v128[i] >> in2);
			}

			return out;
		}
		const cryptanalysislib::_Xint64x2_t<__unsigned> helper = cryptanalysislib::_Xint64x2_t<__unsigned>::set1(-in2);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			out.v128[i] = vshlq_u64(in1.v128[i], helper.v128);
		}

		return out;
	}


	/// \param in1[in]: vector element
	/// \param in2[in]:
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S ror(const S in1,
	                                             const uint8_t in2) noexcept {
		constexpr uint32_t bits = sizeof(limb_type) * 8u;
		const uint32_t s = in2 % bits;
		if (s == 0) {
			return in1;
		}
		return S::or_(S::srli(in1, s), S::slli(in1, bits - s));
	}

	/// \param in1[in]: vector element
	/// \param in2[in]:
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S rol(const S in1,
	                                             const uint8_t in2) noexcept {
		constexpr uint32_t bits = sizeof(limb_type) * 8u;
		const uint32_t s = in2 % bits;
		if (s == 0) {
			return in1;
		}
		return S::or_(S::slli(in1, s), S::srli(in1, bits - s));
	}
	///
	/// \param in1
	/// \param in2
	/// \return
	/// runtime version of `permute<in2>`: ret[i] = in1[(in2 >> 2i) & 3]
	constexpr static inline S permute(const S in1,
	                                  const uint32_t in2) noexcept {
		S ret;
		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = in1.v64[(in2 >> (2 * i)) & 0b11];
		}
		return ret;
	}

	/// lane-wise comparisons of the (unsigned) registers;
	/// signed lanes are compared as signed values if `!__unsigned`
	[[nodiscard]] constexpr static inline uint64x2_t cmpgt128(const uint64x2_t a, const uint64x2_t b) noexcept {
		if constexpr (__unsigned) {
			return vcgtq_u64(a, b);
		} else {
			return vcgtq_s64(vreinterpretq_s64_u64(a), vreinterpretq_s64_u64(b));
		}
	}
	[[nodiscard]] constexpr static inline uint64x2_t cmpge128(const uint64x2_t a, const uint64x2_t b) noexcept {
		if constexpr (__unsigned) {
			return vcgeq_u64(a, b);
		} else {
			return vcgeq_s64(vreinterpretq_s64_u64(a), vreinterpretq_s64_u64(b));
		}
	}
	[[nodiscard]] constexpr static inline uint64x2_t cmplt128(const uint64x2_t a, const uint64x2_t b) noexcept { return cmpgt128(b, a); }
	[[nodiscard]] constexpr static inline uint64x2_t cmple128(const uint64x2_t a, const uint64x2_t b) noexcept { return cmpge128(b, a); }

	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline int gt(const S in1,
	                               const S in2) noexcept {
		int ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint64x2_t tmp = cmpgt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi64(tmp) << i * 2;
#else
			const uint64x2_t tmp = cmpgt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi64(tmp) << i * 2;
#endif
		}
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline S gt_(const S in1,
	                              const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmpgt128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmpgt128(in1.v128[i], in2.v128[i]);
#endif
		}

		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline int ge(const S in1,
	                               const S in2) noexcept {
		int ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint64x2_t tmp = cmpge128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi64(tmp) << i * 2;
#else
			const uint64x2_t tmp = cmpge128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi64(tmp) << i * 2;
#endif
		}
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline S ge_(const S in1,
	                              const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmpge128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmpge128(in1.v128[i], in2.v128[i]);
#endif
		}

		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline int lt(const S in1,
	                               const S in2) noexcept {
		int ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint64x2_t tmp = cmplt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi64(tmp) << i * 2;
#else
			const uint64x2_t tmp = cmplt128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi64(tmp) << i * 2;
#endif
		}
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline S lt_(const S in1,
	                              const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmplt128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmplt128(in1.v128[i], in2.v128[i]);
#endif
		}

		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline int le(const S in1,
	                               const S in2) noexcept {
		int ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint64x2_t tmp = cmple128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi64(tmp) << i * 2;
#else
			const uint64x2_t tmp = cmple128(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi64(tmp) << i * 2;
#endif
		}
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline S le_(const S in1,
	                              const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = cmple128(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = cmple128(in1.v128[i], in2.v128[i]);
#endif
		}

		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline int eq(const S in1,
                                   const S in2) noexcept {
		int ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint64x2_t tmp = vceqq_u64(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi64(tmp) << i * 2;
#else
			const uint64x2_t tmp = in1.v128[i] == in2.v128[i];
			ret ^= _mm_movemask_epi64(tmp) << i * 2;
#endif
		}
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline S eq_(const S in1,
								  const S in2) noexcept {
		S ret;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			ret.v128[i] = vceqq_u64(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = in1.v128[i] == in2.v128[i];
#endif
		}
		return ret;
	}
    
    /// \param in1
	/// \param in2
	/// \return {-1, 0, 1}
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
												 const S in2) noexcept {
		S ret;
    	if constexpr (__unsigned) {
			ret.v128[0]  = vcltq_u64(in1.v128[0], in2.v128[0]);
			ret.v128[0] ^= vcgtq_u64(in1.v128[0], in2.v128[0]);
			ret.v128[1]  = vcltq_u64(in1.v128[1], in2.v128[1]);
			ret.v128[1] ^= vcgtq_u64(in1.v128[1], in2.v128[1]);
    	} else {
			ret.v128[0]  = vcltq_s64(in1.v128[0], in2.v128[0]);
			ret.v128[0] ^= vcgtq_s64(in1.v128[0], in2.v128[0]);
			ret.v128[1]  = vcltq_s64(in1.v128[1], in2.v128[1]);
			ret.v128[1] ^= vcgtq_s64(in1.v128[1], in2.v128[1]);
    	}
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
												       const S in2) noexcept {
        S ret = S::cmp_(in1, in2);
		return _mm_movemask_epi64(ret.v128[0]) ^ (_mm_movemask_epi64(ret.v128[1]) << 2u);
	}

	[[nodiscard]] constexpr static inline uint8_t move(const S in1) noexcept {
		uint8_t ret = 0;
		for (uint32_t i = 0; i < 2; i++) {
			ret ^= _mm_movemask_epi64(in1.v128[i]) << i * 2;
		}

		return ret;
	}

	/// same semantics as `vpconflictq` (AVX512CD):
	/// lane i is the bitmask of all lanes j < i with in1[j] == in1[i].
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S conflict(const S in1) noexcept {
		// compare all lanes with the broadcasted lane j, keep the lanes > j
		const uint64x2_t idx0 = {0, 1};
		const uint64x2_t idx1 = {2, 3};
		uint64x2_t r0 = vdupq_n_u64(0), r1 = vdupq_n_u64(0);
		for (uint32_t j = 0; j < LIMBS - 1u; ++j) {
			const uint64x2_t b = vdupq_n_u64(in1.d[j]);
			const uint64x2_t jj = vdupq_n_u64(j);
			const uint64x2_t bit = vdupq_n_u64(uint64_t(1u) << j);
			r0 = vorrq_u64(r0, vandq_u64(vandq_u64(vceqq_u64(in1.v128[0], b), vcgtq_u64(idx0, jj)), bit));
			r1 = vorrq_u64(r1, vandq_u64(vandq_u64(vceqq_u64(in1.v128[1], b), vcgtq_u64(idx1, jj)), bit));
		}
		S ret;
		ret.v128[0] = r0;
		ret.v128[1] = r1;
		return ret;
	}

	/// gathers one 64 bit limb per lane from byte offset data[i] * scale of `ptr`
	/// \param ptr base pointer
	/// \param data offsets, in units of `scale` bytes
	/// \return
	// NOTE: `scale` defaults to 1 (byte offsets) like the AVX2 gathers
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
												   const cryptanalysislib::_uint32x4_t data) {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);

		S ret;
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			// NOTE: byte offsets, so the address may not be aligned
			__builtin_memcpy(&ret.v64[i], ptr8 + data.v32[i] * scale, sizeof(uint64_t));
		}

		return ret;
	}

	/// gathers one 64 bit limb per lane from byte offset data[i] * scale of `ptr`
	/// \param ptr base pointer
	/// \param data offsets, in units of `scale` bytes
	/// \return
	// NOTE: `scale` defaults to 1 (byte offsets) like the AVX2 gathers
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
												   const S data) {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);

		S ret;
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			// NOTE: byte offsets, so the address may not be aligned
			__builtin_memcpy(&ret.v64[i], ptr8 + data.v64[i] * scale, sizeof(uint64_t));
		}

		return ret;
	}

	/// gathers one 64 bit limb per lane from byte offset data[i] * scale of
	/// `ptr`, the offsets are the first 4 bytes of `data`
	/// NOTE: NEON has no gather instruction (only SVE has), so this is a
	/// 	scalar loop.
	/// \param ptr base pointer
	/// \param data offsets, in units of `scale` bytes
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
	                                               const uint8x32_t data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);

		S ret;
		const uint8_t *ptr8 = (const uint8_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			__builtin_memcpy(&ret.v64[i], ptr8 + data.d[i] * scale, sizeof(uint64_t));
		}

		return ret;
	}

	/// writes data[i] to the byte offset `offset[i] * scale` of `ptr`
	/// \tparam scale
	/// \param ptr base pointer
	/// \param offset offsets, in units of `scale` bytes
	/// \param data
	template<const uint32_t scale = 1>
	constexpr static inline void scatter(void *ptr,
	                                     const S offset,
	                                     const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		// NOTE: NEON has no scatter instruction
		uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			__builtin_memcpy(ptr8 + offset.v64[i] * scale, &data.v64[i], sizeof(uint64_t));
		}
	}

	/// permutes the 4 64 bit lanes, like `_mm256_permute4x64_epi64`:
	/// ret[i] = in1[(in2 >> 2i) & 3]
	/// \tparam in2 8 bit immediate, 2 bits per lane
	/// \param in1
	/// \return
	template<const uint32_t in2>
	[[nodiscard]] constexpr static inline S permute(const S in1) {
		S ret;

		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = in1.v64[(in2 >> (2 * i)) & 0b11];
		}
		return ret;
	}

	///
	/// \param in
	/// \return
	constexpr static inline S popcnt(const S in) noexcept {
		S ret;

		const cryptanalysislib::_Xint16x8_t<__unsigned> mask = cryptanalysislib::_Xint16x8_t<__unsigned>::set1(0xff);
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			const uint16x8_t tmp1 = (uint16x8_t) vcntq_u8((uint8x16_t) in.v128[i]);
			const uint16x8_t tmp2 = vaddq_u16(vshrq_n_u16(tmp1, 8), vandq_u16(tmp1, mask.v128));
			// fold 16 -> 32 -> 64 bit counts, masking the low half at every step
			const uint32x4_t tmp3 = vaddq_u32(vshrq_n_u32((uint32x4_t) tmp2, 16), vandq_u32((uint32x4_t) tmp2, vdupq_n_u32(0xffffu)));
			ret.v128[i] = vaddq_u64(vshrq_n_u64((uint64x2_t) tmp3, 32), vandq_u64((uint64x2_t) tmp3, vdupq_n_u64(0xffffffffull)));
		}
		return ret;
	}

	/// all bits below the highest set bit are set, then the zeros are counted
	/// (NEON has no 64 bit `clz`)
	/// \param in
	/// \return the number of leading zeros of each 64 bit lane (64 for 0)
	[[nodiscard]] constexpr static inline S lzcnt(const S in) noexcept {
		S t = in;
		t = S::or_(t, S::srli(t, 1));
		t = S::or_(t, S::srli(t, 2));
		t = S::or_(t, S::srli(t, 4));
		t = S::or_(t, S::srli(t, 8));
		t = S::or_(t, S::srli(t, 16));
		t = S::or_(t, S::srli(t, 32));
		return S::popcnt(S::not_(t));
	}

	/// tzcnt(x) = popcnt((x - 1) & ~x)
	/// \param in
	/// \return the number of trailing zeros of each 64 bit lane (64 for 0)
	[[nodiscard]] constexpr static inline S tzcnt(const S in) noexcept {
		const S t = S::and_(S::sub(in, S::set1(1)), S::not_(in));
		return S::popcnt(t);
	}

	/// \param in
	/// \return true if all lanes of `in` are equal
	[[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
		for (uint32_t i = 1; i < LIMBS; ++i) {
			if (in.d[0] != in.d[i]) {
				return false;
			}
		}

		return true;
	}

	[[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[LIMBS - 1 - i] = in.d[i];
		}

		return out;
	}

    /// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			c.d[i] = std::min(a.d[i], b.d[i]);
		}

        return c;
    }

	/// as `_mm512_test_epi64_mask`
	/// \param in1[in]: register to test
	/// \param bit_pos[in]: bit position to test
	/// \return bit i is set, if bit `bit_pos` of `in1[i]` is set
	[[nodiscard]] constexpr static inline Mask<LIMBS> test(const S in1,
	                                                       const uint32_t bit_pos) noexcept {
		const uint64x2_t b = vdupq_n_u64(uint64_t(1u) << bit_pos);
		S t;
		t.v128[0] = vtstq_u64(in1.v128[0], b);
		t.v128[1] = vtstq_u64(in1.v128[1], b);
		return Mask<LIMBS>(MaskType<LIMBS>(S::move(t)));
	}

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			c.d[i] = std::max(a.d[i], b.d[i]);
		}

        return c;
    }

	/// \param in1[in]:
	/// \return the minimum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_min(const S in1) noexcept {
		// NEON has no 64 bit min: compare + select
		uint64x2_t gt;
		if constexpr (__unsigned) {
			gt = vcgtq_u64(in1.v128[0], in1.v128[1]);
		} else {
			gt = vcgtq_s64(vreinterpretq_s64_u64(in1.v128[0]), vreinterpretq_s64_u64(in1.v128[1]));
		}
		const uint64x2_t t = vbslq_u64(gt, in1.v128[1], in1.v128[0]);
		const limb_type a = limb_type(vgetq_lane_u64(t, 0)), b = limb_type(vgetq_lane_u64(t, 1));
		return a < b ? a : b;
	}

	/// \param in1[in]:
	/// \return the maximum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_max(const S in1) noexcept {
		// NEON has no 64 bit max: compare + select
		uint64x2_t gt;
		if constexpr (__unsigned) {
			gt = vcgtq_u64(in1.v128[0], in1.v128[1]);
		} else {
			gt = vcgtq_s64(vreinterpretq_s64_u64(in1.v128[0]), vreinterpretq_s64_u64(in1.v128[1]));
		}
		const uint64x2_t t = vbslq_u64(gt, in1.v128[0], in1.v128[1]);
		const limb_type a = limb_type(vgetq_lane_u64(t, 0)), b = limb_type(vgetq_lane_u64(t, 1));
		return a > b ? a : b;
	}

	/// \param in1[in]: vector element
	/// \param in2[in]: divisor, must not be zero
	/// \return in1 / in2, lane-wise
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
		assert(in2 != 0);
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = in1.d[i] / in2;
		}
		return out;
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are zero, as `_mm512_maskz_compress_epi*`
	/// \param in1[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const Mask<LIMBS> &m) noexcept {
		return compress(in1, S::set1(0), m);
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are taken from `in2`, as `_mm512_mask_compress_epi*`
	/// \param in1[in]:
	/// \param in2[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const S in2,
	                                                 const Mask<LIMBS> &m) noexcept {
		// NOTE: NEON has no compress instruction (only SVE has `compact`)
		const uint64_t k = uint64_t(MaskType<LIMBS>(m));
		S ret = in2;
		uint32_t j = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			if ((k >> i) & 1u) { ret.d[j++] = in1.d[i]; }
		}
		return ret;
	}

	/// writes the lanes of `in1` selected by `m` contiguously to `ptr`,
	/// as `_mm512_mask_compressstoreu_epi*`
	/// \param ptr[out]:
	/// \param in1[in]:
	/// \param m[in]:
	constexpr static inline void compress(limb_type *ptr,
	                                      const S in1,
	                                      const Mask<LIMBS> &m) noexcept {
		const uint64_t k = uint64_t(MaskType<LIMBS>(m));
		uint32_t j = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			if ((k >> i) & 1u) { ptr[j++] = in1.d[i]; }
		}
	}
};

///
using uint64x4_t = Xint64x4_t<true>;
using  int64x4_t = Xint64x4_t<false>;
#endif
