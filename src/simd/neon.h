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
			uint8_t d[16];

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

		[[nodiscard]] constexpr static inline _Xint8x16_t set1(const limb_type i) noexcept {
			_Xint8x16_t ret;
			for (uint32_t j = 0; j < 16u; ++j) {
				ret.v8[j] = i;
			}
			return ret;
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
			auto *ptr128 = (poly128_t *) ptr;
			_Xint8x16_t out;
#ifndef __clang__
			out.v128 = (uint8x16_t) (*ptr128);
#else
			out.v128 = (uint8x16_t) __builtin_neon_vldrq_p128(ptr128);
#endif
			return out;
		}

		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint8x16_t unaligned_load(const void *ptr) noexcept {
			auto *ptr128 = (uint8x16_t *) ptr;
			_Xint8x16_t out;
			out.v128 = (uint8x16_t) *ptr128;
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
			auto *ptr128 = (uint8x16_t *) ptr;
			*ptr128 = in.v128;
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
        	if constexpr (__unsigned) {
        		out.v128 = vmulq_n_u8(in1.v128, in2);
        	} else {
        		out.v128 = vmulq_n_s8(in1.v128, in2);
        	}
        	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]:
	    /// \return TODO optimize
	    [[nodiscard]] constexpr static inline S div(const S in1,
	                                                const limb_type in2) noexcept {
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
	    	S out;
        	out.v128 = vshlq_n_u8(in1.v128, in2);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
        	S out;
        	out.v128 = vshrq_n_u8(in1.v128, in2);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]:
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S ror(const S in1,
	                                                 const limb_type in2) noexcept {

	    	S out, t = S::set1(in2);
        	// TODO out.v128 = vrshrq_s8(in1.v128, t.v128);
	    	return out;

        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]:
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S rol(const S in1,
	                                                 const uint8_t in2) noexcept {
	    	S out;
        	// TODO out.v128 = vrshrq_n_s8(in1.v128, in2);
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
		/// \return in1 == in2 ucompressed
		[[nodiscard]] constexpr static inline S cmp_(const S in1,
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
		[[nodiscard]] constexpr static inline uint32_t cmp(const _Xint8x16_t in1,
		                                                   const _Xint8x16_t in2) noexcept {
        	if constexpr (__unsigned) {
				return _mm_movemask_epi8(vceqq_u8(in1.v128, in2.v128));
        	} else {
				return _mm_movemask_epi8(vceqq_s8(in1.v128, in2.v128));
        	}
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
        	// TODO
        	return 0;
        }

        // just shuffle the 16 u8 elements
	    [[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
	    	S ret;
        	if constexpr (__unsigned) {
        		ret.v128 = vrbitq_u8(in.v128);
        	} else {
        		ret.v128 = vrbitq_u8(in.v128);
        	}
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
	    	for (uint32_t i = 0; i < 8; i++) {
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

        constexpr inline _Xint16x8_t operator=(const _Xint8x16_t<> &b) noexcept;
		constexpr inline _Xint16x8_t operator=(const _Xint32x4_t<> &b) noexcept;
		constexpr inline _Xint16x8_t operator=(const _Xint64x2_t<> &b) noexcept;

		constexpr _Xint16x8_t() noexcept = default;
		constexpr _Xint16x8_t(const _Xint8x16_t<> &b) noexcept;
		constexpr _Xint16x8_t(const _Xint32x4_t<> &b) noexcept;
		constexpr _Xint16x8_t(const _Xint64x2_t<> &b) noexcept;

		union {
			// compatibility to `TxN_t`
			uint16_t d[8];

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

		[[nodiscard]] constexpr static inline _Xint16x8_t set1(const uint16_t a) noexcept {
			_Xint16x8_t ret;
			for (uint32_t i = 0; i < 8; ++i) {
				ret.v16[i] = a;
			}
			return ret;
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
			auto *ptr128 = (poly128_t *) ptr;
			_Xint16x8_t out;
#ifndef __clang__
			out.v128 = (uint16x8_t) (*ptr128);
#else
			out.v128 = (uint16x8_t) __builtin_neon_vldrq_p128(ptr128);
#endif
			return out;
		}


		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint16x8_t unaligned_load(const void *ptr) noexcept {
			auto *ptr128 = (poly128_t *) ptr;
			_Xint16x8_t out;
#ifndef __clang__
			out.v128 = (uint16x8_t) (*ptr128);
#else
			out.v128 = (uint16x8_t) __builtin_neon_vldrq_p128(ptr128);
#endif
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
			auto *ptr128 = (uint16x8_t *) ptr;
			*ptr128 = in.v128;
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr, const _Xint16x8_t in) noexcept {
			auto *ptr128 = (uint16x8_t *) ptr;
			*ptr128 = in.v128;
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
			uint32_t d[4];

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

		[[nodiscard]] constexpr static inline _Xint32x4_t set1(const uint32_t a) noexcept {
			_Xint32x4_t ret;
			for (uint32_t i = 0; i < 4; ++i) {
				ret.v32[i] = a;
			}
			return ret;
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
			auto *ptr128 = (poly128_t *) ptr;
			_Xint8x16_t out;
#ifndef __clang__
			out.v128 = (uint32x4_t) (*ptr128);
#else
			out.v128 = (uint32x4_t) __builtin_neon_vldrq_p128(ptr128);
#endif
			return out;
		}


		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint32x4_t unaligned_load(const void *ptr) noexcept {
			auto *ptr128 = (poly128_t *) ptr;
			_Xint8x16_t out;
#ifndef __clang__
			out.v128 = (uint32x4_t) (*ptr128);
#else
			out.v128 = (uint32x4_t) __builtin_neon_vldrq_p128(ptr128);
#endif
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
			auto *ptr128 = (_Xint32x4_t *) ptr;
			*ptr128 = in;
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr, const _Xint32x4_t in) noexcept {
			auto *ptr128 = (_Xint32x4_t *) ptr;
			*ptr128 = in;
		}
	};

    template<const bool __unsigned>
	struct _Xint64x2_t {
		constexpr static uint32_t LIMBS = 2;
		using limb_type = std::conditional_t<__unsigned, uint64_t, int64_t>;

		constexpr inline _Xint64x2_t operator=(const _Xint8x16_t<> &b) noexcept;
		constexpr inline _Xint64x2_t operator=(const _Xint16x8_t<> &b) noexcept;
		constexpr inline _Xint64x2_t operator=(const _Xint32x4_t<> &b) noexcept;

		constexpr _Xint64x2_t() noexcept = default;
		constexpr _Xint64x2_t(const _Xint8x16_t<> &b) noexcept;
		constexpr _Xint64x2_t(const _Xint16x8_t<> &b) noexcept;
		constexpr _Xint64x2_t(const _Xint32x4_t<> &b) noexcept;

		union {
			uint64_t d[2];

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

		[[nodiscard]] constexpr static inline _Xint64x2_t set1(const uint64_t a) noexcept {
			_Xint64x2_t ret;
			for (uint32_t i = 0; i < 2; ++i) {
				ret.v64[i] = a;
			}
			return ret;
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
			auto *ptr128 = (poly128_t *) ptr;
			_Xint8x16_t out;
#ifndef __clang__
			out.v128 = (uint64x2_t) (*ptr128);
#else
			out.v128 = (uint64x2_t) __builtin_neon_vldrq_p128(ptr128);
#endif
			return out;
		}


		///
		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint64x2_t unaligned_load(const void *ptr) noexcept {
			auto *ptr128 = (poly128_t *) ptr;
			_Xint8x16_t out;
#ifndef __clang__
			out.v128 = (uint64x2_t) (*ptr128);
#else
			out.v128 = (uint64x2_t) __builtin_neon_vldrq_p128(ptr128);
#endif
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
			auto *ptr128 = (uint64x2_t *) ptr;
			*ptr128 = in.v128;
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr,
													 const _Xint64x2_t in) noexcept {
			auto *ptr128 = (uint64x2_t *) ptr;
			*ptr128 = in.v128;
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

// implementation of `_mm_shuffle_epi16`
inline uint16x8_t shuffle_epi16(const uint16x8_t a,
								const uint16x8_t b) {
    const uint16x8_t tmp = b*2;
    const uint16x8_t s  = tmp ^ vshlq_n_u16(tmp, 8);
    const uint16x8_t ss = vaddq_u16(s, vdupq_n_u16(0x100));
    return ss; // TODO shuffle_epi8(a, ss);
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
			out1.v128[0] = u8tom128(out.d +  0);
			out1.v128[1] = u8tom128(out.d + 16);
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
			out1.v128[0] = u8tom128(out.d +  0);
			out1.v128[1] = u8tom128(out.d + 16);
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

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	constexpr static inline S load(const uint8_t *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline S aligned_load(const uint8_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u8tom128(ptr +  0);
			out.v128[1] = u8tom128(ptr + 16);
			return out;
		}

		auto *ptr128 = (poly128_t *) ptr;
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (uint8x16_t) vldrq_p128(ptr128);
#else
			out.v128[i] = (uint8x16_t) __builtin_neon_vldrq_p128(ptr128 + i);
#endif
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	constexpr static inline S unaligned_load(const uint8_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u8tom128(ptr +  0);
			out.v128[1] = u8tom128(ptr + 16);
			return out;
		}

		auto *ptr128 = (poly128_t *) ptr;
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (uint8x16_t) vldrq_p128(ptr128);
#else
			out.v128[i] = (uint8x16_t) __builtin_neon_vldrq_p128(ptr128 + i);
#endif
		}
		return out;
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

		aligned_store(ptr, in);
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
		auto *ptr128 = (uint8x16_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			ptr128[i] = in.v128[i];
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
			out.v128[i] = ~(in1.v128[i] & in2.v128[i]);
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
			/// TODO not correct, carry and sruff
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
	                                              const uint8_t in2) {
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
			const uint8_t tmp = (1u<<in2) - 1;
			uint8x16_t t = {tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp};

			for (uint32_t i = 0; i < 2; i++) {
				out.v128[i] = in1.v128[i] >> in2;
				out.v128[i] &= t;
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

		S out;
    	// out.v128[0] = vrshrq_n_u8(in1.v128[0], in2);
    	// out.v128[1] = vrshrq_n_u8(in1.v128[1], in2);
		return out;

    }

	/// \param in1[in]: vector element
	/// \param in2[in]:
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S rol(const S in1,
	                                             const uint8_t in2) noexcept {
		S out;// TODO
    	// out.v128[0] = vrshrq_n_u8(in1.v128[0], in2);
    	// out.v128[1] = vrshrq_n_u8(in1.v128[1], in2);
		return out;
    }

	[[nodiscard]] constexpr static inline uint32_t gt(const S in1,
	                                                  const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint8x16_t tmp = vcgtq_u8(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi8(tmp) << i * 16;
#else
			const uint8x16_t tmp = in1.v128[i] > in2.v128[i];
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
			ret.v128[i] = vcgtq_u8(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = in1.v128[i] > in2.v128[i];
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
			const uint8x16_t tmp = vcltq_u8(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi8(tmp) << i * 16;
#else
			const uint8x16_t tmp = in1.v128[i] < in2.v128[i];
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
			ret.v128[i] = vcltq_u8(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = in1.v128[i] < in2.v128[i];
#endif
		}
		return ret;
	}


	[[nodiscard]] constexpr static inline int cmp(const S in1,
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

	[[nodiscard]] constexpr static inline uint32_t move(const S in1) noexcept {
		uint32_t ret = 0;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			ret ^= _mm_movemask_epi8(in1.v128[i]) << i * 16;
		}
		return ret;
	}

	/// TODO
	/// \param in
	/// \return
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

    /// TODO: not optimized
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

    /// TODO: not optimized
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

	[[nodiscard]] constexpr static inline S set(uint16_t __q31, uint16_t __q30, uint16_t __q29, uint16_t __q28,
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
			out1.v128[0] = u16tom128(out.d + 0);
			out1.v128[1] = u16tom128(out.d + 8);
			return out1;
		}
		return out;
	}

	[[nodiscard]] constexpr static inline S setr(uint16_t __q31, uint16_t __q30, uint16_t __q29, uint16_t __q28,
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
			out1.v128[0] = u16tom128(out.d + 0);
			out1.v128[1] = u16tom128(out.d + 8);
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

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	constexpr static inline S load(const uint16_t *ptr) {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline S aligned_load(const uint16_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u16tom128(ptr + 0);
			out.v128[1] = u16tom128(ptr + 8);
			return out;
		}

		auto *ptr128 = (poly128_t *) ptr;
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (uint16x8_t) vldrq_p128(ptr128 + i);
#else
			out.v128[i] = (uint16x8_t) __builtin_neon_vldrq_p128(ptr128 + i);
#endif
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	constexpr static inline S unaligned_load(const uint16_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u16tom128(ptr + 0);
			out.v128[1] = u16tom128(ptr + 8);
			return out;
		}

		auto *ptr128 = (poly128_t *) ptr;
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (uint16x8_t) vldrq_p128(ptr128 + i);
#else
			out.v128[i] = (uint16x8_t) __builtin_neon_vldrq_p128(ptr128 + i);
#endif
		}
		return out;
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

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr,
                                               const S in) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			vstrq_p128(ptr128 + i, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128 + i, (poly128_t) in.v128[i]);
#endif
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const S in) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __GNUC__
			vstrq_p128(ptr128 + i, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128 + i, (poly128_t) in.v128[i]);
#endif
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
			out.v128[i] = ~(in1.v128[i] & in2.v128[i]);
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
	                                              const uint8_t in2) {
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
			const uint16_t tmp = ((1u<<in2) - 1);
			uint16x8_t t = {tmp,tmp,tmp,tmp,tmp,tmp,tmp,tmp};

			for (uint32_t i = 0; i < 2; i++) {
				out.v128[i] = in1.v128[i] >> in2;
				out.v128[i] &= t;
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

		S out;
    	// out.v128[0] = vrshrq_n_u16(in1.v128[0], in2);
    	// out.v128[1] = vrshrq_n_u16(in1.v128[1], in2);
		return out;

    }

	/// \param in1[in]: vector element
	/// \param in2[in]:
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S rol(const S in1,
	                                             const uint8_t in2) noexcept {
		S out;
    	// out.v128[0] = vrshrq_n_u16(in1.v128[0], in2);
    	// out.v128[1] = vrshrq_n_u16(in1.v128[1], in2);
		return out;
    }

	constexpr static inline int gt(const S in1,
								   const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint16x8_t tmp = vcgtq_u16(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi16(tmp) << i * 8;
#else
			const uint16x8_t tmp = in1.v128[i] > in2.v128[i];
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
			ret.v128[i] = vcgtq_u16(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = in1.v128[i] > in2.v128[i];
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
			const uint16x8_t tmp = vcltq_u16(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi16(tmp) << i * 8;
#else
			const uint16x8_t tmp = in1.v128[i] < in2.v128[i];
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
			ret.v128[i] = vcltq_u16(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = in1.v128[i] < in2.v128[i];
#endif
		}

		return ret;
	}

	constexpr static inline int cmp(const S in1,
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

	constexpr static inline S cmp_(const S in1,
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

	/// TODO
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
		for (uint32_t i = 1; i < LIMBS; ++i) {
			if (in.v16[0] != in.v16[i]) {
				return false;
			}
		}

		return true;
	}

	[[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
		S out;
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.v16[LIMBS - 1 - i] = in.v16[i];
		}

		return out;
	}

	constexpr static inline uint16_t move(const S in1) noexcept {
		uint32_t ret = 0;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
			ret ^= _mm_movemask_epi16(in1.v128[i]) << i * 8;
		}
		return ret;
	}

    /// TODO
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

    /// TODO
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

	[[nodiscard]] constexpr static inline S set(uint32_t __q31, uint32_t __q30, uint32_t __q29, uint32_t __q28,
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
			out1.v128[0] = u32tom128(out.d + 0);
			out1.v128[1] = u32tom128(out.d + 4);
			return out1;
		}
		return out;
	}

	[[nodiscard]] constexpr static inline S setr(uint32_t __q31, uint32_t __q30, uint32_t __q29, uint32_t __q28,
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
			out1.v128[0] = u32tom128(out.d + 0);
			out1.v128[1] = u32tom128(out.d + 4);
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

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	constexpr static inline S load(const uint32_t *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline S aligned_load(const uint32_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u32tom128(ptr + 0);
			out.v128[1] = u32tom128(ptr + 4);
			return out;
		}

		auto *ptr128 = (poly128_t *) ptr;
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (uint32x4_t) vldrq_p128(ptr128 + i);
#else
			out.v128[i] = (uint32x4_t) __builtin_neon_vldrq_p128(ptr128 + i);
#endif
		}
		return out;
	}


	///
	/// \param ptr
	/// \return
	constexpr static inline S unaligned_load(const uint32_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u32tom128(ptr + 0);
			out.v128[1] = u32tom128(ptr + 4);
			return out;
		}

		auto *ptr128 = (poly128_t *) ptr;
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (uint32x4_t) vldrq_p128(ptr128 + i);
#else
			out.v128[i] = (uint32x4_t) __builtin_neon_vldrq_p128(ptr128 + i);
#endif
		}
		return out;
	}


	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = true>
	constexpr static inline void store(void *ptr,
                                       const S in) noexcept {
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
                                     const S in) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			vstrq_p128(ptr128 + i, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128 + i, (poly128_t) in.v128[i]);
#endif
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr,
                                                 const S in) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			vstrq_p128(ptr128 + i, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128 + i, (poly128_t) in.v128[i]);
#endif
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
			out.v128[i] = ~(in1.v128[i] & in2.v128[i]);
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
			/// TODO not correct, carry and sruff
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
	                                              const uint8_t in2) {
		S rs = S::set1(in2);
		return S::mullo(in1, rs);
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
			const uint32_t tmp = (1u<<in2) - 1u;
			uint32x4_t t = {tmp,tmp,tmp,tmp};

			for (uint32_t i = 0; i < 2; i++) {
				out.v128[i] = in1.v128[i] >> in2;
				out.v128[i] &= t;
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
		S out;
    	// out.v128[0] = vrshrq_n_u32(in1.v128[0], in2);
    	// out.v128[1] = vrshrq_n_u32(in1.v128[1], in2);
		return out;

    }

	/// \param in1[in]: vector element
	/// \param in2[in]:
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S rol(const S in1,
	                                             const uint8_t in2) noexcept {
		S out;
    	// out.v128[0] = vrshrq_n_u32(in1.v128[0], in2);
    	// out.v128[1] = vrshrq_n_u32(in1.v128[1], in2);
		return out;
    }

	constexpr static inline int gt(const S in1,
                                   const S in2) noexcept {
		uint32_t ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint32x4_t tmp = vcgtq_u32(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi32(tmp) << i * 4;
#else
			const uint32x4_t tmp = in1.v128[i] > in2.v128[i];
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
			ret.v128[i] = vcgtq_u32(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = in1.v128[i] > in2.v128[i];
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
			const uint32x4_t tmp = vcltq_u32(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi32(tmp) << i * 4;
#else
			const uint32x4_t tmp = in1.v128[i] < in2.v128[i];
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
			ret.v128[i] = vcltq_u32(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = in1.v128[i] < in2.v128[i];
#endif
		}

		return ret;
	}

	constexpr static inline int cmp(const S in1,
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
	constexpr static inline S cmp_(const S in1,
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

	[[nodiscard]] constexpr static inline uint16_t move(const S in1) noexcept {
		uint16_t ret = 0;
		for (uint32_t i = 0; i < 2; i++) {
			ret ^= _mm_movemask_epi32(in1.v128[i]) << i * 8;
		}

		return ret;
	}

	// TODO arm instruction
	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 4>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
												   const S data) {
		S ret;
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 8; i++) {
			ret.d[i] = *(uint32_t *) (ptr8 + (data.d[i] * scale));
		}

		return ret;
	}

	/// TODO
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

	constexpr static inline S popcnt(const S in) {
		S out;
		const cryptanalysislib::_uint16x8_t mask = cryptanalysislib::_uint16x8_t::set1(0xff);

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint16x8_t tmp1 = (uint16x8_t) vcntq_u8((uint8x16_t) in.v128[i]);
			const uint16x8_t tmp2 = vaddq_u16(vshrq_n_u16(tmp1, 8), vandq_u16(tmp1, mask.v128));
			out.v128[i] = vaddq_u32(vshrq_n_u32((uint32x4_t) tmp2, 16), (uint32x4_t) tmp2);
#else

			const uint16x8_t tmp1 = (uint16x8_t) __builtin_aarch64_popcountv16qi((uint8x16_t) in.v128[i]);
			const uint16x8_t tmp2 = __builtin_aarch64_lshrv8hi_uus(tmp1, 8) + (tmp1 & mask.v128);
			out.v128[i] = __builtin_aarch64_lshrv4si_uus((uint32x4_t) tmp2, 16) + (uint32x4_t) tmp2;
#endif
		}

		return out;
	}
	/// TODO
	/// \param in
	/// \return
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

	/// TODO implement everywhere
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
        return c;
    }

	/// TODO implement everywhere
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
        return c;
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
		ret.d[0] = i0;
		ret.d[1] = i1;
		ret.d[2] = i2;
		ret.d[3] = i3;
		if (std::is_constant_evaluated()) {
			S out1;
			out1.v128[0] = u64tom128(ret.d + 0);
			out1.v128[1] = u64tom128(ret.d + 2);
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

	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = true>
	constexpr static inline S load(const uint64_t *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline S aligned_load(const uint64_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u64tom128(ptr + 0);
			out.v128[1] = u64tom128(ptr + 2);
			return out;
		}

		auto *ptr128 = (poly128_t *) ptr;
		S out;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (uint64x2_t) vldrq_p128(ptr128 + i);
#else
			out.v128[i] = (uint64x2_t) __builtin_neon_vldrq_p128(ptr128 + i);
#endif
		}
		return out;
	}

	///
	/// \param ptr
	/// \return
	constexpr static inline S unaligned_load(const uint64_t *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			S out;
			out.v128[0] = u64tom128(ptr + 0);
			out.v128[1] = u64tom128(ptr + 2);
			return out;
		}

		auto *ptr128 = (poly128_t *) ptr;
		S out;
		for (uint32_t i = 0; i < 2u; ++i) {
#ifndef __clang__
			out.v128[i] = (uint64x2_t) vldrq_p128(ptr128 + i);
#else
			out.v128[i] = (uint64x2_t) __builtin_neon_vldrq_p128(ptr128 + i);
#endif
		}
		return out;
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = true>
	constexpr static inline void store(void *ptr, const S in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr,
	                                           const S in) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			vstrq_p128(ptr128 + i, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128 + i, (poly128_t) in.v128[i]);
#endif
		}
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr,
	                                             const S in) noexcept {
		auto *ptr128 = (poly128_t *) ptr;
		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifndef __clang__
			vstrq_p128(ptr128 + i, (poly128_t) in.v128[i]);
#else
			__builtin_neon_vstrq_p128(ptr128 + i, (poly128_t) in.v128[i]);
#endif
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
			out.v128[i] = ~(in1.v128[i] & in2.v128[i]);
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
		assert(in2 <= 8);
		S out;
		if (std::is_constant_evaluated()) {
			const uint64_t tmp = (1ull<<in2) - 1ull;
			uint64x2_t t = {tmp,tmp};

			for (uint32_t i = 0; i < 2; i++) {
				out.v128[i] = in1.v128[i] >> in2;
				out.v128[i] &= t;
			}

			return out;
		}
		const cryptanalysislib::_Xint64x2_t<__unsigned> helper = cryptanalysislib::_Xint64x2_t<__unsigned>::set1(in2);

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
		S out;
    	// out.v128[0] = vrshrq_n_u64(in1.v128[0], in2);
    	// out.v128[1] = vrshrq_n_u64(in1.v128[1], in2);
		return out;

    }

	/// \param in1[in]: vector element
	/// \param in2[in]:
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S rol(const S in1,
	                                             const uint8_t in2) noexcept {
		S out;
    	//out.v128[0] = vrshrq_n_u64(in1.v128[0], in2);
    	//out.v128[1] = vrshrq_n_u64(in1.v128[1], in2);
		return out;//
    }
	///
	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline S permute(const S in1,
	                                  const uint32_t in2) noexcept {
		S ret;
		(void) in1;
		(void) in2;

		assert(0); // TODO
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline int gt(const S in1,
	                               const S in2) noexcept {
		int ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint64x2_t tmp = vcgtq_u64(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi64(tmp) << i * 2;
#else
			const uint64x2_t tmp = in1.v128[i] > in2.v128[i];
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
			ret.v128[i] = vcgtq_u64(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = in1.v128[i] > in2.v128[i];
#endif
		}

		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline int lt(const S in1,
	                               const S in2) noexcept {
		int ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint64x2_t tmp = vcltq_u64(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi64(tmp) << i * 2;
#else
			const uint64x2_t tmp = in1.v128[i] < in2.v128[i];
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
			ret.v128[i] = vcltq_u64(in1.v128[i], in2.v128[i]);
#else
			ret.v128[i] = in1.v128[i] < in2.v128[i];
#endif
		}

		return ret;
	}
	///
	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline int cmp(const S in1,
                                    const S in2) noexcept {
		int ret = 0;

		LOOP_UNROLL()
		for (uint32_t i = 0; i < 2; ++i) {
#ifdef __clang__
			const uint64x2_t tmp = vceqq_u64(in1.v128[i], in2.v128[i]);
			ret ^= _mm_movemask_epi64(tmp) << i * 4;
#else
			const uint64x2_t tmp = in1.v128[i] == in2.v128[i];
			ret ^= _mm_movemask_epi64(tmp) << i * 4;
#endif
		}
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	constexpr static inline S cmp_(const S in1,
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

	[[nodiscard]] constexpr static inline uint8_t move(const S in1) noexcept {
		uint8_t ret = 0;
		for (uint32_t i = 0; i < 2; i++) {
			ret ^= _mm_movemask_epi64(in1.v128[i]) << i * 4;
		}

		return ret;
	}

	/// TODO
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 8>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
												   const cryptanalysislib::_uint32x4_t data) {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);

		S ret;
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = *(uint64_t *) (ptr8 + data.v32[i] * scale);
		}

		return ret;
	}

	/// TODO
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 8>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
												   const S data) {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);

		S ret;
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			ret.v64[i] = *(uint64_t *) (ptr8 + data.v64[i] * scale);
		}

		return ret;
	}

	/// TODO
	/// \tparam in2
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
#ifndef __clang__
			const uint16x8_t tmp1 = (uint16x8_t) vcntq_u8((uint8x16_t) in.v128[i]);
			const uint16x8_t tmp2 = vaddq_u16(vshrq_n_u16(tmp1, 8), vandq_u16(tmp1, mask.v128));
			const uint32x4_t tmp3 = vaddq_u32(vshrq_n_u32((uint32x4_t) tmp2, 16), (uint32x4_t) tmp2);
			ret.v128[i] = vaddq_u64(vshrq_n_u64((uint64x2_t) tmp3, 32), (uint64x2_t) tmp3);
#else
			// TODO
			assert(false);
#endif
		}
		return ret;
	}

	/// TODO
	/// \param in
	/// \return
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
};

///
using uint64x4_t = Xint64x4_t<true>;
using  int64x4_t = Xint64x4_t<false>;
#endif
