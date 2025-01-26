#ifndef CRYPTANALYSISLIB_SIMD_AVX2_H
#define CRYPTANALYSISLIB_SIMD_AVX2_H

#ifndef CRYPTANALYSISLIB_SIMD_H
#error "dont include this file directly. Use `#include <simd/simd.h>`"
#endif

#ifndef USE_AVX2
#error "no avx2 enabled."
#endif

#include <immintrin.h>
#include <type_traits>


#include <cstdint>
#include <cstdio>
#include <immintrin.h>

#include "helper.h"
#include "algorithm/bits/popcount.h"
#include "random.h"


/// justification for the unsinged comparison
/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIAMxcpK4AMngMmAByPgBGmMQgAGwapAAOqAqETgwe3r4BQemZjgJhEdEscQnJtpj2JQxCBEzEBLk%2BfoG19dlNLQRlUbHxSSkKza3t%2BV3j/YMVVaMAlLaoXsTI7Bzm/uHI3lgA1Cb%2BbngsLOEExOEAdAgn2CYaAILPLwD6HyxmAKyJeEOCloBAAHBA0Axxocvj9/oCmKRDpDobC/gDDjElscAOxWV7PACcxEwBHWDEOEDR8OxUC%2BADd/GYAI4KJZMY6nSkMpmspZYk74t44gAi72pGK8IPBKIIMO%2B6IRSNl8rhGKxuKFRJJZOIFKpCpplINjJZXnZnLc3I%2BpuZ5oF/i1ove4sNGIUwDBEIEqLdSuRPrlEsBGpMeJdGmJpPJ1rVeFpJt5bI5j2ttrZDqdYtewcOXk9MsDqsVh0RAahQb9mOxYa1kZ1MYNcYTPLNFtTibbmZdzoJfbeOarwIIXES3orxfhpeVRdzofDfajuv1wZbNtHCHjKa5Jo38e7BN7nyrUpHY5VubLF6r87rS8bq%2BNDI35u3Vt3iQQ9qWgp72YHx5xkCnqjuOvpAVes43jWC4Dveeqxoqa70nu7b%2BNgaaoQewr/rm%2BZnmBlYQTOE5zjBd4NghTZIU%2B66fq%2BnIYR%2BX78j%2Bjp/hGAG5sOTKEZOGKQaR0GahG8Ern6yGgpkaHvgyUn7mxWaukBp68dexHluBJa3qJlHic2tH0lJDEnLJNrGaxv6Hv%2Byklh6BBqVBGnqdp5G6dGVGPnS5nSduTFydJ2FhrhJ6eo5QnOU5rkiYuemIUa3lGQoJnoWmFlBUenHcSCiSSHxl4kVpU46bFHn6TRiWSPgMlplVClWThtlTqeuX5VWglFeqbmlcu8UAshVUpWZ9KDZZ7HWVlQ6eq1LlTh1RHRbW7m9dRCUmnVaH%2BTaG0ZSFKnTXls0CYVC3Fd1cFxat/WGaNfm1eg34NcFHArLQnC/LwfjcLwqCcG41jWECawbJgxxmIEvAEJoL0rAA1iAvwpG9HCSJ90OkL9HC8AoIApFDHBaCscCwEgaAsKkdDxOQlBkxT9AJPshjAFwoJcCkNAgvEOMQDE6MxOELQAJ6cDwpD88wxCCwA8jE2iYA4Iu8GTbCCFLDC0MLBO8FgMT5m4Yi0Dj32kFgLBM%2BIWsm3gJIOHg9KYEbWjBKo8teAQWyi1cdTo7QeAxMQQseFg6PXOciukPbxAxBkmAipgZtGL7RjQysVAGMACgAGp4JgADuUupIw4f8IIIhiOwUgyIIigqOolu6EEBjJ6YljWPofs45AKyoKkDRGwAtGb9KqIcg9MMPZij5UlxTzElyT/3Uv%2BD9kc3FgncQCsdjyw0LgMO4ngdHooThEMlQjNIRRZAIUx%2BIUGTXww8zDAk0jb7bAh9JMh/5EE78NF/AYp8FgX1sBMNoP875gLmMAl%2BIBJBb2BpsCQr13po0tpjQ4qhQSJH7rlZETdgCHBZrcLgtwNCUlwIQEgYNAhLEhinOGCMkacFRqQL6TtMbY1xqQfGhNUEcDMOgzhnAGFayWCsSOmRnCSCAA%3D%3D%3D

using namespace cryptanalysislib::popcount::internal;



constexpr static __m256i u8tom256(const uint8_t t[32]) noexcept {
	long long __t[4];
	__t[0] = (long long) t[0] | (((long long) t[1]) << 8) | ((long long) t[2] << 16) | ((long long) t[3] << 24) | ((long long) t[4] << 32) | ((long long) t[5] << 40) | ((long long) t[6] << 48) | ((long long) t[7] << 56);
	__t[1] = (long long) t[8] | (((long long) t[9]) << 8) | ((long long) t[10] << 16) | ((long long) t[11] << 24) | ((long long) t[12] << 32) | ((long long) t[13] << 40) | ((long long) t[14] << 48) | ((long long) t[15] << 56);
	__t[2] = (long long) t[16] | (((long long) t[17]) << 8) | ((long long) t[18] << 16) | ((long long) t[19] << 24) | ((long long) t[20] << 32) | ((long long) t[21] << 40) | ((long long) t[22] << 48) | ((long long) t[23] << 56);
	__t[3] = (long long) t[24] | (((long long) t[25]) << 8) | ((long long) t[26] << 16) | ((long long) t[27] << 24) | ((long long) t[28] << 32) | ((long long) t[29] << 40) | ((long long) t[30] << 48) | ((long long) t[31] << 56);
	__m256i tmp = {__t[0], __t[1], __t[2], __t[3]};
	return tmp;
}

constexpr static __m128i u8tom128(const uint8_t t[16]) noexcept {
	long long __t[2];
	__t[0] = (long long) t[0] | (((long long) t[1]) << 8) | ((long long) t[2] << 16) | ((long long) t[3] << 24) | ((long long) t[4] << 32) | ((long long) t[5] << 40) | ((long long) t[6] << 48) | ((long long) t[7] << 56);
	__t[1] = (long long) t[8] | (((long long) t[9]) << 8) | ((long long) t[10] << 16) | ((long long) t[11] << 24) | ((long long) t[12] << 32) | ((long long) t[13] << 40) | ((long long) t[14] << 48) | ((long long) t[15] << 56);
	__m128i tmp = {__t[0], __t[1]};
	return tmp;
}

constexpr static __m256i u16tom256(const uint16_t t[16]) noexcept {
	long long __t[4];
	__t[0] = (long long) t[0] | (((long long) t[1]) << 16) | ((long long) t[2] << 32) | ((long long) t[3] << 48);
	__t[1] = (long long) t[4] | (((long long) t[5]) << 16) | ((long long) t[6] << 32) | ((long long) t[7] << 48);
	__t[2] = (long long) t[8] | (((long long) t[9]) << 16) | ((long long) t[10] << 32) | ((long long) t[11] << 48);
	__t[3] = (long long) t[12] | (((long long) t[13]) << 16) | ((long long) t[14] << 32) | ((long long) t[15] << 48);
	__m256i tmp = {__t[0], __t[1], __t[2], __t[3]};
	return tmp;
}

constexpr static __m128i u16tom128(const uint16_t t[16]) noexcept {
	long long __t[2];
	__t[0] = (long long) t[0] | (((long long) t[1]) << 16) | ((long long) t[2] << 32) | ((long long) t[3] << 48);
	__t[1] = (long long) t[4] | (((long long) t[5]) << 16) | ((long long) t[6] << 32) | ((long long) t[7] << 48);
	__m128i tmp = {__t[0], __t[1]};
	return tmp;
}

constexpr static __m256i u32tom256(const uint32_t t[8]) noexcept {
	long long __t[4];
	__t[0] = (long long) t[0] | (((long long) t[1]) << 32);
	__t[1] = (long long) t[2] | (((long long) t[3]) << 32);
	__t[2] = (long long) t[4] | (((long long) t[5]) << 32);
	__t[3] = (long long) t[6] | (((long long) t[7]) << 32);
	__m256i tmp = {__t[0], __t[1], __t[2], __t[3]};
	return tmp;
}

constexpr static __m128i u32tom128(const uint32_t t[8]) noexcept {
	long long __t[4];
	__t[0] = (long long) t[0] | (((long long) t[1]) << 32);
	__t[1] = (long long) t[2] | (((long long) t[3]) << 32);
	__m128i tmp = {__t[0], __t[1]};
	return tmp;
}

constexpr static __m256i u64tom256(const uint64_t t[4]) noexcept {
	__m256i tmp = {(long long) t[0], (long long) t[1], (long long) t[2], (long long) t[3]};
	return tmp;
}

constexpr static __m128i u64tom128(const uint64_t t[2]) noexcept {
	__m128i tmp = {(long long) t[0], (long long) t[1]};
	return tmp;
}

/// NOTE: not working
//constexpr static void m256tou16(uint16_t t[16],
//								  const __m256i m) noexcept {
//	const __v4di mm = m;
//	long long d0 = mm[0], d1 = 1, d2 = 2, d3 = 3;
//	t[0] = d0;
//	t[1] = d0 >> 16;
//	t[2] = d0 >> 32;
//	t[3] = d0 >> 48;
//	t[4] = d1;
//	t[5] = d1 >> 16;
//	t[6] = d1 >> 32;
//	t[7] = d1 >> 48;
//	t[8] = d2;
//	t[9] = d2 >> 16;
//	t[10] = d2 >> 32;
//	t[11] = d2 >> 48;
//	t[12] = d3;
//	t[13] = d3 >> 16;
//	t[14] = d3 >> 32;
//	t[15] = d3 >> 48;
//}

namespace internal {
	/// helper function. This enforces the compiler to emit a `vmovdqu` instruction
	/// \param ptr pointer to data.
	///				No alignment needed
	/// 			but 32 bytes should be readable
	/// \return unaligned `__m256i`
	constexpr static inline __m256i_u unaligned_load_wrapper(const __m256i_u *ptr) noexcept {
		return *ptr;
	}

	/// helper function. This enforces the compiler to emit a unaligned instruction
	/// \param ptr[out]: pointer to data
	/// \param data[in]: data to store
	/// \return nothing
	constexpr static inline void unaligned_store_wrapper(__m256i_u *ptr,
                                                         const __m256i_u data) noexcept {
		*ptr = data;
	}

    /// \param ptr[in]:
	constexpr static inline __m128i_u unaligned_load_wrapper_128(__m128i_u const *ptr) noexcept {
		return *ptr;
	}

    /// \param ptr[out]:
    /// \param data[in]:
	constexpr static inline void unaligned_store_wrapper_128(__m128i_u *ptr, 
                                                             __m128i_u data) noexcept {
		*ptr = data;
	}
}// namespace internal


namespace cryptanalysislib {
    template<const bool __unsigned=true>
	struct _Xint16x8_t;
    template<const bool __unsigned=true>
	struct _Xint32x4_t;
    template<const bool __unsigned=true>
	struct _Xint64x2_t;

    template<const bool __unsigned=true>
	struct _Xint8x16_t {
		constexpr static uint32_t LIMBS = 16;
		using limb_type = std::conditional_t<__unsigned, uint8_t, int8_t>;
	    using S = _Xint8x16_t;
        
        using V = std::conditional<__unsigned, __v16qu, __v16hi>::type;

        using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
        using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
        using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
        using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

		union {
			// compatibility to `TxN_t`
			T8   d[16];
			T8  v8[16];
			T16 v16[8];
			T32 v32[4];
			T64 v64[2];

			__m128i v128;
		};
	    
        [[nodiscard]] constexpr inline static size_t size() noexcept { 
            return LIMBS; 
        }

	    [[nodiscard]] constexpr inline static bool is_unsigned() noexcept { 
            return __unsigned; 
        }
        
        constexpr inline S operator=(const _Xint16x8_t<> &b) noexcept;
		constexpr inline S operator=(const _Xint32x4_t<> &b) noexcept;
		constexpr inline S operator=(const _Xint64x2_t<> &b) noexcept;

		constexpr _Xint8x16_t() = default;
		constexpr _Xint8x16_t(const _Xint16x8_t<> &b) noexcept;
		constexpr _Xint8x16_t(const _Xint32x4_t<> &b) noexcept;
		constexpr _Xint8x16_t(const _Xint64x2_t<> &b) noexcept;

        /// \param i[in]: position of the limb to return
        /// \return __m256i[i]
		[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
			assert(i < LIMBS);
			return d[i];
		}

        /// \param i[in]: position of the limb to return
        /// \return __m256i[i]
		[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
			assert(i < LIMBS);
			return d[i];
		}
		
		/// \return random element
		static inline S random() noexcept {
			S ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = rng();
			}

			return ret;
		}

	    /// \param binary[in]:
	    /// \param hex[in]:
	    constexpr inline void print(bool binary = false,
	                                bool hex = false) const;

        /// \param a-p[in]: 
        /// \return 
		[[nodiscard]] constexpr static inline S set(const limb_type a, const limb_type b, const limb_type c, const limb_type d,
		                                            const limb_type e, const limb_type f, const limb_type g, const limb_type h,
		                                            const limb_type i, const limb_type j, const limb_type k, const limb_type l,
		                                            const limb_type m, const limb_type n, const limb_type o, const limb_type p) noexcept {
			S ret;
            ret.v128 = __extension__ (__m128i)(__v16qi){
                (char)p, (char)o, (char)n, (char)m, (char)l, (char)k, (char)j, (char)i,
                (char)h, (char)g, (char)f, (char)e, (char)d, (char)c, (char)b, (char)a,
            };
            return ret;
		}

        /// \param a-p[in]: 
        /// \return 
		[[nodiscard]] constexpr static inline S setr(const limb_type a, const limb_type b, const limb_type c, const limb_type d,
		                                             const limb_type e, const limb_type f, const limb_type g, const limb_type h,
		                                             const limb_type i, const limb_type j, const limb_type k, const limb_type l,
		                                             const limb_type m, const limb_type n, const limb_type o, const limb_type p) noexcept {
			S ret;
            ret.v128 = __extension__ (__m128i)(__v16qi){
                (char)a, (char)b, (char)c, (char)d, (char)e, (char)f, (char)g, (char)h,
                (char)i, (char)j, (char)k, (char)l, (char)m, (char)n, (char)o, (char)p,
            };
            return ret;
		}

        /// \param a-p[in]: 
        /// \return 
		[[nodiscard]] constexpr static inline S set1(const limb_type i) noexcept {
            return S::set(i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i);
		}

		/// \tparam aligned[in]: if true a alied instruction will be emitted
		/// \param ptr[in]: pointer to (aligned )16 bytes 
		/// \return vector element
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline S load(const limb_type *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		/// \param ptr[in]: pointer to 16 aligned bytes
		/// \return: vector element
		[[nodiscard]] constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
		    if (std::is_constant_evaluated()) {
		    	const __m128i tmp = u8tom128(ptr);
		    	S out;
		    	out.v128 = tmp;
		    	return out;
		    } else {
		    	auto *ptr128 = (__m128i *) ptr;
		    	S out;
		    	out.v128 = *ptr128;
		    	return out;
            }
		}

		/// \param ptr[in]: pointer to 16 unaligned bytes
		/// \return: vector element
		[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		    if (std::is_constant_evaluated()) {
		    	const __m128i tmp = u8tom128(ptr);
		    	S out;
		    	out.v128 = tmp;
		    	return out;
		    } else {
		    	__m128i_u const *ptr128 = (__m128i_u const *) ptr;
		    	const __m128i_u tmp = ::internal::unaligned_load_wrapper_128(ptr128);
		    	S out;
		    	out.v128 = tmp;
		    	return out;
            }
		}

	    /// NOTE: the store can never be constexpr ans its needs to access
	    /// given memory
	    /// \tparam aligned[in]: 
	    /// \param ptr[in/out]: pointer to 16 (aligned) bytes
	    /// \param in[in]: vector element
		template<const bool aligned = false>
		static inline void store(limb_type *ptr,
                                 const S in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

	    /// \param ptr[in/out]: pointer to 16 aligned bytes
	    /// \param in[in]: vector element
		constexpr static inline void aligned_store(void *ptr,
                                                   const S in) noexcept {
			auto *ptr128 = (__m128i *) ptr;
			*ptr128 = in.v128;
		}

	    /// \param ptr[in/out]: pointer to 16 unaligned bytes
	    /// \param in[in]: vector element
		constexpr static inline void unaligned_store(void *ptr,
                                                     const S in) noexcept {
			auto *ptr128 = (__m128i_u *) ptr;
			::internal::unaligned_store_wrapper_128(ptr128, in.v128);
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 ^ in2
	    [[nodiscard]] constexpr static inline S xor_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 ^ (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 & in2
	    [[nodiscard]] constexpr static inline S and_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 & (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 | in2
	    [[nodiscard]] constexpr static inline S or_(const S in1,
	                                                const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 | (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return (~in1) & in2
	    [[nodiscard]] constexpr static inline S andnot(const S in1,
	                                                   const S in2) noexcept {
	    	S out;
#ifndef __clang__
		    out.v128 = _mm_andnot_si128(in1.v128, in2.v128);
#else
		    out.v128 = (__m128i) (~(V) in1.v128 & (V) in2.v128);
#endif
		return out;
	    }

	    /// \param in1[in]: vector element
	    /// \return ~in1
	    [[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
	    	S out;
	    	const S minus_one = set1(-1);
	    	out.v128 = (__m128i) ((V) in1.v128 ^ (V) minus_one.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 + in2
	    [[nodiscard]] constexpr static inline S add(const S in1,
	                                                const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 + (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 - in2
	    [[nodiscard]] constexpr static inline S sub(const S in1,
	                                                const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 - (V) in2.v128);
	    	return out;
	    }

	    /// 8 bit mul lo
	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1*in2
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const S in2) noexcept {
		    S out;
		    out.v128 = ((__m128i) ((V) in1.v128 * (V) in2.v128));
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const limb_type in2) noexcept {
	    	const S rs = S::set1(in2);
	    	return S::mullo(in1, rs);
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
	    	const S mask = set1((1u << in2) - 1u);
	    	out = S::and_(in1, mask);
	    	// if (std::is_constant_evaluated()) {
	    	// 	out.v128 = (__m128i)((__v32qi)out.v128) << in2;
	    	// 	return out;
	    	// }
	    	// out.v128 = (__m128i) __builtin_ia32_psllwi128((__v16hi) out.v128, in2);

	    	out.v128 = (__m128i) ((V) out.v128) << in2;
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
	    	constexpr S mask1 = set1(((1u << (8u - in2)) - 1u) << in2);
	    	constexpr S mask2 = set1((1u << (8u - in2)) - 1u);
	    	S out = S::and_(in1, mask1);
	    	if (std::is_constant_evaluated()) {
	    		out.v128 = (__m128i) ((V) out.v128) >> in2;
	    		return out;
	    	}
	    	out.v128 = (__m128i) ((V) out.v128) >> in2;
	    	out = S::and_(out, mask2);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S ror(const S in1,
	                                                 const uint8_t in2) noexcept {

	    	S out;
            const __m128i mask = _mm_set1_epi8((1u << (8u-in2)) -1u);
            out.v128 = _mm_slli_epi16(in1.v128, in2) ^ (_mm_srli_epi16(in1.v128, 8u-in2) & mask);
	    	return out;

        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S rol(const S in1,
	                                                 const uint8_t in2) noexcept {
	    	S out;
            const __m128i mask = _mm_set1_epi8((1u << (8-in2)) -1u);
            out.v128 = _mm_slli_epi16(in1.v128, in2) ^ (_mm_srli_epi16(in1.v128, 8u-in2) & mask);
	    	return out;
        }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S gt_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 > (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t gt(const S in1,
		                                                  const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 > (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}
	    
        /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S lt_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 < (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t lt(const S in1,
		                                                  const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 < (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}
        
        /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S cmp_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 == (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
		                                                   const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 == (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}

	    /// \param in[in]: vector element
		/// \return [popcnt(in[0]), ..., popcnt(in[7])]
	    [[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
	    	S ret;
	    	ret.v128 = popcount_sse_u8x16(in.v128);
	    	return ret;
	    }

	    [[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
            const __m128i rotated = _mm_alignr_epi8(in.v128, in.v128, 1);
            const __m128i eq = _mm_cmpeq_epi8(in.v128, rotated);

            return ((uint16_t)_mm_movemask_epi8(eq) == 0xffff);
        }
        
        // just shuffle the 16 u8 elements 
	    [[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
	    	S ret;
            const __m128i shuffle = _mm_setr_epi8(15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0);
            ret.v128 = _mm_shuffle_epi8(in.v128, shuffle);
	    	return ret;
        }

	    /// kmoves the msb into each bit
	    [[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
	    	return __builtin_ia32_pmovmskb128((__v16qi) in.v128);
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
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_min((__v16qu)a.v128, (__v16qu)b.v128);
#else
			    c.v128 = (__m128i)_mm_min_epu8(a.v128, b.v128);
#endif
		    } else {
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_min((__v16qi)a.v128, (__v16qi)b.v128);
#else
			    c.v128 = (__m128i)_mm_min_epi8(a.v128, b.v128);
#endif
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
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_max((__v16qu)a.v128, (__v16qu)b.v128);
#else
			    c.v128 = (__m128i)_mm_max_epu8(a.v128, b.v128);
#endif
		    } else {
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_max((__v16qi)a.v128, (__v16qi)b.v128);
#else
			    c.v128 = (__m128i)_mm_max_epi8(a.v128, b.v128);
#endif
	    	}
            return c;
        }

	};

    /// 
    using _uint8x16_t = _Xint8x16_t<true>;
    using  _int8x16_t = _Xint8x16_t<false>;

    template<const bool __unsigned>
	struct _Xint16x8_t {
		constexpr static uint32_t LIMBS = 8;
	    using limb_type = std::conditional<__unsigned, uint16_t, int16_t>::type;
	    using S = _Xint16x8_t;
	    using simd_type = S;

        using V   = std::conditional<__unsigned, __v16qu, __v8qi>::type;
        using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
        using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
        using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
        using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

		constexpr inline S operator=(const _Xint8x16_t<> &b) noexcept;
		constexpr inline S operator=(const _Xint32x4_t<> &b) noexcept;
		constexpr inline S operator=(const _Xint64x2_t<> &b) noexcept;

		constexpr _Xint16x8_t() noexcept = default;
		constexpr _Xint16x8_t(const _Xint8x16_t<> &b) noexcept;
		constexpr _Xint16x8_t(const _Xint32x4_t<> &b) noexcept;
		constexpr _Xint16x8_t(const _Xint64x2_t<> &b) noexcept;

		union {
			// compatibility to `TxN_t`
			T16 d[8];

			T8 v8[16];
			T16 v16[8];
			T32 v32[4];
			T64 v64[2];

			__m128i v128;
		};

        /// \param i[in]: position of the limb to return
        /// \return __m128i[i]
		[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
			assert(i < LIMBS);
			return d[i];
		}

        /// \param i[in]: position of the limb to return
        /// \return __m128i[i]
		[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
			assert(i < LIMBS);
			return d[i];
		}

		/// \return
		static inline S random() noexcept {
			S ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = rng();
			}

			return ret;
		}

	    /// \param binary[in]:
	    /// \param hex[in]:
	    constexpr inline void print(bool binary = false,
	                                bool hex = false) const;
        
        /// \return
		[[nodiscard]] constexpr static inline S set(
		        const limb_type a, const limb_type b, const limb_type c, const limb_type d,
		        const limb_type e, const limb_type f, const limb_type g, const limb_type h) noexcept {
			S ret;
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

        /// \return
		[[nodiscard]] constexpr static inline S setr(
		        const limb_type a, const limb_type b, const limb_type c, const limb_type d,
		        const limb_type e, const limb_type f, const limb_type g, const limb_type h) noexcept {
			S ret;
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
		
        /// \return
        [[nodiscard]] constexpr static inline S set1(const limb_type a) noexcept {
			S ret;
			ret.v16[0] = a;
			ret.v16[1] = a;
			ret.v16[2] = a;
			ret.v16[3] = a;
			ret.v16[4] = a;
			ret.v16[5] = a;
			ret.v16[6] = a;
			ret.v16[7] = a;
			return ret;
        }

		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline S load(const limb_type *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
			auto *ptr128 = (__m128i *) ptr;
			S out;
			out.v128 = *ptr128;
			return out;
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
			__m128i_u const *ptr128 = (__m128i_u const *) ptr;
			const __m128i_u tmp = ::internal::unaligned_load_wrapper_128(ptr128);
			S out;
			out.v128 = tmp;
			return out;
		}

		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(limb_type *ptr,
                                           const S in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(limb_type *ptr,
                                                   const S in) noexcept {
			auto *ptr128 = (__m128i *) ptr;
			*ptr128 = in.v128;
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr, const S in) noexcept {
			auto *ptr128 = (__m128i_u *) ptr;
			::internal::unaligned_store_wrapper_128(ptr128, in.v128);
		}
	    
    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 ^ in2
	    [[nodiscard]] constexpr static inline S xor_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 ^ (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 & in2
	    [[nodiscard]] constexpr static inline S and_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 & (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 | in2
	    [[nodiscard]] constexpr static inline S or_(const S in1,
	                                                const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 | (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return (~in1) & in2
	    [[nodiscard]] constexpr static inline S andnot(const S in1,
	                                                   const S in2) noexcept {
	    	S out;
#ifndef __clang__
		    out.v128 = _mm_andnot_si128(in1.v128, in2.v128);
#else
		    out.v128 = (__m128i) (~(V) in1.v128 & (V) in2.v128);
#endif
		return out;
	    }

	    /// \param in1[in]: vector element
	    /// \return ~in1
	    [[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
	    	S out;
	    	const S minus_one = set1(-1);
	    	out.v128 = (__m128i) ((V) in1.v128 ^ (V) minus_one.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 + in2
	    [[nodiscard]] constexpr static inline S add(const S in1,
	                                                const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 + (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 - in2
	    [[nodiscard]] constexpr static inline S sub(const S in1,
	                                                const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 - (V) in2.v128);
	    	return out;
	    }

	    /// 8 bit mul lo
	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1*in2
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const S in2) noexcept {
		    S out;
		    out.v128 = ((__m128i) ((V) in1.v128 * (V) in2.v128));
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const limb_type in2) noexcept {
	    	const S rs = S::set1(in2);
	    	return S::mullo(in1, rs);
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return
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
	    	constexpr S mask = set1((1u << in2) - 1u);
	    	out = S::and_(in1, mask);
	    	// if (std::is_constant_evaluated()) {
	    	// 	out.v128 = (__m128i)((__v32qi)out.v128) << in2;
	    	// 	return out;
	    	// }
	    	// out.v128 = (__m128i) __builtin_ia32_psllwi128((__v16hi) out.v128, in2);

	    	out.v128 = (__m128i) ((V) out.v128) << in2;
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
	    	constexpr S mask1 = set1(((1u << (8u - in2)) - 1u) << in2);
	    	constexpr S mask2 = set1((1u << (8u - in2)) - 1u);
	    	S out = S::and_(in1, mask1);
	    	if (std::is_constant_evaluated()) {
	    		out.v128 = (__m128i) ((V) out.v128) >> in2;
	    		return out;
	    	}
	    	out.v128 = (__m128i) ((V) out.v128) >> in2;
	    	out = S::and_(out, mask2);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S gt_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 > (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t gt(const S in1,
		                                                  const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 > (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}
	    
        /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S lt_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 < (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t lt(const S in1,
		                                                  const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 < (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}
        
        /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S cmp_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 == (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
		                                                   const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 == (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}

	    /// \param in[in]: vector element
		/// \return [popcnt(in[0]), ..., popcnt(in[7])]
	    [[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
	    	S ret;
	    	ret.v128 = popcount_sse_u8x16(in.v128);
	    	return ret;
	    }

        /// \param in[in]:
        /// \return  
	    [[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
            return _Xint8x16_t<__unsigned>::reverse(in);
        }
        
        /// \param in[in]:
        /// \return  
	    [[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
	    	S ret;
            const __m128i shuffle = _mm_setr_epi8(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1);
            ret.v128 = _mm_shuffle_epi8(in.v128, shuffle);
	    	return ret;
        }

	    /// kmoves the msb into each bit
	    [[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
	    	return __builtin_ia32_pmovmskb128((__v16qi) in.v128);
	    }

        /// \tparam scale[in]:
        /// \param ptr[in]:
        /// \param data[in]:
        /// \return
	    template<const uint32_t scale = 1>
	    [[nodiscard]] constexpr static inline S gather(const void *ptr,
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
	    constexpr static inline void scatter(const void *ptr,
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
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_min((__v16qu)a.v128, (__v16qu)b.v128);
#else
			    c.v128 = (__m128i)_mm_min_epu8(a.v128, b.v128);
#endif
		    } else {
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_min((__v16qi)a.v128, (__v16qi)b.v128);
#else
			    c.v128 = (__m128i)_mm_min_epi8(a.v128, b.v128);
#endif
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
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_max((__v16qu)a.v128, (__v16qu)b.v128);
#else
			    c.v128 = (__m128i)_mm_max_epu8(a.v128, b.v128);
#endif
		    } else {
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_max((__v16qi)a.v128, (__v16qi)b.v128);
#else
			    c.v128 = (__m128i)_mm_max_epi8(a.v128, b.v128);
#endif
	    	}
            return c;
        }
	};

    /// 
    using _uint16x8_t = _Xint16x8_t<true>;
    using  _int16x8_t = _Xint16x8_t<false>;

    template<const bool __unsigned>
	struct _Xint32x4_t {
		constexpr static uint32_t LIMBS = 4;
	    using limb_type = std::conditional<__unsigned, uint32_t, int32_t>::type;
	    using S = _Xint32x4_t;
	    using simd_type = S;

        using V   = std::conditional<__unsigned, __v32qu, __v32qi>::type;
        using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
        using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
        using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
        using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

		constexpr inline _Xint32x4_t operator=(const _Xint8x16_t<> &b) noexcept;
		constexpr inline _Xint32x4_t operator=(const _Xint16x8_t<> &b) noexcept;
		constexpr inline _Xint32x4_t operator=(const _Xint64x2_t<> &b) noexcept;

		constexpr _Xint32x4_t() noexcept = default;
		constexpr _Xint32x4_t(const _Xint8x16_t<> &b) noexcept;
		constexpr _Xint32x4_t(const _Xint16x8_t<> &b) noexcept;
		constexpr _Xint32x4_t(const _Xint64x2_t<> &b) noexcept;

		union {
		    // compatibility with TxN_t
		    T16  d[32];

			T8  v8[16];
			T16 v16[8];
			T32 v32[4];
			T64 v64[2];
			__m128i v128;
		};

		[[nodiscard]] constexpr static inline S set(const limb_type a,
                                                    const limb_type b,
                                                    const limb_type c,
                                                    const limb_type d) {
			S ret;
			ret.v32[0] = d;
			ret.v32[1] = c;
			ret.v32[2] = b;
			ret.v32[3] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline S setr(const limb_type a,
                                                     const limb_type b,
                                                     const limb_type c,
                                                     const limb_type d) {
			S ret;
			ret.v32[0] = a;
			ret.v32[1] = b;
			ret.v32[2] = c;
			ret.v32[3] = d;
			return ret;
		}
		
        [[nodiscard]] constexpr static inline S set1(const limb_type a) {
			S ret;
			ret.v32[0] = a;
			ret.v32[1] = a;
			ret.v32[2] = a;
			ret.v32[3] = a;
			return ret;
		}

		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline S load(const limb_type *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
			auto *ptr128 = (__m128i *) ptr;
			S out;
			out.v128 = *ptr128;
			return out;
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
			__m128i_u const *ptr128 = (__m128i_u const *) ptr;
			const __m128i_u tmp = ::internal::unaligned_load_wrapper_128(ptr128);
			S out;
			out.v128 = tmp;
			return out;
		}

		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(limb_type *ptr,
                                           const S in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(limb_type *ptr,
                                                   const S in) noexcept {
			auto *ptr128 = (__m128i *) ptr;
			*ptr128 = in.v128;
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr, const S in) noexcept {
			auto *ptr128 = (__m128i_u *) ptr;
			::internal::unaligned_store_wrapper_128(ptr128, in.v128);
		}
	    
        /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 ^ in2
	    [[nodiscard]] constexpr static inline S xor_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 ^ (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 & in2
	    [[nodiscard]] constexpr static inline S and_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 & (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 | in2
	    [[nodiscard]] constexpr static inline S or_(const S in1,
	                                                const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 | (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return (~in1) & in2
	    [[nodiscard]] constexpr static inline S andnot(const S in1,
	                                                   const S in2) noexcept {
	    	S out;
#ifndef __clang__
		    out.v128 = _mm_andnot_si128(in1.v128, in2.v128);
#else
		    out.v128 = (__m128i) (~(V) in1.v128 & (V) in2.v128);
#endif
		return out;
	    }

	    /// \param in1[in]: vector element
	    /// \return ~in1
	    [[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
	    	S out;
	    	const S minus_one = set1(-1);
	    	out.v128 = (__m128i) ((V) in1.v128 ^ (V) minus_one.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 + in2
	    [[nodiscard]] constexpr static inline S add(const S in1,
	                                                const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 + (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 - in2
	    [[nodiscard]] constexpr static inline S sub(const S in1,
	                                                const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 - (V) in2.v128);
	    	return out;
	    }

	    /// 8 bit mul lo
	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1*in2
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const S in2) noexcept {
		    S out;
		    out.v128 = ((__m128i) ((V) in1.v128 * (V) in2.v128));
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const limb_type in2) noexcept {
	    	const S rs = S::set1(in2);
	    	return S::mullo(in1, rs);
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return
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
	    	constexpr S mask = set1((1u << in2) - 1u);
	    	out = S::and_(in1, mask);
	    	// if (std::is_constant_evaluated()) {
	    	// 	out.v128 = (__m128i)((__v32qi)out.v128) << in2;
	    	// 	return out;
	    	// }
	    	// out.v128 = (__m128i) __builtin_ia32_psllwi128((__v16hi) out.v128, in2);

	    	out.v128 = (__m128i) ((V) out.v128) << in2;
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
	    	constexpr S mask1 = set1(((1u << (8u - in2)) - 1u) << in2);
	    	constexpr S mask2 = set1((1u << (8u - in2)) - 1u);
	    	S out = S::and_(in1, mask1);
	    	if (std::is_constant_evaluated()) {
	    		out.v128 = (__m128i) ((V) out.v128) >> in2;
	    		return out;
	    	}
	    	out.v128 = (__m128i) ((V) out.v128) >> in2;
	    	out = S::and_(out, mask2);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S gt_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 > (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t gt(const S in1,
		                                                  const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 > (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}
	    
        /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S lt_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 < (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t lt(const S in1,
		                                                  const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 < (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}
        
        /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S cmp_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 == (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
		                                                   const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 == (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}

	    /// \param in[in]: vector element
		/// \return [popcnt(in[0]), ..., popcnt(in[7])]
	    [[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
	    	S ret;
	    	ret.v128 = popcount_sse_u8x16(in.v128);
	    	return ret;
	    }

	    /// \param in[in]: vector element
	    [[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
            return _Xint8x16_t<__unsigned>::all_equal(in);
        }
        
	    /// \param in[in]: vector element
	    [[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
	    	S ret;
            const __m128i shuffle = _mm_setr_epi8(12,13,14,15,8,9,10,11,4,5,6,7,0,1,2,3);
            ret.v128 = _mm_shuffle_epi8(in.v128, shuffle);
	    	return ret;
        }

	    /// kmoves the msb into each bit
	    [[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
	    	return __builtin_ia32_pmovmskb128((__v16qi) in.v128);
	    }

        /// \tparam scale[in]:
        /// \param ptr[in]:
        /// \param data[in]:
        /// \return
	    template<const uint32_t scale = 1>
	    [[nodiscard]] constexpr static inline S gather(const void *ptr,
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
	    constexpr static inline void scatter(const void *ptr,
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
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_min((__v16qu)a.v128, (__v16qu)b.v128);
#else
			    c.v128 = (__m128i)_mm_min_epu8(a.v128, b.v128);
#endif
		    } else {
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_min((__v16qi)a.v128, (__v16qi)b.v128);
#else
			    c.v128 = (__m128i)_mm_min_epi8(a.v128, b.v128);
#endif
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
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_max((__v16qu)a.v128, (__v16qu)b.v128);
#else
			    c.v128 = (__m128i)_mm_max_epu8(a.v128, b.v128);
#endif
		    } else {
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_max((__v16qi)a.v128, (__v16qi)b.v128);
#else
			    c.v128 = (__m128i)_mm_max_epi8(a.v128, b.v128);
#endif
	    	}
            return c;
        }
	};


    /// 
    using _uint32x4_t = _Xint32x4_t<true>;
    using  _int32x4_t = _Xint32x4_t<false>;

    template<const bool __unsigned>
	struct _Xint64x2_t {
		constexpr static uint32_t LIMBS = 2;
	    using limb_type = std::conditional<__unsigned, uint64_t, int64_t>::type;
	    using S = _Xint64x2_t;
	    using simd_type = S;

        using V   = std::conditional<__unsigned, __v32qu, __v32qi>::type;
        using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
        using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
        using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
        using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

		constexpr inline _Xint64x2_t operator=(const _Xint8x16_t<> &b) noexcept;
		constexpr inline _Xint64x2_t operator=(const _Xint16x8_t<> &b) noexcept;
		constexpr inline _Xint64x2_t operator=(const _Xint32x4_t<> &b) noexcept;

		constexpr _Xint64x2_t() noexcept = default;
		constexpr _Xint64x2_t(const _Xint8x16_t<> &b) noexcept;
		constexpr _Xint64x2_t(const _Xint16x8_t<> &b) noexcept;
		constexpr _Xint64x2_t(const _Xint32x4_t<> &b) noexcept;

		union {
			uint8_t v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];
			__m128i v128;
		};

		[[nodiscard]] constexpr static inline _Xint64x2_t set1(const limb_type a) {
			_Xint64x2_t ret;
			ret.v64[0] = a;
			ret.v64[1] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint64x2_t set(const limb_type a, const limb_type b) {
			_Xint64x2_t ret;
			ret.v64[0] = b;
			ret.v64[1] = a;
			return ret;
		}

		[[nodiscard]] constexpr static inline _Xint64x2_t setr(const limb_type a, const limb_type b) {
			_Xint64x2_t ret;
			ret.v64[0] = a;
			ret.v64[1] = b;
			return ret;
		}

		/// \tparam aligned
		/// \param ptr
		/// \return
		template<const bool aligned = false>
		[[nodiscard]] constexpr static inline _Xint64x2_t load(const limb_type *ptr) noexcept {
			if constexpr (aligned) {
				return aligned_load(ptr);
			}

			return unaligned_load(ptr);
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint64x2_t aligned_load(const limb_type *ptr) noexcept {
			auto *ptr128 = (__m128i *) ptr;
			_Xint64x2_t out;
			out.v128 = *ptr128;
			return out;
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint64x2_t unaligned_load(const limb_type *ptr) noexcept {
			__m128i_u const *ptr128 = (__m128i_u const *) ptr;
			const __m128i_u tmp = ::internal::unaligned_load_wrapper_128(ptr128);
			_Xint64x2_t out;
			out.v128 = tmp;
			return out;
		}

		/// \tparam aligned
		/// \param ptr
		/// \param in
		template<const bool aligned = false>
		constexpr static inline void store(limb_type *ptr,
                                           const S in) noexcept {
			if constexpr (aligned) {
				aligned_store(ptr, in);
				return;
			}

			unaligned_store(ptr, in);
		}

		/// \param ptr
		/// \param in
		constexpr static inline void aligned_store(limb_type *ptr,
                                                   const S in) noexcept {
			auto *ptr128 = (__m128i *) ptr;
			*ptr128 = in.v128;
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(limb_type *ptr,
                                                     const S in) noexcept {
			auto *ptr128 = (__m128i_u *) ptr;
			::internal::unaligned_store_wrapper_128(ptr128, in.v128);
		}

        /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 ^ in2
	    [[nodiscard]] constexpr static inline S xor_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 ^ (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 & in2
	    [[nodiscard]] constexpr static inline S and_(const S in1,
	                                                 const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 & (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 | in2
	    [[nodiscard]] constexpr static inline S or_(const S in1,
	                                                const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 | (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return (~in1) & in2
	    [[nodiscard]] constexpr static inline S andnot(const S in1,
	                                                   const S in2) noexcept {
	    	S out;
#ifndef __clang__
		    out.v128 = _mm_andnot_si128(in1.v128, in2.v128);
#else
		    out.v128 = (__m128i) (~(V) in1.v128 & (V) in2.v128);
#endif
		return out;
	    }

	    /// \param in1[in]: vector element
	    /// \return ~in1
	    [[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
	    	S out;
	    	const S minus_one = set1(-1);
	    	out.v128 = (__m128i) ((V) in1.v128 ^ (V) minus_one.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 + in2
	    [[nodiscard]] constexpr static inline S add(const S in1,
	                                                const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 + (V) in2.v128);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 - in2
	    [[nodiscard]] constexpr static inline S sub(const S in1,
	                                                const S in2) noexcept {
	    	S out;
	    	out.v128 = (__m128i) ((V) in1.v128 - (V) in2.v128);
	    	return out;
	    }

	    /// 8 bit mul lo
	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1*in2
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const S in2) noexcept {
		    S out;
		    out.v128 = ((__m128i) ((V) in1.v128 * (V) in2.v128));
		    return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return
	    [[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                  const limb_type in2) noexcept {
	    	const S rs = S::set1(in2);
	    	return S::mullo(in1, rs);
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return
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
	    	constexpr S mask = set1((1u << in2) - 1u);
	    	out = S::and_(in1, mask);
	    	// if (std::is_constant_evaluated()) {
	    	// 	out.v128 = (__m128i)((__v32qi)out.v128) << in2;
	    	// 	return out;
	    	// }
	    	// out.v128 = (__m128i) __builtin_ia32_psllwi128((__v16hi) out.v128, in2);

	    	out.v128 = (__m128i) ((V) out.v128) << in2;
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
	    	constexpr S mask1 = set1(((1u << (8u - in2)) - 1u) << in2);
	    	constexpr S mask2 = set1((1u << (8u - in2)) - 1u);
	    	S out = S::and_(in1, mask1);
	    	if (std::is_constant_evaluated()) {
	    		out.v128 = (__m128i) ((V) out.v128) >> in2;
	    		return out;
	    	}
	    	out.v128 = (__m128i) ((V) out.v128) >> in2;
	    	out = S::and_(out, mask2);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S gt_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 > (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t gt(const S in1,
		                                                  const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 > (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}
	    
        /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S lt_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 < (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 > in2 compressed
		[[nodiscard]] constexpr static inline uint32_t lt(const S in1,
		                                                  const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 < (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}
        
        /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S cmp_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 == (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
		                                                   const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 == (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}

	    /// \param in[in]: vector element
		/// \return [popcnt(in[0]), ..., popcnt(in[7])]
	    [[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
	    	S ret;
	    	ret.v128 = popcount_sse_u8x16(in.v128);
	    	return ret;
	    }
	    
        /// \param in[in]: vector element
	    [[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
            return _Xint8x16_t<__unsigned>::all_equal(in);
        }
        
	    /// \param in[in]: vector element
	    [[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
	    	S ret;
            const __m128i shuffle = _mm_setr_epi8(8,9,10,11,12,13,14,15,0,1,2,3,4,5,6,7);
            ret.v128 = _mm_shuffle_epi8(in.v128, shuffle);
	    	return ret;
        }

	    /// kmoves the msb into each bit
	    [[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
	    	return __builtin_ia32_pmovmskb128((__v16qi) in.v128);
	    }

        /// \tparam scale[in]:
        /// \param ptr[in]:
        /// \param data[in]:
        /// \return
	    template<const uint32_t scale = 1>
	    [[nodiscard]] constexpr static inline S gather(const void *ptr,
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
	    constexpr static inline void scatter(const void *ptr,
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
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_min((__v16qu)a.v128, (__v16qu)b.v128);
#else
			    c.v128 = (__m128i)_mm_min_epu8(a.v128, b.v128);
#endif
		    } else {
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_min((__v16qi)a.v128, (__v16qi)b.v128);
#else
			    c.v128 = (__m128i)_mm_min_epi8(a.v128, b.v128);
#endif
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
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_max((__v16qu)a.v128, (__v16qu)b.v128);
#else
			    c.v128 = (__m128i)_mm_max_epu8(a.v128, b.v128);
#endif
		    } else {
#ifdef __clang__
			    c.v128 = (__m128i)__builtin_elementwise_max((__v16qi)a.v128, (__v16qi)b.v128);
#else
			    c.v128 = (__m128i)_mm_max_epi8(a.v128, b.v128);
#endif
	    	}
            return c;
        }
	};


    /// 
    using _uint64x2_t = _Xint64x2_t<true>;
    using  _int64x2_t = _Xint64x2_t<false>;
}// namespace cryptanalysislib



template<const bool __unsigned=true>
struct Xint8x32_t {
	constexpr static uint32_t LIMBS = 32;
	using limb_type = std::conditional<__unsigned, uint8_t, int8_t>::type;
	using S = Xint8x32_t;
	using simd_type = S;

    using V   = std::conditional<__unsigned, __v32qu, __v32qi>::type;
    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		// compatibility with TxN_t
		T8  d[32];

		T8  v8[32];
		T16 v16[16];
		T32 v32[8];
		T64 v64[4];
		__m256i v256;
	};

	[[nodiscard]] constexpr inline static size_t size() noexcept { 
        return LIMBS; 
    }
	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
        return __unsigned; 
    }

    /// \param i[in]: position of the limb to return
    /// \return __m256i[i]
	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const {
		assert(i < LIMBS);
		return d[i];
	}

    /// \param i[in]: position of the limb to return
    /// \return __m256i[i]
	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) {
		assert(i < LIMBS);
		return d[i];
	}

	/// Example of how the constexpr implementation works:
	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGEgBykrgAyeAyYAHI%2BAEaYxCCSZqQADqgKhE4MHt6%2BASlpGQKh4VEssfGJtpj2jgJCBEzEBNk%2BflyBdpgOmfWNBMWRMXEJSQoNTS257bbj/WGDZcOJAJS2qF7EyOwc5gDMYcjeWADUJrtuY/iCAHQIZ9gmGgCCewdHmKfneCwsYQTEYVu90eLzM%2BwYhy8JzObmQlycwOeILGxC8DmOXj%2B/lUuzMAH0CKcAOxWJEaACcXgYmWJpJeFMelKxBOOxwAbv5TgBWCy4kxcgAiZzpjMZmMEXAAbCy2VL%2BRY5YLhSDRRTxQRcTK%2BTzjv5%2BULdiKGWq/pLJDKzfLjpJ9cqyeTGXi8SwzFzJXh2a7JXb6Q6iQa6azWSCg8c0AwxphVMliMcxkxHMhjmFaPMMVicfjCaoSHiIOGxunBNjNYSwlxSCqKaGa7W6/WC4T1SWs8mGGZlrSqw6TcXMyz1gQfaryYPrmyvZ8BccIE6XW68J2oE62ZJ0F5luXx5P%2BdgZyu1xuwmZt27lsPjeSAPRXscTt1T47O%2BfSnPEPHpL0QLf3yWkNsnr%2B56Gt2jLEJgBAbAwxyDhefpCmSvo3le3bIdybjJI0rDHMk/yoTe6HgZBxAMN2jZRjGcYNImbapuERYEC2LJUmIeDAOE6B4rQqBMOg%2BYCIWbKoHg6DHAAVLhxCdiYJKgRSTBeEQ4mSTuuzTrOzpeh6YnLJJcFihmpYwYp%2BkUneqnThJ/xeqZ163opp6So%2Bz5elxPHrh%2BeBfhpL7abp/zAUaPbkkRUHGUOIH2jJCFPKGIZBsh%2BFXuhmHENhen2mh/IYVhLBtmRAkEBRsbxjRKZpkJIkYswqbsZgnFjCQmAQJVolWWQYaFQxTFlgw0myVFFJzlpynWQ%2BZzqcNC7if5xC2e1Flto5Ppxf6KprZF9JPOR0axq1xxFWMEDNiyYmDv%2Bjbdad5YXV1J2EmJx63RGTZ/EZfVds8ob8LGx1va2HoTccGjCsm1i4p8bjHKRhpg5YE24v1gZ1pdzb9oSTCPmjpYgCALG1Rxbm8d%2BDBcKcljJmJiMrSjd2Ga20RY/TBK4/jbGE9xxPHuTFiU9Tm0NnTfZGUmQPY1muNvnmTD/tEgXxbW4ss3jNXs/VH5EOBEDnWG8tfUG0XrTFIJ/McLBMGEEBIwrnUvcVDHvY%2BXBmHqAtBjtlH3eF8qkYKj4yRYIMbcjrIe7GXvlj7%2Br%2BySQcBjbYdXb1ZhR37QMB3HF6xUGh0EFAXs6TrN0AaQfU06yMZ/FQEDmCnrroNlpFmEk3s8iDgp6y8/ocKstCcFyvB%2BBwWikKgnBuNY1hxusmwfHsPCkAQmg96sADWIC7Ls1waP4kiSkSkhEv4GhH0ff59xwkiD8vo%2BcLwCggBoi/L6scCwEgaAsMkdBxOQlCf9/eg8RgBcFxKQLAbI8BbAAGp4EwAAdwAPLJEYJwBeNBaBFWIA/CA0Qb7RDCI0AAnmg3gBDmDECIYg6I2guhL24LwT%2BbBBCIIYLQEhw9eBYGiF4YAbgxC0Afgw8BmBzZGHEJw8BeBwLdDZJgIRI8oxdEUtsBefxqg31TNENKlCPBYBvv8b4pDVhUAMMABQsCEHINQcI/gggRBiHYFIGQghFAqHUJI3QFYDBGBQJPSw%2Bg8DRAfpAVYqBcKZCEQAWnNmyVQZhjhRMQbsXgqA5HEABFgEJVsqg1EyC4Bg7hPCtAkESIIhSBilHKBIJ%2BqR0i1CyMUqYZS6mFAYJUoY8QuBP06N0OosxJhtDKb0hpvQmgdMWF0npAymlDJmH0CZ1TumrAUDPLYEhe792vpIseHBjiqH8JKKJZpjjAGQEmUB1wEkQFwIQEg5NdhcGWLwehWhlhrw3lvLk/guDknJO0fw3yj4/OkBfK%2Bpcb67Pvo/Z%2BnDX4wEQCAQcyRFJ/34l/H%2BxAIisG2Aco5JyzkXK3mYXg9U7mZL0HY4QohxDOKpW4tQN8vGkHgWlZIxj9BbIhTszgiDFIosJKgKg%2BzDnHMkKc85xxLnXI8BioBDynkvJfh8ze1xN7qo1Zqzll9tkjyhbYGFryV7apJdyvVd9YVvNWOk9IzhJBAA%3D
	constexpr inline Xint8x32_t() noexcept = default;

	/// NOTE: currently cannot be constexpr
	/// \return a uniform rng element
	[[nodiscard]] static inline S random() noexcept {
		S ret;
		for (size_t i = 0; i < 4; ++i) {
			ret.v64[i] = cryptanalysislib::rng();
		}
		return ret;
	}

	/// \param binary[in]:
	/// \param hex[in]:
	constexpr inline void print(bool binary = false,
	                            bool hex = false) const;

	/// \return __m256i_set_epi8()
	[[nodiscard]] constexpr static inline S set(const limb_type __q31, const limb_type __q30, const limb_type __q29, const limb_type __q28,
	                                            const limb_type __q27, const limb_type __q26, const limb_type __q25, const limb_type __q24,
	                                            const limb_type __q23, const limb_type __q22, const limb_type __q21, const limb_type __q20,
	                                            const limb_type __q19, const limb_type __q18, const limb_type __q17, const limb_type __q16,
	                                            const limb_type __q15, const limb_type __q14, const limb_type __q13, const limb_type __q12,
	                                            const limb_type __q11, const limb_type __q10, const limb_type __q09, const limb_type __q08,
	                                            const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
	                                            const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		S out;
		out.v256 = __extension__(__m256i)(__v32qi){
		        (char)__q00, (char)__q01, (char)__q02, (char)__q03, (char)__q04, (char)__q05, (char)__q06, (char)__q07,
		        (char)__q08, (char)__q09, (char)__q10, (char)__q11, (char)__q12, (char)__q13, (char)__q14, (char)__q15,
		        (char)__q16, (char)__q17, (char)__q18, (char)__q19, (char)__q20, (char)__q21, (char)__q22, (char)__q23,
		        (char)__q24, (char)__q25, (char)__q26, (char)__q27, (char)__q28, (char)__q29, (char)__q30, (char)__q31};

		return out;
	}

	/// \return _m256i_setr_epi8
	[[nodiscard]] constexpr static inline S setr(const limb_type __q31, const limb_type __q30, const limb_type __q29, const limb_type __q28,
	                                             const limb_type __q27, const limb_type __q26, const limb_type __q25, const limb_type __q24,
	                                             const limb_type __q23, const limb_type __q22, const limb_type __q21, const limb_type __q20,
	                                             const limb_type __q19, const limb_type __q18, const limb_type __q17, const limb_type __q16,
	                                             const limb_type __q15, const limb_type __q14, const limb_type __q13, const limb_type __q12,
	                                             const limb_type __q11, const limb_type __q10, const limb_type __q09, const limb_type __q08,
	                                             const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
	                                             const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		return set(__q00, __q01, __q02, __q03, __q04, __q05, __q06, __q07, __q08,
		           __q09, __q10, __q11, __q12, __q13, __q14, __q15, __q16, __q17,
		           __q18, __q19, __q20, __q21, __q22, __q23, __q24, __q25, __q26,
		           __q27, __q28, __q29, __q30, __q31);
	}

	/// sets all 32 8bit limbs to `a`
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline S set1(const limb_type a) noexcept {
		S out;
		out = S::set(a, a, a, a, a, a, a, a,
		             a, a, a, a, a, a, a, a,
		             a, a, a, a, a, a, a, a,
		             a, a, a, a, a, a, a, a);
		return out;
	}

	/// \tparam aligned[in]: if true a alied instruction will be emitted
	/// \param ptr[in]: pointer to (aligned) 32 bytes 
	/// \return vector element
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline S load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	/// the is `is_constant_evaluated()' is removed with `-O3`
	/// https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM6SuADJ4DJgAcj4ARpjEIACsABykAA6oCoRODB7evv6p6ZkCIWGRLDFxSbaY9o4CQgRMxAQ5Pn4BdpgOWQ1NBCUR0bEJyQqNza15HeP9oYPlw0kAlLaoXsTI7BzmAMyhyN5YANQmO254LCyhBMShAHQIp9gmGgCCu/uHmCdnY/iCDyeL3eZj2DAOXmOpzcBAAnilMAB9G5MQgKIFvYFoBhjTCqFLEI5jJiOZBHRGIlhmeIANjwRy8iSIVNpEGxYwZ10SyKOBBM8QsOzM/IAIksjgxUHjNikCCcAOxWTEaACctAEwCO6qM5OR/IsklFpyV71VFL5Ao0op%2BIqOEG1modSwtFiOVvitpM8rcdqgDq1Gud%2BqOXFF4uh0KOiXD3t9/qdLqOwo9Pzcka4NJjPr9GoDRiDAqOO2tEbOSckWbjuYTwcNKdLPqFlZzOprhfiJbOkckGmb9urgcTNM7abLkmjCuz/dbg%2BD8pHkdpS2NwJeKvN%2BtD9Z2tunjtnhcSJdjUD3eeABddKrDqcjE69U/jB4sXHdnq7ZYzfaf%2BZdXC376jj6ZgVpOVYzr%2Bm7JoBkZNmBLb7pBApcMW25AUcPbfgOSEvnWMFjveJ5nm2L4dmhi6ZiuyrrnqArQTado/hef7DvWRHEc%2BXDzh64Yfj6hGPthzGbke5Gfpm8EcThXDXmJwGgQ%2B4GIcJdFvreZZwYpCHnpeZgAepPqYZJTG6fRDYYQJSk6S6oILmWS5UaaNEuqh767iZNl4ZJUkqRY1I3uZlnaSRZisfhPpfsZQm6dx4XllhEG%2BWYolxZpREefqZiyXFRlaT5l47Gp5njglykFfp5kOTsJprhSLJ0ryLApAxXoWBulqiqQ7Uvp13XQV1tGCka8oio5a7EJgBDrAwjUpGNI2rsqrxjMQXgOJygiJKoQo8q1i0quyBB4gSRKNKSG0EDtcpBAAkgAshYQgMUKY2ql4GSthcUTIvC3ynLaXhcnq1X7e9oSak9/0XVtV2vU5XgMFkCo1aqa4APRo0caBNSSeBRHQhCwkcADuhAIEcAAqqjhHq1FroDm08ug%2BovR6cNrvTQNygAbqJgrQa915vdcGY8tzGabmFguc4IV1HNzrMWCl0vC4INKSGL6v6nhKs0fV9Lc9Sw4g3TC0m6aTkY2j%2B1Wyc8RuCkTSsEcsrEDbGN224E1TcQDD7fq%2BqSvgCiiMQzNsymh3HYSxLnaEtBzND21mDyYh4MAYToIi6pMOgbICByDMENycoAFSu%2BKkrSpgsrI/7qp4FQdp/CAIB4AoiKHYYBCIpg3NiF4JKYHnSwxijQsTx7oS8gg3xR/ihKiEoRIXCktBE%2BnkoTTPJIz986AkkwRzt0cacZ8P9eT5jhAk3QtASqgcoIEw3PfIYRPmGYq2CBcmCfy7jAxCOEwOiOmqNVSHV1PrWaDFGTMiNhACuus1xPTWHyc2HNwEqjQXcQ2tIGIECasg1U3tppHDQcgkaRwajLz2mAieTAvBECOOXG4RsGIQDqkbekpdxSu2ISqVBTCBE4LwTSBirDiBGwEaQ325DhEYKwV6UapsVHvDNijV4RcYYpzlEdMYXB844jlNonkkix7AiOFYo4siZraOTsiVuZ9M7Z1QLnRBNxlwYIWkteeJ17Fy30QQMwRjC5cxYRXOubxrE2MmmQgJuinEJ3PlnHOeckHeJFBwFYtBODxF4H4bgvBUCcDTJYawRI1gbD%2BqCHgpACCaGySsAA1iAHYKo7g7HlBoGkKpEgoS4jSDQGhki5I4JIApjTSAlI4LwBQIAND1MaSsOAsAkDYxSHQWI5BKAbK2XEYAUguB8DoEdYg8yIBRCmfjZgxBYScDqTcposIADyURtBdAaUU0g2M2CCBeQwdeUysBRC8MANwYhaDzO%2BVgFghhgDiA4FoUg%2BAJrdFftC5FeIuhMK2HU64NQpkJyiMQZ5HgsBTJuBcB5vBX7ECiOkTAIpMBwqMAnIwyy%2BAGGAAoAAangTAxMXkIkKXU/gggRBiHYFIGQghFAqHUEinQegDActMOUyw%2Bg8bzMgCsVAsosjQoALQvJ2EcI1cLuaqHNSwKgcKbVRCuGYG1L9VDOvNX8f6GcvDWGsGYZFqA6W3CwDqiAKxOjdGcBAVwkw/DHOCHMMoFQ9BpAyHUbIng2gpsKOmgYSbhjHIjem3oExM15ELTUT5PQZh5qGHEQtMxY16GJM0WtCx63hqqZsCQOS8mTKVdMzgRxVCJBpEa9WRxgDIDJFIO4XA7S4EICQE4tSli8C%2BVoUepBWlmHlHcLK8QVQ0kSPEHYkgVQ9MPQEMZEzSCFIDZwOZCyllKpWTARAIA0EpCYTs/OTV9nhFYFsEdY6J1TpnZIOdvBh5LuDXocVwhRDiBlQh%2BVagpm6GOcTUlKQaW9o4Pku9UyZkvKYd%2BuUqAm4gfHZISd06QyQfnRADw/76CEl2FwNdL7N0tLaTsTpOxBNCeE0J/QnBb33uKY%2B2wz6N1NLExwMw/aH2zO4/JulGRnCSCAA%3D
	/// \param ptr[in]: pointer to 32 aligned bytes
	/// \return: vector element
	[[nodiscard]] constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m256i tmp = u8tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			auto *ptr256 = (__m256i *) ptr;
			S out;
			out.v256 = *ptr256;
			return out;
		}
	}

	/// \param ptr[in]: pointer to 32 unaligned bytes
	/// \return: vector element
	[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			const __m256i tmp = u8tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			S out;
			out.v256 = ::internal::unaligned_load_wrapper((__m256i_u *) ptr);
			return out;
		}
	}

	/// NOTE: the store can never be constexpr ans its needs to access
	/// given memory
	/// \tparam aligned[in]: 
	/// \param ptr[in/out]: pointer to 32 (aligned) bytes
	/// \param in[in]: vector element
	template<const bool aligned = false>
	static inline void store(limb_type *ptr, 
                             const S in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	/// \param ptr[in/out]: pointer to 32 aligned bytes
	/// \param in[in]: vector element
	static inline void aligned_store(limb_type *ptr,
                                     const S in) noexcept {
		auto *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	/// \param ptr[in/out]: pointer to 32 unaligned bytes
	/// \param in[in]: vector element
	static inline void unaligned_store(limb_type *ptr, 
                                       const S in) noexcept {
		auto *ptr256 = (__m256i_u *) ptr;
		::internal::unaligned_store_wrapper(ptr256, in.v256);
	}

    /// Argument to use 
    ///     `((V) in1.v256 ^ (V) in2.v256)` instead of 
    ///     `((__v4du) in1.v256 ^ (__v4du) in2.v256);`        
    /// https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,selection:(endColumn:8,endLineNumber:3,positionColumn:8,positionLineNumber:3,selectionStartColumn:8,selectionStartLineNumber:3,startColumn:8,startLineNumber:3),source:'%23include+%3Cimmintrin.h%3E%0A%0A__m256i+square(__m256i+in1,+__m256i+in2)+%7B%0A%09return+(__m256i)+((__v4du)+in1+%5E+(__v4du)+in2)%3B%0A%7D'),l:'5',n:'0',o:'C%2B%2B+source+%231',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:g142,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,libs:!(),options:'-mavx+-mavx2+-O3',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+x86-64+gcc+14.2+(Editor+%231)',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4
	/// \param in1[in]: vector element
	/// \param in2[in]: vector element
	/// \return in1 ^ in2
	[[nodiscard]] constexpr static inline S xor_(const S in1,
	                                             const S in2) noexcept {
		S out;
		out.v256 = (__m256i) ((V) in1.v256 ^ (V) in2.v256);
		return out;
	}

	/// \param in1[in]: vector element
	/// \param in2[in]: vector element
	/// \return in1 & in2
	[[nodiscard]] constexpr static inline S and_(const S in1,
	                                             const S in2) noexcept {
		S out;
		out.v256 = (__m256i) ((V) in1.v256 & (V) in2.v256);
		return out;
	}

	/// \param in1[in]: vector element
	/// \param in2[in]: vector element
	/// \return in1 | in2
	[[nodiscard]] constexpr static inline S or_(const S in1,
	                                            const S in2) noexcept {
		S out;
		out.v256 = (__m256i) ((V) in1.v256 | (V) in2.v256);
		return out;
	}

	/// \param in1[in]: vector element
	/// \param in2[in]: vector element
	/// \return (~in1) & in2
	[[nodiscard]] constexpr static inline S andnot(const S in1,
	                                               const S in2) noexcept {
		S out;
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_andnotsi256((__v4di) in1.v256, (__v4di) in2.v256);
#else
		out.v256 = (__m256i) (~(V) in1.v256 & (V) in2.v256);
#endif
		return out;
	}

	/// \param in1[in]: vector element
	/// \return ~in1
	[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
		S out;
		const S minus_one = set1(-1);
		out.v256 = (__m256i) ((V) in1.v256 ^ (V) minus_one.v256);
		return out;
	}

	/// \param in1[in]: vector element
	/// \param in2[in]: vector element
	/// \return in1 + in2
	[[nodiscard]] constexpr static inline S add(const S in1,
	                                            const S in2) noexcept {
		S out;
		out.v256 = (__m256i) ((V) in1.v256 + (V) in2.v256);
		return out;
	}

	/// \param in1[in]: vector element
	/// \param in2[in]: vector element
	/// \return in1 - in2
	[[nodiscard]] constexpr static inline S sub(const S in1,
	                                            const S in2) noexcept {
		S out;
		out.v256 = (__m256i) ((V) in1.v256 - (V) in2.v256);
		return out;
	}

	/// 8 bit mul lo
	/// \param in1[in]: vector element
	/// \param in2[in]: vector element
	/// \return in1*in2
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const S in2) noexcept {
		S out;
		out.v256 = ((__m256i) ((V) in1.v256 * (V) in2.v256));
		return out;
		const __m256i maskl = __extension__(__m256i)(__v16hi){0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff};
		const __m256i maskh = __extension__(__m256i)(__v16hi){(short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00, (short) 0xff00};

		const __m256i in1l = (__m256i) ((V) in1.v256 & (V) maskl);
		const __m256i in2l = (__m256i) ((V) in2.v256 & (V) maskl);
		const __m256i in1h = (__m256i) ((V) in1.v256 & (V) maskh);
		const __m256i in2h = (__m256i) ((V) in2.v256 & (V) maskh);


		out.v256 = ((__m256i) ((__v16hu) in1l * (__v16hu) in2l)) & maskl;
		out.v256 ^= ((__m256i) ((__v16hu) in1h * (__v16hu) in2h)) & maskh;
		return out;
	}

	/// \param in1[in]: vector element
	/// \param in2[in]: vector element
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const limb_type in2) noexcept {
		const S rs = S::set1(in2);
		return S::mullo(in1, rs);
	}

    /// TODO impl: https://github.com/ridiculousfish/libdivide/blob/master/libdivide.h#L144
    /// definition of magic/more: https://github.com/ridiculousfish/libdivide/blob/af1db190fe740f33e08a0b541146cda85dbd5006/libdivide.h#L1382
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
		const S mask = set1((1u << in2) - 1u);
		out = S::and_(in1, mask);
		// if (std::is_constant_evaluated()) {
		// 	out.v256 = (__m256i)((__v32qi)out.v256) << in2;
		// 	return out;
		// }
		// out.v256 = (__m256i) __builtin_ia32_psllwi256((__v16hi) out.v256, in2);

		out.v256 = (__m256i) ((V) out.v256) << in2;
		return out;
	}

	/// \param in1[in]: vector element
	/// \param in2[in]: vector element
	/// \return in1 >> in2
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const uint8_t in2) noexcept {
		assert(in2 <= 8);
		const S mask1 = set1(((1u << (8u - in2)) - 1u) << in2);
		const S mask2 = set1((1u << (8u - in2)) - 1u);
		S out = S::and_(in1, mask1);
		if (std::is_constant_evaluated()) {
			out.v256 = (__m256i) ((V) out.v256) >> in2;
			return out;
		}
		out.v256 = (__m256i) __builtin_ia32_psrlwi256((__v16hi) out.v256, in2);
		out = S::and_(out, mask2);
		return out;
	}

	/// \param in1[in]: vector element
	/// \param in2[in]: 
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S ror(const S in1,
	                                             const uint8_t in2) noexcept {

		S out;
        const __m256i mask = _mm256_set1_epi8((1u << (8u-in2)) -1u);
        out.v256 = _mm256_slli_epi16(in1.v256, in2) ^ (_mm256_srli_epi16(in1.v256, 8u-in2) & mask);
		return out;

    }

	/// \param in1[in]: vector element
	/// \param in2[in]: 
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S rol(const S in1,
	                                             const uint8_t in2) noexcept {
		S out;
        const __m256i mask = _mm256_set1_epi8((1u << (8-in2)) -1u);
        out.v256 = _mm256_slli_epi16(in1.v256, in2) ^ (_mm256_srli_epi16(in1.v256, 8u-in2) & mask);
		return out;
    }

    /// TODO
	/// \param in1[in]: vector element
	/// \param in2[in]: vector element
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline S gt_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 > (V) in2.v256);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return in1 > in2 compressed
	[[nodiscard]] constexpr static inline uint32_t gt(const S in1,
	                                                  const S in2) noexcept {
		const __m256i tmp = (__m256i) ((V) in1.v256 > (V) in2.v256);
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
	}

	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline S lt_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 < (V) in2.v256);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return in1 > in2 compressed
	[[nodiscard]] constexpr static inline uint32_t lt(const S in1,
	                                                  const S in2) noexcept {
		const __m256i tmp = (__m256i) ((V) in1.v256 < (V) in2.v256);
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 == in2 compressed
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
	                                                      const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((__v32qs) in1.v256 == (__v32qs) in2.v256);
		return ret;
	}

	/// \param in1[in]:
	/// \param in2[in]:
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
	                                                   const S in2) noexcept {
		const __m256i tmp = (__m256i) ((__v32qi) in1.v256 == (__v32qi) in2.v256);
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
	}

	/// \param in[in]:
	/// \return 
	[[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
		S ret;
		ret.v256 = popcount_avx2_8(in.v256);
		return ret;
	}

	/// checks if all bytes are equal
	/// source: https://github.com/WojciechMula/toys/tree/master/simd-all-bytes-equal
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
#ifdef __clang__
		// no cost, 0th lane is mapped to an XMM reg
		const __m128i lane0 = __builtin_shufflevector((__v4di) in.v256, (__v4di) in.v256, 0, 1);
		const __m128i tmp = (__m128i) __builtin_ia32_pshufb128((__v16qi) lane0,
		                                                       (__v16qi) __extension__(__m128i)(__v4si){0, 0, 0, 0});
		const __m256i populated_0th_byte = (__m256i) __builtin_shufflevector((__v2di) tmp, (__v2di) tmp, 0, 1, 2, 3);
		const __m256i eq = (__m256i) ((__v32qi) in.v256 == (__v32qi) populated_0th_byte);
		return (uint32_t) __builtin_ia32_pmovmskb256((__v32qi) eq) == 0xffffffff;
#else
		const __m128i lane0 = (__m128i) __builtin_ia32_si_si256((__v8si) in.v256);
		const __m128i tmp = (__m128i) __builtin_ia32_pshufb128((__v16qi) lane0,
		                                                       (__v16qi) __extension__(__m128i)(__v4si){0, 0, 0, 0});
		const __m256i populated_0th_byte = ((__m256i) __builtin_ia32_vinsertf128_si256(
		        (__v8si) (__m256i) (__builtin_ia32_si256_si((__v4si) tmp)),
		        (__v4si) (__m128i) (tmp),
		        (int) (1)));
		const __m256i eq = (__m256i) ((__v32qi) in.v256 == (__v32qi) populated_0th_byte);
		return (uint32_t) __builtin_ia32_pmovmskb256((__v32qi) eq) == 0xffffffff;
#endif
	}

	/// only reverses the u8 limbs
	/// source:  https://github.com/WojciechMula/toys/blob/master/simd-basic/reverse-bytes/reverse.avx2.cpp
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
		// extract 128-bit lanes
		const __m128i lo = ((__m128i) __builtin_ia32_extract128i256((__v4di) (__m256i) (in.v256), (int) (0)));
		const __m128i hi = ((__m128i) __builtin_ia32_extract128i256((__v4di) (__m256i) (in.v256), (int) (1)));

		// reverse them using SSE instructions
		const __m128i indices = __extension__(__m128i)(__v16qi){15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0};
		const __m128i lo_rev = (__m128i) __builtin_ia32_pshufb128((__v16qi) lo, (__v16qi) indices);
		const __m128i hi_rev = (__m128i) __builtin_ia32_pshufb128((__v16qi) hi, (__v16qi) indices);

		// build the new AVX2 vector
#ifdef __clang__
		__m256i ret = __builtin_shufflevector((__v2di) hi_rev, (__v2di) hi_rev, 0, 1, 2, 3);
#else
		__m256i ret = (__m256i) __builtin_ia32_si256_si((__v4si) hi_rev);
#endif
		ret = ((__m256i) __builtin_ia32_insert128i256((__v4di) (__m256i) (ret),
		                                              (__v2di) (__m128i) (lo_rev), (int) (1)));
		S ret2;
		ret2.v256 = ret;
		return ret2;
	}
	/// \param in
	/// \param perm
	/// \return
	[[nodiscard]] constexpr static inline S permute(const S in,
	                                                const S perm) noexcept {
		S ret;
		ret.v256 = _mm256_shuffle_epi8(in.v256, perm.v256);
		return ret;
	}

	/// kmoves the msb into each bit
	[[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
		return __builtin_ia32_pmovmskb256((__v32qi) in.v256);
	}

    /// \tparam scale[in]:
    /// \param ptr[in]:
    /// \param data[in]:
    /// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
												   const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		S ret;

		const limb_type *ptr8 = (limb_type *) ptr;
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
	constexpr static inline void scatter(const void *ptr,
										 const S offset,
										 const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		limb_type *ptr8 = (limb_type *) ptr;
		for (uint32_t i = 0; i < 8; i++) {
			*(ptr8 + offset.d[i] * scale) = data.d[i];
		}
	}

	/// \param a[in]:
	/// \param b[in]:
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
		if constexpr (__unsigned) {
#ifdef __clang__
			c.v256 = (__m256i)__builtin_elementwise_min((__v32qu)a.v256, (__v32qu)b.v256);
#else
			c.v256 = (__m256i)_mm256_min_epu8(a.v256, b.v256);
#endif
		} else {
#ifdef __clang__
			c.v256 = (__m256i)__builtin_elementwise_min((__v32qi)a.v256, (__v32qi)b.v256);
#else
			c.v256 = (__m256i)_mm256_min_epi8(a.v256, b.v256);
#endif
		}
        return c;
    }

	/// \param a[in]:
	/// \param b[in]:
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
		if constexpr (__unsigned) {
#ifdef __clang__
			c.v256 = (__m256i)__builtin_elementwise_max((__v32qu)a.v256, (__v32qu)b.v256);
#else
			c.v256 = (__m256i)_mm256_max_epu8(a.v256, b.v256);
#endif
		} else {
#ifdef __clang__
			c.v256 = (__m256i)__builtin_elementwise_max((__v32qi)a.v256, (__v32qi)b.v256);
#else
			c.v256 = (__m256i)_mm256_max_epi8(a.v256, b.v256);
#endif
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
	using limb_type = std::conditional<__unsigned, uint16_t, int16_t>::type; 
	using S = Xint16x16_t;
    
    using V = std::conditional<__unsigned, __v16hu, __v16hi>::type;

    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		// compatibility with TxN_t
		T16 d[16];

		T8  v8[32];
		T16 v16[16];
		T32 v32[8];
		T64 v64[4];
		__m256i v256;
	};

	constexpr inline Xint16x16_t() noexcept = default;

	[[nodiscard]] constexpr inline static size_t size() noexcept { 
        return LIMBS; 
    }
	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept { 
        return __unsigned; 
    }

    /// \param i[in]: position of the limb to return
    /// \return __m256i[i]
	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const {
		assert(i < LIMBS);
		return d[i];
	}

    /// \param i[in]: position of the limb to return
    /// \return __m256i[i]
	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) {
		assert(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline S random() noexcept {
		S ret;
		for (size_t i = 0; i < 4; ++i) {
			ret.v64[i] = cryptanalysislib::rng();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	///
	[[nodiscard]] constexpr static inline S set(const limb_type __q15, const limb_type __q14, const limb_type __q13, const limb_type __q12,
	                                            const limb_type __q11, const limb_type __q10, const limb_type __q09, const limb_type __q08,
	                                            const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
	                                            const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		S out;
		out.v256 = __extension__(__m256i)(__v16hi){
		        (short)__q00, (short)__q01, (short)__q02, (short)__q03, (short)__q04, (short)__q05, (short)__q06, (short)__q07,
		        (short)__q08, (short)__q09, (short)__q10, (short)__q11, (short)__q12, (short)__q13, (short)__q14, (short)__q15};
		return out;
	}

	///
	[[nodiscard]] constexpr static inline S setr(const limb_type __q15, const limb_type __q14, const limb_type __q13, const limb_type __q12,
	                                             const limb_type __q11, const limb_type __q10, const limb_type __q09, const limb_type __q08,
	                                             const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
	                                             const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		return S::set(__q00, __q01, __q02, __q03, __q04, __q05, __q06, __q07, 
                      __q08, __q09, __q10, __q11, __q12, __q13, __q14, __q15);
	}

	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline S set1(const limb_type a) noexcept {
		return S::set(a, a, a, a, a, a, a, a, a, a, a, a, a, a, a, a);
	}

	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline S load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline S aligned_load(const limb_type ptr[16]) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m256i tmp = u16tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			auto *ptr256 = (__m256i *) ptr;
			S out;
			out.v256 = *ptr256;
			return out;
		}
	}

	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			const __m256i tmp = u16tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			S out;
			out.v256 = ::internal::unaligned_load_wrapper((__m256i_u *) ptr);
			return out;
		}
	}

	/// NOTE: can never be constexpr
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	static inline void store(limb_type *ptr,
                             const S in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		aligned_store(ptr, in);
	}

	/// \param ptr
	/// \param in
	static inline void aligned_store(limb_type *ptr,
                                     const S in) noexcept {
		auto *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	/// \param ptr
	/// \param in
	static inline void unaligned_store(limb_type *ptr,
                                       const S in) noexcept {
		auto *ptr256 = (__m256i_u *) ptr;
		::internal::unaligned_store_wrapper(ptr256, in.v256);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S xor_(const S in1,
	                                             const S in2) noexcept {
		S out;
		out.v256 = in1.v256 ^ in2.v256;
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S and_(const S in1,
	                                             const S in2) noexcept {
		S out;
		out.v256 = in1.v256 & in2.v256;
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S or_(const S in1,
	                                            const S in2) noexcept {
		S out;
		out.v256 = in1.v256 | in2.v256;
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S andnot(const S in1,
	                                               const S in2) noexcept {
		S out;
		out.v256 = ~(in1.v256 & in2.v256);
		return out;
	}

	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
		S out;
		const S minus_one = S::set1(-1);
		out.v256 = in1.v256 ^ minus_one.v256;
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S add(const S in1,
	                                            const S in2) noexcept {
		S out;
		out.v256 = (__m256i) ((V) in1.v256 + (V) in2.v256);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S sub(const S in1,
	                                            const S in2) noexcept {
		S out;
		out.v256 = (__m256i) ((V) in1.v256 - (V) in2.v256);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const S in2) noexcept {
		S out;
		out.v256 = (__m256i) ((V) in1.v256 * (V) in2.v256);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const limb_type in2) noexcept {
		auto rs = S::set1(in2);
		return mullo(in1, rs);
	}

    /// TODO: https://stackoverflow.com/questions/16822757/sse-integer-division
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
        S out;
        const __m256i vb = _mm256_set1_epi16(32768 / in2);
        out.v256 = _mm256_mulhrs_epi16(in1.v256, vb);
        return out;
    }

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S slli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 <= 16);
		const S mask = set1((1u << ((16u - in2) & 15u)) - 1u);
		S out = S::and_(in1, mask);
#ifndef __clang__
		// NOTE: there is no typecast to V, because gcc does things
		out.v256 = (__m256i) __builtin_ia32_psllwi256((__v16hi)out.v256, in2);
#else
		out.v256 = (__m256i) (((V) out.v256) << in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 <= 16);
		const S mask = set1(~((1u << in2) - 1u));
		S out;
		out = S::and_(in1, mask);
#ifndef __clang__
		// NOTE: there is no typecast to V, because gcc does things
		out.v256 = (__m256i) __builtin_ia32_psrlwi256((__v16hi) out.v256, in2);
#else
		out.v256 = (__m256i) (((V) out.v256) >> in2);
#endif
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S ror(const S in1,
												const limb_type in2) noexcept {
		S out;
        out.v256 = _mm256_slli_epi16(in1.v256, 16u - in2) ^ _mm256_srli_epi16(in1.v256, in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S rol(const S in1,
												const limb_type in2) noexcept {
		S out;
        out.v256 = _mm256_slli_epi16(in1.v256, in2) ^ _mm256_srli_epi16(in1.v256, 16u-in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline S gt_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 > (V) in2.v256);
		return ret;
	}

	/// NOTE: this is a function which cannot be vectorized
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t gt(const S in1,
	                                                  const S in2) noexcept {
		S tmp;
		tmp.v256 = (__m256i) ((V) in1.v256 > (V) in2.v256);
		return S::move(tmp);
	}

	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline S lt_(const S in1,
	                                                      const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 < (V) in2.v256);
		return ret;
	}

	/// NOTE: this is a function which cannot be vectorized
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t lt(const S in1,
	                                                  const S in2) noexcept {
		S tmp;
		tmp.v256 = (__m256i) ((V) in1.v256 < (V) in2.v256);
		return S::move(tmp);
	}

	/// \param in1
	/// \param in2
	/// \return in1 == in2 uncompressed
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
	                                                       const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 == (V) in2.v256);
		return ret;
	}

	/// NOTE: this is a function which cannot be vectorized
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline int cmp(const S in1,
	                                              const S in2) noexcept {
		S tmp;
		tmp.v256 = (__m256i) ((V) in1.v256 == (V) in2.v256);
		uint32_t t = _mm256_movemask_epi8(tmp.v256);
		uint16_t ret = _pdep_u32(t, 0b01010101010101010101010101010101);
		return ret;
	}

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
		S ret;
		ret.v256 = popcount_avx2_16(in.v256);
		return ret;
	}
	
	/// \param in
	/// \return
    [[nodiscard]] constexpr static inline S clz(const S in) noexcept {
		S ret;
		// TODO
		return ret;
    }
	
    /// \param in
	/// \return
    [[nodiscard]] constexpr static inline S ctz(const S in) noexcept {
		S ret;
		// TODO
		return ret;
    }

	/// checks if all bytes are equal
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
		const __m256i tmp1 = _mm256_permute4x64_epi64(in.v256, 0);
		const __m256i tmp2 = _mm256_shufflelo_epi16(tmp1, 0);
		const __m256i tmp3 = (__m256i) ((V)in.v256 == (V)tmp2);
		const uint32_t t = _mm256_movemask_epi8(tmp3);
		return t == 0xFFFFFFFF;
	}

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
		S ret;
		constexpr uint8x32_t shuffle = uint8x32_t::setr(14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1,
														14,15,12,13,10,11,8,9,6,7,4,5,2,3,0,1);
		__m256i tmp =_mm256_permute4x64_epi64(in.v256, 0b01001110);
		ret.v256 = _mm256_shuffle_epi8(tmp, shuffle.v256);
		return ret;
	}

    /// not vectorized: https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,selection:(endColumn:56,endLineNumber:4,positionColumn:56,positionLineNumber:4,selectionStartColumn:56,selectionStartLineNumber:4,startColumn:56,startLineNumber:4),source:'%23include+%3Cimmintrin.h%3E%0A%23include+%3Cstdint.h%3E%0A%0Avoid+perm(uint16_t+a%5B16%5D,+uint16_t+b%5B16%5D,+uint16_t+c%5B16%5D)+%7B%0A++++for+(uint32_t+i+%3D+0%3B+i+%3C+16%3B+i%2B%2B)+%7B%0A++++++++a%5Bb%5Bi%5D%5D+%3D+c%5Bi%5D%3B%0A++++%7D%0A%7D'),l:'5',n:'0',o:'C%2B%2B+source+%231',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:g142,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:c%2B%2B,libs:!(),options:'-mavx+-mavx2+-O3',overrides:!(),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+x86-64+gcc+14.2+(Editor+%231)',t:'0')),k:50,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4
	/// \param in
	/// \param perm
	/// \return
	[[nodiscard]] constexpr static inline S permute(const S in,
	                                                const S perm) noexcept {
		S ret;
        for (uint32_t i = 0; i < LIMBS; i++) {
            ret.d[perm.d[i]] = in[i];
        }

        return ret;
    }

    ///
	[[nodiscard]] constexpr static inline S conflict(const S in1) noexcept {
		S ret;
        return ret;
    }

	/// kmoves the msb into each bit
	[[nodiscard]] constexpr static inline limb_type move(const S in) noexcept {
		uint32_t t = _mm256_movemask_epi8(in.v256);
		uint16_t ret = _pext_u32(t, 0b01010101010101010101010101010101);
		return ret;
	}

    /// \tparam scale[in]:
    /// \param ptr[in]:
    /// \param data[in]:
    /// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
												   const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		S ret;

		const limb_type *ptr8 = (limb_type *) ptr;
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
	constexpr static inline void scatter(const void *ptr,
										 const S offset,
										 const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		limb_type *ptr8 = (limb_type *) ptr;
		for (uint32_t i = 0; i < 8; i++) {
			*(ptr8 + offset.d[i] * scale) = data.d[i];
		}
	}

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
		c.v256 = (__m256i)__builtin_elementwise_min((V)a.v256, (V)b.v256);
        return c;
    }

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
		c.v256 = (__m256i)__builtin_elementwise_max((V)a.v256, (V)b.v256);
        return c;
    }
};


using uint16x16_t = Xint16x16_t<true>;
using  int16x16_t = Xint16x16_t<false>;

template<const bool __unsigned=true>
struct Xint32x8_t {
	constexpr static uint32_t LIMBS = 8;
	using limb_type = std::conditional<__unsigned, uint32_t, int32_t>::type; 
	using S = Xint32x8_t;
    using V = std::conditional<__unsigned, __v8su, __v8si>::type;

    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		// compatibility to TxN_t
		T32 d[8];

		T8  v8[32];
		T16 v16[16];
		T32 v32[8];
		T64 v64[4];
		cryptanalysislib::_Xint32x4_t<__unsigned> v128[2];
		__m256i v256;
	};

	[[nodiscard]] constexpr inline static size_t size() noexcept { 
        return LIMBS; 
    }
	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
        return __unsigned; 
    }

	/// \param i
	/// \return
	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	/// \param i
	/// \return
	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline S random() noexcept {
		S ret{};
		for (size_t i = 0; i < 4; ++i) {
			ret.v64[i] =cryptanalysislib::rng();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	/// \param a0
	/// \param a1
	/// \param a2
	/// \param a3
	/// \param a4
	/// \param a5
	/// \param a6
	/// \param a7
	/// \return
	[[nodiscard]] constexpr inline static S set(const limb_type a0,
	                                            const limb_type a1,
	                                            const limb_type a2,
	                                            const limb_type a3,
	                                            const limb_type a4,
	                                            const limb_type a5,
	                                            const limb_type a6,
	                                            const limb_type a7) noexcept {
		S out{};
		out.v256 = __extension__(__m256i)(__v8si){(int) a7, (int) a6, (int) a5, (int) a4, (int) a3, (int) a2, (int) a1, (int) a0};
		return out;
	}

	/// \param a0
	/// \param a1
	/// \param a2
	/// \param a3
	/// \param a4
	/// \param a5
	/// \param a6
	/// \param a7
	/// \return
	[[nodiscard]] constexpr inline static S setr(const limb_type a0,
	                                             const limb_type a1,
	                                             const limb_type a2,
	                                             const limb_type a3,
	                                             const limb_type a4,
	                                             const limb_type a5,
	                                             const limb_type a6,
	                                             const limb_type a7) noexcept {
		return set(a7, a6, a5, a4, a3, a2, a1, a0);
	}

	///
	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline S set1(const limb_type a) noexcept {
		return set(a, a, a, a, a, a, a, a);
	}

	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline S load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m256i tmp = u32tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			auto *ptr256 = (__m256i *) ptr;
			S out;
			out.v256 = *ptr256;
			return out;
		}
	}

	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			const __m256i tmp = u32tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			S out;
			out.v256 = ::internal::unaligned_load_wrapper((__m256i_u *) ptr);
			return out;
		}
	}

	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(limb_type *ptr,
                                       const S in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(limb_type *ptr,
                                               const S in) noexcept {
		auto *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(limb_type *ptr,
                                                 const S in) noexcept {
		auto *ptr256 = (__m256i_u *) ptr;
		::internal::unaligned_store_wrapper(ptr256, in.v256);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S xor_(const S in1,
	                                             const S in2) noexcept {
		S out{};
		out.v256 = in1.v256 ^ in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S and_(const S in1,
	                                             const S in2) noexcept {
		S out{};
		out.v256 = in1.v256 & in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S or_(const S in1,
	                                            const S in2) noexcept {
		S out{};
		out.v256 = in1.v256 | in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S andnot(const S in1,
	                                               const S in2) noexcept {
		S out{};
		out.v256 = (__m256i) (~(__v4du) in1.v256 & (__v4du) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
		S out{};
		constexpr S minus_one = S::set1(-1);
		out.v256 = in1.v256 ^ minus_one.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S add(const S in1,
	                                            const S in2) noexcept {
		S out{};
		out.v256 = (__m256i) ((V) in1.v256 + (V) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S sub(const S in1,
	                                            const S in2) noexcept {
		S out{};
		out.v256 = (__m256i) ((V) in1.v256 - (V) in2.v256);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mul(const S in1,
	                                            const S in2) noexcept {
		S out{};
		// todo
		out.v256 = (__m256i) ((V) in1.v256 * (V) in2.v256);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mulhi(const S in1,
	                                              const S in2) noexcept {
		S out{};
		// todo
		out.v256 = (__m256i) ((V) in1.v256 * (V) in2.v256);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const S in2) noexcept {
		S out{};
		out.v256 = (__m256i) ((V) in1.v256 * (V) in2.v256);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const uint32_t in2) noexcept {
		auto m = S::set1(in2);
		return mullo(in1, m);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
        S out;
        for (uint32_t i = 0; i < LIMBS; i++) {
            out[i] = in1[i] / in2;
        }
        return out;
    }

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S slli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 <= 8);
		S out{};
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psllwi256((__v16hi) in1.v256, in2);
#else
		out.v256 = (__m256i) ((V) in1.v256 << in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 <= 8);
		S out{};
#ifndef __clang__
		// NOTE: there is no typecast to V, because gcc does things
		out.v256 = (__m256i) __builtin_ia32_psrldi256((__v8si) in1.v256, in2);
#else
		out.v256 = (__m256i) ((V) in1.v256 >> in2);
#endif
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S ror(const S in1,
												const limb_type in2) noexcept {
		S out;
        out.v256 = _mm256_slli_epi32(in1.v256, 32 - in2) ^ _mm256_srli_epi32(in1.v256, in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S rol(const S in1,
												const limb_type in2) noexcept {
		S out;
        out.v256 = _mm256_slli_epi32(in1.v256, in2) ^ _mm256_srli_epi32(in1.v256, 32u-in2);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompress
	[[nodiscard]] constexpr static inline S gt_(const S in1,
	                                            const S in2) noexcept {
		S ret{};
		ret.v256 = (__m256i) ((V) in1.v256 > (V) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t gt(const S &in1,
                                                      const S &in2) noexcept {
		const __m256i tmp = (__m256i) ((V) in1.v256 > (V) in2.v256);
		return __builtin_ia32_movmskps256((__v8sf) tmp);
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompress
	[[nodiscard]] constexpr static inline S lt_(const S in1,
	                                            const S in2) noexcept {
		S ret{};
		ret.v256 = (__m256i) ((V) in1.v256 < (V) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t lt(const S in1,
	                                                  const S in2) noexcept {
		const __m256i tmp = (__m256i) ((V) in1.v256 < (V) in2.v256);
		return __builtin_ia32_movmskps256((__v8sf) tmp);
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 == in2 uncompress
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
	                                             const S in2) noexcept {
		S ret{};
		ret.v256 = (__m256i) ((V) in1.v256 == (V) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline int cmp(const S in1,
                                                  const S in2) noexcept {
		const __m256i tmp = (__m256i) ((V) in1.v256 == (V) in2.v256);
#ifndef __clang__
		return __builtin_ia32_movmskps256((__v8sf) tmp);
#else
		return _mm256_movemask_ps((__m256) tmp);
#endif
	}

	///
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
		S ret{};

#ifdef USE_AVX512
#ifndef __clang__
		ret.v256 = (__m256i) __builtin_ia32_vpopcountd_v8si((V) in.v256);
#else
		ret.v256 = __builtin_ia32_vpopcntd_256((V) in.v256);
#endif
#else
		ret.v256 = popcount_avx2_32(in.v256);
#endif
		return ret;
	}

	/// checks if all bytes are equal
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
		constexpr S shuffle = S::set1(0);
		const __m256i tmp1 = _mm256_permutevar8x32_epi32(in.v256, shuffle.v256);
		const __m256i tmp2 = (__m256i) ((V)in.v256 == (V)tmp1);
#ifndef __clang__
		return __builtin_ia32_movmskps256((__v8sf) tmp2) == 0xFF;
#else
		return _mm256_movemask_ps((__m256) tmp2) == 0xFF;
#endif
	}

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
		S ret;
		constexpr S shuffle = S::setr(7,6,5,4,3,2,1,0);
		ret.v256 = _mm256_permutevar8x32_epi32(in.v256, shuffle.v256);
		return ret;
	}

	/// \param in
	/// \param perm
	/// \return
	[[nodiscard]] constexpr static inline S permute(const S in,
	                                                const S perm) noexcept {
		S ret;
#ifdef __clang__
		ret.v256 = (__m256i) __builtin_ia32_permvarsi256((V) in.v256, (V) perm.v256);
#else 

		ret.v256 = (__m256i) __builtin_ia32_permvarsi256((__v8si) in.v256, (__v8si) perm.v256);
#endif
		return ret;
	}

	/// moves the msb into each bit
	[[nodiscard]] constexpr static inline uint8_t move(const S in) noexcept {
		return __builtin_ia32_movmskps256((__v8sf) in.v256);
	}

	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
	                                               const S data) noexcept {
		S ret;
#ifndef __clang__
		// NOTE: there is no typecast to V, because gcc does things
		ret.v256 = (__m256i) __builtin_ia32_gathersiv8si((__v8si) _mm256_setzero_si256(),
		                                                 (int const *) (ptr),
		                                                 (__v8si) (__m256i) (data.v256),
		                                                 (__v8si) _mm256_set1_epi32(-1),
		                                                 (int) (scale));
#else
		ret.v256 = _mm256_i32gather_epi32((int *) ptr, data.v256, scale);
#endif
		return ret;
	}

	/// \tparam scale
	/// \param ptr
	/// \param offset
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	constexpr static inline void scatter(const void *ptr,
	                                     const S offset,
	                                     const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 8; i++) {
			*(uint32_t *) (ptr8 + offset.v32[i] * scale) = data.v32[i];
		}
	}

	/// needs BMI2
	/// src: https://stackoverflow.com/questions/36932240/avx2-what-is-the-most-efficient-way-to-pack-left-based-on-a-mask
	/// input:
	/// 	mask: 0b010101010
	/// output: a permutation mask s.t, applied on in =  [ x0, x1, x2, x3, x4, x5, x6, x7 ],
	/// 			S::permute(in, permutation_mask) will result int
	///  	[x1, x3, x5, x7, 0, 0, 0, 0]
	[[nodiscard]] constexpr static inline S pack(const uint32_t mask) noexcept {
		S ret{};
#ifdef USE_BMI2
		uint64_t expanded_mask = __builtin_ia32_pdep_di(mask, 0x0101010101010101);
		expanded_mask *= 0xFFU;
		const uint64_t identity_indices = 0x0706050403020100;
		uint64_t wanted_indices = __builtin_ia32_pext_di(identity_indices, expanded_mask);
		const __m128i bytevec = __extension__(__m128i)(__v2di){0, (long long int) wanted_indices};
#ifdef __clang__
		ret.v256 = (__m256i) __builtin_convertvector((__v8hi) bytevec, V);
#else
		ret.v256 = (__m256i) __builtin_ia32_pmovzxbd256((__v16qi) bytevec);
#endif
#else
		assert(false);
#endif
		return ret;
	}

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S cvtepu8(const cryptanalysislib::_uint8x16_t in) noexcept {
		S ret{};
#ifdef __clang__
		ret.v256 = (__m256i) __builtin_convertvector((__v8hi) in.v128, V);
#else
		ret.v256 = (__m256i) __builtin_ia32_pmovzxbd256((__v16qi) in.v128);
#endif
		return ret;
	}

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
        c.v256 = _mm256_min_epi32(a.v256, b.v256);
        return c;
    }

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
        c.v256 = _mm256_max_epi32(a.v256, b.v256);
        return c;
    }
};

using uint32x8_t = Xint32x8_t<true>;
using  int32x8_t = Xint32x8_t<false>;

template<const bool __unsigned=true>
struct Xint64x4_t {
	constexpr static uint32_t LIMBS = 4;
	using limb_type = std::conditional<__unsigned, uint64_t, int64_t>::type; 
	using S = Xint64x4_t;
    using V = std::conditional<__unsigned, __v4du, __v4di>::type;

    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		// compatibility with TxN_t
		T64 d[4];

		T8 v8[32];
		T16 v16[16];
		T32 v32[8];
		T64 v64[4];
		__m256i v256;
	};

	[[nodiscard]] constexpr inline static size_t size() noexcept { 
        return LIMBS; 
    }

	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept { 
        return __unsigned; 
    }

    /// \param i[in]: position of the limb to return
    /// \return __m256i[i]
	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		assert(i < LIMBS);
		return d[i];
	}

    /// \param i[in]: position of the limb to return
    /// \return __m256i[i]
	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline S random() noexcept {
		S ret;
		for (size_t i = 0; i < 4; ++i) {
			ret.v64[i] = cryptanalysislib::rng();
		}
		return ret;
	}

	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	/// \param a
	/// \param b
	/// \param c
	/// \param d
	/// \return
	[[nodiscard]] constexpr static inline S set(const limb_type a,
	                                            const limb_type b,
	                                            const limb_type c,
	                                            const limb_type d) noexcept {
		S out;
		out.v256 = __extension__(__m256i)(__v4di){(long long) d,
		                                          (long long) c,
		                                          (long long) b,
		                                          (long long) a};
		return out;
	}

	/// \param a
	/// \param b
	/// \param c
	/// \param d
	/// \return
	[[nodiscard]] constexpr static inline S setr(const limb_type a,
                                                 const limb_type b,
                                                 const limb_type c,
                                                 const limb_type d) noexcept {
		return set(d, c, b, a);
	}

	/// \param a
	/// \return
	[[nodiscard]] constexpr static inline S set1(const limb_type a) noexcept {
		return set(a, a, a, a);
	}

	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline S load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline S aligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m256i tmp = u64tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			auto *ptr256 = (__m256i *) ptr;
			S out;
			out.v256 = *ptr256;
			return out;
		}
	}

	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			const __m256i tmp = u64tom256(ptr);
			S out;
			out.v256 = tmp;
			return out;
		} else {
			S out;
			out.v256 = ::internal::unaligned_load_wrapper((const __m256i_u *) ptr);
			return out;
		}
	}

	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	static inline void store(limb_type *ptr, 
                             const S in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	/// \param ptr
	/// \param in
	static inline void aligned_store(limb_type *ptr,
                                     const S in) noexcept {
		auto *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	/// \param ptr
	/// \param in
	static inline void unaligned_store(limb_type *ptr,
                                       const S in) noexcept {
		auto *ptr256 = (__m256i_u *) ptr;
		::internal::unaligned_store_wrapper(ptr256, in.v256);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S xor_(const S in1,
	                                             const S in2) noexcept {
		S out;
		out.v256 = in1.v256 ^ in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S and_(const S in1,
	                                             const S in2) noexcept {
		S out;
		out.v256 = in1.v256 & in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S or_(const S in1,
	                                            const S in2) noexcept {
		S out;
		out.v256 = in1.v256 | in2.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S andnot(const S in1,
	                                               const S in2) noexcept {
		S out;
		out.v256 = (__m256i) (~(__v4du) in1.v256 & (__v4du) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
		S out;
		constexpr S minus_one = S::set1(-1);
		out.v256 = in1.v256 ^ minus_one.v256;
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S add(const S in1,
	                                            const S in2) noexcept {
		S out;
		out.v256 = (__m256i) ((V) in1.v256 + (V) in2.v256);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S sub(const S in1,
	                                            const S in2) noexcept {
		S out;
		out.v256 = (__m256i) ((V) in1.v256 - (V) in2.v256);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const S in2) noexcept {
		S out;
		out.v256 = (__m256i) ((V) in1.v256 * (V) in2.v256);
		// if (std::is_constant_evaluated()) {
		// 	for (uint32_t i = 0; i < 4; i++) {
		// 		out.v64[i] = in1.v64[i] * in2.v64[i];
		// 	}
		// } else {
		// 	for (uint32_t i = 0; i < 4; i++) {
		// 		out.v64[i] = in1.v64[i] * in2.v64[i];
		// 	}
		// }
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
        S out;
        const __m256i vb = _mm256_set1_epi16(32768 / in2);
        out.v256 = _mm256_mulhrs_epi16(in1.v256, vb);
        return out;
    }

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                              const limb_type in2) noexcept {
		auto m = S::set1(in2);
		return mullo(in1, m);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S slli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 <= 8);
		S out;
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psllqi256((__v4di) in1.v256, in2);
#else
		out.v256 = (__m256i) ((__v4di) in1.v256 << in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 <= 8);
		//const S mask = set1(((1u << (8u - in2)) - 1u) << in2);
		S out;
		//out = S::and_(in1, mask);
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psrlqi256((__v4di) in1.v256, in2);
#else
		out.v256 = (__m256i) ((__v4di) in1.v256 >> in2);
#endif
		return out;
	}
	
    /// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S ror(const S in1,
												const limb_type in2) noexcept {
		S out;
        out.v256 = _mm256_slli_epi64(in1.v256, 64u - in2) ^ _mm256_srli_epi64(in1.v256, in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S rol(const S in1,
												const limb_type in2) noexcept {
		S out;
        out.v256 = _mm256_slli_epi64(in1.v256, in2) ^ _mm256_srli_epi64(in1.v256, 64u-in2);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline S gt_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 > (V) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t gt(const S in1,
	                                                  const S in2) noexcept {
		const auto tmp = (__m256i) ((V) in1.v256 > (V) in2.v256);
		return __builtin_ia32_movmskpd256((__v4df) tmp);
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline S lt_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 < (V) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t lt(const S in1,
	                                                  const S in2) noexcept {
		const auto tmp = (__m256i) ((V) in1.v256 < (V) in2.v256);
		return __builtin_ia32_movmskpd256((__v4df) tmp);
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 == in2 uncompressed
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
	                                             const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((__v4di) in1.v256 == (__v4di) in2.v256);
		return ret;
	}

	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
	                                                   const S in2) noexcept {
#ifndef __clang__
		const __m256i tmp = (__m256i) ((__v4di) in1.v256 == (__v4di) in2.v256);
		return __builtin_ia32_movmskpd256((__v4df) tmp);
#else
		const __m256i tmp = _mm256_cmpeq_epi64(in1.v256, in2.v256);
		return _mm256_movemask_pd((__m256d) tmp);
#endif
	}

	/// \param in
	/// \return
	constexpr static inline S popcnt(const S in) noexcept {
		S ret;
#ifdef USE_AVX512
#ifdef __clang__
		ret.v256 = (__m256i) __builtin_ia32_vpopcntq_256((__v4di) in.v256);
#else
		ret.v256 = (__m256i) __builtin_ia32_vpopcountq_v4di((__v4di) in.v256);
#endif
#else
		ret.v256 = popcount_avx2_64(in.v256);
#endif
		return ret;
	}

	/// checks if all 64bits are equal
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const S in) noexcept {
		const __m256i tmp1 = _mm256_permute4x64_epi64(in.v256, 0);
		const __m256i tmp2 = _mm256_cmpeq_epi64(in.v256, tmp1);
		return _mm256_movemask_pd((__m256d) tmp2) == 0b1111;
	}

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
		S ret;
		ret.v256 = _mm256_permute4x64_epi64(in.v256, 0b00011011);
		return ret;
	}

	/// \tparam in2
	/// \param in1
	/// \return
	template<const uint32_t in2>
	[[nodiscard]] constexpr static inline S permute(const S in1) noexcept {
		S ret;
		ret.v256 = ((__m256i) __builtin_ia32_permdi256((__v4di) (__m256i) (in1.v256), (int) (in2)));
		return ret;
	}

	/// \param in1[in]:
	/// \return
	[[nodiscard]] constexpr static inline uint8_t move(const S in1) noexcept {
#ifndef __clang__
		return __builtin_ia32_movmskpd256((__v4df) in1.v256);
#else
		return _mm256_movemask_pd((__m256d) in1.v256);
#endif
	}

	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
                                                   const cryptanalysislib::_uint32x4_t data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		S ret;
#ifndef __clang__
		ret.v256 = (__m256i) __builtin_ia32_gathersiv4di((__v4di) _mm256_setzero_si256(),
		                                                 (long long const *) (ptr),
		                                                 (__v4si) (__m128i) (data.v128),
		                                                 (__v4di) _mm256_set1_epi64x(-1),
		                                                 (int) (scale));
#else
		ret.v256 = _mm256_i32gather_epi64((long long *) ptr, data.v128, scale);
#endif
		return ret;
	}

	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
                                                   const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		S ret;
#ifndef __clang__
		ret.v256 = (__m256i) __builtin_ia32_gatherdiv4di((__v4di) _mm256_setzero_si256(),
		                                                 (long long const *) (ptr),
		                                                 (__v4di) (__m256i) (data.v256),
		                                                 (__v4di) _mm256_set1_epi64x(-1),
		                                                 (int) (scale));
#else
		ret.v256 = _mm256_i64gather_epi64((long long *) ptr, data.v256, scale);
#endif
		return ret;
	}

	/// \tparam scale
	/// \param ptr
	/// \param data
	/// \return  NOTE: correct?
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
												   const uint8x32_t data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		S ret;
		const auto *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			ret.d[i] = *(limb_type *)(ptr8 + data.d[i]);
		}
		return ret;
	}
	///
	/// \tparam scale
	/// \param ptr
	/// \param offset
	/// \param data
	/// \return
	template<const uint32_t scale = 1>
	constexpr static inline void scatter(const void *ptr,
	                                     const S offset,
	                                     const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		const uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < 4; i++) {
			*(uint64_t *) (ptr8 + offset.v64[i] * scale) = data.v64[i];
		}
	}

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
#ifdef USE_AVX512F
		if constexpr (__unsigned) {
			c.v256 = (__m256i)_mm256_min_epu64(a.v256, b.v256);
		} else {
			c.v256 = (__m256i)_mm256_min_epi64(a.v256, b.v256);
		}
		return c;
#else
#ifdef __clang__
		c.v256 = (__m256i)__builtin_elementwise_min((__v4df)a.v256, (__v4df)b.v256);
#else
        for (uint32_t i = 0; i < LIMBS; i++) {
            c.v64[i] = a.v64[i] < b.v64[i] ? a.v64[i] : b.v64[i];
        }
#endif
#endif

        return c;
    }

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
#ifdef USE_AVX512F
		if constexpr (__unsigned) {
			c.v256 = (__m256i)_mm256_max_epu64(a.v256, b.v256);
		} else {
			c.v256 = (__m256i)_mm256_max_epi64(a.v256, b.v256);
		}
		return c;
#else

#ifdef __clang__
		c.v256 = (__m256i)__builtin_elementwise_max((__v4df)a.v256, (__v4df)b.v256);
#else
        for (uint32_t i = 0; i < LIMBS; i++) {
            c.v64[i] = a.v64[i] < b.v64[i] ? a.v64[i] : b.v64[i];
        }
#endif
#endif

        return c;
    }
};

using uint64x4_t = Xint64x4_t<true>;
using  int64x4_t = Xint64x4_t<false>;


/// loads `element_count` f32 elements from array + index*8
/// \param array base pointer to the data
/// \param index number of __m256 already loaded
/// \param element_count  number of f32 to load
/// \return __m256 register with the first `element_count` f32
/// 		fields loaded, the rest is set to inf.
static inline __m256 avx2_load_f32x8(const float *array,
                                     const uint32_t index,
                                     const uint32_t element_count) noexcept {
	if (element_count == 8) {
		return _mm256_loadu_ps(array + index * 8);
	}

	__m256 inf_mask = _mm256_cvtepi32_ps(_mm256_set_epi32(0x7F800000,
	                                                      (element_count > 6) ? 0 : 0x7F800000,
	                                                      (element_count > 5) ? 0 : 0x7F800000,
	                                                      (element_count > 4) ? 0 : 0x7F800000,
	                                                      (element_count > 3) ? 0 : 0x7F800000,
	                                                      (element_count > 2) ? 0 : 0x7F800000,
	                                                      (element_count > 1) ? 0 : 0x7F800000,
	                                                      (element_count > 0) ? 0 : 0x7F800000));

	__m256i loadstoremask = _mm256_set_epi32(0,
	                                         (element_count > 6) ? 0xffffffff : 0,
	                                         (element_count > 5) ? 0xffffffff : 0,
	                                         (element_count > 4) ? 0xffffffff : 0,
	                                         (element_count > 3) ? 0xffffffff : 0,
	                                         (element_count > 2) ? 0xffffffff : 0,
	                                         (element_count > 1) ? 0xffffffff : 0,
	                                         (element_count > 0) ? 0xffffffff : 0);
	__m256 a = _mm256_maskload_ps(array + index * 8, loadstoremask);
	return _mm256_or_ps(a, inf_mask);
}

/// \param array base pointer to the data
/// \param a data to store
/// \param index number of `__m256` already stored
/// \param element_count numbe of f32 to store from `a`
static inline void avx2_store_f32x8(float *array,
                                    const __m256 a,
                                    const uint32_t index,
                                    const uint32_t element_count) {
	if (element_count == 8) {
		_mm256_storeu_ps(array + index * 8, a);
	} else {
		__m256i loadstoremask = _mm256_set_epi32(0,
		                                         (element_count > 6) ? 0xffffffff : 0,
		                                         (element_count > 5) ? 0xffffffff : 0,
		                                         (element_count > 4) ? 0xffffffff : 0,
		                                         (element_count > 3) ? 0xffffffff : 0,
		                                         (element_count > 2) ? 0xffffffff : 0,
		                                         (element_count > 1) ? 0xffffffff : 0,
		                                         (element_count > 0) ? 0xffffffff : 0);
		_mm256_maskstore_ps(array + index * 8, loadstoremask, a);
	}
}

/// Transpose a bit-matrix using the vpmovmskb instruction.
///
/// See Bitshuffle - https://github.com/kiyo-masui/bitshuffle (MIT)
/// Copyright (c) 2014 Kiyoshi Masui (kiyo@physics.ubc.ca)
void matrix_transpose(uint64_t At,
					  const uint64_t A,
					  const size_t nrows,
                      const size_t ncols) {
  uint8_t in[32] __attribute__((aligned(32)));

	for (size_t i = 0; i < (nrows + 31) / 32; i += 1) {
		for (size_t j = 0; j < (ncols + 7) / 8; j += 1) {
			for (size_t k = 0; k < 32; ++k) {
				in[k] = (((uint8_t **) A)[k + i * 32])[j];
			}

			__m256i vec_x = *((__m256i *) in);
			for (size_t k = 8; k-- > 0;) {
				int32_t hi = _mm256_movemask_epi8(vec_x);
				vec_x = _mm256_slli_epi16(vec_x, 1);
				(((int32_t **) At)[k + j * 8])[i] = hi;
			}
		}
	}
}


/* Transpose bytes within elements, starting partway through input. */
static constexpr int64_t bshuf_trans_byte_elem_remainder(const void* in,
                                                         void* out,
                                        				 const size_t size,
                                        				 const size_t elem_size,
                                                         const size_t start) noexcept {

	size_t ii, jj, kk;
	const char* in_b = (const char*) in;
	char* out_b = (char*) out;

	// CHECK_MULT_EIGHT(start);

	if (size > start) {
		// ii loop separated into 2 loops so the compiler can unroll
		// the inner one.
		for (ii = start; ii + 7 < size; ii += 8) {
			for (jj = 0; jj < elem_size; jj++) {
				for (kk = 0; kk < 8; kk++) {
					out_b[jj * size + ii + kk]
					        = in_b[ii * elem_size + kk * elem_size + jj];
				}
			}
		}
		for (ii = size - size % 8; ii < size; ii ++) {
			for (jj = 0; jj < elem_size; jj++) {
				out_b[jj * size + ii] = in_b[ii * elem_size + jj];
			}
		}
	}
	return size * elem_size;
}

// Transpose bytes within elements for 16 byte elements.
/// input: [0, 1, 2, 3, ..., 31] // each number is 8 bits
/// output:[1, 3, ...,  0, 2, ... ]
/// 	pos:0, 1,      16,17
/// \param out
/// \param in
/// \param size
/// \return
uint64_t bshuf_trans_byte_elem_SSE_16(void* out,
        							  const void* in,
                                      const size_t size) {
    size_t ii;
    const char *in_b = (const char*) in;
    char *out_b = (char*) out;
    __m128i a0, b0, a1, b1;

    for (ii=0; ii + 15 < size; ii += 16) {
        a0 = _mm_loadu_si128((__m128i *) &in_b[2*ii + 0*16]);
        b0 = _mm_loadu_si128((__m128i *) &in_b[2*ii + 1*16]);

        a1 = _mm_unpacklo_epi8(a0, b0);
        b1 = _mm_unpackhi_epi8(a0, b0);

        a0 = _mm_unpacklo_epi8(a1, b1);
        b0 = _mm_unpackhi_epi8(a1, b1);

        a1 = _mm_unpacklo_epi8(a0, b0);
        b1 = _mm_unpackhi_epi8(a0, b0);

        a0 = _mm_unpacklo_epi8(a1, b1);
        b0 = _mm_unpackhi_epi8(a1, b1);

        _mm_storeu_si128((__m128i *) &out_b[0*size + ii], a0);
        _mm_storeu_si128((__m128i *) &out_b[1*size + ii], b0);
    }

    return bshuf_trans_byte_elem_remainder(in, out, size, 2,
            size - size % 16);
}


#endif
