#ifndef CRYPTANALYSISLIB_SIMD_AVX2_H
#define CRYPTANALYSISLIB_SIMD_AVX2_H

#include <xmmintrin.h>
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


#if !defined(USE_AVX512F) && !defined(__AVX512VLINTRIN_H) && !defined(USE_AVX2)
// AVX2 does not provide scatter, so we emulate it in a most obvious way
void _mm256_i32scatter_epi32(int32_t* base, 
                             __m256i vec_indices,
                             __m256i vec_values, size_t /*scale*/) {
    uint32_t indices[8];
    uint32_t values[8];

    _mm256_storeu_si256((__m256i*)indices, vec_indices);
    _mm256_storeu_si256((__m256i*)values, vec_values);

    for (int i=0; i < 8; i++) {
        const size_t idx = indices[i];
        base[idx] = values[i];
    }
}
#endif


/// NOTE: accepts signed and unsigned limbs; every limb is zero-extended
template<typename T>
	requires (sizeof(T) == 1)
constexpr static __m256i u8tom256(const T t[32]) noexcept {
	long long __t[4];
	__t[0] = (long long) (uint8_t) t[0] | (((long long) (uint8_t) t[1]) << 8) | ((long long) (uint8_t) t[2] << 16) | ((long long) (uint8_t) t[3] << 24) | ((long long) (uint8_t) t[4] << 32) | ((long long) (uint8_t) t[5] << 40) | ((long long) (uint8_t) t[6] << 48) | ((long long) (uint8_t) t[7] << 56);
	__t[1] = (long long) (uint8_t) t[8] | (((long long) (uint8_t) t[9]) << 8) | ((long long) (uint8_t) t[10] << 16) | ((long long) (uint8_t) t[11] << 24) | ((long long) (uint8_t) t[12] << 32) | ((long long) (uint8_t) t[13] << 40) | ((long long) (uint8_t) t[14] << 48) | ((long long) (uint8_t) t[15] << 56);
	__t[2] = (long long) (uint8_t) t[16] | (((long long) (uint8_t) t[17]) << 8) | ((long long) (uint8_t) t[18] << 16) | ((long long) (uint8_t) t[19] << 24) | ((long long) (uint8_t) t[20] << 32) | ((long long) (uint8_t) t[21] << 40) | ((long long) (uint8_t) t[22] << 48) | ((long long) (uint8_t) t[23] << 56);
	__t[3] = (long long) (uint8_t) t[24] | (((long long) (uint8_t) t[25]) << 8) | ((long long) (uint8_t) t[26] << 16) | ((long long) (uint8_t) t[27] << 24) | ((long long) (uint8_t) t[28] << 32) | ((long long) (uint8_t) t[29] << 40) | ((long long) (uint8_t) t[30] << 48) | ((long long) (uint8_t) t[31] << 56);
	__m256i tmp = {__t[0], __t[1], __t[2], __t[3]};
	return tmp;
}

/// NOTE: accepts signed and unsigned limbs; every limb is zero-extended
template<typename T>
	requires (sizeof(T) == 1)
constexpr static __m128i u8tom128(const T t[16]) noexcept {
	long long __t[2];
	__t[0] = (long long) (uint8_t) t[0] | (((long long) (uint8_t) t[1]) << 8) | ((long long) (uint8_t) t[2] << 16) | ((long long) (uint8_t) t[3] << 24) | ((long long) (uint8_t) t[4] << 32) | ((long long) (uint8_t) t[5] << 40) | ((long long) (uint8_t) t[6] << 48) | ((long long) (uint8_t) t[7] << 56);
	__t[1] = (long long) (uint8_t) t[8] | (((long long) (uint8_t) t[9]) << 8) | ((long long) (uint8_t) t[10] << 16) | ((long long) (uint8_t) t[11] << 24) | ((long long) (uint8_t) t[12] << 32) | ((long long) (uint8_t) t[13] << 40) | ((long long) (uint8_t) t[14] << 48) | ((long long) (uint8_t) t[15] << 56);
	__m128i tmp = {__t[0], __t[1]};
	return tmp;
}

/// NOTE: accepts signed and unsigned limbs; every limb is zero-extended
template<typename T>
	requires (sizeof(T) == 2)
constexpr static __m256i u16tom256(const T t[16]) noexcept {
	long long __t[4];
	__t[0] = (long long) (uint16_t) t[0] | (((long long) (uint16_t) t[1]) << 16) | ((long long) (uint16_t) t[2] << 32) | ((long long) (uint16_t) t[3] << 48);
	__t[1] = (long long) (uint16_t) t[4] | (((long long) (uint16_t) t[5]) << 16) | ((long long) (uint16_t) t[6] << 32) | ((long long) (uint16_t) t[7] << 48);
	__t[2] = (long long) (uint16_t) t[8] | (((long long) (uint16_t) t[9]) << 16) | ((long long) (uint16_t) t[10] << 32) | ((long long) (uint16_t) t[11] << 48);
	__t[3] = (long long) (uint16_t) t[12] | (((long long) (uint16_t) t[13]) << 16) | ((long long) (uint16_t) t[14] << 32) | ((long long) (uint16_t) t[15] << 48);
	__m256i tmp = {__t[0], __t[1], __t[2], __t[3]};
	return tmp;
}

/// NOTE: accepts signed and unsigned limbs; every limb is zero-extended
template<typename T>
	requires (sizeof(T) == 2)
constexpr static __m128i u16tom128(const T t[16]) noexcept {
	long long __t[2];
	__t[0] = (long long) (uint16_t) t[0] | (((long long) (uint16_t) t[1]) << 16) | ((long long) (uint16_t) t[2] << 32) | ((long long) (uint16_t) t[3] << 48);
	__t[1] = (long long) (uint16_t) t[4] | (((long long) (uint16_t) t[5]) << 16) | ((long long) (uint16_t) t[6] << 32) | ((long long) (uint16_t) t[7] << 48);
	__m128i tmp = {__t[0], __t[1]};
	return tmp;
}

/// NOTE: accepts signed and unsigned limbs; every limb is zero-extended
template<typename T>
	requires (sizeof(T) == 4)
constexpr static __m256i u32tom256(const T t[8]) noexcept {
	long long __t[4];
	__t[0] = (long long) (uint32_t) t[0] | (((long long) (uint32_t) t[1]) << 32);
	__t[1] = (long long) (uint32_t) t[2] | (((long long) (uint32_t) t[3]) << 32);
	__t[2] = (long long) (uint32_t) t[4] | (((long long) (uint32_t) t[5]) << 32);
	__t[3] = (long long) (uint32_t) t[6] | (((long long) (uint32_t) t[7]) << 32);
	__m256i tmp = {__t[0], __t[1], __t[2], __t[3]};
	return tmp;
}

/// NOTE: accepts signed and unsigned limbs; every limb is zero-extended
template<typename T>
	requires (sizeof(T) == 4)
constexpr static __m128i u32tom128(const T t[8]) noexcept {
	long long __t[4];
	__t[0] = (long long) (uint32_t) t[0] | (((long long) (uint32_t) t[1]) << 32);
	__t[1] = (long long) (uint32_t) t[2] | (((long long) (uint32_t) t[3]) << 32);
	__m128i tmp = {__t[0], __t[1]};
	return tmp;
}

/// NOTE: accepts signed and unsigned limbs; every limb is zero-extended
template<typename T>
	requires (sizeof(T) == 8)
constexpr static __m256i u64tom256(const T t[4]) noexcept {
	__m256i tmp = {(long long) (uint64_t) t[0], (long long) (uint64_t) t[1], (long long) (uint64_t) t[2], (long long) (uint64_t) t[3]};
	return tmp;
}

/// NOTE: accepts signed and unsigned limbs; every limb is zero-extended
template<typename T>
	requires (sizeof(T) == 8)
constexpr static __m128i u64tom128(const T t[2]) noexcept {
	__m128i tmp = {(long long) (uint64_t) t[0], (long long) (uint64_t) t[1]};
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

namespace cryptanalysislib::internal {
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
}// namespace cryptanalysislib::internal


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
        
        using V = std::conditional<__unsigned, __v16qu, __v16qi>::type;

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

        /// \param i[in]: 
        /// \return 
		[[nodiscard]] constexpr static inline S set1(const limb_type i) noexcept {
            return S::set(i,i,i,i,i,i,i,i,i,i,i,i,i,i,i,i);
		}

        /// sets the four 32 bit words, `a` is the highest one
        /// \return `_mm_set_epi32(a, b, c, d)`
		[[nodiscard]] constexpr static inline S set(const uint32_t a, const uint32_t b,
		                                            const uint32_t c, const uint32_t d) noexcept {
			S ret;
			ret.v128 = (__m128i)(__v4su){d, c, b, a};
			return ret;
		}

        /// sets the four 32 bit words, `a` is the lowest one
        /// \return `_mm_setr_epi32(a, b, c, d)`
		[[nodiscard]] constexpr static inline S setr(const uint32_t a, const uint32_t b,
		                                             const uint32_t c, const uint32_t d) noexcept {
			S ret;
			ret.v128 = (__m128i)(__v4su){a, b, c, d};
			return ret;
		}

        /// set the `i`-th bit in each limb
        /// \param pos[in]: bit position to set
        constexpr inline void set_bit(const uint32_t pos) noexcept {
            v128 = _mm_set1_epi8(1u << pos); 
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
		    	__m128i *ptr128 = (__m128i *) ptr;
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
		    	const __m128i_u tmp = cryptanalysislib::internal::unaligned_load_wrapper_128(ptr128);
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
			__m128i *ptr128 = (__m128i *) ptr;
			*ptr128 = in.v128;
		}

	    /// \param ptr[in/out]: pointer to 16 unaligned bytes
	    /// \param in[in]: vector element
		constexpr static inline void unaligned_store(void *ptr,
                                                     const S in) noexcept {
			__m128i_u *ptr128 = (__m128i_u *) ptr;
			cryptanalysislib::internal::unaligned_store_wrapper_128(ptr128, in.v128);
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
	    /// \return in1 / in2, lane-wise
	    [[nodiscard]] constexpr static inline S div(const S in1,
	                                                const limb_type in2) noexcept {
            if constexpr (__unsigned) {
                // NOTE: division by the invariant `in2`, see `internal::udiv_invariant`
                using VT = cryptanalysislib::internal::vec_t<limb_type, LIMBS>;
                S out;
                out.v128 = (__m128i) cryptanalysislib::internal::udiv_invariant<limb_type, LIMBS>((VT) in1.v128, in2);
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
	    	S out;
	    	if consteval {
	    		// NOTE: the intrinsics below are not usable in constant expressions
	    		out.v128 = (in2 >= 8) ? (__m128i)(__v16qu){} : (__m128i)((__v16qu)in1.v128 << (uint8_t)in2);
	    		return out;
	    	}
	    	// shift 16-bit lanes and drop the bits that crossed into the next byte
	    	out.v128 = _mm_and_si128(_mm_slli_epi16(in1.v128, in2),
	    	                         _mm_set1_epi8((char)(uint8_t)(0xFFu << in2)));
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 <= 8);
	    	S out;
	    	if consteval {
	    		// NOTE: the intrinsics below are not usable in constant expressions
	    		out.v128 = (in2 >= 8) ? (__m128i)(__v16qu){} : (__m128i)((__v16qu)in1.v128 >> (uint8_t)in2);
	    		return out;
	    	}
	    	// shift 16-bit lanes and drop the bits that crossed into the previous byte
	    	out.v128 = _mm_and_si128(_mm_srli_epi16(in1.v128, in2),
	    	                         _mm_set1_epi8((char)(uint8_t)(0xFFu >> in2)));
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S ror(const S in1,
	                                                 const uint8_t in2) noexcept {
	    	const uint8_t s = in2 % 8u;
	    	if (s == 0) {
	    		return in1;
	    	}
	    	return S::or_(S::srli(in1, s), S::slli(in1, 8u - s));
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: 
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S rol(const S in1,
	                                                 const uint8_t in2) noexcept {
	    	const uint8_t s = in2 % 8u;
	    	if (s == 0) {
	    		return in1;
	    	}
	    	return S::or_(S::slli(in1, s), S::srli(in1, 8u - s));
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
	    /// \return in1 >= in2 uncompressed
	    [[nodiscard]] constexpr static inline S ge_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 >= (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 >= in2 compressed
		[[nodiscard]] constexpr static inline uint32_t ge(const S in1,
		                                                  const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 >= (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 <= in2 uncompressed
	    [[nodiscard]] constexpr static inline S le_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 <= (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 <= in2 compressed
		[[nodiscard]] constexpr static inline uint32_t le(const S in1,
		                                                  const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 <= (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}

		/// \param in1[in]: vector element
		/// \param in2[in]: vector element
		/// \return in1 > in2 uncompressed
		[[nodiscard]] constexpr static inline S eq_(const S in1,
		                                            const S in2) noexcept {
			S ret;
			ret.v128 = (__m128i) ((V) in1.v128 == (V) in2.v128);
			return ret;
		}

		/// \param in1[in]: vector element
		/// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t eq(const S in1,
		                                                  const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 == (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}

		/// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S cmp_(const S in1,
	                                                 const S in2) noexcept {
	    	S ret;
	    	ret.v128  = (__m128i) ((V) in1.v128 < (V) in2.v128);
			ret.v128 ^= (__m128i) ((V) in1.v128 > (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
		                                                   const S in2) noexcept {
			S ret = S::cmp_(in1, in2);
			return __builtin_ia32_pmovmskb128((__v16qi) ret.v128);
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

        using V   = std::conditional<__unsigned, __v8hu, __v8hi>::type;
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

        [[nodiscard]] constexpr inline static size_t size() noexcept {
            return LIMBS;
        }

	    [[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
            return __unsigned;
        }

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

        /// sets the four 32 bit words, `a` is the highest one
        /// \return `_mm_set_epi32(a, b, c, d)`
		[[nodiscard]] constexpr static inline S set(const uint32_t a, const uint32_t b,
		                                            const uint32_t c, const uint32_t d) noexcept {
			S ret;
			ret.v128 = (__m128i)(__v4su){d, c, b, a};
			return ret;
		}

        /// sets the four 32 bit words, `a` is the lowest one
        /// \return `_mm_setr_epi32(a, b, c, d)`
		[[nodiscard]] constexpr static inline S setr(const uint32_t a, const uint32_t b,
		                                             const uint32_t c, const uint32_t d) noexcept {
			S ret;
			ret.v128 = (__m128i)(__v4su){a, b, c, d};
			return ret;
		}

        /// set the `i`-th bit in each limb
        /// \param pos[in]: bit position to set
        constexpr inline void set_bit(const uint32_t pos) noexcept {
            v128 = _mm_set1_epi16(1u << pos); 
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
			__m128i *ptr128 = (__m128i *) ptr;
			S out;
			out.v128 = *ptr128;
			return out;
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
			__m128i_u const *ptr128 = (__m128i_u const *) ptr;
			const __m128i_u tmp = cryptanalysislib::internal::unaligned_load_wrapper_128(ptr128);
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
			__m128i *ptr128 = (__m128i *) ptr;
			*ptr128 = in.v128;
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr, const S in) noexcept {
			__m128i_u *ptr128 = (__m128i_u *) ptr;
			cryptanalysislib::internal::unaligned_store_wrapper_128(ptr128, in.v128);
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
            if constexpr (__unsigned) {
                // NOTE: division by the invariant `in2`, see `internal::udiv_invariant`
                using VT = cryptanalysislib::internal::vec_t<limb_type, LIMBS>;
                S out;
                out.v128 = (__m128i) cryptanalysislib::internal::udiv_invariant<limb_type, LIMBS>((VT) in1.v128, in2);
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
	    	assert(in2 < 16);
	    	S out;
	    	out.v128 = (__m128i) ((__v8hu) in1.v128 << in2);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 < 16);
	    	// logical shift: shift the unsigned lanes
	    	S out;
	    	out.v128 = (__m128i) ((__v8hu) in1.v128 >> in2);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]:
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S ror(const S in1,
	                                                const uint8_t in2) noexcept {
	    	const uint8_t s = in2 % 16u;
#if defined(USE_AVX512VBMI2) && defined(USE_AVX512VL)
	    	S ret;
	    	ret.v128 = _mm_shrdv_epi16(in1.v128, in1.v128, _mm_set1_epi16(s));
	    	return ret;
#else
	    	if (s == 0) {
	    		return in1;
	    	}
	    	return S::or_(S::srli(in1, s), S::slli(in1, 16u - s));
#endif
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]:
	    /// \return in1 <<< in2 uncompressed
	    [[nodiscard]] constexpr static inline S rol(const S in1,
	                                                const uint8_t in2) noexcept {
	    	const uint8_t s = in2 % 16u;
#if defined(USE_AVX512VBMI2) && defined(USE_AVX512VL)
	    	S ret;
	    	ret.v128 = _mm_shldv_epi16(in1.v128, in1.v128, _mm_set1_epi16(s));
	    	return ret;
#else
	    	if (s == 0) {
	    		return in1;
	    	}
	    	return S::or_(S::slli(in1, s), S::srli(in1, 16u - s));
#endif
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
			S tmp;
			tmp.v128 = (__m128i) ((V) in1.v128 > (V) in2.v128);
			return S::move(tmp);
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
			S tmp;
			tmp.v128 = (__m128i) ((V) in1.v128 < (V) in2.v128);
			return S::move(tmp);
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >= in2 uncompressed
	    [[nodiscard]] constexpr static inline S ge_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 >= (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 >= in2 compressed
		[[nodiscard]] constexpr static inline uint32_t ge(const S in1,
		                                                  const S in2) noexcept {
			return S::move(S::ge_(in1, in2));
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 <= in2 uncompressed
	    [[nodiscard]] constexpr static inline S le_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 <= (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 <= in2 compressed
		[[nodiscard]] constexpr static inline uint32_t le(const S in1,
		                                                  const S in2) noexcept {
			return S::move(S::le_(in1, in2));
		}
        
        /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S eq_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 == (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t eq(const S in1,
		                                                  const S in2) noexcept {
			S tmp;
			tmp.v128 = (__m128i) ((V) in1.v128 == (V) in2.v128);
			return S::move(tmp);
		}

		/// \param in1[in]: vector element
		/// \param in2[in]: vector element
		/// \return in1 > in2 uncompressed
		[[nodiscard]] constexpr static inline S cmp_(const S in1,
													 const S in2) noexcept {
			S ret;
			ret.v128  = (__m128i) ((V) in1.v128 < (V) in2.v128);
			ret.v128 ^= (__m128i) ((V) in1.v128 > (V) in2.v128);
			return ret;
		}

		/// \param in1[in]: vector element
		/// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
														   const S in2) noexcept {
			S ret = S::cmp_(in1, in2);
			return __builtin_ia32_pmovmskb128((__v16qi) ret.v128);
		}

	    /// \param in[in]: vector element
		/// \return [popcnt(in[0]), ..., popcnt(in[7])]
	    [[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
	    	// popcount per byte, then add the two bytes of each 16-bit lane
	    	const __v8hu t = (__v8hu) popcount_sse_u8x16(in.v128);
	    	S ret;
	    	ret.v128 = (__m128i) ((t & (uint16_t) 0xFF) + (t >> 8));
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
	    	// the sign bit of a 16-bit lane is the msb of its high byte (odd byte)
	    	return _pext_u32(_mm_movemask_epi8(in.v128), 0b1010101010101010u);
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

	    	const uint8_t *ptr8 = (const uint8_t *) ptr;
	    	for (uint32_t i = 0; i < S::LIMBS; i++) {
	    		// byte offsets, so the address may not be aligned
	    		__builtin_memcpy(&ret.d[i], ptr8 + data.d[i] * scale, sizeof(limb_type));
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
	    	for (uint32_t i = 0; i < S::LIMBS; i++) {
	    		__builtin_memcpy(ptr8 + offset.d[i] * scale, &data.d[i], sizeof(limb_type));
	    	}
	    }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [min(a[0], b[0]), ..., min(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S min(const S a,
	                                                const S b) noexcept {
	    	S c;
#ifdef __clang__
	    	// V has the (un)signed lane type of this vector
	    	c.v128 = (__m128i)__builtin_elementwise_min((V)a.v128, (V)b.v128);
#else
	    	if constexpr (__unsigned) {
	    		c.v128 = _mm_min_epu16(a.v128, b.v128);
	    	} else {
	    		c.v128 = _mm_min_epi16(a.v128, b.v128);
	    	}
#endif
	    	return c;
	    }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [max(a[0], b[0]), ..., max(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S max(const S a,
	                                                const S b) noexcept {
	    	S c;
#ifdef __clang__
	    	// V has the (un)signed lane type of this vector
	    	c.v128 = (__m128i)__builtin_elementwise_max((V)a.v128, (V)b.v128);
#else
	    	if constexpr (__unsigned) {
	    		c.v128 = _mm_max_epu16(a.v128, b.v128);
	    	} else {
	    		c.v128 = _mm_max_epi16(a.v128, b.v128);
	    	}
#endif
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

        using V   = std::conditional<__unsigned, __v4su, __v4si>::type;
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
		    T32 d[4];

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

		/// \return random element
		static inline S random() noexcept {
			S ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = rng();
			}

			return ret;
		}

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

	    /// \param binary[in]:
	    /// \param hex[in]:
	    constexpr inline void print(bool binary = false,
	                                bool hex = false) const;

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

        /// sets the two 64 bit words, `a` is the highest one
        /// \return `_mm_set_epi64x(a, b)`
		[[nodiscard]] constexpr static inline S set(const uint64_t a,
		                                            const uint64_t b) noexcept {
			S ret;
			ret.v128 = (__m128i)(__v2du){b, a};
			return ret;
		}

        /// sets the two 64 bit words, `a` is the lowest one
        /// \return `_mm_set_epi64x(b, a)`
		[[nodiscard]] constexpr static inline S setr(const uint64_t a,
		                                             const uint64_t b) noexcept {
			S ret;
			ret.v128 = (__m128i)(__v2du){a, b};
			return ret;
		}

        /// set the `i`-th bit in each limb
        /// \param pos[in]: bit position to set
        constexpr inline void set_bit(const uint32_t pos) noexcept {
            v128 = _mm_set1_epi32(1u << pos); 
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
			__m128i *ptr128 = (__m128i *) ptr;
			S out;
			out.v128 = *ptr128;
			return out;
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
			__m128i_u const *ptr128 = (__m128i_u const *) ptr;
			const __m128i_u tmp = cryptanalysislib::internal::unaligned_load_wrapper_128(ptr128);
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
			__m128i *ptr128 = (__m128i *) ptr;
			*ptr128 = in.v128;
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(void *ptr, const S in) noexcept {
			__m128i_u *ptr128 = (__m128i_u *) ptr;
			cryptanalysislib::internal::unaligned_store_wrapper_128(ptr128, in.v128);
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
            if constexpr (__unsigned) {
                // NOTE: division by the invariant `in2`, see `internal::udiv_invariant`
                using VT = cryptanalysislib::internal::vec_t<limb_type, LIMBS>;
                S out;
                out.v128 = (__m128i) cryptanalysislib::internal::udiv_invariant<limb_type, LIMBS>((VT) in1.v128, in2);
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
	    	assert(in2 < 32);
	    	S out;
	    	out.v128 = (__m128i) ((__v4su) in1.v128 << in2);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 < 32);
	    	// logical shift: shift the unsigned lanes
	    	S out;
	    	out.v128 = (__m128i) ((__v4su) in1.v128 >> in2);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]:
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S ror(const S in1,
	                                                const uint8_t in2) noexcept {
	    	const uint8_t s = in2 % 32u;
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
	    	S ret;
	    	ret.v128 = _mm_rorv_epi32(in1.v128, _mm_set1_epi32(s));
	    	return ret;
#else
	    	if (s == 0) {
	    		return in1;
	    	}
	    	return S::or_(S::srli(in1, s), S::slli(in1, 32u - s));
#endif
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]:
	    /// \return in1 <<< in2 uncompressed
	    [[nodiscard]] constexpr static inline S rol(const S in1,
	                                                const uint8_t in2) noexcept {
	    	const uint8_t s = in2 % 32u;
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
	    	S ret;
	    	ret.v128 = _mm_rolv_epi32(in1.v128, _mm_set1_epi32(s));
	    	return ret;
#else
	    	if (s == 0) {
	    		return in1;
	    	}
	    	return S::or_(S::slli(in1, s), S::srli(in1, 32u - s));
#endif
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
			S tmp;
			tmp.v128 = (__m128i) ((V) in1.v128 > (V) in2.v128);
			return S::move(tmp);
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
			S tmp;
			tmp.v128 = (__m128i) ((V) in1.v128 < (V) in2.v128);
			return S::move(tmp);
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >= in2 uncompressed
	    [[nodiscard]] constexpr static inline S ge_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 >= (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 >= in2 compressed
		[[nodiscard]] constexpr static inline uint32_t ge(const S in1,
		                                                  const S in2) noexcept {
			return S::move(S::ge_(in1, in2));
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 <= in2 uncompressed
	    [[nodiscard]] constexpr static inline S le_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 <= (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 <= in2 compressed
		[[nodiscard]] constexpr static inline uint32_t le(const S in1,
		                                                  const S in2) noexcept {
			return S::move(S::le_(in1, in2));
		}
        
        /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S eq_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 == (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t eq(const S in1,
		                                                  const S in2) noexcept {
			S tmp;
			tmp.v128 = (__m128i) ((V) in1.v128 == (V) in2.v128);
			return S::move(tmp);
		}

		/// \param in1[in]: vector element
		/// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
														   const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 == (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}

		/// \param in1[in]: vector element
		/// \param in2[in]: vector element
		/// \return in1 > in2 uncompressed
		[[nodiscard]] constexpr static inline S cmp_(const S in1,
													 const S in2) noexcept {
			S ret;
			ret.v128  = (__m128i) ((V) in1.v128 < (V) in2.v128);
			ret.v128 ^= (__m128i) ((V) in1.v128 > (V) in2.v128);
			return ret;
		}

	    /// \param in[in]: vector element
		/// \return [popcnt(in[0]), ..., popcnt(in[7])]
	    [[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
	    	// popcount per byte, then sum the four bytes of each 32-bit lane
	    	__v8hu t = (__v8hu) popcount_sse_u8x16(in.v128);
	    	t = (t & (uint16_t) 0xFF) + (t >> 8);
	    	const __v4su u = (__v4su) t;
	    	S ret;
	    	ret.v128 = (__m128i) ((u & 0xFFFFu) + (u >> 16));
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
	    	return _mm_movemask_ps(_mm_castsi128_ps(in.v128));
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
	    	ret.v128 = _mm_i32gather_epi32((const int *) ptr, data.v128, scale);
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
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
	    	_mm_i32scatter_epi32((void *) ptr, offset.v128, data.v128, scale);
#else
	    	// NOTE: AVX2 has no scatter instruction
	    	uint8_t *ptr8 = (uint8_t *) ptr;
	    	for (uint32_t i = 0; i < S::LIMBS; i++) {
	    		__builtin_memcpy(ptr8 + offset.d[i] * scale, &data.d[i], sizeof(limb_type));
	    	}
#endif
	    }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [min(a[0], b[0]), ..., min(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S min(const S a,
	                                                const S b) noexcept {
	    	S c;
#ifdef __clang__
	    	// V has the (un)signed lane type of this vector
	    	c.v128 = (__m128i)__builtin_elementwise_min((V)a.v128, (V)b.v128);
#else
	    	if constexpr (__unsigned) {
	    		c.v128 = _mm_min_epu32(a.v128, b.v128);
	    	} else {
	    		c.v128 = _mm_min_epi32(a.v128, b.v128);
	    	}
#endif
	    	return c;
	    }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [max(a[0], b[0]), ..., max(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S max(const S a,
	                                                const S b) noexcept {
	    	S c;
#ifdef __clang__
	    	// V has the (un)signed lane type of this vector
	    	c.v128 = (__m128i)__builtin_elementwise_max((V)a.v128, (V)b.v128);
#else
	    	if constexpr (__unsigned) {
	    		c.v128 = _mm_max_epu32(a.v128, b.v128);
	    	} else {
	    		c.v128 = _mm_max_epi32(a.v128, b.v128);
	    	}
#endif
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

        using V   = std::conditional<__unsigned, __v2du, __v2di>::type;
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
            T64 d[2];
			uint8_t  v8[16];
			uint16_t v16[8];
			uint32_t v32[4];
			uint64_t v64[2];
			__m128i v128;
		};

        [[nodiscard]] constexpr inline static size_t size() noexcept {
            return LIMBS;
        }

	    [[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
            return __unsigned;
        }

		/// \return random element
		static inline S random() noexcept {
			S ret;
			for (uint32_t i = 0; i < 2; i++) {
				ret.v64[i] = rng();
			}

			return ret;
		}

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

	    /// \param binary[in]:
	    /// \param hex[in]:
	    constexpr inline void print(bool binary = false,
	                                bool hex = false) const;

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


        /// set the `i`-th bit in each limb
        /// \param pos[in]: bit position to set
        constexpr inline void set_bit(const uint32_t pos) noexcept {
            v128 = _mm_set1_epi64x(1ull << pos);
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
			__m128i *ptr128 = (__m128i *) ptr;
			_Xint64x2_t out;
			out.v128 = *ptr128;
			return out;
		}

		/// \param ptr
		/// \return
		[[nodiscard]] constexpr static inline _Xint64x2_t unaligned_load(const limb_type *ptr) noexcept {
			__m128i_u const *ptr128 = (__m128i_u const *) ptr;
			const __m128i_u tmp = cryptanalysislib::internal::unaligned_load_wrapper_128(ptr128);
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
			__m128i *ptr128 = (__m128i *) ptr;
			*ptr128 = in.v128;
		}

		///
		/// \param ptr
		/// \param in
		constexpr static inline void unaligned_store(limb_type *ptr,
                                                     const S in) noexcept {
			__m128i_u *ptr128 = (__m128i_u *) ptr;
			cryptanalysislib::internal::unaligned_store_wrapper_128(ptr128, in.v128);
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
	    	assert(in2 < 64);
	    	S out;
	    	out.v128 = (__m128i) ((__v2du) in1.v128 << in2);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >> in2
	    [[nodiscard]] constexpr static inline S srli(const S in1,
	                                                 const limb_type in2) noexcept {
	    	assert(in2 < 64);
	    	// logical shift: shift the unsigned lanes
	    	S out;
	    	out.v128 = (__m128i) ((__v2du) in1.v128 >> in2);
	    	return out;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]:
	    /// \return in1 >>> in2 uncompressed
	    [[nodiscard]] constexpr static inline S ror(const S in1,
	                                                const uint8_t in2) noexcept {
	    	const uint8_t s = in2 % 64u;
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
	    	S ret;
	    	ret.v128 = _mm_rorv_epi64(in1.v128, _mm_set1_epi64x(s));
	    	return ret;
#else
	    	if (s == 0) {
	    		return in1;
	    	}
	    	return S::or_(S::srli(in1, s), S::slli(in1, 64u - s));
#endif
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]:
	    /// \return in1 <<< in2 uncompressed
	    [[nodiscard]] constexpr static inline S rol(const S in1,
	                                                const uint8_t in2) noexcept {
	    	const uint8_t s = in2 % 64u;
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
	    	S ret;
	    	ret.v128 = _mm_rolv_epi64(in1.v128, _mm_set1_epi64x(s));
	    	return ret;
#else
	    	if (s == 0) {
	    		return in1;
	    	}
	    	return S::or_(S::slli(in1, s), S::srli(in1, 64u - s));
#endif
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
			S tmp;
			tmp.v128 = (__m128i) ((V) in1.v128 > (V) in2.v128);
			return S::move(tmp);
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
			S tmp;
			tmp.v128 = (__m128i) ((V) in1.v128 < (V) in2.v128);
			return S::move(tmp);
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 >= in2 uncompressed
	    [[nodiscard]] constexpr static inline S ge_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 >= (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 >= in2 compressed
		[[nodiscard]] constexpr static inline uint32_t ge(const S in1,
		                                                  const S in2) noexcept {
			return S::move(S::ge_(in1, in2));
		}

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 <= in2 uncompressed
	    [[nodiscard]] constexpr static inline S le_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 <= (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 <= in2 compressed
		[[nodiscard]] constexpr static inline uint32_t le(const S in1,
		                                                  const S in2) noexcept {
			return S::move(S::le_(in1, in2));
		}
        
        /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
	    /// \return in1 > in2 uncompressed
	    [[nodiscard]] constexpr static inline S eq_(const S in1,
	                                                const S in2) noexcept {
	    	S ret;
	    	ret.v128 = (__m128i) ((V) in1.v128 == (V) in2.v128);
	    	return ret;
	    }

	    /// \param in1[in]: vector element
	    /// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t eq(const S in1,
		                                                  const S in2) noexcept {
			S tmp;
			tmp.v128 = (__m128i) ((V) in1.v128 == (V) in2.v128);
			return S::move(tmp);
		}

		/// \param in1[in]: vector element
		/// \param in2[in]: vector element
		/// \return in1 == in2 compressed
		[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
														  const S in2) noexcept {
			const __m128i tmp = (__m128i) ((V) in1.v128 == (V) in2.v128);
			return __builtin_ia32_pmovmskb128((__v16qi) tmp);
		}

		/// \param in1[in]: vector element
		/// \param in2[in]: vector element
		/// \return in1 > in2 uncompressed
		[[nodiscard]] constexpr static inline S cmp_(const S in1,
		                                            const S in2) noexcept {
			S ret;
			ret.v128  = (__m128i) ((V) in1.v128 < (V) in2.v128);
			ret.v128 ^= (__m128i) ((V) in1.v128 > (V) in2.v128);
			return ret;
		}

	    /// \param in[in]: vector element
		/// \return [popcnt(in[0]), ..., popcnt(in[7])]
	    [[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
	    	// popcount per byte, then sum the eight bytes of each 64-bit lane
	    	S ret;
	    	ret.v128 = _mm_sad_epu8(popcount_sse_u8x16(in.v128), _mm_setzero_si128());
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
	    	return _mm_movemask_pd(_mm_castsi128_pd(in.v128));
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
	    	ret.v128 = _mm_i64gather_epi64((const long long *) ptr, data.v128, scale);
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
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
	    	_mm_i64scatter_epi64((void *) ptr, offset.v128, data.v128, scale);
#else
	    	// NOTE: AVX2 has no scatter instruction
	    	uint8_t *ptr8 = (uint8_t *) ptr;
	    	for (uint32_t i = 0; i < S::LIMBS; i++) {
	    		__builtin_memcpy(ptr8 + offset.d[i] * scale, &data.d[i], sizeof(limb_type));
	    	}
#endif
	    }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [min(a[0], b[0]), ..., min(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S min(const S a,
	                                                const S b) noexcept {
	    	S c;
#ifdef __clang__
	    	// V has the (un)signed lane type of this vector
	    	c.v128 = (__m128i)__builtin_elementwise_min((V)a.v128, (V)b.v128);
#else
	    	for (uint32_t i = 0; i < LIMBS; i++) {
	    		c.d[i] = a.d[i] < b.d[i] ? a.d[i] : b.d[i];
	    	}
#endif
	    	return c;
	    }

	    /// \param a[in]:
	    /// \param b[in]:
	    /// \return [max(a[0], b[0]), ..., max(a[7], b[7])]
	    [[nodiscard]] constexpr static inline S max(const S a,
	                                                const S b) noexcept {
	    	S c;
#ifdef __clang__
	    	// V has the (un)signed lane type of this vector
	    	c.v128 = (__m128i)__builtin_elementwise_max((V)a.v128, (V)b.v128);
#else
	    	for (uint32_t i = 0; i < LIMBS; i++) {
	    		c.d[i] = a.d[i] > b.d[i] ? a.d[i] : b.d[i];
	    	}
#endif
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

    /// set the `i`-th bit in each limb
    /// \param pos[in]: bit position to set
    constexpr inline void set_bit(const uint32_t pos) noexcept {
        v256 = _mm256_set1_epi8(1u << pos); 
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
			__m256i *ptr256 = (__m256i *) ptr;
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
			out.v256 = cryptanalysislib::internal::unaligned_load_wrapper((__m256i_u *) ptr);
			return out;
		}
	}

	/// masked load: lane i is loaded if bit i of `m` is set, otherwise 0.
	/// Masked out lanes are not accessed (no fault), as in the AVX-512 version.
	/// \param ptr
	/// \param m lane mask
	/// \return
	[[nodiscard]] static inline S maskz_unaligned_load(const limb_type *ptr,
	                                                   const Mask<LIMBS> &m) noexcept {
		S out;
#if defined(USE_AVX512BW) && defined(USE_AVX512VL)
		if constexpr (sizeof(limb_type) == 1) {
			out.v256 = _mm256_maskz_loadu_epi8(MaskType<LIMBS>(m), ptr);
		} else {
			out.v256 = _mm256_maskz_loadu_epi16(MaskType<LIMBS>(m), ptr);
		}
#else
		// NOTE: AVX2 has no 8/16 bit masked load (`vpmaskmov` is 32/64 bit) and
		// 	a full load could fault on the masked out lanes
		const uint32_t k = uint32_t(MaskType<LIMBS>(m));
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ((k >> i) & 1u) ? ptr[i] : limb_type(0);
		}
#endif
		return out;
	}

	/// same as `maskz_unaligned_load` (no alignment needed)
	[[nodiscard]] static inline S maskz_aligned_load(const limb_type *ptr,
	                                                 const Mask<LIMBS> &m) noexcept {
		return maskz_unaligned_load(ptr, m);
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
		__m256i *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	/// \param ptr[in/out]: pointer to 32 unaligned bytes
	/// \param in[in]: vector element
	static inline void unaligned_store(limb_type *ptr, 
                                       const S in) noexcept {
		__m256i_u *ptr256 = (__m256i_u *) ptr;
		cryptanalysislib::internal::unaligned_store_wrapper(ptr256, in.v256);
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

    /// \return in1 / in2, lane-wise
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
        if constexpr (__unsigned) {
            // NOTE: division by the invariant `in2`, see `internal::udiv_invariant`
            using VT = cryptanalysislib::internal::vec_t<limb_type, LIMBS>;
            S out;
            out.v256 = (__m256i) cryptanalysislib::internal::udiv_invariant<limb_type, LIMBS>((VT) in1.v256, in2);
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
		S out;
		if consteval {
			// NOTE: the intrinsics below are not usable in constant expressions
			out.v256 = (in2 >= 8) ? (__m256i)(__v32qu){} : (__m256i)((__v32qu)in1.v256 << (uint8_t)in2);
			return out;
		}
		// shift 16-bit lanes and drop the bits that crossed into the next byte
		out.v256 = _mm256_and_si256(_mm256_slli_epi16(in1.v256, in2),
		                            _mm256_set1_epi8((char)(uint8_t)(0xFFu << in2)));
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
		const uint8_t s = in2 % 8u;
		if (s == 0) {
			return in1;
		}
		return S::or_(S::srli(in1, s), S::slli(in1, 8u - s));
	}

	/// \param in1[in]: vector element
	/// \param in2[in]: 
	/// \return in1 >>> in2 uncompressed
	[[nodiscard]] constexpr static inline S rol(const S in1,
	                                             const uint8_t in2) noexcept {
		const uint8_t s = in2 % 8u;
		if (s == 0) {
			return in1;
		}
		return S::or_(S::slli(in1, s), S::srli(in1, 8u - s));
	}

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

	/// \param in1[in]: vector element
	/// \param in2[in]: vector element
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline S ge_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 >= (V) in2.v256);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return in1 > in2 compressed
	[[nodiscard]] constexpr static inline uint32_t ge(const S in1,
	                                                  const S in2) noexcept {
		const __m256i tmp = (__m256i) ((V) in1.v256 >= (V) in2.v256);
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

	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline S le_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 <= (V) in2.v256);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return in1 > in2 compressed
	[[nodiscard]] constexpr static inline uint32_t le(const S in1,
	                                                  const S in2) noexcept {
		const __m256i tmp = (__m256i) ((V) in1.v256 <= (V) in2.v256);
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
	}


	/// \param in1
	/// \param in2
	/// \return in1 == in2 compressed
	[[nodiscard]] constexpr static inline S eq_(const S in1,
												 const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((__v32qs) in1.v256 == (__v32qs) in2.v256);
		return ret;
	}

	/// \param in1[in]:
	/// \param in2[in]:
	/// \return in1 == in2 compressed
	[[nodiscard]] constexpr static inline uint32_t eq(const S in1,
													   const S in2) noexcept {
		const __m256i tmp = (__m256i) ((__v32qi) in1.v256 == (__v32qi) in2.v256);
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
	}

	/// \param in1
	/// \param in2
	/// \return  -1 <
	// 			  0 ==
	// 			  1 >
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
	                                             const S in2) noexcept {
		S ret;
		ret.v256  = (__m256i) ((V) in1.v256 < (V) in2.v256);
		ret.v256 ^= (__m256i) ((V) in1.v256 > (V) in2.v256);
		return ret;
	}

	/// \param in1[in]:
	/// \param in2[in]:
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
	                                                   const S in2) noexcept {
		__m256i tmp = (__m256i) ((V) in1.v256 < (V) in2.v256);
		tmp ^= (__m256i) ((V) in1.v256 > (V) in2.v256);
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
	}

	/// \param in[in]:
	/// \return 
	[[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
		S ret;
		ret.v256 = popcount_avx2_8(in.v256);
		return ret;
	}

	/// all bits below the highest set bit are set, then the zeros are counted
	/// \param in[in]:
	/// \return the number of leading zeros of each 8 bit lane (8 for 0)
	[[nodiscard]] constexpr static inline S lzcnt(const S in) noexcept {
		S t = in;
		t = S::or_(t, S::srli(t, 1));
		t = S::or_(t, S::srli(t, 2));
		t = S::or_(t, S::srli(t, 4));
		return S::popcnt(S::not_(t));
	}

	/// tzcnt(x) = popcnt((x - 1) & ~x)
	/// \param in[in]:
	/// \return the number of trailing zeros of each 8 bit lane (8 for 0)
	[[nodiscard]] constexpr static inline S tzcnt(const S in) noexcept {
		const S t = S::and_(S::sub(in, S::set1(1)), S::not_(in));
		return S::popcnt(t);
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

	/// transposes the 8x8 bit matrix in each 64 bit word, as the AVX-512
	/// version: bit i of output byte j is bit j of input byte (7 - i)
	/// \param input
	/// \return
	[[nodiscard]] constexpr static inline S transpose(const S input) noexcept {
		S ret;
#ifdef USE_GFNI
		const __m256i select = _mm256_set1_epi64x(0x8040201008040201ll);
		ret.v256 = _mm256_gf2p8affine_epi64_epi8(select, input.v256, 0);
#else
		// reverse the bytes of each 64 bit word, then transpose with the
		// delta swaps of Hacker's Delight `transpose8`
		const __m256i rev = _mm256_setr_epi8(7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8,
		                                     7, 6, 5, 4, 3, 2, 1, 0, 15, 14, 13, 12, 11, 10, 9, 8);
		__m256i x = _mm256_shuffle_epi8(input.v256, rev);
		__m256i t;
		t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi64(x, 7)), _mm256_set1_epi64x(0x00AA00AA00AA00AAll));
		x = _mm256_xor_si256(x, _mm256_xor_si256(t, _mm256_slli_epi64(t, 7)));
		t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi64(x, 14)), _mm256_set1_epi64x(0x0000CCCC0000CCCCll));
		x = _mm256_xor_si256(x, _mm256_xor_si256(t, _mm256_slli_epi64(t, 14)));
		t = _mm256_and_si256(_mm256_xor_si256(x, _mm256_srli_epi64(x, 28)), _mm256_set1_epi64x(0x00000000F0F0F0F0ll));
		ret.v256 = _mm256_xor_si256(x, _mm256_xor_si256(t, _mm256_slli_epi64(t, 28)));
#endif
		return ret;
	}

	/// as `_mm256_sign_epi8`
	/// \return per lane: 0 if b == 0, -a if b < 0, a if b > 0
	[[nodiscard]] constexpr static inline S comp_sign(const S a,
	                                                  const S b) noexcept {
		S ret;
		ret.v256 = _mm256_sign_epi8(a.v256, b.v256);
		return ret;
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
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			*(ptr8 + offset.d[i] * scale) = data.d[i];
		}
	}

	/// \param in1[in]:
	/// \return the minimum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_min(const S in1) noexcept {
		// the xor maps the order onto the unsigned order of `vphminposuw`
		constexpr uint8_t flip = __unsigned ? 0u : 0x80u;
		const __m128i f = _mm_set1_epi8(char(flip));
		__m128i t = _mm_min_epu8(_mm_xor_si128(_mm256_castsi256_si128(in1.v256), f),
		                         _mm_xor_si128(_mm256_extracti128_si256(in1.v256, 1), f));
		t = _mm_min_epu8(t, _mm_srli_epi16(t, 8));
		return limb_type(uint8_t(_mm_cvtsi128_si32(_mm_minpos_epu16(t))) ^ flip);
	}

	/// \param in1[in]:
	/// \return the maximum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_max(const S in1) noexcept {
		// the xor maps the reversed order onto the unsigned order of `vphminposuw`
		constexpr uint8_t flip = __unsigned ? 0xFFu : 0x7Fu;
		const __m128i f = _mm_set1_epi8(char(flip));
		__m128i t = _mm_min_epu8(_mm_xor_si128(_mm256_castsi256_si128(in1.v256), f),
		                         _mm_xor_si128(_mm256_extracti128_si256(in1.v256, 1), f));
		t = _mm_min_epu8(t, _mm_srli_epi16(t, 8));
		return limb_type(uint8_t(_mm_cvtsi128_si32(_mm_minpos_epu16(t))) ^ flip);
	}

	/// \param in1[in]:
	/// \return the sum of all lanes (modulo 2**8)
	[[nodiscard]] constexpr static inline limb_type reduce_add(const S in1) noexcept {
		// the sum of the 8 bytes of each 64 bit lane, then of the 4 lanes
		const __m256i t = _mm256_sad_epu8(in1.v256, _mm256_setzero_si256());
		const __m128i u = _mm_add_epi64(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
		return limb_type(uint8_t(_mm_cvtsi128_si32(_mm_add_epi64(u, _mm_unpackhi_epi64(u, u)))));
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

	/// as `_mm512_test_epi8_mask`
	/// \param in1[in]: register to test
	/// \param bit_pos[in]: bit position to test
	/// \return bit i is set, if bit `bit_pos` of `in1[i]` is set
	[[nodiscard]] constexpr static inline Mask<LIMBS> test(const S in1,
	                                                       const uint32_t bit_pos) noexcept {
#if defined(USE_AVX512BW) && defined(USE_AVX512VL)
		return Mask<LIMBS>(MaskType<LIMBS>(_mm256_test_epi8_mask(in1.v256, _mm256_set1_epi8(char(1u << bit_pos)))));
#else
		// shifting the 16 bit lanes by 7 - bit_pos moves bit `bit_pos` of
		// each byte into its msb
		const __m256i t = _mm256_sll_epi16(in1.v256, _mm_cvtsi32_si128(int(7u - bit_pos)));
		return Mask<LIMBS>(MaskType<LIMBS>(uint32_t(_mm256_movemask_epi8(t))));
#endif
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

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are zero, as `_mm512_maskz_compress_epi8`
	/// \param in1[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const Mask<LIMBS> &m) noexcept {
		return compress(in1, S::set1(0), m);
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are taken from `in2`, as `_mm512_mask_compress_epi8`
	/// \param in1[in]:
	/// \param in2[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const S in2,
	                                                 const Mask<LIMBS> &m) noexcept {
		const uint32_t k = uint32_t(MaskType<LIMBS>(m));
		S ret;
#if defined(USE_AVX512VBMI2) && defined(USE_AVX512VL)
		ret.v256 = _mm256_mask_compress_epi8(in2.v256, k, in1.v256);
#elif defined(USE_BMI2)
		// `pext` packs each group of 8 lanes, then `in2` is blended into
		// the lanes >= popcount(k)
		alignas(32) uint8_t buf[LIMBS + 8];
		uint32_t j = 0;
		for (uint32_t g = 0; g < 4; g++) {
			const uint32_t kg = (k >> (8u * g)) & 0xFFu;
			const uint64_t bm = _pdep_u64(kg, 0x0101010101010101ull) * 0xFFu;
			const uint64_t t = _pext_u64(uint64_t(in1.v64[g]), bm);
			__builtin_memcpy(buf + j, &t, 8);
			j += __builtin_popcount(kg);
		}
		const __m256i idx = _mm256_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
		                                     16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31);
		const __m256i lt = _mm256_cmpgt_epi8(_mm256_set1_epi8(char(j)), idx);
		ret.v256 = _mm256_blendv_epi8(in2.v256, _mm256_load_si256((const __m256i *) buf), lt);
#else
		// NOTE: AVX2 has no compress instruction, needs BMI2 for `pext`
		ret = in2;
		uint32_t j = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			if ((k >> i) & 1u) { ret.d[j++] = in1.d[i]; }
		}
#endif
		return ret;
	}

	/// writes the lanes of `in1` selected by `m` contiguously to `ptr`,
	/// as `_mm512_mask_compressstoreu_epi8`
	/// \param ptr[out]:
	/// \param in1[in]:
	/// \param m[in]:
	constexpr static inline void compress(limb_type *ptr,
	                                      const S in1,
	                                      const Mask<LIMBS> &m) noexcept {
#if defined(USE_AVX512VBMI2) && defined(USE_AVX512VL)
		_mm256_mask_compressstoreu_epi8(ptr, uint32_t(MaskType<LIMBS>(m)), in1.v256);
#else
		// AVX2 has no 8 bit masked store: only the selected lanes are written
		const S t = compress(in1, m);
		__builtin_memcpy(ptr, t.d, __builtin_popcount(uint32_t(MaskType<LIMBS>(m))));
#endif
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

    /// set the `i`-th bit in each limb
    /// \param pos[in]: bit position to set
    constexpr inline void set_bit(const uint32_t pos) noexcept {
        v256 = _mm256_set1_epi16(1u << pos); 
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
			__m256i *ptr256 = (__m256i *) ptr;
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
			out.v256 = cryptanalysislib::internal::unaligned_load_wrapper((__m256i_u *) ptr);
			return out;
		}
	}

	/// masked load: lane i is loaded if bit i of `m` is set, otherwise 0.
	/// Masked out lanes are not accessed (no fault), as in the AVX-512 version.
	/// \param ptr
	/// \param m lane mask
	/// \return
	[[nodiscard]] static inline S maskz_unaligned_load(const limb_type *ptr,
	                                                   const Mask<LIMBS> &m) noexcept {
		S out;
#if defined(USE_AVX512BW) && defined(USE_AVX512VL)
		if constexpr (sizeof(limb_type) == 1) {
			out.v256 = _mm256_maskz_loadu_epi8(MaskType<LIMBS>(m), ptr);
		} else {
			out.v256 = _mm256_maskz_loadu_epi16(MaskType<LIMBS>(m), ptr);
		}
#else
		// NOTE: AVX2 has no 8/16 bit masked load (`vpmaskmov` is 32/64 bit) and
		// 	a full load could fault on the masked out lanes
		const uint32_t k = uint32_t(MaskType<LIMBS>(m));
		for (uint32_t i = 0; i < LIMBS; i++) {
			out.d[i] = ((k >> i) & 1u) ? ptr[i] : limb_type(0);
		}
#endif
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
		__m256i *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	/// \param ptr
	/// \param in
	static inline void unaligned_store(limb_type *ptr,
                                       const S in) noexcept {
		__m256i_u *ptr256 = (__m256i_u *) ptr;
		cryptanalysislib::internal::unaligned_store_wrapper(ptr256, in.v256);
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
		out.v256 = _mm256_andnot_si256(in1.v256, in2.v256);
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

    /// \return in1 / in2, lane-wise
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
        if constexpr (__unsigned) {
            // NOTE: division by the invariant `in2`, see `internal::udiv_invariant`
            using VT = cryptanalysislib::internal::vec_t<limb_type, LIMBS>;
            S out;
            out.v256 = (__m256i) cryptanalysislib::internal::udiv_invariant<limb_type, LIMBS>((VT) in1.v256, in2);
            return out;
        }

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
		assert(in2 < 16);
		S out;
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psllwi256((__v16hi) in1.v256, in2);
#else
		out.v256 = (__m256i) ((__v16hu) in1.v256 << in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 < 16);
		S out;
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psrlwi256((__v16hi) in1.v256, in2);
#else
		// logical shift: shift the unsigned lanes
		out.v256 = (__m256i) ((__v16hu) in1.v256 >> in2);
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
	[[nodiscard]] constexpr static inline S ge_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 >= (V) in2.v256);
		return ret;
	}

	/// NOTE: this is a function which cannot be vectorized
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t ge(const S in1,
	                                                  const S in2) noexcept {
		S tmp;
		tmp.v256 = (__m256i) ((V) in1.v256 >= (V) in2.v256);
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
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline S le_(const S in1,
	                                                      const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 <= (V) in2.v256);
		return ret;
	}

	/// NOTE: this is a function which cannot be vectorized
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t le(const S in1,
	                                                  const S in2) noexcept {
		S tmp;
		tmp.v256 = (__m256i) ((V) in1.v256 <= (V) in2.v256);
		return S::move(tmp);
	}

	/// \param in1
	/// \param in2
	/// \return in1 == in2 uncompressed
	[[nodiscard]] constexpr static inline S eq_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 == (V) in2.v256);
		return ret;
	}

	/// NOTE: this is a function which cannot be vectorized
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline int eq(const S in1,
	                                             const S in2) noexcept {
		S tmp;
		tmp.v256 = (__m256i) ((V) in1.v256 == (V) in2.v256);
		uint32_t t = _mm256_movemask_epi8(tmp.v256);
		uint16_t ret = _pext_u32(t, 0b01010101010101010101010101010101);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return  -1 <
	// 			  0 ==
	// 			  1 >
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
												 const S in2) noexcept {
		S ret;
		ret.v256  = (__m256i) ((V) in1.v256 < (V) in2.v256);
		ret.v256 ^= (__m256i) ((V) in1.v256 > (V) in2.v256);
		return ret;
	}

	/// \param in1[in]:
	/// \param in2[in]:
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
													   const S in2) noexcept {
		__m256i tmp = (__m256i) ((V) in1.v256 < (V) in2.v256);
		tmp ^= (__m256i) ((V) in1.v256 > (V) in2.v256);
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
	}

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S popcnt(const S in) noexcept {
		S ret;
		ret.v256 = popcount_avx2_16(in.v256);
		return ret;
	}
	
	/// Source: https://stackoverflow.com/questions/58823140/count-leading-zero-bits-for-each-element-in-avx2-vector-emulate-mm256-lzcnt-ep
	/// Every 16 bit value is exactly representable as a float, so its
	/// exponent is floor(log2(x)) and clz(x) = 15 - floor(log2(x)).
	/// \param in
	/// \return the number of leading zeros of each 16 bit lane (16 for 0)
    [[nodiscard]] constexpr static inline S clz(const S in) noexcept {
		typedef uint32_t v16u32 __attribute__((vector_size(64)));
		typedef int32_t v16i32 __attribute__((vector_size(64)));
		typedef float v16f __attribute__((vector_size(64)));
		const v16u32 x = __builtin_convertvector((__v16hu) in.v256, v16u32);
		const v16f f = __builtin_convertvector(x, v16f);
		const v16i32 e = ((v16i32) f >> 23) - 127;
		const v16i32 r = (x == 0) ? v16i32{} + 16 : 15 - e;
		S ret;
		ret.v256 = (__m256i) __builtin_convertvector(r, __v16hu);
		return ret;
    }
	
	/// ctz(x) = popcnt((x - 1) & ~x)
    /// \param in
	/// \return the number of trailing zeros of each 16 bit lane (16 for 0)
    [[nodiscard]] constexpr static inline S ctz(const S in) noexcept {
		S t;
		t.v256 = (__m256i) (((__v16hu) in.v256 - 1u) & ~((__v16hu) in.v256));
		return S::popcnt(t);
    }

	/// same as `clz`, same name as in the AVX-512 backend
	[[nodiscard]] constexpr static inline S lzcnt(const S in) noexcept { return clz(in); }

	/// same as `ctz`, same name as in the AVX-512 backend
	[[nodiscard]] constexpr static inline S tzcnt(const S in) noexcept { return ctz(in); }

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

	/// same semantics as `vpconflict` (AVX512CD), which has no 16 bit version:
	/// lane i is the bitmask of all lanes j < i with in1[j] == in1[i].
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S conflict(const S in1) noexcept {
		typedef uint16_t VU __attribute__((vector_size(32)));
		const VU x = (VU) in1.v256;
		constexpr VU idx = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
		constexpr VU one = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
		VU r = {};
		for (uint32_t j = 0; j < LIMBS; ++j) {
			const VU eq = (VU) (x == x[j]);
			const VU later = (VU) (idx > (uint16_t) j);
			r |= (eq & later) & (one << j);
		}

		S ret;
		ret.v256 = (__m256i) r;
		return ret;
	}

	/// kmoves the msb into each bit
	[[nodiscard]] constexpr static inline uint32_t move(const S in) noexcept {
		uint32_t t = _mm256_movemask_epi8(in.v256);
		// the sign bit of a 16-bit lane is the msb of its high byte (odd byte)
		uint16_t ret = _pext_u32(t, 0b10101010101010101010101010101010u);
		return ret;
	}

    /// \tparam scale[in]:
    /// \param ptr[in]:
    /// \param data[in]: offsets, in units of `scale` bytes
    /// \return ret[i] = limb at byte offset data[i] * scale of `ptr`
	template<const uint32_t scale = 1>
	[[nodiscard]] constexpr static inline S gather(const void *ptr,
												   const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		S ret;

		const uint8_t *ptr8 = (const uint8_t *) ptr;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			// byte offsets, so the address may not be aligned
			__builtin_memcpy(&ret.d[i], ptr8 + data.d[i] * scale, sizeof(limb_type));
		}
		return ret;
	}

    /// writes data[i] to the byte offset `offset[i] * scale` of `ptr`
    /// \tparam scale[in]:
    /// \param ptr[in]:
    /// \param offset[in]: offsets, in units of `scale` bytes
    /// \param data[in]:
	template<const uint32_t scale = 1>
	constexpr static inline void scatter(const void *ptr,
										 const S offset,
										 const S data) noexcept {
		static_assert(scale == 1 || scale == 2 || scale == 4 || scale == 8);
		uint8_t *ptr8 = (uint8_t *) ptr;
		for (uint32_t i = 0; i < S::LIMBS; i++) {
			__builtin_memcpy(ptr8 + offset.d[i] * scale, &data.d[i], sizeof(limb_type));
		}
	}

	/// \param in1[in]:
	/// \return the minimum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_min(const S in1) noexcept {
		// the xor maps the order onto the unsigned order of `vphminposuw`
		constexpr uint16_t flip = __unsigned ? 0u : 0x8000u;
		const __m128i f = _mm_set1_epi16(short(flip));
		const __m128i t = _mm_min_epu16(_mm_xor_si128(_mm256_castsi256_si128(in1.v256), f),
		                                _mm_xor_si128(_mm256_extracti128_si256(in1.v256, 1), f));
		return limb_type(uint16_t(_mm_cvtsi128_si32(_mm_minpos_epu16(t))) ^ flip);
	}

	/// \param in1[in]:
	/// \return the maximum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_max(const S in1) noexcept {
		// the xor maps the reversed order onto the unsigned order of `vphminposuw`
		constexpr uint16_t flip = __unsigned ? 0xFFFFu : 0x7FFFu;
		const __m128i f = _mm_set1_epi16(short(flip));
		const __m128i t = _mm_min_epu16(_mm_xor_si128(_mm256_castsi256_si128(in1.v256), f),
		                                _mm_xor_si128(_mm256_extracti128_si256(in1.v256, 1), f));
		return limb_type(uint16_t(_mm_cvtsi128_si32(_mm_minpos_epu16(t))) ^ flip);
	}

	/// \param in1[in]:
	/// \return the sum of all lanes (modulo 2**16)
	[[nodiscard]] constexpr static inline limb_type reduce_add(const S in1) noexcept {
		// pairs of 16 bit lanes -> 32 bit sums (signed, the low 16 bits are the same)
		const __m256i t = _mm256_madd_epi16(in1.v256, _mm256_set1_epi16(1));
		__m128i u = _mm_add_epi32(_mm256_castsi256_si128(t), _mm256_extracti128_si256(t, 1));
		u = _mm_add_epi32(u, _mm_shuffle_epi32(u, 0b01001110));
		u = _mm_add_epi32(u, _mm_shuffle_epi32(u, 0b10110001));
		return limb_type(uint16_t(_mm_cvtsi128_si32(u)));
	}

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
#ifdef __clang__
		c.v256 = (__m256i)__builtin_elementwise_min((V)a.v256, (V)b.v256);
#else
		if constexpr (__unsigned) {
			c.v256 = _mm256_min_epu16(a.v256, b.v256);
		} else {
			c.v256 = _mm256_min_epi16(a.v256, b.v256);
		}
#endif
        return c;
    }

	/// as `_mm512_test_epi16_mask`
	/// \param in1[in]: register to test
	/// \param bit_pos[in]: bit position to test
	/// \return bit i is set, if bit `bit_pos` of `in1[i]` is set
	[[nodiscard]] constexpr static inline Mask<LIMBS> test(const S in1,
	                                                       const uint32_t bit_pos) noexcept {
#if defined(USE_AVX512BW) && defined(USE_AVX512VL)
		return Mask<LIMBS>(MaskType<LIMBS>(_mm256_test_epi16_mask(in1.v256, _mm256_set1_epi16(short(1u << bit_pos)))));
#else
		// move bit `bit_pos` into the msb, saturate to bytes:
		// bytes 0-7 and 16-23 hold the lanes 0-7 and 8-15
		const __m256i t = _mm256_sll_epi16(in1.v256, _mm_cvtsi32_si128(int(15u - bit_pos)));
		const uint32_t m = uint32_t(_mm256_movemask_epi8(_mm256_packs_epi16(t, _mm256_setzero_si256())));
		return Mask<LIMBS>(MaskType<LIMBS>((m & 0xFFu) | ((m >> 8u) & 0xFF00u)));
#endif
	}

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
#ifdef __clang__
		c.v256 = (__m256i)__builtin_elementwise_max((V)a.v256, (V)b.v256);
#else
		if constexpr (__unsigned) {
			c.v256 = _mm256_max_epu16(a.v256, b.v256);
		} else {
			c.v256 = _mm256_max_epi16(a.v256, b.v256);
		}
#endif
        return c;
    }

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are zero, as `_mm512_maskz_compress_epi16`
	/// \param in1[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const Mask<LIMBS> &m) noexcept {
		return compress(in1, S::set1(0), m);
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are taken from `in2`, as `_mm512_mask_compress_epi16`
	/// \param in1[in]:
	/// \param in2[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const S in2,
	                                                 const Mask<LIMBS> &m) noexcept {
		const uint32_t k = uint32_t(MaskType<LIMBS>(m)) & 0xFFFFu;
		S ret;
#if defined(USE_AVX512VBMI2) && defined(USE_AVX512VL)
		ret.v256 = _mm256_mask_compress_epi16(in2.v256, k, in1.v256);
#elif defined(USE_BMI2)
		// `pext` packs each group of 4 lanes, then `in2` is blended into
		// the lanes >= popcount(k)
		alignas(32) uint16_t buf[LIMBS + 4];
		uint32_t j = 0;
		for (uint32_t g = 0; g < 4; g++) {
			const uint32_t kg = (k >> (4u * g)) & 0xFu;
			const uint64_t bm = _pdep_u64(kg, 0x0001000100010001ull) * 0xFFFFu;
			const uint64_t t = _pext_u64(uint64_t(in1.v64[g]), bm);
			__builtin_memcpy(buf + j, &t, 8);
			j += __builtin_popcount(kg);
		}
		const __m256i idx = _mm256_setr_epi16(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
		const __m256i lt = _mm256_cmpgt_epi16(_mm256_set1_epi16(short(j)), idx);
		ret.v256 = _mm256_blendv_epi8(in2.v256, _mm256_load_si256((const __m256i *) buf), lt);
#else
		// NOTE: AVX2 has no compress instruction, needs BMI2 for `pext`
		ret = in2;
		uint32_t j = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			if ((k >> i) & 1u) { ret.d[j++] = in1.d[i]; }
		}
#endif
		return ret;
	}

	/// writes the lanes of `in1` selected by `m` contiguously to `ptr`,
	/// as `_mm512_mask_compressstoreu_epi16`
	/// \param ptr[out]:
	/// \param in1[in]:
	/// \param m[in]:
	constexpr static inline void compress(limb_type *ptr,
	                                      const S in1,
	                                      const Mask<LIMBS> &m) noexcept {
#if defined(USE_AVX512VBMI2) && defined(USE_AVX512VL)
		_mm256_mask_compressstoreu_epi16(ptr, uint32_t(MaskType<LIMBS>(m)), in1.v256);
#else
		// AVX2 has no 16 bit masked store: only the selected lanes are written
		const S t = compress(in1, m);
		const uint32_t c = __builtin_popcount(uint32_t(MaskType<LIMBS>(m)) & 0xFFFFu);
		__builtin_memcpy(ptr, t.d, c * sizeof(limb_type));
#endif
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
    
    /// set the `i`-th bit in each limb
    /// \param pos[in]: bit position to set
    constexpr inline void set_bit(const uint32_t pos) noexcept {
        v256 = _mm256_set1_epi32(1u << pos); 
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
			__m256i *ptr256 = (__m256i *) ptr;
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
			out.v256 = cryptanalysislib::internal::unaligned_load_wrapper((__m256i_u *) ptr);
			return out;
		}
	}

	/// masked load: lane i is loaded if bit i of `m` is set, otherwise 0.
	/// Masked out lanes are not accessed (no fault), as in the AVX-512 version.
	/// \param ptr
	/// \param m lane mask
	/// \return
	[[nodiscard]] static inline S maskz_unaligned_load(const limb_type *ptr,
	                                                   const Mask<LIMBS> &m) noexcept {
		// the msb of each 32 bit lane selects it
		const uint32_t k = uint32_t(MaskType<LIMBS>(m));
		const __v8su sel = (__v8su){k << 31u, k << 30u, k << 29u, k << 28u,
		                            k << 27u, k << 26u, k << 25u, k << 24u};
		S out;
		out.v256 = _mm256_maskload_epi32((const int *) ptr, (__m256i) sel);
		return out;
	}

	/// same as `maskz_unaligned_load` (no alignment needed)
	[[nodiscard]] static inline S maskz_aligned_load(const limb_type *ptr,
	                                                 const Mask<LIMBS> &m) noexcept {
		return maskz_unaligned_load(ptr, m);
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
		__m256i *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(limb_type *ptr,
                                                 const S in) noexcept {
		__m256i_u *ptr256 = (__m256i_u *) ptr;
		cryptanalysislib::internal::unaligned_store_wrapper(ptr256, in.v256);
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
		// lower 32 bits of each product, same as `mullo`
		out.v256 = (__m256i) ((V) in1.v256 * (V) in2.v256);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return upper 32 bits of the 64 bit products in1[i]*in2[i]
	[[nodiscard]] constexpr static inline S mulhi(const S in1,
	                                              const S in2) noexcept {
		S out{};
		// products of the even/odd lanes; the odd lanes are moved down first
		const __m256i a_odd = _mm256_srli_epi64(in1.v256, 32);
		const __m256i b_odd = _mm256_srli_epi64(in2.v256, 32);
		__m256i even, odd;
		if constexpr (__unsigned) {
			even = _mm256_mul_epu32(in1.v256, in2.v256);
			odd = _mm256_mul_epu32(a_odd, b_odd);
		} else {
			even = _mm256_mul_epi32(in1.v256, in2.v256);
			odd = _mm256_mul_epi32(a_odd, b_odd);
		}
		// upper halves: even lanes shifted down, odd lanes already in place
		out.v256 = _mm256_blend_epi32(_mm256_srli_epi64(even, 32), odd, 0b10101010);
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
        if constexpr (__unsigned) {
            // NOTE: division by the invariant `in2`, see `internal::udiv_invariant`
            using VT = cryptanalysislib::internal::vec_t<limb_type, LIMBS>;
            S out;
            out.v256 = (__m256i) cryptanalysislib::internal::udiv_invariant<limb_type, LIMBS>((VT) in1.v256, in2);
            return out;
        }

        S out;
        for (uint32_t i = 0; i < LIMBS; i++) {
            out[i] = in1[i] / in2;
        }
        return out;
    }

	/// horizontal add of the 4 unsigned bytes of each 32 bit lane
	/// \param in1
	/// \return sum of the bytes, per 32 bit lane
	[[nodiscard]] constexpr static inline S hadd_epu8(const S in1) noexcept {
		// byte pairs -> 16 bit sums (<= 510, no saturation) -> 32 bit sums
		const __m256i t = _mm256_maddubs_epi16(in1.v256, _mm256_set1_epi8(1));
		S ret;
		ret.v256 = _mm256_madd_epi16(t, _mm256_set1_epi16(1));
		return ret;
	}

	/// horizontal add of the 2 unsigned 16 bit values of each 32 bit lane
	/// \param in1
	/// \return sum, per 32 bit lane
	[[nodiscard]] constexpr static inline S hadd_epu16(const S in1) noexcept {
		const __v8su x = (__v8su) in1.v256;
		S ret;
		ret.v256 = (__m256i) ((x & 0xFFFFu) + (x >> 16u));
		return ret;
	}

	/// number of 4 bit values (nibbles) equal to `in2` in each 32 bit lane
	/// \param in1
	/// \param in2 nibble to count, < 16
	/// \return per 32 bit lane: #{i : nibble i of the lane == in2}, in [0, 8]
	[[nodiscard]] constexpr static inline S histogram_epi4(const S in1,
	                                                       const uint8_t in2) noexcept {
		assert(in2 < 16);
		const __v8su z = (__v8su) in1.v256 ^ (0x11111111u * in2);
		const __v8su nz = (z | (z >> 1u) | (z >> 2u) | (z >> 3u)) & 0x11111111u;
		S t;
		t.v256 = (__m256i) nz;
		t = S::popcnt(t);
		t.v256 = (__m256i) (8u - (__v8su) t.v256);
		return t;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S slli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 < 32);
		S out;
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_pslldi256((__v8si) in1.v256, in2);
#else
		out.v256 = (__m256i) ((__v8su) in1.v256 << in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 < 32);
		S out;
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psrldi256((__v8si) in1.v256, in2);
#else
		// logical shift: shift the unsigned lanes
		out.v256 = (__m256i) ((__v8su) in1.v256 >> in2);
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
	[[nodiscard]] constexpr static inline S ge_(const S in1,
	                                            const S in2) noexcept {
		S ret{};
		ret.v256 = (__m256i) ((V) in1.v256 >= (V) in2.v256);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t ge(const S &in1,
                                                      const S &in2) noexcept {
		const __m256i tmp = (__m256i) ((V) in1.v256 >= (V) in2.v256);
		return __builtin_ia32_movmskps256((__v8sf) tmp);
	}

	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompress
	[[nodiscard]] constexpr static inline S lt_(const S in1,
	                                            const S in2) noexcept {
		S ret{};
		ret.v256 = (__m256i) ((V) in1.v256 < (V) in2.v256);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t lt(const S in1,
	                                                  const S in2) noexcept {
		const __m256i tmp = (__m256i) ((V) in1.v256 < (V) in2.v256);
		return __builtin_ia32_movmskps256((__v8sf) tmp);
	}
	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompress
	[[nodiscard]] constexpr static inline S le_(const S in1,
	                                            const S in2) noexcept {
		S ret{};
		ret.v256 = (__m256i) ((V) in1.v256 <= (V) in2.v256);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t le(const S in1,
	                                                  const S in2) noexcept {
		const __m256i tmp = (__m256i) ((V) in1.v256 <= (V) in2.v256);
		return __builtin_ia32_movmskps256((__v8sf) tmp);
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 == in2 uncompress
	[[nodiscard]] constexpr static inline S eq_(const S in1,
	                                            const S in2) noexcept {
		S ret{};
		ret.v256 = (__m256i) ((V) in1.v256 == (V) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline int eq(const S in1,
	                                             const S in2) noexcept {
		const __m256i tmp = (__m256i) ((V) in1.v256 == (V) in2.v256);
#ifndef __clang__
		return __builtin_ia32_movmskps256((__v8sf) tmp);
#else
		return _mm256_movemask_ps((__m256) tmp);
#endif
	}

	/// \param in1
	/// \param in2
	/// \return  -1 <
	// 			  0 ==
	// 			  1 >
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
												 const S in2) noexcept {
		S ret;
		ret.v256  = (__m256i) ((V) in1.v256 < (V) in2.v256);
		ret.v256 ^= (__m256i) ((V) in1.v256 > (V) in2.v256);
		return ret;
	}

	/// \param in1[in]:
	/// \param in2[in]:
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
													   const S in2) noexcept {
		__m256i tmp = (__m256i) ((V) in1.v256 < (V) in2.v256);
		tmp ^= (__m256i) ((V) in1.v256 > (V) in2.v256);
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
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

	/// ctz(x) = popcnt((x - 1) & ~x)
	/// \param in
	/// \return the number of trailing zeros of each 32 bit lane (32 for 0)
	[[nodiscard]] constexpr static inline S ctz(const S in) noexcept {
		const S t = S::and_(S::sub(in, S::set1(1)), S::not_(in));
		return S::popcnt(t);
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

	/// same semantics as `vpconflictd` (AVX512CD):
	/// lane i is the bitmask of all lanes j < i with in1[j] == in1[i].
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S conflict(const S in1) noexcept {
		S ret;
#if defined(USE_AVX512CD) && defined(USE_AVX512VL)
		ret.v256 = _mm256_conflict_epi32(in1.v256);
#else
		// compare all lanes with the broadcasted lane j, keep the lanes > j
		const __m256i idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
		__m256i r = _mm256_setzero_si256();
		for (uint32_t j = 0; j < LIMBS - 1u; ++j) {
			const __m256i jj = _mm256_set1_epi32(int(j));
			const __m256i b = _mm256_permutevar8x32_epi32(in1.v256, jj);
			const __m256i t = _mm256_and_si256(_mm256_cmpeq_epi32(in1.v256, b),
			                                   _mm256_cmpgt_epi32(idx, jj));
			r = _mm256_or_si256(r, _mm256_and_si256(t, _mm256_set1_epi32(int(1u << j))));
		}
		ret.v256 = r;
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
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
		ret.v256 = _mm256_maskz_compress_epi32((__mmask8) mask, _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
#elif defined(USE_BMI2)
		uint64_t expanded_mask = __builtin_ia32_pdep_di(mask, 0x0101010101010101);
		expanded_mask *= 0xFFU;
		const uint64_t identity_indices = 0x0706050403020100;
		uint64_t wanted_indices = __builtin_ia32_pext_di(identity_indices, expanded_mask);
		// the 8 indices are the lower 8 bytes, zero extended to 32 bits
		const __m128i bytevec = __extension__(__m128i)(__v2di){(long long int) wanted_indices, 0};
		ret.v256 = _mm256_cvtepu8_epi32(bytevec);
#else
		// NOTE: needs BMI2 for `pext`
		uint32_t k = 0;
		for (uint32_t i = 0; i < 8; i++) {
			if ((mask >> i) & 1u) {
				ret.v32[k++] = i;
			}
		}
#endif
		return ret;
	}

	/// zero extends the lower 8 bytes of `in` to 32 bits
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S cvtepu8(const cryptanalysislib::_uint8x16_t in) noexcept {
		S ret{};
		ret.v256 = _mm256_cvtepu8_epi32(in.v128);
		return ret;
	}

	/// \param in1[in]:
	/// \return the minimum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_min(const S in1) noexcept {
		const __m128i lo = _mm256_castsi256_si128(in1.v256);
		const __m128i hi = _mm256_extracti128_si256(in1.v256, 1);
		__m128i t;
		if constexpr (__unsigned) {
			t = _mm_min_epu32(lo, hi);
			t = _mm_min_epu32(t, _mm_shuffle_epi32(t, 0b01001110));
			t = _mm_min_epu32(t, _mm_shuffle_epi32(t, 0b10110001));
		} else {
			t = _mm_min_epi32(lo, hi);
			t = _mm_min_epi32(t, _mm_shuffle_epi32(t, 0b01001110));
			t = _mm_min_epi32(t, _mm_shuffle_epi32(t, 0b10110001));
		}
		return limb_type(_mm_cvtsi128_si32(t));
	}

	/// \param in1[in]:
	/// \return the maximum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_max(const S in1) noexcept {
		const __m128i lo = _mm256_castsi256_si128(in1.v256);
		const __m128i hi = _mm256_extracti128_si256(in1.v256, 1);
		__m128i t;
		if constexpr (__unsigned) {
			t = _mm_max_epu32(lo, hi);
			t = _mm_max_epu32(t, _mm_shuffle_epi32(t, 0b01001110));
			t = _mm_max_epu32(t, _mm_shuffle_epi32(t, 0b10110001));
		} else {
			t = _mm_max_epi32(lo, hi);
			t = _mm_max_epi32(t, _mm_shuffle_epi32(t, 0b01001110));
			t = _mm_max_epi32(t, _mm_shuffle_epi32(t, 0b10110001));
		}
		return limb_type(_mm_cvtsi128_si32(t));
	}

	/// \param in1[in]:
	/// \return the sum of all lanes (modulo 2**32)
	[[nodiscard]] constexpr static inline limb_type reduce_add(const S in1) noexcept {
		__m128i t = _mm_add_epi32(_mm256_castsi256_si128(in1.v256), _mm256_extracti128_si256(in1.v256, 1));
		t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0b01001110));
		t = _mm_add_epi32(t, _mm_shuffle_epi32(t, 0b10110001));
		return limb_type(_mm_cvtsi128_si32(t));
	}

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S min(const S a,
                                                const S b) noexcept {
        S c;
        if constexpr (__unsigned) {
			c.v256 = _mm256_min_epu32(a.v256, b.v256);
		} else {
			c.v256 = _mm256_min_epi32(a.v256, b.v256);
		}
        return c;
    }

	/// as `_mm512_test_epi32_mask`
	/// \param in1[in]: register to test
	/// \param bit_pos[in]: bit position to test
	/// \return bit i is set, if bit `bit_pos` of `in1[i]` is set
	[[nodiscard]] constexpr static inline Mask<LIMBS> test(const S in1,
	                                                       const uint32_t bit_pos) noexcept {
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
		return Mask<LIMBS>(MaskType<LIMBS>(_mm256_test_epi32_mask(in1.v256, _mm256_set1_epi32(int(1u << bit_pos)))));
#else
		// move bit `bit_pos` into the msb
		const __m256i t = _mm256_sll_epi32(in1.v256, _mm_cvtsi32_si128(int(31u - bit_pos)));
		return Mask<LIMBS>(MaskType<LIMBS>(uint32_t(_mm256_movemask_ps(_mm256_castsi256_ps(t)))));
#endif
	}

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline S max(const S a,
                                                const S b) noexcept {
        S c;
        if constexpr (__unsigned) {
			c.v256 = _mm256_max_epu32(a.v256, b.v256);
		} else {
			c.v256 = _mm256_max_epi32(a.v256, b.v256);
		}
        return c;
    }

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are zero, as `_mm512_maskz_compress_epi32`
	/// \param in1[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const Mask<LIMBS> &m) noexcept {
		return compress(in1, S::set1(0), m);
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are taken from `in2`, as `_mm512_mask_compress_epi32`
	/// \param in1[in]:
	/// \param in2[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const S in2,
	                                                 const Mask<LIMBS> &m) noexcept {
		const uint32_t k = uint32_t(MaskType<LIMBS>(m)) & 0xFFu;
		S ret;
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
		ret.v256 = _mm256_mask_compress_epi32(in2.v256, k, in1.v256);
#elif defined(USE_BMI2)
		// `pext` packs the lane indices, then `in2` is blended into the
		// lanes >= popcount(k)
		const uint64_t bm = _pdep_u64(k, 0x0101010101010101ull) * 0xFFu;
		const uint64_t sel = _pext_u64(0x0706050403020100ull, bm);
		const __m256i idx = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) sel));
		const __m256i lt = _mm256_cmpgt_epi32(_mm256_set1_epi32(__builtin_popcount(k)),
		                                      _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
		ret.v256 = _mm256_blendv_epi8(in2.v256, _mm256_permutevar8x32_epi32(in1.v256, idx), lt);
#else
		// NOTE: AVX2 has no compress instruction, needs BMI2 for `pext`
		ret = in2;
		uint32_t j = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			if ((k >> i) & 1u) { ret.d[j++] = in1.d[i]; }
		}
#endif
		return ret;
	}

	/// writes the lanes of `in1` selected by `m` contiguously to `ptr`,
	/// as `_mm512_mask_compressstoreu_epi32`
	/// \param ptr[out]:
	/// \param in1[in]:
	/// \param m[in]:
	constexpr static inline void compress(limb_type *ptr,
	                                      const S in1,
	                                      const Mask<LIMBS> &m) noexcept {
		const uint32_t k = uint32_t(MaskType<LIMBS>(m)) & 0xFFu;
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
		_mm256_mask_compressstoreu_epi32(ptr, k, in1.v256);
#else
		// only the first popcount(k) lanes are written
		const S t = compress(in1, m);
		const __m256i lt = _mm256_cmpgt_epi32(_mm256_set1_epi32(__builtin_popcount(k)),
		                                      _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
		_mm256_maskstore_epi32((int *) ptr, lt, t.v256);
#endif
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

    /// set the `i`-th bit in each limb
    /// \param pos[in]: bit position to set
    constexpr inline void set_bit(const uint32_t pos) noexcept {
        v256 = _mm256_set1_epi64x(1ull << pos);
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
			__m256i *ptr256 = (__m256i *) ptr;
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
			out.v256 = cryptanalysislib::internal::unaligned_load_wrapper((const __m256i_u *) ptr);
			return out;
		}
	}

	/// masked load: lane i is loaded if bit i of `m` is set, otherwise 0.
	/// Masked out lanes are not accessed (no fault), as in the AVX-512 version.
	/// \param ptr
	/// \param m lane mask
	/// \return
	[[nodiscard]] static inline S maskz_unaligned_load(const limb_type *ptr,
	                                                   const Mask<LIMBS> &m) noexcept {
		// the msb of each 64 bit lane selects it
		const uint64_t k = uint64_t(MaskType<LIMBS>(m));
		const __v4du sel = (__v4du){k << 63u, k << 62u, k << 61u, k << 60u};
		S out;
		out.v256 = _mm256_maskload_epi64((const long long *) ptr, (__m256i) sel);
		return out;
	}

	/// same as `maskz_unaligned_load` (no alignment needed)
	[[nodiscard]] static inline S maskz_aligned_load(const limb_type *ptr,
	                                                 const Mask<LIMBS> &m) noexcept {
		return maskz_unaligned_load(ptr, m);
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
		__m256i *ptr256 = (__m256i *) ptr;
		*ptr256 = in.v256;
	}

	/// \param ptr
	/// \param in
	static inline void unaligned_store(limb_type *ptr,
                                       const S in) noexcept {
		__m256i_u *ptr256 = (__m256i_u *) ptr;
		cryptanalysislib::internal::unaligned_store_wrapper(ptr256, in.v256);
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
        // NOTE: exact, per lane. Was `mulhrs_epi16(in1, 32768 / in2)`, an
        // 	approximation (and for 64 bit lanes even on 16 bit lanes).
        S out;
        for (uint32_t i = 0; i < LIMBS; i++) {
            out[i] = in1[i] / in2;
        }
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

	/// horizontal add of the 8 unsigned bytes of each 64 bit lane
	/// \param in1
	/// \return sum of the bytes, per 64 bit lane
	[[nodiscard]] constexpr static inline S hadd_epu8(const S in1) noexcept {
		S ret;
		ret.v256 = _mm256_sad_epu8(in1.v256, _mm256_setzero_si256());
		return ret;
	}

	/// horizontal add of the 4 unsigned 16 bit values of each 64 bit lane
	/// \param in1
	/// \return sum, per 64 bit lane
	[[nodiscard]] constexpr static inline S hadd_epu16(const S in1) noexcept {
		const __v4du x = (__v4du) in1.v256;
		constexpr uint64_t m = 0x0000FFFF0000FFFFull;
		const __v4du t = (x & m) + ((x >> 16u) & m);
		S ret;
		ret.v256 = (__m256i) ((t & 0xFFFFFFFFull) + (t >> 32u));
		return ret;
	}

	/// horizontal add of the 2 unsigned 32 bit values of each 64 bit lane
	/// \param in1
	/// \return sum, per 64 bit lane
	[[nodiscard]] constexpr static inline S hadd_epu32(const S in1) noexcept {
		const __v4du x = (__v4du) in1.v256;
		S ret;
		ret.v256 = (__m256i) ((x & 0xFFFFFFFFull) + (x >> 32u));
		return ret;
	}

	/// number of 4 bit values (nibbles) equal to `in2` in each 64 bit lane
	/// \param in1
	/// \param in2 nibble to count, < 16
	/// \return per 64 bit lane: #{i : nibble i of the lane == in2}, in [0, 16]
	[[nodiscard]] constexpr static inline S histogram_epi4(const S in1,
	                                                       const uint8_t in2) noexcept {
		assert(in2 < 16);
		const __v4du z = (__v4du) in1.v256 ^ (0x1111111111111111ull * in2);
		const __v4du nz = (z | (z >> 1u) | (z >> 2u) | (z >> 3u)) & 0x1111111111111111ull;
		S t;
		t.v256 = (__m256i) nz;
		t = S::popcnt(t);
		t.v256 = (__m256i) (16u - (__v4du) t.v256);
		return t;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S slli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 < 64);
		S out;
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psllqi256((__v4di) in1.v256, in2);
#else
		out.v256 = (__m256i) ((__v4du) in1.v256 << in2);
#endif
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S srli(const S in1,
	                                             const limb_type in2) noexcept {
		assert(in2 < 64);
		S out;
#ifndef __clang__
		out.v256 = (__m256i) __builtin_ia32_psrlqi256((__v4di) in1.v256, in2);
#else
		// logical shift: shift the unsigned lanes
		out.v256 = (__m256i) ((__v4du) in1.v256 >> in2);
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

	/// \param in1
	/// \param in2
	/// \return in1 > in2 uncompressed
	[[nodiscard]] constexpr static inline S ge_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 >= (V) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t ge(const S in1,
	                                                  const S in2) noexcept {
		const auto tmp = (__m256i) ((V) in1.v256 >= (V) in2.v256);
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
	/// \return in1 <= in2 uncompressed
	[[nodiscard]] constexpr static inline S le_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((V) in1.v256 <= (V) in2.v256);
		return ret;
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 <= in2 compressed
	[[nodiscard]] constexpr static inline uint32_t le(const S in1,
	                                                  const S in2) noexcept {
		const auto tmp = (__m256i) ((V) in1.v256 <= (V) in2.v256);
		return __builtin_ia32_movmskpd256((__v4df) tmp);
	}

	///
	/// \param in1
	/// \param in2
	/// \return in1 == in2 uncompressed
	[[nodiscard]] constexpr static inline S eq_(const S in1,
	                                            const S in2) noexcept {
		S ret;
		ret.v256 = (__m256i) ((__v4di) in1.v256 == (__v4di) in2.v256);
		return ret;
	}

	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline uint32_t eq(const S in1,
	                                                  const S in2) noexcept {
#ifndef __clang__
		const __m256i tmp = (__m256i) ((__v4di) in1.v256 == (__v4di) in2.v256);
		return __builtin_ia32_movmskpd256((__v4df) tmp);
#else
		const __m256i tmp = _mm256_cmpeq_epi64(in1.v256, in2.v256);
		return _mm256_movemask_pd((__m256d) tmp);
#endif
	}

	/// \param in1
	/// \param in2
	/// \return  -1 <
	// 			  0 ==
	// 			  1 >
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
												 const S in2) noexcept {
		S ret;
		ret.v256  = (__m256i) ((V) in1.v256 < (V) in2.v256);
		ret.v256 ^= (__m256i) ((V) in1.v256 > (V) in2.v256);
		return ret;
	}

	/// \param in1[in]:
	/// \param in2[in]:
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
													   const S in2) noexcept {
		__m256i tmp = (__m256i) ((V) in1.v256 < (V) in2.v256);
		tmp ^= (__m256i) ((V) in1.v256 > (V) in2.v256);
		return __builtin_ia32_pmovmskb256((__v32qi) tmp);
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

	/// without AVX512CD: all bits below the highest set bit are set, then the
	/// zeros are counted
	/// \param in
	/// \return the number of leading zeros of each 64 bit lane (64 for 0)
	[[nodiscard]] constexpr static inline S lzcnt(const S in) noexcept {
#if defined(USE_AVX512CD) && defined(USE_AVX512VL)
		S ret;
		ret.v256 = _mm256_lzcnt_epi64(in.v256);
		return ret;
#else
		S t = in;
		t = S::or_(t, S::srli(t, 1));
		t = S::or_(t, S::srli(t, 2));
		t = S::or_(t, S::srli(t, 4));
		t = S::or_(t, S::srli(t, 8));
		t = S::or_(t, S::srli(t, 16));
		t = S::or_(t, S::srli(t, 32));
		return S::popcnt(S::not_(t));
#endif
	}

	/// tzcnt(x) = popcnt((x - 1) & ~x)
	/// \param in
	/// \return the number of trailing zeros of each 64 bit lane (64 for 0)
	[[nodiscard]] constexpr static inline S tzcnt(const S in) noexcept {
		const S t = S::and_(S::sub(in, S::set1(1)), S::not_(in));
		return S::popcnt(t);
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

	/// runtime version of `permute<in2>`: ret[i] = in1[(in2 >> 2i) & 3]
	/// \param in1
	/// \param in2 8 bit, 2 bits per lane
	/// \return
	[[nodiscard]] constexpr static inline S permute(const S in1,
	                                                const uint32_t in2) noexcept {
		// 32 bit lane indices {2*sel_i, 2*sel_i + 1} with sel_i = (in2 >> 2i) & 3
		const __m256i shift = _mm256_setr_epi32(0, 0, 2, 2, 4, 4, 6, 6);
		const __m256i sel = _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(int(in2)), shift),
		                                     _mm256_set1_epi32(3));
		const __m256i idx = _mm256_add_epi32(_mm256_slli_epi32(sel, 1),
		                                     _mm256_setr_epi32(0, 1, 0, 1, 0, 1, 0, 1));
		S ret;
		ret.v256 = _mm256_permutevar8x32_epi32(in1.v256, idx);
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

	/// same semantics as `vpconflictq` (AVX512CD):
	/// lane i is the bitmask of all lanes j < i with in1[j] == in1[i].
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S conflict(const S in1) noexcept {
		S ret;
#if defined(USE_AVX512CD) && defined(USE_AVX512VL)
		ret.v256 = _mm256_conflict_epi64(in1.v256);
#else
		// compare all lanes with the broadcasted lane j, keep the lanes > j
		const __m256i idx = _mm256_setr_epi64x(0, 1, 2, 3);
		__m256i r = _mm256_setzero_si256();
		for (uint32_t j = 0; j < LIMBS - 1u; ++j) {
			const __m256i b = _mm256_permutevar8x32_epi32(in1.v256, _mm256_set1_epi64x(((long long)(2 * j + 1) << 32) | (2 * j)));
			const __m256i t = _mm256_and_si256(_mm256_cmpeq_epi64(in1.v256, b),
			                                   _mm256_cmpgt_epi64(idx, _mm256_set1_epi64x(j)));
			r = _mm256_or_si256(r, _mm256_and_si256(t, _mm256_set1_epi64x(1ll << j)));
		}
		ret.v256 = r;
#endif
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
		// zero extend the first 4 bytes to 32 bit offsets
		const __m128i off = _mm_cvtepu8_epi32(_mm256_castsi256_si128(data.v256));
		S ret;
		ret.v256 = _mm256_i32gather_epi64((const long long *) ptr, off, scale);
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

	/// \param in1[in]:
	/// \return the minimum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_min(const S in1) noexcept {
		const __m128i lo = _mm256_castsi256_si128(in1.v256);
		const __m128i hi = _mm256_extracti128_si256(in1.v256, 1);
		__m128i t;
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
		if constexpr (__unsigned) {
			t = _mm_min_epu64(lo, hi);
			t = _mm_min_epu64(t, _mm_unpackhi_epi64(t, t));
		} else {
			t = _mm_min_epi64(lo, hi);
			t = _mm_min_epi64(t, _mm_unpackhi_epi64(t, t));
		}
#else
		// AVX2 has no 64 bit min: signed compare + blend, the xor maps the
		// unsigned order onto the signed one
		const __m128i f = _mm_set1_epi64x(__unsigned ? (long long) (1ull << 63u) : 0ll);
		const __m128i a = _mm_xor_si128(lo, f), b = _mm_xor_si128(hi, f);
		t = _mm_blendv_epi8(a, b, _mm_cmpgt_epi64(a, b));
		const __m128i u = _mm_unpackhi_epi64(t, t);
		t = _mm_xor_si128(_mm_blendv_epi8(t, u, _mm_cmpgt_epi64(t, u)), f);
#endif
		return limb_type(_mm_cvtsi128_si64(t));
	}

	/// \param in1[in]:
	/// \return the maximum of all lanes
	[[nodiscard]] constexpr static inline limb_type reduce_max(const S in1) noexcept {
		const __m128i lo = _mm256_castsi256_si128(in1.v256);
		const __m128i hi = _mm256_extracti128_si256(in1.v256, 1);
		__m128i t;
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
		if constexpr (__unsigned) {
			t = _mm_max_epu64(lo, hi);
			t = _mm_max_epu64(t, _mm_unpackhi_epi64(t, t));
		} else {
			t = _mm_max_epi64(lo, hi);
			t = _mm_max_epi64(t, _mm_unpackhi_epi64(t, t));
		}
#else
		// AVX2 has no 64 bit max: signed compare + blend, the xor maps the
		// unsigned order onto the signed one
		const __m128i f = _mm_set1_epi64x(__unsigned ? (long long) (1ull << 63u) : 0ll);
		const __m128i a = _mm_xor_si128(lo, f), b = _mm_xor_si128(hi, f);
		t = _mm_blendv_epi8(a, b, _mm_cmpgt_epi64(b, a));
		const __m128i u = _mm_unpackhi_epi64(t, t);
		t = _mm_xor_si128(_mm_blendv_epi8(t, u, _mm_cmpgt_epi64(u, t)), f);
#endif
		return limb_type(_mm_cvtsi128_si64(t));
	}

	/// \param in1[in]:
	/// \return the sum of all lanes (modulo 2**64)
	[[nodiscard]] constexpr static inline limb_type reduce_add(const S in1) noexcept {
		__m128i t = _mm_add_epi64(_mm256_castsi256_si128(in1.v256), _mm256_extracti128_si256(in1.v256, 1));
		t = _mm_add_epi64(t, _mm_unpackhi_epi64(t, t));
		return limb_type(_mm_cvtsi128_si64(t));
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
		// compare as (un)signed 64-bit integers, not as doubles
		c.v256 = (__m256i)__builtin_elementwise_min((V)a.v256, (V)b.v256);
#else
        for (uint32_t i = 0; i < LIMBS; i++) {
            c.v64[i] = a.v64[i] < b.v64[i] ? a.v64[i] : b.v64[i];
        }
#endif
#endif

        return c;
    }

	/// as `_mm512_test_epi64_mask`
	/// \param in1[in]: register to test
	/// \param bit_pos[in]: bit position to test
	/// \return bit i is set, if bit `bit_pos` of `in1[i]` is set
	[[nodiscard]] constexpr static inline Mask<LIMBS> test(const S in1,
	                                                       const uint32_t bit_pos) noexcept {
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
		return Mask<LIMBS>(MaskType<LIMBS>(_mm256_test_epi64_mask(in1.v256, _mm256_set1_epi64x((long long) (uint64_t(1u) << bit_pos)))));
#else
		// move bit `bit_pos` into the msb
		const __m256i t = _mm256_sll_epi64(in1.v256, _mm_cvtsi32_si128(int(63u - bit_pos)));
		return Mask<LIMBS>(MaskType<LIMBS>(uint32_t(_mm256_movemask_pd(_mm256_castsi256_pd(t)))));
#endif
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
		// compare as (un)signed 64-bit integers, not as doubles
		c.v256 = (__m256i)__builtin_elementwise_max((V)a.v256, (V)b.v256);
#else
        for (uint32_t i = 0; i < LIMBS; i++) {
            c.v64[i] = a.v64[i] > b.v64[i] ? a.v64[i] : b.v64[i];
        }
#endif
#endif

        return c;
    }

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are zero, as `_mm512_maskz_compress_epi64`
	/// \param in1[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const Mask<LIMBS> &m) noexcept {
		return compress(in1, S::set1(0), m);
	}

	/// packs the lanes of `in1` selected by `m` to the front (in order),
	/// the other lanes are taken from `in2`, as `_mm512_mask_compress_epi64`
	/// \param in1[in]:
	/// \param in2[in]:
	/// \param m[in]:
	/// \return
	[[nodiscard]] constexpr static inline S compress(const S in1,
	                                                 const S in2,
	                                                 const Mask<LIMBS> &m) noexcept {
		const uint32_t k = uint32_t(MaskType<LIMBS>(m)) & 0xFu;
		S ret;
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
		ret.v256 = _mm256_mask_compress_epi64(in2.v256, k, in1.v256);
#elif defined(USE_BMI2)
		// `pext` packs the 32 bit lane indices (2 per 64 bit lane), then
		// `in2` is blended into the lanes >= popcount(k)
		const uint64_t k2 = _pdep_u64(k, 0x55u) * 3u;
		const uint64_t bm = _pdep_u64(k2, 0x0101010101010101ull) * 0xFFu;
		const uint64_t sel = _pext_u64(0x0706050403020100ull, bm);
		const __m256i idx = _mm256_cvtepu8_epi32(_mm_cvtsi64_si128((long long) sel));
		const __m256i lt = _mm256_cmpgt_epi64(_mm256_set1_epi64x(__builtin_popcount(k)),
		                                      _mm256_setr_epi64x(0, 1, 2, 3));
		ret.v256 = _mm256_blendv_epi8(in2.v256, _mm256_permutevar8x32_epi32(in1.v256, idx), lt);
#else
		// NOTE: AVX2 has no compress instruction, needs BMI2 for `pext`
		ret = in2;
		uint32_t j = 0;
		for (uint32_t i = 0; i < LIMBS; i++) {
			if ((k >> i) & 1u) { ret.d[j++] = in1.d[i]; }
		}
#endif
		return ret;
	}

	/// writes the lanes of `in1` selected by `m` contiguously to `ptr`,
	/// as `_mm512_mask_compressstoreu_epi64`
	/// \param ptr[out]:
	/// \param in1[in]:
	/// \param m[in]:
	constexpr static inline void compress(limb_type *ptr,
	                                      const S in1,
	                                      const Mask<LIMBS> &m) noexcept {
		const uint32_t k = uint32_t(MaskType<LIMBS>(m)) & 0xFu;
#if defined(USE_AVX512F) && defined(USE_AVX512VL)
		_mm256_mask_compressstoreu_epi64(ptr, k, in1.v256);
#else
		// only the first popcount(k) lanes are written
		const S t = compress(in1, m);
		const __m256i lt = _mm256_cmpgt_epi64(_mm256_set1_epi64x(__builtin_popcount(k)),
		                                      _mm256_setr_epi64x(0, 1, 2, 3));
		_mm256_maskstore_epi64((long long *) ptr, lt, t.v256);
#endif
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

// bit/byte matrix transposes (bitshuffle)
#include "simd/avx2_transpose.h"

#endif
