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

/// NOTE: apparently this function is not in the normal list of intrinsics?
static inline 
__m512i _mm512_setr_epi8 (char __e63, char __e62, char __e61, char __e60, char __e59,            
                          char __e58, char __e57, char __e56, char __e55, char __e54, char __e53,
                          char __e52, char __e51, char __e50, char __e49, char __e48, char __e47,
                          char __e46, char __e45, char __e44, char __e43, char __e42, char __e41,
                          char __e40, char __e39, char __e38, char __e37, char __e36, char __e35,
                          char __e34, char __e33, char __e32, char __e31, char __e30, char __e29,
                          char __e28, char __e27, char __e26, char __e25, char __e24, char __e23,
                          char __e22, char __e21, char __e20, char __e19, char __e18, char __e17,
                          char __e16, char __e15, char __e14, char __e13, char __e12, char __e11,
                          char __e10, char __e9,  char __e8,  char __e7,  char __e6,  char __e5,
                          char __e4,  char __e3,  char __e2,  char __e1,  char __e0) {
  return __extension__ (__m512i)(__v64qi) {
    __e63, __e62, __e61, __e60, __e59,            
    __e58, __e57, __e56, __e55, __e54, __e53,
    __e52, __e51, __e50, __e49, __e48, __e47,
    __e46, __e45, __e44, __e43, __e42, __e41,
    __e40, __e39, __e38, __e37, __e36, __e35,
    __e34, __e33, __e32, __e31, __e30, __e29,
    __e28, __e27, __e26, __e25, __e24, __e23,
    __e22, __e21, __e20, __e19, __e18, __e17,
    __e16, __e15, __e14, __e13, __e12, __e11,
    __e10, __e9,  __e8,  __e7,  __e6,  __e5,
    __e4,  __e3,  __e2,  __e1,  __e0};
}

/// translates 64 bytes into a singe __m512i register as constexpr
[[nodiscard]] constexpr static __m512i u8tom512(const uint8_t t[64]) noexcept {
	long long __t[8];
	__t[0] = (long long)t[ 0] | (((long long)t[ 1]) << 8) | ((long long)t[ 2] << 16) | ((long long)t[ 3] << 24) | ((long long)t[ 4] << 32) | ((long long)t[ 5] << 40) | ((long long)t[ 6] << 48) | ((long long)t[ 7] << 56);
	__t[1] = (long long)t[ 8] | (((long long)t[ 9]) << 8) | ((long long)t[10] << 16) | ((long long)t[11] << 24) | ((long long)t[12] << 32) | ((long long)t[13] << 40) | ((long long)t[14] << 48) | ((long long)t[15] << 56);
	__t[2] = (long long)t[16] | (((long long)t[17]) << 8) | ((long long)t[18] << 16) | ((long long)t[19] << 24) | ((long long)t[20] << 32) | ((long long)t[21] << 40) | ((long long)t[22] << 48) | ((long long)t[23] << 56);
	__t[3] = (long long)t[24] | (((long long)t[25]) << 8) | ((long long)t[26] << 16) | ((long long)t[27] << 24) | ((long long)t[28] << 32) | ((long long)t[29] << 40) | ((long long)t[30] << 48) | ((long long)t[31] << 56);

	__t[4] = (long long)t[32] | (((long long)t[33]) << 8) | ((long long)t[34] << 16) | ((long long)t[35] << 24) | ((long long)t[36] << 32) | ((long long)t[37] << 40) | ((long long)t[38] << 48) | ((long long)t[39] << 56);
	__t[5] = (long long)t[40] | (((long long)t[41]) << 8) | ((long long)t[42] << 16) | ((long long)t[43] << 24) | ((long long)t[44] << 32) | ((long long)t[45] << 40) | ((long long)t[46] << 48) | ((long long)t[47] << 56);
	__t[6] = (long long)t[48] | (((long long)t[49]) << 8) | ((long long)t[50] << 16) | ((long long)t[51] << 24) | ((long long)t[52] << 32) | ((long long)t[53] << 40) | ((long long)t[54] << 48) | ((long long)t[55] << 56);
	__t[7] = (long long)t[56] | (((long long)t[57]) << 8) | ((long long)t[58] << 16) | ((long long)t[59] << 24) | ((long long)t[60] << 32) | ((long long)t[61] << 40) | ((long long)t[62] << 48) | ((long long)t[63] << 56);
	__m512i tmp = {__t[0],__t[1],__t[2],__t[3],
	               __t[4],__t[5],__t[6],__t[7]};
	return tmp;
}

/// translates 32 uint16_t into a singe __m512i register as constexpr
constexpr static __m512i u16tom512(const uint16_t t[32]) noexcept {
	long long __t[8];
	__t[0] = (long long)t[ 0] | (((long long)t[ 1]) << 16) | ((long long)t[ 2] << 32) | ((long long)t[ 3] << 48);
	__t[1] = (long long)t[ 4] | (((long long)t[ 5]) << 16) | ((long long)t[ 6] << 32) | ((long long)t[ 7] << 48);
	__t[2] = (long long)t[ 8] | (((long long)t[ 9]) << 16) | ((long long)t[10] << 32) | ((long long)t[11] << 48);
	__t[3] = (long long)t[12] | (((long long)t[13]) << 16) | ((long long)t[14] << 32) | ((long long)t[15] << 48);
	__t[4] = (long long)t[16] | (((long long)t[17]) << 16) | ((long long)t[18] << 32) | ((long long)t[19] << 48);
	__t[5] = (long long)t[20] | (((long long)t[21]) << 16) | ((long long)t[22] << 32) | ((long long)t[23] << 48);
	__t[6] = (long long)t[24] | (((long long)t[25]) << 16) | ((long long)t[26] << 32) | ((long long)t[27] << 48);
	__t[7] = (long long)t[28] | (((long long)t[29]) << 16) | ((long long)t[30] << 32) | ((long long)t[31] << 48);
	__m512i tmp = {__t[0],__t[1],__t[2],__t[3],
				   __t[4],__t[5],__t[6],__t[7]};
	return tmp;
}

/// translates 16 uint32_t into a singe __m512i register as constexpr
constexpr static __m512i u32tom512(const uint32_t t[16]) noexcept {
	long long __t[8];
	__t[0] = (long long)t[ 0] | (((long long)t[ 1]) << 32);
	__t[1] = (long long)t[ 2] | (((long long)t[ 3]) << 32);
	__t[2] = (long long)t[ 4] | (((long long)t[ 5]) << 32);
	__t[3] = (long long)t[ 6] | (((long long)t[ 7]) << 32);
	__t[4] = (long long)t[ 8] | (((long long)t[ 9]) << 32);
	__t[5] = (long long)t[10] | (((long long)t[11]) << 32);
	__t[6] = (long long)t[12] | (((long long)t[13]) << 32);
	__t[7] = (long long)t[14] | (((long long)t[15]) << 32);
	__m512i tmp = {__t[0],__t[1],__t[2],__t[3],
				   __t[4],__t[5],__t[6],__t[7]};
	return tmp;
}

/// translates 8 uint64_t into a singe __m512i register as constexpr
constexpr static __m512i u64tom512(const uint64_t t[8]) noexcept {
	__m512i tmp = {(long long)t[0],(long long)t[1],(long long)t[2],(long long)t[3],
				   (long long)t[4],(long long)t[5],(long long)t[6],(long long)t[7]};
	return tmp;
}


template<const bool __unsigned=true>
struct Xint8x64_t {
	constexpr static uint32_t LIMBS = 64;
	using limb_type = std::conditional<__unsigned, uint8_t, int8_t>::type;
	using S = Xint8x64_t;
	using simd_type = S;

    using V   = std::conditional<__unsigned, __v32qu, __v32qi>::type;
    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		// compatibility with TxN_t
		T8   d[64];
		T8  v8[64];
		T16 v16[32];
		T32 v32[16];
		T64 v64[8];
		cryptanalysislib::_uint8x16_t v128[4];
		__m256i v256[2];
		__m512i v512;
	};
	
    [[nodiscard]] constexpr inline static size_t size() noexcept { 
        return LIMBS; 
    }

	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept {
        return __unsigned; 
    }

	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		assert(i < LIMBS);
		return d[i];
	}
	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline S random() noexcept {
		S ret;
		for (size_t i = 0; i < 8; ++i) {
			ret.v64[i] = rng();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	/// \return
	[[nodiscard]] constexpr static inline S set(
	        const limb_type __q63, const limb_type __q62, const limb_type __q61, const limb_type __q60,
	        const limb_type __q59, const limb_type __q58, const limb_type __q57, const limb_type __q56,
	        const limb_type __q55, const limb_type __q54, const limb_type __q53, const limb_type __q52,
	        const limb_type __q51, const limb_type __q50, const limb_type __q49, const limb_type __q48,
	        const limb_type __q47, const limb_type __q46, const limb_type __q45, const limb_type __q44,
	        const limb_type __q43, const limb_type __q42, const limb_type __q41, const limb_type __q40,
	        const limb_type __q39, const limb_type __q38, const limb_type __q37, const limb_type __q36,
	        const limb_type __q35, const limb_type __q34, const limb_type __q33, const limb_type __q32,
	        const limb_type __q31, const limb_type __q30, const limb_type __q29, const limb_type __q28,
	        const limb_type __q27, const limb_type __q26, const limb_type __q25, const limb_type __q24,
	        const limb_type __q23, const limb_type __q22, const limb_type __q21, const limb_type __q20,
	        const limb_type __q19, const limb_type __q18, const limb_type __q17, const limb_type __q16,
	        const limb_type __q15, const limb_type __q14, const limb_type __q13, const limb_type __q12,
	        const limb_type __q11, const limb_type __q10, const limb_type __q09, const limb_type __q08,
	        const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
	        const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		S out;
		out.v512 = __extension__(__m512i)(__v64qi){
		        (char)__q00, (char)__q01, (char)__q02, (char)__q03, (char)__q04, (char)__q05, (char)__q06, (char)__q07,
		        (char)__q08, (char)__q09, (char)__q10, (char)__q11, (char)__q12, (char)__q13, (char)__q14, (char)__q15,
		        (char)__q16, (char)__q17, (char)__q18, (char)__q19, (char)__q20, (char)__q21, (char)__q22, (char)__q23,
		        (char)__q24, (char)__q25, (char)__q26, (char)__q27, (char)__q28, (char)__q29, (char)__q30, (char)__q31,
		        (char)__q32, (char)__q33, (char)__q34, (char)__q35, (char)__q36, (char)__q37, (char)__q38, (char)__q39,
		        (char)__q40, (char)__q41, (char)__q42, (char)__q43, (char)__q44, (char)__q45, (char)__q46, (char)__q47,
		        (char)__q48, (char)__q49, (char)__q50, (char)__q51, (char)__q52, (char)__q53, (char)__q54, (char)__q55,
		        (char)__q56, (char)__q57, (char)__q58, (char)__q59, (char)__q60, (char)__q61, (char)__q62, (char)__q63};
		return out;
	}

	/// \return
	[[nodiscard]] constexpr static inline S setr(
	        const limb_type __q63, const limb_type __q62, const limb_type __q61, const limb_type __q60,
	        const limb_type __q59, const limb_type __q58, const limb_type __q57, const limb_type __q56,
	        const limb_type __q55, const limb_type __q54, const limb_type __q53, const limb_type __q52,
	        const limb_type __q51, const limb_type __q50, const limb_type __q49, const limb_type __q48,
	        const limb_type __q47, const limb_type __q46, const limb_type __q45, const limb_type __q44,
	        const limb_type __q43, const limb_type __q42, const limb_type __q41, const limb_type __q40,
	        const limb_type __q39, const limb_type __q38, const limb_type __q37, const limb_type __q36,
	        const limb_type __q35, const limb_type __q34, const limb_type __q33, const limb_type __q32,
	        const limb_type __q31, const limb_type __q30, const limb_type __q29, const limb_type __q28,
	        const limb_type __q27, const limb_type __q26, const limb_type __q25, const limb_type __q24,
	        const limb_type __q23, const limb_type __q22, const limb_type __q21, const limb_type __q20,
	        const limb_type __q19, const limb_type __q18, const limb_type __q17, const limb_type __q16,
	        const limb_type __q15, const limb_type __q14, const limb_type __q13, const limb_type __q12,
	        const limb_type __q11, const limb_type __q10, const limb_type __q09, const limb_type __q08,
	        const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
	        const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		S out;
		out.v512 = __extension__(__m512i)(__v64qi){
		        (char)__q63,
		        (char)__q62,
		        (char)__q61,
		        (char)__q60,
		        (char)__q59,
		        (char)__q58,
		        (char)__q57,
		        (char)__q56,
		        (char)__q55,
		        (char)__q54,
		        (char)__q53,
		        (char)__q52,
		        (char)__q51,
		        (char)__q50,
		        (char)__q49,
		        (char)__q48,
		        (char)__q47,
		        (char)__q46,
		        (char)__q45,
		        (char)__q44,
		        (char)__q43,
		        (char)__q42,
		        (char)__q41,
		        (char)__q40,
		        (char)__q39,
		        (char)__q38,
		        (char)__q37,
		        (char)__q36,
		        (char)__q35,
		        (char)__q34,
		        (char)__q33,
		        (char)__q32,
		        (char)__q31,
		        (char)__q30,
		        (char)__q29,
		        (char)__q28,
		        (char)__q27,
		        (char)__q26,
		        (char)__q25,
		        (char)__q24,
		        (char)__q23,
		        (char)__q22,
		        (char)__q21,
		        (char)__q20,
		        (char)__q19,
		        (char)__q18,
		        (char)__q17,
		        (char)__q16,
		        (char)__q15,
		        (char)__q14,
		        (char)__q13,
		        (char)__q12,
		        (char)__q11,
		        (char)__q10,
		        (char)__q09,
		        (char)__q08,
		        (char)__q07,
		        (char)__q06,
		        (char)__q05,
		        (char)__q04,
		        (char)__q03,
		        (char)__q02,
		        (char)__q01,
		        (char)__q00,
		};
		return out;
	}

	/// \param a[in]: single integer
	/// \return vector register with: [a, ..., a]
	[[nodiscard]] constexpr static inline S set1(const limb_type a) noexcept {
		S out;
		out = set(a, a, a, a, a, a, a, a,
		          a, a, a, a, a, a, a, a,
		          a, a, a, a, a, a, a, a,
		          a, a, a, a, a, a, a, a,
		          a, a, a, a, a, a, a, a,
		          a, a, a, a, a, a, a, a,
		          a, a, a, a, a, a, a, a,
		          a, a, a, a, a, a, a, a);
		return out;
	}


	/// \tparam aligned[in]: if true a aligned instruction will be emmited.
	/// \param ptr[in]: pointer to memory
	/// \return __m512_load{u}_si512(ptr);
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
			const __m512i tmp = u8tom512(ptr);
			S out;
			out.v512 = tmp;
			return out;
		} else {
			assert(is_aligned(ptr, 64));
			const __m512i tmp = *(__m512i *) ptr;
			S out;
			out.v512 = tmp;
			return out;
		}
	}

	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline S unaligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m512i tmp = u8tom512(ptr);
			S out;
			out.v512 = tmp;
			return out;
		} else {
			const __m512i tmp = (__m512i) (*(__v64qi_u *) ptr);
			S out;
			out.v512 = tmp;
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

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(limb_type *ptr,
                                               const S in) noexcept {
		auto *ptr512 = (__m512i *) ptr;
		*ptr512 = in.v512;
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(limb_type *ptr,
                                                 const S in) noexcept {
		auto *ptr512 = (__m512i_u *) ptr;
		*(__m512i_u *) ptr512 = (__m512i_u) in.v512;
	}


	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S xor_(const S in1,
	                                             const S in2) noexcept {
		S out;
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S and_(const S in1,
	                                                      const S in2) noexcept {
		S out;
		out.v512 = (__m512i) ((__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S or_(const S in1,
	                                            const S in2) noexcept {
		S out;
		out.v512 = (__m512i) ((__v16su) in1.v512 | (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S andnot(const S in1,
	                                               const S in2) noexcept {
		S out;
		out.v512 = (__m512i) (~(__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S not_(const S in1) noexcept {
		S out;
		const S minus_one = set1(-1);
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) minus_one.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S add(const S in1,
	                                                     const S in2) noexcept {
		S out;
		out.v512 = (__m512i) ((__v64qu) in1.v512 + (__v64qu) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S sub(const S in1,
	                                                     const S in2) noexcept {
		S out;
		out.v512 = (__m512i) ((__v64qu) in1.v512 - (__v64qu) in2.v512);
		return out;
	}

	/// 8 bit mul lo
	/// \param in1 first input
	/// \param in2
	/// \return in1*in2
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                       const S in2) noexcept {
		S out;
		out.v512  = (__m512i) ((__v64qi) in1.v512 * (__v64qi) in2.v512);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S mullo(const S in1,
	                                                       const uint8_t in2) noexcept {
		const S rs = S::set1(in2);
		return S::mullo(in1, rs);
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
														  const uint8_t in2) noexcept {
		assert(in2 <= 8);
		S out;
		const S mask = S::set1(~((1u << in2) - 1u));
		// out.v512 = _mm512_slli_epi16(in1.v512, in2);
		// out.v512 =  (__m512i)__builtin_ia32_psllwi512((__v32hi)in1.v512, (int)in2);
		out.v512 = (__m512i)((__v64qu)in1.v512 << (int)in2);
		out = S::and_(out, mask);
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
		//const S mask = S::set1((1u << ((8u - in2) & 7u)) - 1u);
		// out.v512 = _mm512_srli_epi16(in1.v512, in2);
		out.v512 = (__m512i)(((__v64qu)in1.v512) >> (int)in2);
		// out = S::and_(out, mask);
		return out;
	}



	///
	/// source:https://github.com/WojciechMula/toys/blob/master/avx512/avx512bw-rotate-by1.cpp
	/// needs `avx512bw`
	/// \param input
	/// \return
	[[nodiscard]] constexpr static inline S ror1(const S input) noexcept {
		S ret;
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


	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S ror(const S in1,
												const limb_type in2) noexcept {
		S out;
        __m512i mask = _mm512_set1_epi8(-1u << in2);
        out.v512 = (mask & _mm512_slli_epi16(in1.v512, 8u - in2)) ^ _mm512_srli_epi16(in1.v512, in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S rol(const S in1,
												const limb_type in2) noexcept {
		S out;
        __m512i mask = _mm512_set1_epi8((1u << in2) -1u);
        out.v512 = _mm512_slli_epi16(in1.v512, in2) ^ (_mm512_srli_epi16(in1.v512, 8u-in2) & mask);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S gt_(const S in1,
												const S in2) noexcept {
		S ret;
		ret.v512 = (__m512i) ((__v64qu) in1.v512 > (__v64qu) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64_t gt(const S in1,
													  const S in2) noexcept {
		__m512i v512 = (__m512i) ((__v64qu) in1.v512 > (__v64qu) in2.v512);
		return (uint64_t)(__mmask64) __builtin_ia32_cvtb2mask512 ((__v64qi)v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S ge_(const S in1,
												const S in2) noexcept {
		S ret;
		ret.v512 = (__m512i) ((__v64qu) in1.v512 >= (__v64qu) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64_t ge(const S in1,
													  const S in2) noexcept {
		__m512i v512 = (__m512i) ((__v64qu) in1.v512 >= (__v64qu) in2.v512);
		return (uint64_t)(__mmask64) __builtin_ia32_cvtb2mask512 ((__v64qi)v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S lt_(const S in1,
												const S in2) noexcept {
		S ret;
		ret.v512 = (__m512i) ((__v64qu) in1.v512 < (__v64qu) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64_t lt(const S in1,
	                                                  const S in2) noexcept {
		__m512i v512 = (__m512i) ((__v64qu) in1.v512 < (__v64qu) in2.v512);
		return (uint64_t)(__mmask64) __builtin_ia32_cvtb2mask512 ((__v64qi)v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S le_(const S in1,
												const S in2) noexcept {
		S ret;
		ret.v512 = (__m512i) ((__v64qu) in1.v512 <= (__v64qu) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64_t le(const S in1,
	                                                  const S in2) noexcept {
		__m512i v512 = (__m512i) ((__v64qu) in1.v512 <= (__v64qu) in2.v512);
		return (uint64_t)(__mmask64) __builtin_ia32_cvtb2mask512 ((__v64qi)v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S eq_(const S in1,
												const S in2) noexcept {
		S ret;
		ret.v512 = (__m512i) ((__v64qi) in1.v512 == (__v64qi) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64_t eq(const S in1,
												      const S in2) noexcept {
		__m512i v512 = (__m512i) ((__v64qi) in1.v512 == (__v64qi) in2.v512);
		return (uint64_t)(__mmask64) __builtin_ia32_cvtb2mask512 ((__v64qi)v512);
	}

	/// \param in1
	/// \param in2
	/// \return {-1, 0, 1}
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
												 const S in2) noexcept {
		S ret;
		ret.v512  = (__m512i) ((__v64qi) in1.v512 < (__v64qi) in2.v512);
		ret.v512 ^= (__m512i) ((__v64qi) in1.v512 > (__v64qi) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint64_t cmp(const S in1,
												       const S in2) noexcept {
        S ret = S::cmp_(in1, in2);
		return (uint64_t)(__mmask64) __builtin_ia32_cvtb2mask512 ((__v64qi)ret.v512);
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S popcnt(const S in1) noexcept {
		S ret;
#ifdef USE_AVX512BITALG
#ifdef __clang__
		ret.v512 = (__m512i) __builtin_ia32_vpopcntb_512((__v64qi) in1.v512);
#else
  		ret.v512 = (__m512i) __builtin_ia32_vpopcountb_v64qi ((__v64qi)in1.v512);
#endif
#else
		for (uint32_t i = 0; i < S::LIMBS; ++i) {
			ret.v8[i] = cryptanalysislib::popcount::popcount(in1.v8[i]);
		}
#endif
		return ret;
	}

	/// Source:http://0x80.pl/notesen/2023-01-31-avx512-bsf.html
	/// needs`AVX512VPOPCNTDQ`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S lzcnt(const S in1) noexcept {
		S ret;
		constexpr S one = S::set1(1);
		ret = S::sub(in1, one);
		ret = S::and_(ret, S::not_(in1));
		ret = S::popcnt(ret);
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
	[[nodiscard]] constexpr static inline S transpose(const S input) noexcept {
		S ret;
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

	/// needs `avx512bw`
	/// source:  http://0x80.pl/notesen/2021-02-02-all-bytes-in-reg-are-equal.html
	/// \param input
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const S input) noexcept {
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

	/// https://github.com/WojciechMula/toys/blob/master/avx512-galois/reverse.cpp
	/// \param input = [0, 1, ..., 62, 63]
	/// \return [63, 62, ..., 1, 0]
	[[nodiscard]] constexpr static inline S reverse(const S in) noexcept {
		S ret;
#ifdef USE_EVEX512
		const long long __d = bit_shuffle_const(7, 6, 5, 4, 3, 2, 1, 0);
		const __m512i select = __extension__(__m512i)(__v8di){__d, __d, __d, __d, __d, __d, __d, __d};
		ret.v512 = ((__m512i) __builtin_ia32_vgf2p8affineqb_v64qi((__v64qi) (__m512i) (select),
		                                                          (__v64qi) (__m512i) (in.v512), (char) (0x00)));
#else
		for (uint32_t i = 0; i < S::LIMBS; ++i) {
			ret.d[LIMBS - 1 - i] = in.d[i];
		}
#endif
		return ret;
	}

	/// source: https://github.com/WojciechMula/toys/tree/master/simd-basic/reverse-bytes
	/// \param input
	/// \return
	[[nodiscard]] constexpr static inline S reverse_(const S input) noexcept {
#if defined(USE_AVX512VBMI)
		const __m512i indices_byte = _mm512_set_epi64(
		        0x0001020304050607llu, 0x08090a0b0c0d0e0fllu,
		        0x1011121314151617llu, 0x18191a1b1c1d1e1fllu,
		        0x2021222324252627llu, 0x28292a2b2c2d2e2fllu,
		        0x3031323334353637llu, 0x38393a3b3c3d3e3fllu);

		S ret;
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

		S ret;
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

		S ret;
		ret.v512 = (__m512i) __builtin_ia32_pshufb512_mask ((__v64qi)swap_128,
						  (__v64qi)indices_byte,
						  (__v64qi)Y,
						  (__mmask64) -1);
		return ret;
#endif
#else
		S ret;
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

	/// \param in[in]:
	/// \param perm[in]:
	/// \return
	[[nodiscard]] constexpr static inline S permute(const S in,
	                                                const S perm) noexcept {
		S ret;
        ret.v512 = _mm512_permutexvar_epi8(in.v512, perm.v512);
        return ret;
    }

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline uint64_t move(const S in) noexcept {
		const __mmask64 t = _mm512_movepi8_mask(in.v512);
		return t;
	}


	/// finds the first limb in lane which is equal to in2
	/// SOURCE: http://0x80.pl/notesen/2023-02-06-avx512-find-first-byte-in-lane.html
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr inline static S find_first_byte_in_lane(const S in1,
	                                                                  const uint8_t in2) noexcept {
		S tmp1 = S::setr(
				0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,
				0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,
				0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1,
				1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1
		);

		S tmp = S::set1(in2);
		constexpr S one = S::set1(1u);
		tmp = S::xor_(tmp, in1);
		S tmpp = S::xor_(S::min(one, tmp), one);
		tmp = S::sub(tmpp, tmp1);
		tmp = S::and_(tmp, S::not_(tmpp));
		return S::popcnt(tmp);
	}

	/// returns
	/// if b == 0: return 0
   	/// if b < 0 : return -a
    /// if b > 0 : return a
	[[nodiscard]] constexpr inline static S comp_sign(const S a,
			const S b) {
		S ret;
  		__m512i zero = _mm512_setzero_si512();
  		__mmask64 blt0 = _mm512_movepi8_mask(b.v512);
  		__mmask64 ble0 = _mm512_cmple_epi8_mask(b.v512, zero);
  		__m512i a_blt0 = _mm512_mask_mov_epi8(zero, blt0, a.v512);
		ret.v512 = _mm512_mask_sub_epi8(a.v512, ble0, zero, a_blt0);;
  		return ret;
	}

	/// needs `AVX512BW`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline S min(const S in1,
	                                            const S in2) noexcept {
		S ret;
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
	[[nodiscard]] constexpr static inline S max(const S in1,
	                                            const S in2) noexcept {
		S ret;
#ifdef __clang__
		ret.v512 = (__m512i)__builtin_elementwise_max((__v8du)in1.v512, (__v8du)in2.v512);
  		//ret.v512 = (__m512i)__builtin_ia32_pmaxsb512((__v64qi)in1.v512, (__v64qi)in2.v512);
#else
  		//ret.v512 = (__m512i) __builtin_ia32_pmaxsb512_mask ((__v64qi)in1.v512,
		//				  (__v64qi)in2.v512,
		//				  (__v64qi) __extension__ (__m512i)(__v8di){ 0, 0, 0, 0, 0, 0, 0, 0 },
		//				  (__mmask64) -1);
		ret.v512 = _mm512_max_epu8(in1.v512, in2.v512);
#endif
		return ret;
	}
};

///
using uint8x64_t = Xint8x64_t<true>;
using  int8x64_t = Xint8x64_t<false>;

template<const bool __unsigned=true>
struct Xint16x32_t {
	constexpr static uint32_t LIMBS = 32;
	using limb_type = uint16_t;
	using S = Xint16x32_t;
	using simd_type = S;

	using V   = std::conditional<__unsigned, __v32qu, __v32qi>::type;
    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		// compatibility with TxN_t
		T16 d[32];

		T8  v8[64];
		T16 v16[32];
		T32 v32[16];
		T64 v64[8];
		cryptanalysislib::_uint16x8_t v128[4];
		__m256i v256[2];
		__m512i v512;
	};

	[[nodiscard]] constexpr inline static size_t size() noexcept { 
        return LIMBS; 
    }

	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept { 
        return __unsigned; 
    }

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

	///
	/// \return
	[[nodiscard]] static inline Xint16x32_t random() noexcept {
		Xint16x32_t ret;
		for (size_t i = 0; i < 8; ++i) {
			ret.v64[i] = cryptanalysislib::rng();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t set(
	        const limb_type __q31, const limb_type __q30, const limb_type __q29, const limb_type __q28,
	        const limb_type __q27, const limb_type __q26, const limb_type __q25, const limb_type __q24,
	        const limb_type __q23, const limb_type __q22, const limb_type __q21, const limb_type __q20,
	        const limb_type __q19, const limb_type __q18, const limb_type __q17, const limb_type __q16,
	        const limb_type __q15, const limb_type __q14, const limb_type __q13, const limb_type __q12,
	        const limb_type __q11, const limb_type __q10, const limb_type __q09, const limb_type __q08,
	        const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
	        const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		Xint16x32_t out;
		out.v512 = __extension__(__m512i)(__v32hi){
		        (short)__q00, (short)__q01, (short)__q02, (short)__q03, (short)__q04, (short)__q05, (short)__q06, (short)__q07,
		        (short)__q08, (short)__q09, (short)__q10, (short)__q11, (short)__q12, (short)__q13, (short)__q14, (short)__q15,
		        (short)__q16, (short)__q17, (short)__q18, (short)__q19, (short)__q20, (short)__q21, (short)__q22, (short)__q23,
		        (short)__q24, (short)__q25, (short)__q26, (short)__q27, (short)__q28, (short)__q29, (short)__q30, (short)__q31};
		return out;
	}

	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t setr(
	        const limb_type __q31, const limb_type __q30, const limb_type __q29, const limb_type __q28,
	        const limb_type __q27, const limb_type __q26, const limb_type __q25, const limb_type __q24,
	        const limb_type __q23, const limb_type __q22, const limb_type __q21, const limb_type __q20,
	        const limb_type __q19, const limb_type __q18, const limb_type __q17, const limb_type __q16,
	        const limb_type __q15, const limb_type __q14, const limb_type __q13, const limb_type __q12,
	        const limb_type __q11, const limb_type __q10, const limb_type __q09, const limb_type __q08,
	        const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
	        const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		Xint16x32_t out;
		out.v512 = __extension__(__m512i)(__v32hi){
		        (short)__q31, (short)__q30, (short)__q29, (short)__q28, (short)__q27, (short)__q26, (short)__q25, (short)__q24,
		        (short)__q23, (short)__q22, (short)__q21, (short)__q20, (short)__q19, (short)__q18, (short)__q17, (short)__q16,
		        (short)__q15, (short)__q14, (short)__q13, (short)__q12, (short)__q11, (short)__q10, (short)__q09, (short)__q08,
		        (short)__q07, (short)__q06, (short)__q05, (short)__q04, (short)__q03, (short)__q02, (short)__q01, (short)__q00};
		return out;
	}

	[[nodiscard]] constexpr static inline Xint16x32_t set1(const limb_type __A) noexcept {
		Xint16x32_t out;
		out.v512 = __extension__(__m512i)(__v32hi){(short)__A, (short)__A, (short)__A, (short)__A, (short)__A, (short)__A, (short)__A, (short)__A,
		                                           (short)__A, (short)__A, (short)__A, (short)__A, (short)__A, (short)__A, (short)__A, (short)__A,
		                                           (short)__A, (short)__A, (short)__A, (short)__A, (short)__A, (short)__A, (short)__A, (short)__A,
		                                           (short)__A, (short)__A, (short)__A, (short)__A, (short)__A, (short)__A, (short)__A, (short)__A};

		return out;
	}


	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline Xint16x32_t load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t aligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m512i tmp = u16tom512(ptr);
			S out;
			out.v512 = tmp;
			return out;
		} else {
			const __m512i tmp = *(__m512i *) ptr;
			Xint16x32_t out;
			out.v512 = tmp;
			return out;
		}

	}

	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t unaligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m512i tmp = u16tom512(ptr);
			S out;
			out.v512 = tmp;
			return out;
		} else {
			const __m512i tmp = (__m512i) (*(__v64qi_u *) ptr);
			Xint16x32_t out;
			out.v512 = tmp;
			return out;
		}
	}

	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(limb_type *ptr, 
                                       const Xint16x32_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(limb_type *ptr,
                                               const Xint16x32_t in) noexcept {
		auto *ptr512 = (__m512i *) ptr;
		*ptr512 = in.v512;
	}

	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(limb_type *ptr,
                                                 const Xint16x32_t in) noexcept {
		auto *ptr512 = (__m512i_u *) ptr;
		*(__m512i_u *) ptr512 = (__m512i_u) in.v512;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t xor_(const Xint16x32_t in1,
	                                                       const Xint16x32_t in2) noexcept {
		Xint16x32_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) in2.v512);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t and_(const Xint16x32_t in1,
	                                                       const Xint16x32_t in2) noexcept {
		Xint16x32_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t or_(const Xint16x32_t in1,
	                                                      const Xint16x32_t in2) noexcept {
		Xint16x32_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 | (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t andnot(const Xint16x32_t in1,
	                                                         const Xint16x32_t in2) noexcept {
		Xint16x32_t out;
		out.v512 = (__m512i) (~(__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t not_(const Xint16x32_t in1) noexcept {
		Xint16x32_t out;
		const Xint16x32_t minus_one = set1(-1);
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) minus_one.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t add(const Xint16x32_t in1,
	                                                      const Xint16x32_t in2) noexcept {
		Xint16x32_t out;
		out.v512 = (__m512i) ((__v32hu) in1.v512 + (__v32hu) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t sub(const Xint16x32_t in1,
	                                                      const Xint16x32_t in2) noexcept {
		Xint16x32_t out;
		out.v512 = (__m512i) ((__v32hu) in1.v512 - (__v32hu) in2.v512);
		return out;
	}

	/// 8 bit mul lo
	/// \param in1 first input
	/// \param in2
	/// \return in1*in2
	[[nodiscard]] constexpr static inline Xint16x32_t mullo(const Xint16x32_t in1,
	                                                        const Xint16x32_t in2) noexcept {
		Xint16x32_t out;
		out.v512 = (__m512i) ((__v32hu) in1.v512 * (__v32hu) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t mullo(const Xint16x32_t in1,
	                                                        const uint8_t in2) noexcept {
		const Xint16x32_t rs = Xint16x32_t::set1(in2);
		return Xint16x32_t::mullo(in1, rs);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
        S out;
        const __m512i vb = _mm512_set1_epi16(32768 / in2);
        out.v512 = _mm512_mulhrs_epi16(in1.v512, vb);
        return out;
    }

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t slli(const Xint16x32_t in1,
	                                                      const uint8_t in2) noexcept {
		assert(in2 <= 16);
		Xint16x32_t out;
		// out.v512 = _mm512_slli_epi16(in1.v512, in2);
		out.v512 = (__m512i)((__v32hi)in1.v512 << (int)in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t srli(const Xint16x32_t in1,
	                                                       const uint8_t in2) noexcept {
		assert(in2 <= 16);
		Xint16x32_t out;
		// out.v512 = _mm512_srli_epi16(in1.v512, in2);
		out.v512 = (__m512i)((__v32hi)in1.v512 >> (int)in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S ror(const S in1,
												const limb_type in2) noexcept {
		S out;
        out.v512 = _mm512_slli_epi16(in1.v512, 16u - in2) ^ _mm512_srli_epi16(in1.v512, in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S rol(const S in1,
												const limb_type in2) noexcept {
		S out;
        out.v512 = _mm512_slli_epi16(in1.v512, in2) ^ _mm512_srli_epi16(in1.v512, 16u-in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t gt_(const Xint16x32_t in1,
														  const Xint16x32_t in2) noexcept {
		Xint16x32_t ret;
		ret.v512 = (__m512i) ((__v32hu) in1.v512 > (__v32hu) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t gt(const Xint16x32_t in1,
													  const Xint16x32_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v32hu) in1.v512 > (__v32hu) in2.v512);
		return (uint32_t)(__mmask32) __builtin_ia32_cvtw2mask512 ((__v32hi)v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t ge_(const Xint16x32_t in1,
														  const Xint16x32_t in2) noexcept {
		Xint16x32_t ret;
		ret.v512 = (__m512i) ((__v32hu) in1.v512 >= (__v32hu) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t ge(const Xint16x32_t in1,
													  const Xint16x32_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v32hu) in1.v512 >= (__v32hu) in2.v512);
		return (uint32_t)(__mmask32) __builtin_ia32_cvtw2mask512 ((__v32hi)v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t lt_(const Xint16x32_t in1,
														  const Xint16x32_t in2) noexcept {
		Xint16x32_t ret;
		ret.v512 = (__m512i) ((__v32hu) in1.v512 < (__v32hu) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t lt(const Xint16x32_t in1,
													  const Xint16x32_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v32hu) in1.v512 < (__v32hu) in2.v512);
		return (uint32_t)(__mmask32) __builtin_ia32_cvtw2mask512 ((__v32hi)v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t le_(const Xint16x32_t in1,
														  const Xint16x32_t in2) noexcept {
		Xint16x32_t ret;
		ret.v512 = (__m512i) ((__v32hu) in1.v512 <= (__v32hu) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t le(const Xint16x32_t in1,
													  const Xint16x32_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v32hu) in1.v512 <= (__v32hu) in2.v512);
		return (uint32_t)(__mmask32) __builtin_ia32_cvtw2mask512 ((__v32hi)v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t eq_(const Xint16x32_t in1,
														  const Xint16x32_t in2) noexcept {
		Xint16x32_t ret;
		ret.v512 = (__m512i) ((__v32hi) in1.v512 == (__v32hi) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t eq(const Xint16x32_t in1,
												      const Xint16x32_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v32hi) in1.v512 == (__v32hi) in2.v512);
		return (uint32_t)(__mmask32) __builtin_ia32_cvtw2mask512 ((__v32hi)v512);
	}

	/// \param in1
	/// \param in2
	/// \return {-1, 0, 1}
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
												 const S in2) noexcept {
		S ret;
		ret.v512  = (__m512i) ((V) in1.v512 < (V) in2.v512);
		ret.v512 ^= (__m512i) ((V) in1.v512 > (V) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
												       const S in2) noexcept {
        S ret = S::cmp_(in1, in2);
		return (uint32_t)(__mmask32) __builtin_ia32_cvtw2mask512 ((__v32hi)ret.v512);
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t popcnt(const Xint16x32_t in1) noexcept {
		Xint16x32_t ret;
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
	[[nodiscard]] constexpr static inline Xint16x32_t lzcnt(const Xint16x32_t in1) noexcept {
		Xint16x32_t ret;
		constexpr Xint16x32_t one = Xint16x32_t::set1(1);
		ret = Xint16x32_t::sub(in1, one);
		ret = Xint16x32_t::and_(ret, Xint16x32_t::not_(in1));
		ret = Xint16x32_t::popcnt(ret);
		return ret;
	}

    /// TODO opt
	/// checks if all bytes are equal
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const Xint16x32_t in) noexcept {
		for (uint32_t i = 1; i < LIMBS; ++i) {
			if (in.d[i-1] != in.d[i]) {
				return false;
			}
		}

		return true;
	}

	/// \param in
	/// \param perm
	/// \return TODO optimize
	[[nodiscard]] constexpr static inline S permute(const S in,
	                                                const S perm) noexcept {
		S ret;
        for (uint32_t i = 0; i < LIMBS; i++) {
            ret.d[perm.d[i]] = in[i];
        }

        return ret;
    }

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t reverse(const Xint16x32_t in) noexcept {
		Xint16x32_t out;
		for (uint32_t i = 0; i < LIMBS; ++i) {
			out.d[i] = in.d[LIMBS - 1 - i];
		}

		return out;
	}

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline uint32_t move(const Xint16x32_t in) noexcept {
		const __mmask32 t = _mm512_movepi16_mask(in.v512);
		return t;
	}

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t min(const Xint16x32_t a,
                                                      	  const Xint16x32_t b) noexcept {
        Xint16x32_t c;
        c.v512 = _mm512_min_epi32(a.v512, b.v512);
        return c;
    }

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline Xint16x32_t max(const Xint16x32_t a,
														  const Xint16x32_t b) noexcept {
        Xint16x32_t c;
        c.v512 = _mm512_max_epi16(a.v512, b.v512);
        return c;
    }
};

///
using uint16x32_t = Xint16x32_t<true>;
using  int16x32_t = Xint16x32_t<false>;

template<const bool __unsigned=true>
struct Xint32x16_t {
	constexpr static uint32_t LIMBS = 16;
	using limb_type = uint32_t;
	using S = Xint32x16_t;
	using simd_type = S;

    using V   = std::conditional<__unsigned, __v32qu, __v32qi>::type;
    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		// compatibility with TxN_t
		T32   d[16];

		T8   v8[64];
		T16 v16[32];
		T32 v32[16];
		T64 v64[8];
		cryptanalysislib::_uint32x4_t v128[4];
		__m256i v256[2];
		__m512i v512;
	};

	[[nodiscard]] constexpr inline static size_t size() noexcept { 
        return LIMBS; 
    }

	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept { 
        return __unsigned; 
    }


	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	///
	/// \return
	[[nodiscard]] static inline Xint32x16_t random() noexcept {
		Xint32x16_t ret;
		for (size_t i = 0; i < 8; ++i) {
			ret.v64[i] = rng();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t set(
	        const limb_type __q15, const limb_type __q14, const limb_type __q13, const limb_type __q12,
	        const limb_type __q11, const limb_type __q10, const limb_type __q09, const limb_type __q08,
	        const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
	        const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		Xint32x16_t out;
		out.v512 = __extension__(__m512i)(__v16si){
		        (int)__q00, (int)__q01, (int)__q02, (int)__q03, (int)__q04, (int)__q05, (int)__q06, (int)__q07,
		        (int)__q08, (int)__q09, (int)__q10, (int)__q11, (int)__q12, (int)__q13, (int)__q14, (int)__q15};
		return out;
	}

	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t setr(
	        const limb_type __q15, const limb_type __q14, const limb_type __q13, const limb_type __q12,
	        const limb_type __q11, const limb_type __q10, const limb_type __q09, const limb_type __q08,
	        const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
	        const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		Xint32x16_t out;
		out.v512 = __extension__(__m512i)(__v16si){
		        (int)__q15, (int)__q14, (int)__q13, (int)__q12, (int)__q11, (int)__q10, (int)__q09, (int)__q08,
		        (int)__q07, (int)__q06, (int)__q05, (int)__q04, (int)__q03, (int)__q02, (int)__q01, (int)__q00};
		return out;
	}

	[[nodiscard]] constexpr static inline Xint32x16_t set1(const limb_type __a) noexcept {
		Xint32x16_t out;
		out.v512 = __extension__(__m512i)(__v16si){(int)__a, (int)__a, (int)__a, (int)__a, (int)__a, (int)__a, (int)__a, (int)__a,
		                                           (int)__a, (int)__a, (int)__a, (int)__a, (int)__a, (int)__a, (int)__a, (int)__a};

		return out;
	}


	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline Xint32x16_t load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t aligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m512i tmp = u32tom512(ptr);
			S out;
			out.v512 = tmp;
			return out;
		} else {
			const __m512i tmp = *(__m512i *) ptr;
			Xint32x16_t out;
			out.v512 = tmp;
			return out;
		}
	}


	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t unaligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m512i tmp = u32tom512(ptr);
			S out;
			out.v512 = tmp;
			return out;
		} else {
			const __m512i tmp = (__m512i) (*(__v64qi_u *) ptr);
			Xint32x16_t out;
			out.v512 = tmp;
			return out;
		}
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(void *ptr, const Xint32x16_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(void *ptr, const Xint32x16_t in) noexcept {
		auto *ptr512 = (__m512i *) ptr;
		*ptr512 = in.v512;
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(void *ptr, const Xint32x16_t in) noexcept {
		auto *ptr512 = (__m512i_u *) ptr;
		*(__m512i_u *) ptr512 = (__m512i_u) in.v512;
	}


	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t xor_(const Xint32x16_t in1,
	                                                       const Xint32x16_t in2) noexcept {
		Xint32x16_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t and_(const Xint32x16_t in1,
	                                                       const Xint32x16_t in2) noexcept {
		Xint32x16_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t or_(const Xint32x16_t in1,
	                                                      const Xint32x16_t in2) noexcept {
		Xint32x16_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 | (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t andnot(const Xint32x16_t in1,
	                                                         const Xint32x16_t in2) noexcept {
		Xint32x16_t out;
		out.v512 = (__m512i) (~(__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t not_(const Xint32x16_t in1) noexcept {
		Xint32x16_t out;
		const Xint32x16_t minus_one = set1(-1);
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) minus_one.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t add(const Xint32x16_t in1,
	                                                      const Xint32x16_t in2) noexcept {
		Xint32x16_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 + (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t sub(const Xint32x16_t in1,
	                                                      const Xint32x16_t in2) noexcept {
		Xint32x16_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 - (__v16su) in2.v512);
		return out;
	}

	/// 8 bit mul lo
	/// \param in1 first input
	/// \param in2
	/// \return in1*in2
	[[nodiscard]] constexpr static inline Xint32x16_t mullo(const Xint32x16_t in1,
	                                                        const Xint32x16_t in2) noexcept {
		Xint32x16_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 * (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t mullo(const Xint32x16_t in1,
	                                                        const limb_type in2) noexcept {
		const Xint32x16_t rs = Xint32x16_t::set1(in2);
		return Xint32x16_t::mullo(in1, rs);
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t mullo(const Xint32x16_t in1,
	                                                        const uint8_t in2) noexcept {
		const Xint32x16_t rs = Xint32x16_t::set1(in2);
		return Xint32x16_t::mullo(in1, rs);
	}

	/// \param in1
	/// \param in2
	/// \return TODO
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
        S out;
        return out;
    }

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t slli(const Xint32x16_t in1,
														   const uint8_t in2) noexcept {
		assert(in2 <= 32);
		Xint32x16_t out;
		// out.v512 = _mm512_slli_epi32(in1.v512, in2);
		// out.v512 (__m512i)__builtin_ia32_pslldi512((__v16si)in1.v512, (int)in2);
		out.v512 = (__m512i) ((__v16si) in1.v512 << (int)in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t srli(const Xint32x16_t in1,
														   const uint8_t in2) noexcept {
		assert(in2 <= 32);
		Xint32x16_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 >> (int)in2);
		return out;
	}
	
    /// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S ror(const S in1,
												const limb_type in2) noexcept {
		S out;
        out.v512 = _mm512_slli_epi32(in1.v512, 32u - in2) ^ _mm512_srli_epi32(in1.v512, in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S rol(const S in1,
												const limb_type in2) noexcept {
		S out;
        out.v512 = _mm512_slli_epi32(in1.v512, in2) ^ _mm512_srli_epi32(in1.v512, 32u-in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t gt_(const Xint32x16_t in1,
														  const Xint32x16_t in2) noexcept {
		Xint32x16_t ret;
		ret.v512 = (__m512i) ((__v16su) in1.v512 > (__v16su) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16_t gt(const Xint32x16_t in1,
													  const Xint32x16_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v16su) in1.v512 > (__v16su) in2.v512);
		return (uint16_t)(__mmask16) __builtin_ia32_cvtd2mask512 ((__v16si)v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t ge_(const Xint32x16_t in1,
														  const Xint32x16_t in2) noexcept {
		Xint32x16_t ret;
		ret.v512 = (__m512i) ((__v16su) in1.v512 >= (__v16su) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16_t ge(const Xint32x16_t in1,
													  const Xint32x16_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v16su) in1.v512 >= (__v16su) in2.v512);
		return (uint16_t)(__mmask16) __builtin_ia32_cvtd2mask512 ((__v16si)v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t lt_(const Xint32x16_t in1,
														  const Xint32x16_t in2) noexcept {
		Xint32x16_t ret;
		ret.v512 = (__m512i) ((__v16su) in1.v512 < (__v16su) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16_t lt(const Xint32x16_t in1,
													  const Xint32x16_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v16su) in1.v512 < (__v16su) in2.v512);
		return (uint16_t)(__mmask16) __builtin_ia32_cvtd2mask512 ((__v16si)v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t le_(const Xint32x16_t in1,
														  const Xint32x16_t in2) noexcept {
		Xint32x16_t ret;
		ret.v512 = (__m512i) ((__v16su) in1.v512 <= (__v16su) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16_t le(const Xint32x16_t in1,
													  const Xint32x16_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v16su) in1.v512 <= (__v16su) in2.v512);
		return (uint16_t)(__mmask16) __builtin_ia32_cvtd2mask512 ((__v16si)v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t eq_(const Xint32x16_t in1,
														  const Xint32x16_t in2) noexcept {
		Xint32x16_t ret;
		ret.v512 = (__m512i) ((__v16si) in1.v512 == (__v16si) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16_t eq(const Xint32x16_t in1,
													  const Xint32x16_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v16si) in1.v512 == (__v16si) in2.v512);
		return (uint16_t)(__mmask16) __builtin_ia32_cvtd2mask512 ((__v16si)v512);
	}

	/// \param in1
	/// \param in2
	/// \return {-1, 0, 1}
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
												 const S in2) noexcept {
		S ret;
		ret.v512  = (__m512i) ((V) in1.v512 < (V) in2.v512);
		ret.v512 ^= (__m512i) ((V) in1.v512 > (V) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
												       const S in2) noexcept {
        S ret = S::cmp_(in1, in2);
		return (uint16_t)(__mmask16) __builtin_ia32_cvtd2mask512 ((__v16si)ret.v512);
	}

	/// needs`AVX512VPOPCNTDQ`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t popcnt(const Xint32x16_t in1) noexcept {
		Xint32x16_t ret;
#ifdef __clang__
		ret.v512 = (__m512i) __builtin_ia32_vpopcntd_512((__v16si)in1.v512);
#else
  		ret.v512 = (__m512i) __builtin_ia32_vpopcountd_v16si ((__v16si)in1.v512);
#endif
		return ret;
	}

	/// Source: http://0x80.pl/notesen/2023-01-31-avx512-bsf.html
	/// count trailing zeros
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t ctz(const Xint32x16_t in1) noexcept {
		Xint32x16_t ret;
		const Xint32x16_t one = Xint32x16_t::set1(1u);
		ret = Xint32x16_t::sub(in1, one);
		ret = Xint32x16_t::and_(ret, Xint32x16_t::not_(in1));
		ret = Xint32x16_t::popcnt(ret);
		return ret;
	}

	/// checks if all bytes are equal
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const Xint32x16_t in) noexcept {
		for (uint32_t i = 1; i < LIMBS; ++i) {
			if (in.d[0] != in.d[i]) {
				return false;
			}
		}

		return true;
	}

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t reverse(const Xint32x16_t in) noexcept {
		Xint32x16_t out;
		for (uint32_t i = 0; i < LIMBS; ++i) {
			out.d[i] = in.d[LIMBS - 1 - i];
		}

		return out;
	}

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline S permute(const S in,
	                                                const S perm) noexcept {
        S ret;
        ret.v512 = _mm512_permutexvar_epi32(in.v512, perm.v512);
        return ret;
    }

	/// needs `AVX512CD`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t conflict(const Xint32x16_t in1) noexcept {
		Xint32x16_t ret;
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
	[[nodiscard]] constexpr static inline Xint32x16_t shuffle_32x4(const Xint32x16_t in1,
	                                                               const Xint32x16_t in2) noexcept {
		Xint32x16_t ret;
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

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline uint16_t move(const Xint32x16_t in) noexcept {
		const __mmask16 t = _mm512_movepi32_mask(in.v512);
		return t;
	}

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t min(const Xint32x16_t a,
                                                      	  const Xint32x16_t b) noexcept {
        Xint32x16_t c;
        c.v512 = _mm512_min_epi32(a.v512, b.v512);
        return c;
    }

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline Xint32x16_t max(const Xint32x16_t a,
														  const Xint32x16_t b) noexcept {
        Xint32x16_t c;
        c.v512 = _mm512_max_epi32(a.v512, b.v512);
        return c;
    }
};

///
using uint32x16_t = Xint32x16_t<true>;
using  int32x16_t = Xint32x16_t<false>;

template<const bool __unsigned=true>
struct Xint64x8_t {
	constexpr static uint32_t LIMBS = 8;
	using limb_type = uint64_t;
	using S = Xint64x8_t;
	using simd_type = S;

    using V   = std::conditional<__unsigned, __v64qu, __v64qi>::type;
    using T8  = std::conditional<__unsigned, uint8_t,   int8_t>::type;
    using T16 = std::conditional<__unsigned, uint16_t, int16_t>::type;
    using T32 = std::conditional<__unsigned, uint32_t, int32_t>::type;
    using T64 = std::conditional<__unsigned, uint64_t, int64_t>::type;

	union {
		// compatibility with TxN_t
		T64   d[8];
		T8   v8[64];
		T16 v16[32];
		T32 v32[16];
		T64 v64[8];
		cryptanalysislib::_uint64x2_t v128[4];
		__m256i v256[2];
		__m512i v512;
	};
	
    [[nodiscard]] constexpr inline static size_t size() noexcept { 
        return LIMBS; 
    }
	[[nodiscard]] constexpr inline static bool is_unsigned() noexcept { 
        return __unsigned; 
    }

	[[nodiscard]] constexpr inline limb_type &operator[](const uint32_t i) noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	[[nodiscard]] constexpr inline limb_type operator[](const uint32_t i) const noexcept {
		assert(i < LIMBS);
		return d[i];
	}

	/// \return
	[[nodiscard]] static inline Xint64x8_t random() noexcept {
		Xint64x8_t ret;
		for (size_t i = 0; i < 8; ++i) {
			ret.v64[i] = rng();
		}
		return ret;
	}

	///
	/// \param binary
	/// \param hex
	constexpr inline void print(bool binary = false, bool hex = false) const;

	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t set(
	        const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
	        const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		Xint64x8_t out;
		out.v512 = __extension__(__m512i)(__v8di){
		        (long long)__q00, (long long)__q01, (long long)__q02, (long long)__q03, (long long)__q04, (long long)__q05, (long long)__q06, (long long)__q07};
		return out;
	}

	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t setr(
			const limb_type __q07, const limb_type __q06, const limb_type __q05, const limb_type __q04,
			const limb_type __q03, const limb_type __q02, const limb_type __q01, const limb_type __q00) noexcept {
		Xint64x8_t out;
		out.v512 = __extension__(__m512i)(__v8di){
		        (long long)__q07, (long long)__q06, (long long)__q05, (long long)__q04, (long long)__q03, (long long)__q02, (long long)__q01, (long long)__q00};
		return out;
	}

	[[nodiscard]] constexpr static inline Xint64x8_t set1(const limb_type __a) noexcept {
		Xint64x8_t out;
		const long long t = (long long)__a;
		out.v512 = __extension__(__m512i)(__v8di){t,t,t,t,t,t,t,t};
		return out;
	}


	///
	/// \tparam aligned
	/// \param ptr
	/// \return
	template<const bool aligned = false>
	[[nodiscard]] constexpr static inline Xint64x8_t load(const limb_type *ptr) noexcept {
		if constexpr (aligned) {
			return aligned_load(ptr);
		}

		return unaligned_load(ptr);
	}

	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t aligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m512i tmp = u64tom512(ptr);
			S out;
			out.v512 = tmp;
			return out;
		} else {
			const __m512i tmp = *(__m512i *) ptr;
			Xint64x8_t out;
			out.v512 = tmp;
			return out;
		}
	}


	///
	/// \param ptr
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t unaligned_load(const limb_type *ptr) noexcept {
		if (std::is_constant_evaluated()) {
			// in the constexpr case simply ignore that the data is aligned
			// it will not have any "runtime" penalties
			const __m512i tmp = u64tom512(ptr);
			S out;
			out.v512 = tmp;
			return out;
		} else {
			const __m512i tmp = (__m512i) (*(__v64qi_u *) ptr);
			Xint64x8_t out;
			out.v512 = tmp;
			return out;
		}
	}

	///
	/// \tparam aligned
	/// \param ptr
	/// \param in
	template<const bool aligned = false>
	constexpr static inline void store(limb_type *ptr,
                                       const Xint64x8_t in) noexcept {
		if constexpr (aligned) {
			aligned_store(ptr, in);
			return;
		}

		unaligned_store(ptr, in);
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void aligned_store(limb_type *ptr,
                                               const Xint64x8_t in) noexcept {
		auto *ptr512 = (__m512i *) ptr;
		*ptr512 = in.v512;
	}

	///
	/// \param ptr
	/// \param in
	constexpr static inline void unaligned_store(limb_type *ptr,
                                                 const Xint64x8_t in) noexcept {
		auto *ptr512 = (__m512i_u *) ptr;
		*(__m512i_u *) ptr512 = (__m512i_u) in.v512;
	}


	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t xor_(const Xint64x8_t in1,
	                                                      const Xint64x8_t in2) noexcept {
		Xint64x8_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t and_(const Xint64x8_t in1,
	                                                      const Xint64x8_t in2) noexcept {
		Xint64x8_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t or_(const Xint64x8_t in1,
	                                                     const Xint64x8_t in2) noexcept {
		Xint64x8_t out;
		out.v512 = (__m512i) ((__v16su) in1.v512 | (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t andnot(const Xint64x8_t in1,
	                                                        const Xint64x8_t in2) noexcept {
		Xint64x8_t out;
		out.v512 = (__m512i) (~(__v16su) in1.v512 & (__v16su) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t not_(const Xint64x8_t in1) noexcept {
		Xint64x8_t out;
		const Xint64x8_t minus_one = set1(-1);
		out.v512 = (__m512i) ((__v16su) in1.v512 ^ (__v16su) minus_one.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t add(const Xint64x8_t in1,
	                                                     const Xint64x8_t in2) noexcept {
		Xint64x8_t out;
		out.v512 = (__m512i) ((__v8du) in1.v512 + (__v8du) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t sub(const Xint64x8_t in1,
	                                                     const Xint64x8_t in2) noexcept {
		Xint64x8_t out;
		out.v512 = (__m512i) ((__v8du) in1.v512 - (__v8du) in2.v512);
		return out;
	}

	/// 8 bit mul lo
	/// \param in1 first input
	/// \param in2
	/// \return in1*in2
	[[nodiscard]] constexpr static inline Xint64x8_t mullo(const Xint64x8_t in1,
	                                                       const Xint64x8_t in2) noexcept {
		Xint64x8_t out;
		out.v512 = (__m512i) ((__v8du) in1.v512 * (__v8du) in2.v512);
		return out;
	}

	///
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t mullo(const Xint64x8_t in1,
	                                                       const uint8_t in2) noexcept {
		const Xint64x8_t rs = Xint64x8_t::set1(in2);
		return Xint64x8_t::mullo(in1, rs);
	}

	/// \param in1
	/// \param in2
	/// \return TODO
	[[nodiscard]] constexpr static inline S div(const S in1,
	                                            const limb_type in2) noexcept {
        S out;
        return out;
    }

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t slli(const Xint64x8_t in1,
	                                                      const limb_type in2) noexcept {
		assert(in2 <= 64);
		Xint64x8_t out;
		// out.v512 = _mm512_slli_epi64(in1.v512, in2);
		// out.v512 = (__m512i)__builtin_ia32_psllqi512((__v8di)in1.v512, (int)in2);
		out.v512 = (__m512i) ((V)in1.v512 << (int)in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t srli(const Xint64x8_t in1,
	                                                       const limb_type in2) noexcept {
		assert(in2 <= 64);
		Xint64x8_t out;
		// out.v512 = _mm512_srli_epi64(in1.v512, in2);
		out.v512 = (__m512i) ((__v8di)in1.v512 >> (int)in2);
		return out;
	}
	
    /// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S ror(const S in1,
												const limb_type in2) noexcept {
		S out;
        out.v512 = _mm512_slli_epi64(in1.v512, 64u - in2) ^ _mm512_srli_epi64(in1.v512, in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline S rol(const S in1,
												const limb_type in2) noexcept {
		S out;
        out.v512 = _mm512_slli_epi64(in1.v512, in2) ^ _mm512_srli_epi64(in1.v512, 64u-in2);
		return out;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16_t gt(const Xint64x8_t in1,
													  const Xint64x8_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v8du) in1.v512 > (__v8du) in2.v512);
		return (uint8_t)(__mmask8) __builtin_ia32_cvtq2mask512 ((__v8di) v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t gt_(const Xint64x8_t in1,
														 const Xint64x8_t in2) noexcept {
		Xint64x8_t ret;
		ret.v512 = (__m512i) ((__v8du) in1.v512 > (__v8du) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16_t ge(const Xint64x8_t in1,
													  const Xint64x8_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v8du) in1.v512 >= (__v8du) in2.v512);
		return (uint8_t)(__mmask8) __builtin_ia32_cvtq2mask512 ((__v8di) v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t ge_(const Xint64x8_t in1,
														 const Xint64x8_t in2) noexcept {
		Xint64x8_t ret;
		ret.v512 = (__m512i) ((__v8du) in1.v512 >= (__v8du) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16_t lt(const Xint64x8_t in1,
													  const Xint64x8_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v8du) in1.v512 < (__v8du) in2.v512);
		return (uint8_t)(__mmask8) __builtin_ia32_cvtq2mask512 ((__v8di) v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t lt_(const Xint64x8_t in1,
														 const Xint64x8_t in2) noexcept {
		Xint64x8_t ret;
		ret.v512 = (__m512i) ((__v8du) in1.v512 < (__v8du) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint16_t le(const Xint64x8_t in1,
													  const Xint64x8_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v8du) in1.v512 <= (__v8du) in2.v512);
		return (uint8_t)(__mmask8) __builtin_ia32_cvtq2mask512 ((__v8di) v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t le_(const Xint64x8_t in1,
														 const Xint64x8_t in2) noexcept {
		Xint64x8_t ret;
		ret.v512 = (__m512i) ((__v8du) in1.v512 <= (__v8du) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint8_t eq(const Xint64x8_t in1,
													 const Xint64x8_t in2) noexcept {
		const __m512i v512 = (__m512i) ((__v8di) in1.v512 == (__v8di) in2.v512);
		return (uint8_t)(__mmask8) __builtin_ia32_cvtq2mask512 ((__v8di) v512);
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t eq_(const Xint64x8_t in1,
	                                                     const Xint64x8_t in2) noexcept {
		Xint64x8_t ret;
		ret.v512 = (__m512i) ((__v8di) in1.v512 == (__v8di) in2.v512);
		return ret;
	}
	
    /// \param in1
	/// \param in2
	/// \return {-1, 0, 1}
	[[nodiscard]] constexpr static inline S cmp_(const S in1,
												 const S in2) noexcept {
		S ret;
		ret.v512  = (__m512i) ((V) in1.v512 < (V) in2.v512);
		ret.v512 ^= (__m512i) ((V) in1.v512 > (V) in2.v512);
		return ret;
	}

	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline uint32_t cmp(const S in1,
												       const S in2) noexcept {
        S ret = S::cmp_(in1, in2);
		return (uint8_t)(__mmask8) __builtin_ia32_cvtq2mask512 ((__v8di) ret.v512);
	}

	///
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t popcnt(const Xint64x8_t in1) noexcept {
		Xint64x8_t ret;
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
	[[nodiscard]] constexpr static inline Xint64x8_t lzcnt(const Xint64x8_t in1) noexcept {
		Xint64x8_t ret;
		constexpr Xint64x8_t one = Xint64x8_t::set1(1);
		ret = Xint64x8_t::sub(in1, one);
		ret = Xint64x8_t::and_(ret, Xint64x8_t::not_(in1));
		ret = Xint64x8_t::popcnt(ret);
		return ret;
	}

	/// checks if all bytes are equal
	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline bool all_equal(const Xint64x8_t in) noexcept {
		for (uint32_t i = 1; i < LIMBS; ++i) {
			if (in.d[0] != in.d[i]) {
				return false;
			}
		}

		return true;
	}

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t reverse(const Xint64x8_t in) noexcept {
		Xint64x8_t out;
		for (uint32_t i = 0; i < LIMBS; ++i) {
			out.d[LIMBS - 1 - i] = in.d[i];
		}

		return out;
	}

	/// needs `AVX512CD`
	/// \param in1
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t conflict(const Xint64x8_t in1) noexcept {
		Xint64x8_t ret;
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
	[[nodiscard]] constexpr static inline Xint64x8_t hadd_epu8(const Xint64x8_t in1) noexcept {
		Xint64x8_t ret;
		constexpr Xint64x8_t zero = Xint64x8_t::set1(0);
		ret.v512 = (__m512i) __builtin_ia32_psadbw512((__v64qi) in1.v512, (__v64qi) zero.v512);
		return ret;
	}

	/// TODO test, and implement for uint32x16 and so on
	/// Source: http://0x80.pl/notesen/2023-01-06-avx512-popcount-4bit.html
	/// \param in1
	/// \param in2
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t histogram_epi4(const Xint64x8_t in1,
	                                                                const uint8_t in2) noexcept {
		assert(in2 < 16);
		Xint64x8_t tmp = Xint64x8_t::xor_(in1, Xint64x8_t::set1(in2));
		tmp = Xint64x8_t::sub(tmp, Xint64x8_t::set1(1));
		tmp = Xint64x8_t::popcnt(tmp);
		return tmp;
	}

	/// \param in
	/// \return
	[[nodiscard]] constexpr static inline uint8_t move(const Xint64x8_t in) noexcept {
		const __mmask8 t = _mm512_movepi64_mask(in.v512);
		return t;
	}


	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t min(const Xint64x8_t a,
                                                      	  const Xint64x8_t b) noexcept {
        Xint64x8_t c;
        c.v512 = _mm512_min_epi64(a.v512, b.v512);
        return c;
    }

	/// \param a
	/// \param b
	/// \return
	[[nodiscard]] constexpr static inline Xint64x8_t max(const Xint64x8_t a,
														  const Xint64x8_t b) noexcept {
        Xint64x8_t c;
        c.v512 = _mm512_max_epi64(a.v512, b.v512);
        return c;
    }

};

///
using uint64x8_t = Xint64x8_t<true>;
using  int64x8_t = Xint64x8_t<false>;

/// TODO make generic
constexpr inline uint8x64_t operator*(const uint8x64_t &lhs, const uint8x64_t &rhs) noexcept {
	return uint8x64_t::mullo(lhs, rhs);
}
constexpr inline uint8x64_t operator*(const uint8x64_t &lhs, const uint8_t &rhs)noexcept {
	return uint8x64_t::mullo(lhs, rhs);
}
constexpr inline uint8x64_t operator*(const uint8_t &lhs, const uint8x64_t &rhs)noexcept {
	return uint8x64_t::mullo(rhs, lhs);
}
constexpr inline uint8x64_t operator+(const uint8x64_t &lhs, const uint8x64_t &rhs) noexcept{
	return uint8x64_t::add(lhs, rhs);
}
constexpr inline uint8x64_t operator-(const uint8x64_t &lhs, const uint8x64_t &rhs) noexcept {
	return uint8x64_t::sub(lhs, rhs);
}
constexpr inline uint8x64_t operator&(const uint8x64_t &lhs, const uint8x64_t &rhs) noexcept {
	return uint8x64_t::and_(lhs, rhs);
}
constexpr inline uint8x64_t operator^(const uint8x64_t &lhs, const uint8x64_t &rhs) noexcept {
	return uint8x64_t::xor_(lhs, rhs);
}
constexpr inline uint8x64_t operator|(const uint8x64_t &lhs, const uint8x64_t &rhs) noexcept {
	return uint8x64_t::or_(lhs, rhs);
}
constexpr inline uint8x64_t operator~(const uint8x64_t &lhs) noexcept {
	return uint8x64_t::not_(lhs);
}
constexpr inline uint8x64_t operator>> (const uint8x64_t& lhs, const uint32_t rhs) noexcept {
	return uint8x64_t::srli(lhs, rhs);
}
constexpr inline uint8x64_t operator<< (const uint8x64_t& lhs, const uint32_t rhs) noexcept {
	return uint8x64_t::slli(lhs, rhs);
}
constexpr inline uint8x64_t operator^=(uint8x64_t &lhs, const uint8x64_t &rhs) noexcept {
	lhs = uint8x64_t::xor_(lhs, rhs);
	return lhs;
}
constexpr inline uint8x64_t operator&=(uint8x64_t &lhs, const uint8x64_t &rhs) noexcept {
	lhs = uint8x64_t::and_(lhs, rhs);
	return lhs;
}
constexpr inline uint8x64_t operator|=(uint8x64_t &lhs, const uint8x64_t &rhs) noexcept {
	lhs = uint8x64_t::or_(lhs, rhs);
	return lhs;
}


///
constexpr inline uint16x32_t operator*(const uint16x32_t &lhs, const uint16x32_t &rhs) noexcept {
	return uint16x32_t::mullo(lhs, rhs);
}
constexpr inline uint16x32_t operator*(const uint16x32_t &lhs, const uint8_t &rhs) noexcept {
	return uint16x32_t::mullo(lhs, rhs);
}
constexpr inline uint16x32_t operator*(const uint8_t &lhs, const uint16x32_t &rhs) noexcept {
	return uint16x32_t::mullo(rhs, lhs);
}
constexpr inline uint16x32_t operator+(const uint16x32_t &lhs, const uint16x32_t &rhs) noexcept {
	return uint16x32_t::add(lhs, rhs);
}
constexpr inline uint16x32_t operator-(const uint16x32_t &lhs, const uint16x32_t &rhs) noexcept {
	return uint16x32_t::sub(lhs, rhs);
}
constexpr inline uint16x32_t operator&(const uint16x32_t &lhs, const uint16x32_t &rhs) noexcept {
	return uint16x32_t::and_(lhs, rhs);
}
constexpr inline uint16x32_t operator^(const uint16x32_t &lhs, const uint16x32_t &rhs) noexcept {
	return uint16x32_t::xor_(lhs, rhs);
}
constexpr inline uint16x32_t operator|(const uint16x32_t &lhs, const uint16x32_t &rhs) noexcept {
	return uint16x32_t::or_(lhs, rhs);
}
constexpr inline uint16x32_t operator~(const uint16x32_t &lhs) {
	return uint16x32_t::not_(lhs);
}
constexpr inline uint16x32_t operator>> (const uint16x32_t& lhs, const uint32_t rhs) noexcept {
	return uint16x32_t::srli(lhs, rhs);
}
constexpr inline uint16x32_t operator<< (const uint16x32_t& lhs, const uint32_t rhs) noexcept {
	return uint16x32_t::slli(lhs, rhs);
}
constexpr inline uint16x32_t operator^=(uint16x32_t &lhs, const uint16x32_t &rhs) noexcept {
	lhs = uint16x32_t::xor_(lhs, rhs);
	return lhs;
}
constexpr inline uint16x32_t operator&=(uint16x32_t &lhs, const uint16x32_t &rhs) noexcept{
	lhs = uint16x32_t::and_(lhs, rhs);
	return lhs;
}
constexpr inline uint16x32_t operator|=(uint16x32_t &lhs, const uint16x32_t &rhs) noexcept{
	lhs = uint16x32_t::or_(lhs, rhs);
	return lhs;
}


///
constexpr inline uint32x16_t operator*(const uint32x16_t &lhs, const uint32x16_t &rhs) noexcept{
	return uint32x16_t::mullo(lhs, rhs);
}
constexpr inline uint32x16_t operator*(const uint32x16_t &lhs, const uint8_t &rhs) noexcept{
	return uint32x16_t::mullo(lhs, rhs);
}
constexpr inline uint32x16_t operator*(const uint8_t &lhs, const uint32x16_t &rhs) noexcept{
	return uint32x16_t::mullo(rhs, lhs);
}
constexpr inline uint32x16_t operator+(const uint32x16_t &lhs, const uint32x16_t &rhs) noexcept{
	return uint32x16_t::add(lhs, rhs);
}
constexpr inline uint32x16_t operator-(const uint32x16_t &lhs, const uint32x16_t &rhs)noexcept {
	return uint32x16_t::sub(lhs, rhs);
}
constexpr inline uint32x16_t operator&(const uint32x16_t &lhs, const uint32x16_t &rhs) noexcept{
	return uint32x16_t::and_(lhs, rhs);
}
constexpr inline uint32x16_t operator^(const uint32x16_t &lhs, const uint32x16_t &rhs)noexcept {
	return uint32x16_t::xor_(lhs, rhs);
}
constexpr inline uint32x16_t operator|(const uint32x16_t &lhs, const uint32x16_t &rhs) noexcept{
	return uint32x16_t::or_(lhs, rhs);
}
constexpr inline uint32x16_t operator~(const uint32x16_t &lhs) noexcept {
	return uint32x16_t::not_(lhs);
}
constexpr inline uint32x16_t operator>> (const uint32x16_t& lhs, const uint32_t rhs) noexcept {
	return uint32x16_t::srli(lhs, rhs);
}
constexpr inline uint32x16_t operator<< (const uint32x16_t& lhs, const uint32_t rhs) noexcept {
	return uint32x16_t::slli(lhs, rhs);
}
constexpr inline uint32x16_t operator^=(uint32x16_t &lhs, const uint32x16_t &rhs) noexcept {
	lhs = uint32x16_t::xor_(lhs, rhs);
	return lhs;
}
constexpr inline uint32x16_t operator&=(uint32x16_t &lhs, const uint32x16_t &rhs) noexcept {
	lhs = uint32x16_t::and_(lhs, rhs);
	return lhs;
}
constexpr inline uint32x16_t operator|=(uint32x16_t &lhs, const uint32x16_t &rhs) noexcept {
	lhs = uint32x16_t::or_(lhs, rhs);
	return lhs;
}


///
constexpr inline uint64x8_t operator*(const uint64x8_t &lhs, const uint64x8_t &rhs) noexcept {
	return uint64x8_t::mullo(lhs, rhs);
}
constexpr inline uint64x8_t operator*(const uint64x8_t &lhs, const uint8_t &rhs) noexcept{
	return uint64x8_t::mullo(lhs, rhs);
}
constexpr inline uint64x8_t operator*(const uint8_t &lhs, const uint64x8_t &rhs) noexcept{
	return uint64x8_t::mullo(rhs, lhs);
}
constexpr inline uint64x8_t operator+(const uint64x8_t &lhs, const uint64x8_t &rhs) noexcept{
	return uint64x8_t::add(lhs, rhs);
}
constexpr inline uint64x8_t operator-(const uint64x8_t &lhs, const uint64x8_t &rhs) noexcept{
	return uint64x8_t::sub(lhs, rhs);
}
constexpr inline uint64x8_t operator&(const uint64x8_t &lhs, const uint64x8_t &rhs) noexcept{
	return uint64x8_t::and_(lhs, rhs);
}
constexpr inline uint64x8_t operator^(const uint64x8_t &lhs, const uint64x8_t &rhs) noexcept{
	return uint64x8_t::xor_(lhs, rhs);
}
constexpr inline uint64x8_t operator|(const uint64x8_t &lhs, const uint64x8_t &rhs) noexcept{
	return uint64x8_t::or_(lhs, rhs);
}
constexpr inline uint64x8_t operator~(const uint64x8_t &lhs) noexcept{
	return uint64x8_t::not_(lhs);
}
constexpr inline uint64x8_t operator>> (const uint64x8_t& lhs, const uint32_t rhs) noexcept {
	return uint64x8_t::srli(lhs, rhs);
}
constexpr inline uint64x8_t operator<< (const uint64x8_t& lhs, const uint32_t rhs) noexcept {
	return uint64x8_t::slli(lhs, rhs);
}
constexpr inline uint64x8_t operator^=(uint64x8_t &lhs, const uint64x8_t &rhs)noexcept {
	lhs = uint64x8_t::xor_(lhs, rhs);
	return lhs;
}
constexpr inline uint64x8_t operator&=(uint64x8_t &lhs, const uint64x8_t &rhs) noexcept{
	lhs = uint64x8_t::and_(lhs, rhs);
	return lhs;
}
constexpr inline uint64x8_t operator|=(uint64x8_t &lhs, const uint64x8_t &rhs) noexcept{
	lhs = uint64x8_t::or_(lhs, rhs);
	return lhs;
}

constexpr inline uint64_t operator==(const uint8x64_t &a, const uint8x64_t &b) noexcept {
	return uint8x64_t::cmp(a, b);
}
constexpr inline uint64_t operator!=(const uint8x64_t &a, const uint8x64_t &b) noexcept {
	return -1ull ^ uint8x64_t::cmp(a, b);
}
constexpr inline uint64_t operator<(const uint8x64_t &a, const uint8x64_t &b) noexcept {
	return uint8x64_t::lt(a, b);
}
constexpr inline uint64_t operator>(const uint8x64_t &a, const uint8x64_t &b) noexcept {
	return uint8x64_t::gt(a, b);
}

constexpr inline uint64_t operator==(const uint16x32_t &a, const uint16x32_t &b) noexcept {
	return uint16x32_t::cmp(a, b);
}
constexpr inline uint64_t operator!=(const uint16x32_t &a, const uint16x32_t &b) noexcept {
	return -1ull ^ uint16x32_t::cmp(a, b);
}
constexpr inline uint64_t operator<(const uint16x32_t &a, const uint16x32_t &b) noexcept {
	return uint16x32_t::lt(a, b);
}
constexpr inline uint64_t operator>(const uint16x32_t &a, const uint16x32_t &b) noexcept {
	return uint16x32_t::gt(a, b);
}

constexpr inline uint64_t operator==(const uint32x16_t &a, const uint32x16_t &b) noexcept {
	return uint32x16_t::cmp(a, b);
}
constexpr inline uint64_t operator!=(const uint32x16_t &a, const uint32x16_t &b) noexcept {
	return -1ull ^ uint32x16_t::cmp(a, b);
}
constexpr inline uint64_t operator<(const uint32x16_t &a, const uint32x16_t &b) noexcept {
	return uint32x16_t::lt(a, b);
}
constexpr inline uint64_t operator>(const uint32x16_t &a, const uint32x16_t &b) noexcept {
	return uint32x16_t::gt(a, b);
}

constexpr inline uint64_t operator==(const uint64x8_t &a, const uint64x8_t &b) noexcept {
	return uint64x8_t::cmp(a, b);
}
constexpr inline uint64_t operator!=(const uint64x8_t &a, const uint64x8_t &b) noexcept {
	return -1ull ^ uint64x8_t::cmp(a, b);
}
constexpr inline uint64_t operator<(const uint64x8_t &a, const uint64x8_t &b) noexcept {
	return uint64x8_t::lt(a, b);
}
constexpr inline uint64_t operator>(const uint64x8_t &a, const uint64x8_t &b) noexcept {
	return uint64x8_t::gt(a, b);
}

/// NOTE: this is stupid. gcc does strange thing.
/// \return an uninitialized avx512 register
constexpr inline __m512i
__mm512_undefined_epi32 (void) {
#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wuninitialized"
#else
	#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-self"
#endif
	__m512i __Y = __Y;
#pragma GCC diagnostic pop
	return __Y;
}



/* Transpose bits within bytes. */
/// source: https://github.com/kiyo-masui/bitshuffle/blob/master/src/bitshuffle_core.c
///
int64_t bshuf_trans_bit_byte_AVX512(const void* in,
								   void* out,
								   const size_t size,
         						const size_t elem_size) {

    size_t ii, kk;
    const char* in_b = (const char*) in;
    char* out_b = (char*) out;
    size_t nbyte = elem_size * size;
    int64_t count;

    int64_t* out_i64;
    __m512i zmm;
    __mmask64 bt;
    if (nbyte >= 64) {
        const __m512i mask = _mm512_set1_epi8(0);

       for (ii = 0; ii + 63 < nbyte; ii += 64) {
            zmm = _mm512_loadu_si512((__m512i *) &in_b[ii]);
            for (kk = 0; kk < 8; kk++) {
                bt = _mm512_cmp_epi8_mask(zmm, mask, 1);
                zmm = _mm512_slli_epi16(zmm, 1);
                out_i64 = (int64_t*) &out_b[((7 - kk) * nbyte + ii) / 8];
                *out_i64 = (int64_t)bt;
            }
        }
    }

    __m256i ymm;
    int32_t bt32;
    int32_t* out_i32;
    size_t start = nbyte - nbyte % 64;
    for (ii = start; ii + 31 < nbyte; ii += 32) {
        ymm = _mm256_loadu_si256((__m256i *) &in_b[ii]);
        for (kk = 0; kk < 8; kk++) {
            bt32 = _mm256_movemask_epi8(ymm);
            ymm = _mm256_slli_epi16(ymm, 1);
            out_i32 = (int32_t*) &out_b[((7 - kk) * nbyte + ii) / 8];
            *out_i32 = bt32;
        }
    }


    count = bshuf_trans_byte_elem_remainder(in, out, size, elem_size,
            nbyte - nbyte % 64 % 32);

    return count;
}

template<const uint32_t k>
__m512i _mm512_slli_si512_epi64(const __m512i x) noexcept {
    const __m512i ZERO = _mm512_setzero_si512();
    return _mm512_alignr_epi64(x, ZERO, 8 - k);
}

///
template<const uint32_t k>
__m512i _mm512_slli_si512_epi32(const __m512i x) noexcept {
    const __m512i ZERO = _mm512_setzero_si512();
    return _mm512_alignr_epi32(x, ZERO, 16 - k);
}

template<const uint32_t k>
__m512i _mm512_slli_si128_epi8 (const __m512i x) noexcept {
    const __m512i ZERO = _mm512_setzero_si512();
    return _mm512_alignr_epi8(x, ZERO, 16 - k);
}

__m512i __prefixsum_u32_avx512(__m512i x) noexcept {
    x = _mm512_add_epi32(x, _mm512_slli_si512_epi32<1>(x));
    x = _mm512_add_epi32(x, _mm512_slli_si512_epi32<2>(x));
    x = _mm512_add_epi32(x, _mm512_slli_si512_epi32<4>(x));
    x = _mm512_add_epi32(x, _mm512_slli_si512_epi32<8>(x));
    return x;
}

#endif//CRYPTANALYSISLIB_AVX512_H
