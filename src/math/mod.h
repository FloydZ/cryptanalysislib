#ifndef CRYPTANALYSISLIB_MATH_MOD_H
#define CRYPTANALYSISLIB_MATH_MOD_H

#include <type_traits>
#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

#include <cstdint>

#include "simd/simd.h"


namespace cryptanalysislib::math::internal {
// TODO: - u64 mod/div, simd mod/div
// 		 - docs
// extension by FloydZ
// source: https://github.com/lemire/fastmod

/// \param lowbits
/// \param d
/// \return
constexpr static uint64_t mul128_u32(const uint64_t lowbits,
                                     const uint32_t d) noexcept {
	return ((__uint128_t)lowbits * d) >> 64;
}

///
/// \tparam S
/// \param a
/// \param b
/// \return
template<typename S>
#if __cplusplus > 201709L
	requires SIMDAble<S>
#endif
constexpr static S mul128_u32(const S &a,
                              const S &b) noexcept {
	static_assert(sizeof(S::limb_type));
	(void)a;
	(void)b;
	return a;
}

///
/// \param lowbits
/// \param d
/// \return
constexpr static uint64_t mul128_from_u64(const uint64_t lowbits,
                                          const uint64_t d) noexcept {
	return ((__uint128_t)lowbits * d) >> 64;
}

///
/// \param lowbits
/// \param d
/// \return
constexpr static uint64_t mul128_s32(const uint64_t lowbits,
                                     const int32_t d) noexcept {
	if (d < 0) {
		return mul128_from_u64(lowbits, (int64_t)d) - lowbits;
	}

	return mul128_u32(lowbits, d);
}

/// This is for the 64-bit functions.
/// \param lowbits
/// \param d
/// \return
constexpr static uint64_t mul128_u64(const __uint128_t lowbits,
                              		 const uint64_t d) noexcept {
	__uint128_t bottom_half =
	        (lowbits & UINT64_C(0xFFFFFFFFFFFFFFFF)) * d; // Won't overflow
	// Only need the top 64 bits, as we'll shift the lower half away;
	bottom_half >>= 64;
	__uint128_t top_half = (lowbits >> 64) * d;
	__uint128_t both_halves =
	        bottom_half + top_half; // Both halves are already shifted down by 64
	both_halves >>= 64;         // Get top half of both_halves
	return (uint64_t)both_halves;
}


/// 
inline __uint128_t computeM_u64(uint64_t d) {
    // what follows is just ((__uint128_t)0 - 1) / d) + 1 spelled out
    __uint128_t M = UINT64_C(0xFFFFFFFFFFFFFFFF);
    M <<= 64;
    M |= UINT64_C(0xFFFFFFFFFFFFFFFF);
    M /= d;
    M += 1;
    return M;
}

inline uint64_t fastmod_u64(uint64_t a,
                     __uint128_t M,
                     uint64_t d) {
    __uint128_t lowbits = M * a;
    return mul128_u64(lowbits, d);
}

inline uint64_t fastdiv_u64(uint64_t a, __uint128_t M) {
    return mul128_u64(M, a);
}

// given precomputed M, is_divisible checks whether n % d == 0
inline bool is_divisible_u64(uint64_t n, __uint128_t M) { 
    return n * M <= M - 1; 
}


/**
 * Unsigned integers.
 * Usage:
 *  uint32_t d = ... ; // divisor, should be non-zero
 *  uint64_t M = computeM_u32(d); // do once
 *  fastmod_u32(a,M,d) is a % d for all 32-bit a.
 *
 **/

/// M = ceil( (1<<64) / d ), d > 0
/// \param d
/// \return
constexpr static uint64_t computeM_u32(const uint32_t d) noexcept {
	return UINT64_C(0xFFFFFFFFFFFFFFFF) / d + 1ull;
}

/// fastmod computes (a % d) given precomputed M
/// \param a
/// \param M
/// \param d
/// \return a % d
constexpr static uint32_t fastmod_u32(const uint32_t a,
                                      const uint64_t M,
                                      const uint32_t d) {
	uint64_t lowbits = M * a;
	return (uint32_t)(mul128_u32(lowbits, d));
}

/// fastdiv computes (a / d) given precomputed M for d>1
/// \param a
/// \param M
/// \return
constexpr static uint32_t fastdiv_u32(const uint32_t a,
                                      const uint64_t M) noexcept {
	return (uint32_t)(mul128_u32(M, a));
}

// given precomputed M, is_divisible checks whether n % d == 0
constexpr static bool is_divisible(const uint32_t n,
                                   const uint64_t M) noexcept {
	return n * M <= M - 1;
}

/**
 * signed integers
 * Usage:
 *  int32_t d = ... ; // should be non-zero and between [-2147483647,2147483647]
 *  int32_t positive_d = d < 0 ? -d : d; // absolute value
 *  uint64_t M = computeM_s32(d); // do once
 *  fastmod_s32(a,M,positive_d) is a % d for all 32-bit a.
 **/

// M = floor( (1<<64) / d ) + 1
// you must have that d is different from 0 and -2147483648
// if d = -1 and a = -2147483648, the result is undefined
constexpr static uint64_t computeM_s32(int32_t d) noexcept {
	if (d < 0) {
		d = -d;
	}
	return UINT64_C(0xFFFFFFFFFFFFFFFF) / d + 1 + ((d & (d - 1)) == 0 ? 1 : 0);
}

/// fastmod computes (a % d) given precomputed M,
/// you should pass the absolute value of d
/// \param a
/// \param M
/// \param positive_d
/// \return
constexpr int32_t fastmod_s32(const int32_t a,
                              const uint64_t M,
                              const int32_t positive_d) {
	uint64_t lowbits = M * a;
	int32_t highbits = (int32_t)mul128_u32(lowbits, positive_d);
	return highbits - ((positive_d - 1) & (a >> 31));
}

/// fastdiv computes (a / d) given a precomputed M, assumes that d must not
/// \param a
/// \param M
/// \param d
/// \return
constexpr static int32_t fastdiv_s32(const int32_t a,
                                     const uint64_t M,
                                     const int32_t d) noexcept {
	uint64_t highbits = mul128_s32(M, a);
	highbits += (a < 0 ? 1 : 0);
	if (d < 0) {
		return -(int32_t)(highbits);
    }
	return (int32_t)(highbits);
}

}; // end namespace cryptanalysislib::math::internal


/// \tparam d
/// \param x
/// \return
template <uint32_t d>
constexpr static uint32_t fastmod(const uint32_t x) noexcept {
	constexpr uint64_t v = cryptanalysislib::math::internal::computeM_u32(d);
	return cryptanalysislib::math::internal::fastmod_u32(x, v, d);
}

/// \tparam d
/// \param x
/// \return x/d
template <uint32_t d>
constexpr static uint32_t fastdiv(const uint32_t x) noexcept {
	static_assert(d != 0);
	// NOTE: the fastdiv trick requires d > 1
	if constexpr (d == 1) {
		return x;
	}
	constexpr uint64_t v = cryptanalysislib::math::internal::computeM_u32(d);
	return cryptanalysislib::math::internal::fastdiv_u32(x, v);
}

///
/// \tparam d
/// \param x
/// \return x/d
template <int32_t d>
constexpr static int32_t fastmod(const int32_t x) noexcept {
	static_assert(d != 0 && d != INT32_MIN);
	constexpr uint64_t v = cryptanalysislib::math::internal::computeM_s32(d);
	// NOTE: fastmod_s32 needs the absolute value of d
	constexpr int32_t positive_d = d < 0 ? -d : d;
	return cryptanalysislib::math::internal::fastmod_s32(x, v, positive_d);
}

/// \tparam d
/// \param x
/// \return x/d
template <int32_t d>
constexpr static int32_t fastdiv(const int32_t x) noexcept {
	static_assert(d != 0 && d != INT32_MIN);
	// NOTE: the fastdiv trick does not work for d = 1 and d = -1
	if constexpr (d == 1) {
		return x;
	} else if constexpr (d == -1) {
		return -x;
	}
	constexpr uint64_t v = cryptanalysislib::math::internal::computeM_s32(d);
	return cryptanalysislib::math::internal::fastdiv_s32(x, v, d);
}


namespace cryptanalysislib {
	/// \return a mod m in [0, m), also for negative a (requires m > 0)
	template<typename T>
	    requires std::is_integral_v<T>
	[[nodiscard]] constexpr T pmod(const T a, const T m) noexcept {
		const T r = a % m;
		if constexpr (std::is_signed_v<T>) {
			return r < 0 ? T(r + m) : r;
		}
		return r;
	}

	/// \return (a * b) mod m in [0, m), computed with a 128-bit
	///		intermediate, so it does not overflow for 64-bit T (requires m > 0)
	template<typename T>
	    requires std::is_integral_v<T> && (sizeof(T) <= 8)
	[[nodiscard]] constexpr T mulmod(const T a, const T b, const T m) noexcept {
		if constexpr (std::is_signed_v<T>) {
			__int128 r = (__int128(a) * __int128(b)) % __int128(m);
			return T(r < 0 ? r + m : r);
		} else {
			return T(((unsigned __int128)a * (unsigned __int128)b) % m);
		}
	}

	/// \return (a + b) mod m, without overflow (requires a, b in [0, m))
	template<typename T>
	    requires std::is_unsigned_v<T>
	[[nodiscard]] constexpr T addmod(const T a, const T b, const T m) noexcept {
		return (a >= (m - b)) ? T(a - (m - b)) : T(a + b);
	}

	/// \return (a - b) mod m, without underflow (requires a, b in [0, m))
	template<typename T>
	    requires std::is_unsigned_v<T>
	[[nodiscard]] constexpr T submod(const T a, const T b, const T m) noexcept {
		return (a >= b) ? T(a - b) : T(a + (m - b));
	}

	/// \return base^exp mod m in [0, m) (requires exp >= 0 and m > 0)
	template<typename T>
	    requires std::is_integral_v<T> && (sizeof(T) <= 8)
	[[nodiscard]] constexpr T mod_pow(T base, T exp, const T m) noexcept {
		T r = pmod<T>(T(1), m);
		base = pmod<T>(base, m);
		while (exp > 0) {
			if (exp & 1) {
				r = mulmod<T>(r, base, m);
			}
			base = mulmod<T>(base, base, m);
			exp >>= 1;
		}
		return r;
	}
} // end namespace cryptanalysislib

#endif
