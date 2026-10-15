#ifndef CRYPTANALYSISLIB_MEMCMP_H
#define CRYPTANALYSISLIB_MEMCMP_H

#include <concepts>
#include <cstdlib>
#include <cstdint>
#include <type_traits>

#include "simd/simd.h"

namespace cryptanalysislib {

#ifdef USE_AVX2
    /// compares n bytes, the main loop processes `sizeof(S)` bytes per step.
    /// \tparam S SIMD type whose `eq` returns a bitmask with one bit per
    ///     lane, e.g. `uint64x4_t` (default), `uint8x32_t`, `uint64x8_t`
    /// \param a
    /// \param b
    /// \param n number of bytes
    /// \return true if they differ
    template<typename S = uint64x4_t>
#if __cplusplus > 201709L
        requires requires(const typename S::limb_type *p) {
            S::LIMBS;
            { S::eq(S::load(p), S::load(p)) } -> std::convertible_to<uint64_t>;
        }
#endif
	inline bool memcmp_simd_u8(const uint8_t *__restrict__ a,
                               const uint8_t *__restrict__ b,
		                       const size_t n) noexcept {
        constexpr int64_t bytes = sizeof(S);
        // one bit per lane
        constexpr uint64_t full = (S::LIMBS >= 64) ? ~0ull : ((1ull << S::LIMBS) - 1ull);
        a += n;
        b += n;

        int64_t nn = -n;

        while (nn <= -bytes) {
            const uint64_t t = S::eq(S::load((const typename S::limb_type *)(a + nn)),
                                     S::load((const typename S::limb_type *)(b + nn)));
            if (t != full) { return 1; }

            nn += bytes;
        }

        using A = _uint64x2_t;
        while (nn <= -16) {
            // one bit per 64-bit lane
            const uint32_t t = A::eq(A::load((A::limb_type *)(a + nn)), A::load((A::limb_type *)(b + nn)));
            if (t != 0b11u) { return 1; }

            nn += 16;
        }

        // NOTE: the tails are loaded via memcpy, as `a`/`b` are not aligned
        if (nn <= -8) {
            uint64_t x, y;
            __builtin_memcpy(&x, a + nn, 8); __builtin_memcpy(&y, b + nn, 8);
            if (x != y) { return 1; }
            nn += 8;
        }

        if (nn <= -4) {
            uint32_t x, y;
            __builtin_memcpy(&x, a + nn, 4); __builtin_memcpy(&y, b + nn, 4);
            if (x != y) { return 1; }
            nn += 4;
        }

        if (nn <= -2) {
            uint16_t x, y;
            __builtin_memcpy(&x, a + nn, 2); __builtin_memcpy(&y, b + nn, 2);
            if (x != y) { return 1; }
            nn += 2;
        }

        while (nn != 0) {
            if (*(a + nn) != *(b + nn)) { return 1; }
            nn += 1; 
        }

        return 0;
    }

    /// 256 bit version, see `memcmp_simd_u8`
	inline bool memcmp_u256_u8(const uint8_t *__restrict__ a,
                               const uint8_t *__restrict__ b,
		                       const size_t n) noexcept {
        return memcmp_simd_u8<uint64x4_t>(a, b, n);
    }
#endif

	/// \tparam T type
	/// \param a 
	/// \param b 
	/// \param len number of elements NOT byts
	template<typename T>
	constexpr bool memcmp(const T *a,
                          const T *b, 
	                      const size_t len) noexcept {
#ifdef USE_AVX2 
        return memcmp_u256_u8((uint8_t *)a, (uint8_t *)b, len * sizeof(T));
#endif 
        // fallback impl
        for (size_t i = 0; i < len; i++) {
            if (a[i] != b[i]) {
                return 1;
            }
        }

        return 0;

    }
} // end namespace cryptanalysislib
#endif//CRYPTANALYSISLIB_MEMCMP_H
