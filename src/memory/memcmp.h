#ifndef CRYPTANALYSISLIB_MEMCMP_H
#define CRYPTANALYSISLIB_MEMCMP_H

#include <cstdlib>
#include <cstdint>
#include <type_traits>

#include "simd/simd.h"

namespace cryptanalysislib {

#ifdef USE_AVX2
	bool memcmp_u256_u8(const uint8_t *__restrict__ a,
                        const uint8_t *__restrict__ b,
		                const size_t n) noexcept {
        using S = uint64x4_t;
        a += n;
        b += n;

        int64_t nn = -n;

        while (nn <= -32) {
            uint32_t t = S::load((uint64_t *)(a + nn)) == S::load((uint64_t *)(b + nn));
            t ^= 0xF;
            if (t) { return 1; }

            nn += 32;
        }

        
        using A = _uint64x2_t;
        if (nn <= -16) {
            uint32_t t = A::load(a + nn) == A::load(b + nn);
            t ^= 0xFFFF;
            if (t) { return 1; }

            nn += 16;
        }
        
        if (nn <= -8) {
            bool t = (*((uint64_t *)(a + nn))) == (*((uint64_t *)(b + nn)));
            if (t) { return 1; }
            nn += 8;
        }

        if (nn <= -4) {
            bool t = (*((uint32_t *)(a + nn))) == (*((uint32_t *)(b + nn)));
            if (t) { return 1; }
            nn += 4;
        }

        if (nn < -2) {
            bool t = (*((uint16_t *)(a + nn))) == (*((uint16_t *)(b + nn)));
            if (t) { return 1; }
            nn += 2;
        }

        while (nn != 0) {
            bool t = (*(a + nn)) == (*(b + nn));
            if (t) { return 1; }
            nn += 1; 
        }

        return 0;
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
