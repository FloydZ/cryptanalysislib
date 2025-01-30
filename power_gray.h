#pragma once 

#include <cstdint>

template<typename T>
class PowerGray {
    constexpr static uint32_t BITS = sizeof(T) * 8u;

    // Return (gray_code**e)(x)
    // gray_pow(x, 1) == gray_code(x)
    // gray_pow(x, BITS_PER_LONG-1) == inverse_gray_code(x)
    constexpr static inline T gray_pow(T x, T e) noexcept {
        e &= (BITS-1);  // modulo BITS_PER_LONG
        T s = 1;
        while ( e )
        {
            if ( e & 1 )  x ^= x >> s;  // gray ** s
            s <<= 1;
            e >>= 1;
        }
        return  x;
    }
    
    // Return (inverse_gray_code**(e))(x)
    //   == (gray_code**(-e))(x)
    // inverse_gray_pow(x, 1) == inverse_gray_code(x)
    // inverse_gray_pow(x, BITS_PER_LONG-1) == gray_code(x)
    constexpr static inline T inverse_gray_pow(T x, T e) {
        return  gray_pow(x, -e);
    }
    
    // Return (rev_gray_code**e)(x)
    // rev_gray_pow(x, 1) == rev_gray_code(x)
    // rev_gray_pow(x, BITS_PER_LONG-1) == inverse_rev_gray_code(x)
    constexpr static inline T rev_gray_pow(T x, T e) {
        e &= (BITS-1);  // modulo BITS_PER_LONG
        T s = 1;
        while ( e )
        {
            if ( e & 1 )  x ^= x << s;  // rev_gray ** s
            s <<= 1;
            e >>= 1;
        }
        return  x;
    }
    
    // Return (inverse_rev_gray_code**(e))(x)
    //   == (rev_gray_code**(-e))(x)
    // inverse_rev_gray_pow(x, 1) == inverse_rev_gray_code(x)
    // inverse_rev_gray_pow(x, BITS_PER_LONG-1) == rev_gray_code(x)
    constexpr static inline T inverse_rev_gray_pow(T x, T e) {
        return  rev_gray_pow(x, -e);
    }
};
