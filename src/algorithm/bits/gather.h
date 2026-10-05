#pragma once

#ifdef USE_AVX
#include <immintrin.h>
#endif

// Return  word with bits of w collected as indicated by m:
// Example:
//  w = ABCDEFGH
//  m = 00101100
//  ==> 00000CEF
// This is the inverse of bit_scatter()
template<typename T>
constexpr static inline T bit_gather(T w, T m) noexcept {
#ifdef USE_AVX
    return _pdep_u64(w, m);
#endif
    T z = 0;
    T b = 1;
    while ( m ) {
        T i = m & -m;  // lowest bit
        m ^= i;  // clear lowest bit in m
        z += (i&w ? b : 0);
        b <<= 1;
    }
    return  z;
}

// Return  word with bits of w distributed as indicated by m:
// Example:
//  w = 00000ABC
//  m = 00101100
//  ==> 00A0BC00
// This is the inverse of bit_gather()
template<typename T>
constexpr static inline T bit_scatter(T w, T m) noexcept {
#ifdef USE_AVX
    return _pext_u64(w, m);
#endif
    T z = 0;
    T b = 1;
    while ( m ) {
        T i = m & -m;  // lowest bit
        m ^= i;
        z += (b&w ? i : 0);
        b <<= 1;
    }
    return  z;
}
