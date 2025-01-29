#pragma once 

#include <cstdint>
#include "bit_subset.h"

template<class T>
class enumeration_gray {
private:
    T val = 0; //first_comb();
    constexpr inline void word_gray(T *f, ulong n) noexcept {
        for (ulong k=0;  k<n-1;  ++k)  f[k] ^= f[k+1];
    }
    // -------------------------
    
    constexpr inline void inverse_word_gray(T *f, ulong n) noexcept {
        ulong x = 0,  k = n;
        while ( k-- )  { x ^= f[k];  f[k] = x; }
    }
    
    // result is identical to
    //   for (ulong k=0; k<x; ++k)  word_gray(f, n);
    // Work <= n/2
    void word_gray_pow(T *f, ulong n, ulong x) {
        for (uint32_t s=1; s<n; s*=2) {
            if ( x & 1 ) {
                // word_gray ** s:
                for (uint32_t k=0, j=k+s;  j<n;  ++k, ++j)  f[k] ^= f[j];
            }
            x >>= 1;
        }
    }
    
    void word_rev_gray(T *f, ulong n) {
        for (uint32_t k=n-1; 0!=k; --k)  f[k] ^= f[k-1];
    }
    
    void inverse_word_rev_gray(T *f, ulong n) {
        uint32_t x = 0;
        for (uint32_t k=0;  k<n; ++k)  { 
            x ^= f[k];
            f[k] = x; 
        }
    }
    
    /// result is identical to
    ///   for (ulong k=0; k<x; ++k)  word_rev_gray(f, n);
    /// work <= n/2
    void word_rev_gray_pow(T *f, ulong n, ulong x) {
        x &= (n-1);  // modulo n
        for (uint32_t s=1; s<n; s*=2) {
            if ( x & 1) {
                // word_rev_gray ** s:
                for (uint32_t k=n-1, j=k-s;  k>=s;  --k, --j)  f[k] ^= f[j];
            }
            x >>= 1;
        }
    }
};
