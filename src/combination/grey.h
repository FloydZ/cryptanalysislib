#pragma once 

#include <cstdint>
#include "bit_subset.h"

/// Class providing Gray code enumeration and utilities
///
///   0:   .......    .......   ...... .   ...... .
///   1:   ......1    .....1k   .....1 1   .....+ 1
///   2:   .....11    ....11.   ....11 .   ....+1 .
///   3:   .....1.    ....1.1   ....1. 1   ....1- 1
///   4:   ....11.    ...11..   ...11. .   ...+1. .
///   5:   ....111    ...1111   ...111 1   ...11+ 1
///   6:   ....1.1    ...1.1.   ...1.1 .   ...1-1 .
///   7:   ....1..    ...1..1   ...1.. 1   ...1.- 1
///   8:   ...11..    ..11...   ..11.. .   ..+1.. .
///   9:   ...11.1    ..11.11   ..11.1 1   ..11.+ 1
///  10:   ...1111    ..1111.   ..1111 .   ..11+1 .
///  11:   ...111.    ..111.1   ..111. 1   ..111- 1
///  12:   ...1.1.    ..1.1..   ..1.1. .   ..1-1. .
///  13:   ...1.11    ..1.111   ..1.11 1   ..1.1+ 1
///  14:   ...1..1    ..1..1.   ..1..1 .   ..1.-1 .
///  15:   ...1...    ..1...1   ..1... 1   ..1..- 1
///  16:   ..11...    .11....   .11... .   .+1... .
///  17:   ..11..1    .11..11   .11..1 1   .11..+ 1
/// \tparam T[in]: type to use for representing the codes
template<class T>
class enumeration_gray {

    /// Converts a binary number to its Gray code representation
    /// (Performs a 'bit-wise derivative modulo 2')
    /// \param x[in]: binary number to convert
    /// \return Gray code representation of x
    constexpr static inline T gray_code(T x) noexcept {
        return  x ^ (x>>1);
    }
    
    /// Converts a Gray code back to its binary representation
    /// Inverse of gray_code()
    /// Note: the returned value contains at each bit position
    /// the parity of all bits of the input left from it (including itself)
    /// \param x[in]: Gray code to convert
    /// \return binary representation of the Gray code
    constexpr static inline T inverse_gray_code(T x) noexcept {
        // ----- VERSION 1 (integration modulo 2):
        //    T h=1, r=0;
        //    do
        //    {
        //        if ( x & 1 )  r^=h;
        //        x >>= 1;
        //        h = (h<<1)+1;
        //    }
        //    while ( x!=0 );
        //    return r;
        
        // ----- VERSION 2 (apply graycode BITS_PER_LONG-1 times):
        //    T r = BITS_PER_LONG;
        //    while ( --r )  x ^= x>>1;
        //    return x;
        
        // ----- VERSION 3 (use: gray ** BITSPERLONG == id):
        x ^= x>>1;  // gray ** 1
        x ^= x>>2;  // gray ** 2
        x ^= x>>4;  // gray ** 4
        x ^= x>>8;  // gray ** 8
        x ^= x>>16;  // gray ** 16
        // here: x = gray**31(input)
        // note: the statements can be reordered at will
    
        x ^= x>>32;  // for 64bit words
    
        return  x;
    }
    
    /// Performs Gray code transformation on each byte of the input in parallel
    /// \param x[in]: value to transform
    /// \return Gray code representation of each byte in parallel
    constexpr static inline T byte_gray_code(T x) noexcept {
        return  x ^ ((x & 0xfefefefefefefefeUL)>>1);
    }
    
    /// Performs inverse Gray code transformation on each byte of the input in parallel
    /// \param x[in]: Gray code value to transform
    /// \return binary representation of each byte in parallel
    constexpr static inline T byte_inverse_gray_code(T x) noexcept {
        x ^= ((x & 0xfefefefefefefefeUL)>>1);
        x ^= ((x & 0xfcfcfcfcfcfcfcfcUL)>>2);
        x ^= ((x & 0xf0f0f0f0f0f0f0f0UL)>>4);
        return  x;
    }


    /// Computes the next Gray code in a special sequence
    /// With input x==gray_code(2*k) the return is gray_code(2*k+2).
    /// Let x1 be the word x shifted right once
    /// and i1 its inverse Gray code.
    /// Let r1 be the return r shifted right once.
    /// Then r1 = gray_code(i1+1).
    /// That is, we have a Gray code counter.
    /// The argument must have an even number of bits.
    ///
    ///   k:     g(k)      g(2*k)     g(k) p
    ///   0:   .......    .......   ...... .   ...... .
    ///   1:   ......1    .....1k   .....1 1   .....+ 1
    ///   2:   .....11    ....11.   ....11 .   ....+1 .
    ///   3:   .....1.    ....1.1   ....1. 1   ....1- 1
    ///   4:   ....11.    ...11..   ...11. .   ...+1. .
    ///   5:   ....111    ...1111   ...111 1   ...11+ 1
    ///   6:   ....1.1    ...1.1.   ...1.1 .   ...1-1 .
    ///   7:   ....1..    ...1..1   ...1.. 1   ...1.- 1
    ///   8:   ...11..    ..11...   ..11.. .   ..+1.. .
    ///   9:   ...11.1    ..11.11   ..11.1 1   ..11.+ 1
    ///  10:   ...1111    ..1111.   ..1111 .   ..11+1 .
    ///  11:   ...111.    ..111.1   ..111. 1   ..111- 1
    ///  12:   ...1.1.    ..1.1..   ..1.1. .   ..1-1. .
    ///  13:   ...1.11    ..1.111   ..1.11 1   ..1.1+ 1
    ///  14:   ...1..1    ..1..1.   ..1..1 .   ..1.-1 .
    ///  15:   ...1...    ..1...1   ..1... 1   ..1..- 1
    ///  16:   ..11...    .11....   .11... .   .+1... .
    ///  17:   ..11..1    .11..11   .11..1 1   .11..+ 1
    ///
    /// Note that the changes with increment always
    /// happen one position left of the rightmost bit.
    ///
    /// Convert an arbitrary (Gray code) word g to
    ///   x = (g<<1) ^ parity(g)
    /// in order to use this routine.
    /// \param x[in]: current Gray code (must have even number of bits)
    /// \return next Gray code in the sequence
    constexpr static inline T next_gray2(T x) noexcept {
        x ^= 1;
        x ^= (lowest_one(x) << 1);
        return x;
    }
};


/// Class for performing Gray code operations on arrays of values
/// \tparam T[in]: type to use for array elements
template<class T>
class enumeration_gray_array {
private:
    T val = 0; //first_comb();
    /// Applies Gray code transformation to an array
    /// \param f[in,out]: array to transform
    /// \param n[in]: size of the array
    constexpr inline void word_gray(T *f, T n) noexcept {
        for (T k=0;  k<n-1;  ++k)  f[k] ^= f[k+1];
    }
    // -------------------------
    
    /// Applies inverse Gray code transformation to an array
    /// \param f[in,out]: array to transform
    /// \param n[in]: size of the array
    constexpr inline void inverse_word_gray(T *f, T n) noexcept {
        T x = 0,  k = n;
        while ( k-- )  { x ^= f[k];  f[k] = x; }
    }
    
    /// Applies the word_gray transformation x times efficiently
    /// Result is identical to: for (T k=0; k<x; ++k) word_gray(f, n);
    /// but with work <= n/2
    /// \param f[in,out]: array to transform
    /// \param n[in]: size of the array
    /// \param x[in]: number of times to apply the transformation
    void word_gray_pow(T *f, T n, T x) {
        for (uint32_t s=1; s<n; s*=2) {
            if ( x & 1 ) {
                // word_gray ** s:
                for (uint32_t k=0, j=k+s;  j<n;  ++k, ++j)  f[k] ^= f[j];
            }
            x >>= 1;
        }
    }
    
    /// Applies reverse Gray code transformation to an array
    /// \param f[in,out]: array to transform
    /// \param n[in]: size of the array
    void word_rev_gray(T *f, T n) {
        for (uint32_t k=n-1; 0!=k; --k)  f[k] ^= f[k-1];
    }
    
    /// Applies inverse reverse Gray code transformation to an array
    /// \param f[in,out]: array to transform
    /// \param n[in]: size of the array
    void inverse_word_rev_gray(T *f, T n) {
        uint32_t x = 0;
        for (uint32_t k=0;  k<n; ++k)  { 
            x ^= f[k];
            f[k] = x; 
        }
    }
    
    /// Applies the reverse Gray code transformation x times efficiently
    /// Result is identical to: for (T k=0; k<x; ++k) word_rev_gray(f, n);
    /// but with work <= n/2
    /// \param f[in,out]: array to transform
    /// \param n[in]: size of the array
    /// \param x[in]: number of times to apply the transformation
    void word_rev_gray_pow(T *f, T n, T x) {
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
