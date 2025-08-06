#pragma once 

#include <stdint.h>


/// Class for generating and working with lexicographically ordered subsets
/// Provides utilities to enumerate, convert, and analyze lexicographic representations
class enumeration_lexicographic {

    /// Computes the next word in subset-lexrev order
    /// Start with a one-bit word at position n-1 to generate 2**n subsets of length n
    /// 
    /// Example for n==4 with subsets interpretation:
    ///     word   subset of {0,1,2,3}
    ///     1...   {0}
    ///     11..   {0, 1}
    ///     111.   {0, 1, 2}
    ///     1111   {0, 1, 2, 3}
    ///     11.1   {0, 1, 3}
    ///     1.1.   {0, 2}
    ///     1.11   {0, 2, 3}
    ///     1..1   {0, 3}
    ///     .1..   {1}
    ///     .11.   {1, 2}
    ///     .111   {1, 2, 3}
    ///     .1.1   {1, 3}
    ///     ..1.   {2}
    ///     ..11   {2, 3}
    ///     ...1   {3}
    ///     ....   {}
    /// 
    /// Note (1): The first element of the subset corresponds to the highest set bit.
    ///           When interpreting the binary words via "bit(n)==element n" (as usual),
    ///           the order would be:
    ///     1...           { 3 }
    ///     11..        { 2, 3 }
    ///     111.     { 1, 2, 3 }
    ///     1111  { 0, 1, 2, 3 }
    ///     11.1     { 0, 2, 3 }
    ///     1.1.        { 1, 3 }
    ///     1.11     { 0, 1, 3 }
    ///     1..1        { 0, 3 }
    ///     .1..           { 2 }
    ///     .11.        { 1, 2 }
    ///     .111     { 0, 1, 2 }
    ///     .1.1        { 0, 2 }
    ///     ..1.           { 1 }
    ///     ..11        { 0, 1 }
    ///     ...1           { 0 }
    /// 
    /// Note (2): The lex order for the delta sets would simply be the counting order
    ///           (of the words or reversed words depending on the interpretation as
    ///           explained above).
    /// 
    /// \param x[in]: current word
    /// \return next word in subset-lexrev order
    [[nodiscard]] constexpr static inline ulong next_lexrev(uint64_t x) noexcept {
        ulong x0 = x & -x;  // lowest one
        if ( 1 != x0 ) {  // easy case: set bit right of lowest one
            x0 >>= 1;
            x ^= x0;
            return  x;
        } else  { // lowest one at word end
            x ^= 1;  // clear lowest one
    
            x0 = x & -x;  // new lowest one ...
            x0 >>= 1;  x -= x0;  // ... is moved one to the right
            return  x;
        }
    }

    /// Computes the previous word in subset-lexrev order
    /// Start with zero and use 2**n calls to generate 2**n subsets of length n
    /// 
    /// Example for n==4:
    ///  ....  =  0
    ///  ...1  =  1
    ///  ..11  =  3
    ///  ..1.  =  2
    ///  .1.1  =  5
    ///  .111  =  7
    ///  .11.  =  6
    ///  .1..  =  4
    ///  1..1  =  9
    ///  1.11  = 11
    ///  1.1.  = 10
    ///  11.1  = 13
    ///  1111  = 15
    ///  111.  = 14
    ///  11..  = 12
    ///  1...  =  8
    /// 
    /// \param x[in]: current word
    /// \return previous word in subset-lexrev order
    [[nodiscard]] constexpr static inline 
    uint64_t prev_lexrev(uint64_t x) noexcept {
        uint64_t x0 = x & -x;  // lowest one
        if ( x & (x0<<1) ) { // easy case: next higher bit is set
            x ^= x0;  // clear lowest one
            return x;
        } else {
            x += x0;  // move lowest one to the left
            x |= 1;   // set rightmost bit
            return x;
        }
    }
    
    /// Converts a negative index to a lexicographic-reverse representation
    /// Example conversions:
    ///   k:  negidx2lexrev(k)
    ///   0:  .....
    ///   1:  ....1
    ///   2:  ...11
    ///   3:  ...1.
    ///   4:  ..1.1
    ///   5:  ..111
    ///   6:  ..11.
    ///   7:  ..1..
    ///   8:  .1..1
    ///   9:  .1.11
    ///  10:  .1.1.
    ///  11:  .11.1
    ///  12:  .1111
    ///  13:  .111.
    ///  14:  .11..
    ///  15:  .1...
    ///  16:  1...1
    /// \param k[in]: negative index to convert
    /// \return lexicographic-reverse representation
    [[nodiscard]] constexpr static inline 
    uint64_t negidx2lexrev(uint64_t k) noexcept {
        ulong z = 0;
        ulong h = highest_one(k);
        while ( k ) {
            while ( 0 == (h & k) )  h >>= 1;
            z ^= h;
            ++k;
            k &= h - 1;
        }
    
        return  z;
    }
    
    /// Converts a lexicographic-reverse representation to a negative index
    /// Inverse of negidx2lexrev()
    /// \param x[in]: lexicographic-reverse representation to convert
    /// \return the corresponding negative index
    [[nodiscard]] constexpr static inline uint64_t lexrev2negidx(uint64_t x) noexcept {
        if ( 0==x )  return 0;
        ulong h = x & -x;  // lowest one
        ulong r = (h-1);
        while ( x^=h ) {
            r += (h-1);
            h = x & -x;  // next higher one
        }
        r += h;  // highest one
        return  r;
    }
    
    
    /// Determines if x is a fixed point in the prev_lexrev() sequence
    /// A fixed point is a value that remains unchanged when prev_lexrev() is applied
    /// \param x[in]: value to check
    /// \return true if x is a fixed point, false otherwise
    [[nodiscard]] constexpr static inline bool is_lexrev_fixed_point(uint64_t x) noexcept {
        if (x & 1) {
            return  1 == x;
        }
    
        const uint64_t w = __builtin_popcntll(x);
        if (w != (w & -w)) {
            return  false;
        }

        if (0==x) {
            return  true;
        }

        return  0 != ((x & -x) & w);
    }
};
