#pragma once

#include <cstdint>

#include "combination/enumerator.h"

/// Class for generating and manipulating reverse Gray codes
/// Provides utilities for enumeration and conversion between binary and reverse Gray code
/// \tparam T[in]: integer type to represent code values
/// As an enumerator: 1, rev_gray_code(1), rev_gray_code(rev_gray_code(1)), ...,
/// i.e. the powers (1+x)^i, i = 0, ..., bits(T) - 1, as binary polynomials
/// mod x^bits(T). (1+x)^bits(T) == 1, so the sequence has bits(T) elements.
template<typename T>
class enumeration_reverse_gray_code {
private:
    /// Current value in the enumeration
    T val = 1;
    /// exponent of the current value
    uint32_t i = 0;

    /// Computes the reversed Gray code of x
    /// Performs a bit-wise derivative modulo 2 towards high bits
    /// 
    /// Mathematical interpretations:
    /// - Multiplication by x+1 as binary polynomial
    /// - Returns x^2+x in binary normal basis
    /// - rev_gray_code(x) == revbin( gray_code( revbin(x) ) )
    ///
    /// \param x[in]: value to convert to reverse Gray code
    /// \return the reverse Gray code of x
    constexpr static inline T rev_gray_code(const T x) noexcept {
        return  x ^ (x<<1);
    }

    /// Computes the inverse of the reverse Gray code
    /// The returned value contains at each bit position the parity
    /// of all bits of the input right from it (including itself)
    /// 
    /// Mathematical interpretations:
    /// - Division by x+1 as powers series over GF(2)
    /// - Returns solution of z = x^2+x in binary normal basis
    /// - inverse_rev_gray_code(x) == revbin( inverse_gray_code( revbin(x) ) )
    ///
    /// \param x[in]: reverse Gray code value to convert back
    /// \return the binary value whose reverse Gray code is x
    constexpr static inline T inverse_rev_gray_code(T x) noexcept {
        // use: rev_gray ** BITSPERLONG == id:
        // rev_gray ** (1 + 2 + 4 + ... + bits/2) == rev_gray ** (bits - 1)
        x ^= x<<1;  // rev_gray ** 1
        x ^= x<<2;  // rev_gray ** 2
        x ^= x<<4;  // rev_gray ** 4
        if constexpr (sizeof(T) >= 2) { x ^= x<<8;  } // rev_gray ** 8
        if constexpr (sizeof(T) >= 4) { x ^= x<<16; } // rev_gray ** 16
        if constexpr (sizeof(T) >= 8) { x ^= x<<32; } // rev_gray ** 32
        return  x;
    }
public:
    /// Default constructor initializes the enumeration
    constexpr enumeration_reverse_gray_code() noexcept {};

    /// \return the current value
    [[nodiscard]] constexpr inline T current() const noexcept {
        return val;
    }

    /// Advances to the next value in the reverse Gray code sequence
    /// \return false if the current value is the last one (unchanged)
    constexpr inline bool next() noexcept {
        if (i == sizeof(T) * 8u - 1u) { return false; }
        val = rev_gray_code(val);
        i += 1;
        return true;
    }

    /// Moves to the previous value in the reverse Gray code sequence
    /// \return false if the current value is the first one (1, unchanged)
    constexpr inline bool previous() noexcept {
        if (i == 0) { return false; }
        val = inverse_rev_gray_code(val);
        i -= 1;
        return true;
    }
};
