#pragma once

#include <cstdint>

#include "combination/enumerator.h"

/// Class for computing the Golay-Rudin-Shapiro sequence (OEIS A020985)
/// The sequence determines whether the GRS value is negative at a given index
/// As an enumerator: grs_negative_q(k) for k = 0, 1, 2, ...
class gsr_negative {
    /// current index
    uint64_t k = 0;
    /// grs_negative_q(k)
    uint64_t g = 0;

public:
    /// \return grs_negative_q(index())
    [[nodiscard]] constexpr inline uint64_t current() const noexcept {
        return g;
    }

    /// \return the current index
    [[nodiscard]] constexpr inline uint64_t index() const noexcept {
        return k;
    }

    /// moves to the next index
    /// \return false at the index 2^64 - 1 (unchanged)
    constexpr inline bool next() noexcept {
        if (k == ~uint64_t(0)) { return false; }
        g = grs_next(k, g);
        k += 1;
        return true;
    }

    /// moves to the previous index
    /// \return false at the index 0 (unchanged)
    constexpr inline bool previous() noexcept {
        if (k == 0) { return false; }
        k -= 1;
        g = grs_negative_q(k);
        return true;
    }

/// Determines whether the Golay-Rudin-Shapiro sequence is negative for a given index
/// Returns 1 for indices where the GRS value is negative
/// Examples of indices returning 1:
///   3,6,11,12,13,15,19,22,24,25,26,30,35,38,43,44,45,47,48,49,
///   50,52,53,55,59,60,61,63,67,70,75,76,77,79,83,86,88,89,90,94,
///   96,97,98,100,101,103,104,105,106,110,115,118,120,121,122,
///   126,131,134,139,140, ...
/// Algorithm: counts bit pairs modulo 2
/// \param x[in]: index to check
/// \return 1 if GRS value is negative at index x, 0 otherwise
constexpr static inline uint64_t grs_negative_q(uint64_t x)
{
    return  __builtin_parityll( x & (x>>1) );
}
// -------------------------


/// Computes the next value in the GRS sequence
/// Given g = grs_negative_q(k), computes grs_negative_q(k+1) efficiently
/// \param k[in]: current index
/// \param g[in]: current GRS value at index k
/// \return GRS value at index k+1
constexpr static inline uint64_t grs_next(uint64_t k, uint64_t g)
{
    // NOTE: `BITS_PER_LONG` is not defined here (the 32 bit mask was always
    //  used), the arguments are 64 bit
    const uint64_t cm = 0x5555555555555554UL;
    uint64_t h = ~k;  h &= -h;  // == lowest_zero(k);
    g ^= ( ((h&cm) ^ ((k>>1)&h)) !=0 );
    return  g;
}
// -------------------------
};
