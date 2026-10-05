#pragma once

/// Class for computing the Golay-Rudin-Shapiro sequence (OEIS A020985)
/// The sequence determines whether the GRS value is negative at a given index
class gsr_negative {

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
static inline ulong grs_negative_q(ulong x)
{
    return  parity( x & (x>>1) );
}
// -------------------------


/// Computes the next value in the GRS sequence
/// Given g = grs_negative_q(k), computes grs_negative_q(k+1) efficiently
/// \param k[in]: current index
/// \param g[in]: current GRS value at index k
/// \return GRS value at index k+1
static inline ulong grs_next(ulong k, ulong g)
{
#if BITS_PER_LONG > 32
    const ulong cm = 0x5555555555555554UL;  // 64-bit version
#else
    const ulong cm = 0x55555554UL;
#endif
    ulong h = ~k;  h &= -h;  // == lowest_zero(k);
    g ^= ( ((h&cm) ^ ((k>>1)&h)) !=0 );
    return  g;
}
// -------------------------
};
