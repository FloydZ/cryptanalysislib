#pragma once 

#include <cstdint>

/// Return number of bit blocks.
/// E.g.:
/// ..1..11111...111.  -> 3
/// ...1..11111...111  -> 3
/// ......1.....1.1..  -> 3
/// .........111.1111  -> 2
template<typename T=uint64_t>
constexpr static inline T bit_block_count(T x) noexcept {
    return  (x & 1) + __builtin_popcount( (x^(x>>1)) ) / 2;
}


/// Return number of bit blocks with at least 2 bits.
/// E.g.:
/// ..1..11111...111.  -> 2
/// ...1..11111...111  -> 2
/// ......1.....1.1..  -> 0
/// .........111.1111  -> 2
template<typename T=uint64_t>
constexpr static inline T bit_block_ge2_count(T x) noexcept {
    return  bit_block_count( ( x & (x<<1)) | ( x & (x>>1)) );
}
