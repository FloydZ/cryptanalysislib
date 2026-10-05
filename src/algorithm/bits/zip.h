#pragma once 



// Return word with lower half bits in even indices
//  and upper half bits in odd indices.
template <typename T>
constexpr static inline T bit_zip(T x) noexcept {
    T y =  (x >> 32);
    x &= 0xffffffffUL;
    x = (x | (x<<16)) & 0x0000ffff0000ffffUL;
    y = (y | (y<<16)) & 0x0000ffff0000ffffUL;
    x = (x | (x<<8))  & 0x00ff00ff00ff00ffUL;
    y = (y | (y<<8))  & 0x00ff00ff00ff00ffUL;
    x = (x | (x<<4))  & 0x0f0f0f0f0f0f0f0fUL;
    y = (y | (y<<4))  & 0x0f0f0f0f0f0f0f0fUL;
    x = (x | (x<<2))  & 0x3333333333333333UL;
    y = (y | (y<<2))  & 0x3333333333333333UL;
    x = (x | (x<<1))  & 0x5555555555555555UL;
    y = (y | (y<<1))  & 0x5555555555555555UL;
    x |= (y<<1);
    return  x;
}

// Return word with even indexed bits in lower half
//  and odd indexed bits in upper half.
// Inverse of bit_zip()
template <typename T>
constexpr static inline T bit_unzip(T x) noexcept {
    T y = (x >> 1) & 0x5555555555555555UL;
    x &= 0x5555555555555555UL;
    x = (x | (x>>1))  & 0x3333333333333333UL;
    y = (y | (y>>1))  & 0x3333333333333333UL;
    x = (x | (x>>2))  & 0x0f0f0f0f0f0f0f0fUL;
    y = (y | (y>>2))  & 0x0f0f0f0f0f0f0f0fUL;
    x = (x | (x>>4))  & 0x00ff00ff00ff00ffUL;
    y = (y | (y>>4))  & 0x00ff00ff00ff00ffUL;
    x = (x | (x>>8))  & 0x0000ffff0000ffffUL;
    y = (y | (y>>8))  & 0x0000ffff0000ffffUL;
    x = (x | (x>>16)) & 0x00000000ffffffffUL;
    y = (y | (y>>16)) & 0x00000000ffffffffUL;
    x |= (y<<32);
    return  x;
}

// Return word with lower half bits in even indices.
// upper half must be zero.
// Same effect as bit_zip() but faster.
// 0000abcd --> 0a0b0c0d (a,b,c,d are bits).
template <typename T>
constexpr static inline T bit_zip0(T x) noexcept {
    x = (x | (x<<16)) & 0x0000ffff0000ffffUL;
    x = (x | (x<<8))  & 0x00ff00ff00ff00ffUL;
    x = (x | (x<<4))  & 0x0f0f0f0f0f0f0f0fUL;
    x = (x | (x<<2))  & 0x3333333333333333UL;
    x = (x | (x<<1))  & 0x5555555555555555UL;
    return  x;
}

// Gather bits in even positions into lower half.
// Inverse of bit_zip0().
// Bits at odd positions must be zero.
// 0a0b0c0d --> 0000abcd (a,b,c,d are bits).
template <typename T>
constexpr static inline T bit_unzip0(T x) noexcept {
    x = (x | (x>>1))  & 0x3333333333333333UL;
    x = (x | (x>>2))  & 0x0f0f0f0f0f0f0f0fUL;
    x = (x | (x>>4))  & 0x00ff00ff00ff00ffUL;
    x = (x | (x>>8))  & 0x0000ffff0000ffffUL;
    x = (x | (x>>16)) & 0x00000000ffffffffUL;
    return  x;
}

#define  BPLH  (sizeof(T)*8/2)

// Bits of lower half word spread out into even positions of lo,
// bits of upper half word spread out into even positions of hi.
template <typename T>
constexpr static inline void bit_zip2(T x, T &lo, T &hi) noexcept {
    x = bit_zip(x);
    lo = x & 0x5555555555555555UL;
    hi = (x>>1) & 0x5555555555555555UL;
}

// Inverse of bit_zip2(x, lo, hi).
template <typename T>
constexpr static inline T bit_unzip2(T lo, T hi) noexcept {
    return  bit_unzip( (hi<<1) | lo  );
}

// 2-word version:
// only the lower half of x and y are merged
template <typename T>
static inline T bit_zip2(T x, T y) {
    return  bit_zip( (y<<BPLH) + x );
}

// 2-word version:
// only the lower half of x and y are filled
template <typename T>
static inline void bit_unzip2(T t, T &x, T &y) {
    t = bit_unzip(t);
    y = t >> BPLH;

    x = t & 0x00000000ffffffffUL;
}

// Return word with lower half bits(-pairs) in even (pair-)indices,
// i.e., indices 4k+0, 4k+1.
// Upper half must be zero.
// 0000abcd --> 0a0b0c0d (a,b,c,d are pairs of bits).
template<typename T>
constexpr static inline T bit_zip0_pairs(T x) {
    x = (x | (x<<16)) & 0x0000ffff0000ffffUL;
    x = (x | (x<<8))  & 0x00ff00ff00ff00ffUL;
    x = (x | (x<<4))  & 0x0f0f0f0f0f0f0f0fUL;
    x = (x | (x<<2))  & 0x3333333333333333UL;
//    x = (x | (x<<1))  & 0x5555555555555555UL;
    return  x;
}

// Gather pairs bits in even positions (4k+0, 4k+1) into lower half.
// Inverse of bit_zip0_pairs().
// Bit pairs at odd (pair-)positions (4k+2, 4k+3) must be zero.
// 0a0b0c0d --> 0000abcd (a,b,c,d are pairs of bits).
template<typename T>
static inline T bit_unzip0_pairs(T x) {
//    x = (x | (x>>1))  & 0x3333333333333333UL;
    x = (x | (x>>2))  & 0x0f0f0f0f0f0f0f0fUL;
    x = (x | (x>>4))  & 0x00ff00ff00ff00ffUL;
    x = (x | (x>>8))  & 0x0000ffff0000ffffUL;
    x = (x | (x>>16)) & 0x00000000ffffffffUL;

    return  x;
}

