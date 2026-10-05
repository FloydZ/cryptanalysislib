#pragma once 



// Unranking for binary SL-Gray:
// Convert binary number to corresponding word in SL-Gray order.
// Successive transitions are adjacent (one-close) or three-close.
// See Joerg Arndt, Subset-lex: did we miss an order?, (2014)
//   http://arxiv.org/abs/1405.6503
template<typename T>
constexpr static inline T bin_to_sl_gray(T k, T ldn) {
    if ( ldn==0 )  return 0;

    T b = 1UL << (ldn-1);  // highest bit
    T m = (b<<1) - 1;  // mask for reversing direction
    T z = b;  // Gray code
    k -= 1;  // move all-zero word to begin

    while ( b != 0 )
    {
        const T h = k & b;  // bit under consideration
        z ^= h;  // with one, switch bit in Gray code

        if ( !h )  k ^= m;  // reverse direction with zero

        k += 1;  // SL-Gray

        b >>= 1;  // next lower bit
        m >>= 1;  // next smaller mask
    }

    return z;
}

// Ranking for binary SL-Gray:
// Convert binary word in SL-Gray order to binary number.
// See Joerg Arndt, Subset-lex: did we miss an order?, (2014)
//   http://arxiv.org/abs/1405.6503
template<typename T>
static inline T sl_gray_to_bin(T k, T ldn) {
    if ( k==0 )  return 0;

    T b = 1UL << (ldn-1);  // mask for bit at end
    T h = k & b;  // bit at end
    k ^= h;  // remove bit

    T z = sl_gray_to_bin( k, ldn-1 );  // recursion
    if ( h==0 )  return (b<<1) - z;
    else         return 1 + z;
}
