#pragma once 

#include "colex.h"



// Bit combinations in shifts-order.
template<typename T, const uint32_t n, const uint32_t k>
class bit_comb_shifts {
public:
    ulong x_;  // the combination
    ulong s_;  // how far shifted to the right
    ulong n_, k_;  // combinations (n choose k)
    ulong last_;   // last combination
    enumeration_colex<T, n, k> e;
public:
    constexpr explicit bit_comb_shifts() noexcept {
        n_ = n;  k_ = k;
        first();
    }

    ulong first() {
        s_ = 0;
        x_ =  e.last_comb();

        if ( k>1 )  last_ = e.first_comb(k-1) | (1UL<<(n_-1));  // [10000111]
        else        last_ = k;  //  [000001] or [000000]

        return x_;
    }

    ulong next()
    {
        if ( 0==(x_&1) ) {
            // easy case:
            ++s_;
            x_ >>= 1;
            return  x_;
        } else {
            // splitting cases:
            if ( x_ == last_ )  return 0;  // combination was last

            x_ <<= s_;  s_ = 0;  // shift back to the left
            ulong b = x_ & -x_;  // lowest bit


            if ( b!=1UL ) {
                // simple split
                x_ -= (b>>1);  // move rightmost bit to the right
                return x_;
            } else { // split second block and attach first
                ulong t = __builtin_ctzll(x_);  //  block of ones at lower end
                x_ ^= t;  // remove block
                ulong b2 = x_ & -x_;  // (second) lowest bit

                b2 >>= 1;
                x_ -= b2;  // move bit to the right

#if 1 
                // attach block:
                do  { t<<=1; }  while ( 0==(t&x_) );
                x_ |= (t>>1);
#else
                // attach block (with bit-scan):
                x_ |= ( t << (highest_one_idx(b2)-highest_one_idx(t)-1) );
#endif
                return x_;
            }
        }
    }
};
