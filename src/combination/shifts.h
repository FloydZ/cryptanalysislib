#pragma once 

#include "colex.h"

/// Class for generating bit combinations in shifts-order
/// Provides utilities for enumerating combinations that differ by shifts and splits
/// \tparam T[in]: integer type to represent combinations
/// \tparam n[in]: total number of bits
/// \tparam k[in]: number of bits set to 1
template<typename T, const uint32_t n, const uint32_t k>
class bit_comb_shifts {
public:
    /// Current combination
    ulong x_;
    
    /// How far the combination has been shifted to the right
    ulong s_;
    
    /// Parameters for the combinations (n choose k)
    ulong n_, k_;
    
    /// Last combination in the sequence
    ulong last_;
    
    /// Co-lexicographic enumerator for combinations
    enumeration_colex<T, n, k> e;
    
public:
    /// Constructor initializes the shifts-order combination generator
    constexpr explicit bit_comb_shifts() noexcept {
        n_ = n;  k_ = k;
        first();
    }

    /// Sets the combination to the first one in the shifts-order sequence
    /// \return the first combination
    ulong first() {
        s_ = 0;
        x_ =  e.last_comb();

        if ( k>1 )  last_ = e.first_comb(k-1) | (1UL<<(n_-1));  // [10000111]
        else        last_ = k;  //  [000001] or [000000]

        return x_;
    }

    /// Advances to the next combination in shifts-order
    /// A shifts-order traversal either shifts the current combination right
    /// or performs a split operation when a right shift is not possible
    /// \return the next combination, or 0 if at the end of the sequence
    ulong next()
    {
        if ( 0==(x_&1) ) {
            // Easy case: right shift is possible (rightmost bit is 0)
            ++s_;
            x_ >>= 1;
            return  x_;
        } else {
            // Splitting cases (rightmost bit is 1)
            if ( x_ == last_ )  return 0;  // combination was last

            x_ <<= s_;  s_ = 0;  // shift back to the left
            ulong b = x_ & -x_;  // lowest bit (rightmost 1)

            if ( b!=1UL ) {
                // Simple split: lowest bit is not at position 0
                x_ -= (b>>1);  // move rightmost bit to the right
                return x_;
            } else { 
                // Complex split: lowest bit is at position 0
                // Split second block and attach first
                ulong t = __builtin_ctzll(x_);  // block of ones at lower end
                x_ ^= t;  // remove block
                ulong b2 = x_ & -x_;  // (second) lowest bit

                b2 >>= 1;
                x_ -= b2;  // move bit to the right

#if 1 
                // Attach block by finding the correct position
                do  { t<<=1; }  while ( 0==(t&x_) );
                x_ |= (t>>1);
#else
                // Alternative: attach block using bit-scan operations
                x_ |= ( t << (highest_one_idx(b2)-highest_one_idx(t)-1) );
#endif
                return x_;
            }
        }
    }
};
