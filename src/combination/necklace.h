#pragma once

#include "algorithm/bits/periodic.h"
#include "math/math.h"

// Binary necklaces as binary words.
// The cyclic maximal words are produced.
// Must have 0<n<=BITS_PER_LONG.
template<typename T, const uint32_t n>
class bit_necklace {
private:
    constexpr static uint32_t BITS = sizeof(T)*8;
    static_assert(n > 0);
    static_assert(n <= BITS);
public:
    T a_;   // necklace
    T j_;   // period of the necklace
    T n2_;  // bit representing n: n2==2**(n-1)
    T j2_;  // bit representing j: j2==2**(j-1)
    T n_;   // number of bits in words
    T mm_;  // mask of n ones
    T tfb_;  // for fast factor lookup

public:
    ~bit_necklace()  { ; }

    void init() {
        n_ = n;

        n2_ = 1UL<<(n-1);
        mm_ = (~0UL) >> (BITS-n);
        tfb_ = tiny_factors_tab[n] >> 1;
        tfb_ |= n2_;  // needed for n==BITS_PER_LONG
        first();
    }

    void first()
    {
        a_ = 0;
        j_ = 1;
        j2_ = 1;
    }

    T data() const { return  a_; }
    T period() const { return j_; }

    T next()
    // Create next necklace.
    // Return the period, zero when current necklace is last.
    {
        if ( a_==mm_ )  { first();  return 0; }

        do
        {
#if 0
            j_ = highest_zero_idx( a_ ^ (~mm_)  );
#else  // with explicit bit scan:
            j_ = n_ - 1;
            T jb = 1UL << j_;
            while ( 0!=(a_ & jb) )  { --j_;  jb>>=1; }
#endif
            j2_ = 1UL << j_;
            ++j_;
            a_ |= j2_;
//            a_ = bit_copy_periodic(a_, j_);  a_ &= mm_;
            a_ = bit_copy_periodic(a_, j_, n_);
        }
        while ( 0==(tfb_ & j2_) );  // necklaces only

        return  j_;
    }

    T is_lyndon_word()  const  { return (j2_ & n2_); }

    T next_lyn()
    // Create next Lyndon word.
    // Return the period (==n), zero when current necklace is last.
    {
        if ( a_==mm_ )  { first();  return 0; }
        do  { next(); }  while ( !is_lyndon_word() );
        return  n_;
    }
};
