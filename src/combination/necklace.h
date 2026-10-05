#pragma once

#include "algorithm/bits/periodic.h"
#include "math/math.h"

/// Binary necklace generation class
/// Generates binary necklaces as binary words
/// Produces cyclic maximal words
/// \tparam T[in]: integer type to represent binary words
/// \tparam n[in]: number of bits in necklaces (must have 0 < n <= BITS_PER_LONG)
template<typename T, const uint32_t n>
class bit_necklace {
private:
    /// Number of bits in type T
    constexpr static uint32_t BITS = sizeof(T)*8;
    static_assert(n > 0);
    static_assert(n <= BITS);
public:
    /// Current necklace value
    T a_;
    
    /// Period of the current necklace
    T j_;
    
    /// Bit representing n: n2==2**(n-1)
    T n2_;
    
    /// Bit representing j: j2==2**(j-1)
    T j2_;
    
    /// Number of bits in words
    T n_;
    
    /// Mask of n ones
    T mm_;
    
    /// For fast factor lookup
    T tfb_;

public:
    /// Destructor
    ~bit_necklace()  { ; }

    /// Initializes the necklace generator
    /// Sets up bitmasks and lookup tables
    void init() {
        n_ = n;

        n2_ = 1UL<<(n-1);
        mm_ = (~0UL) >> (BITS-n);
        tfb_ = tiny_factors_tab[n] >> 1;
        tfb_ |= n2_;  // needed for n==BITS_PER_LONG
        first();
    }

    /// Sets the generator to the first necklace
    /// Resets to all zeros with period 1
    void first()
    {
        a_ = 0;
        j_ = 1;
        j2_ = 1;
    }

    /// Gets the current necklace data
    /// \return current necklace as a binary word
    T data() const { return  a_; }
    
    /// Gets the period of the current necklace
    /// \return period of the current necklace
    T period() const { return j_; }

    /// Creates the next necklace in the sequence
    /// \return the period of the new necklace, or 0 if the current necklace was the last
    T next()
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

    /// Determines if the current necklace is a Lyndon word
    /// A Lyndon word is strictly less than all its cyclic rotations
    /// \return non-zero if the current necklace is a Lyndon word, 0 otherwise
    T is_lyndon_word()  const  { return (j2_ & n2_); }

    /// Creates the next Lyndon word in the sequence
    /// \return the period of the new Lyndon word (==n), or 0 if the current necklace was the last
    T next_lyn()
    {
        if ( a_==mm_ )  { first();  return 0; }
        do  { next(); }  while ( !is_lyndon_word() );
        return  n_;
    }
};
