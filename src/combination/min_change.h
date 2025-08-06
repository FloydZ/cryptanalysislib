#pragma once

// TODO replace ulong with uint64_t
// TODO add 

/// Class for generating combinations with minimal change between consecutive elements
/// Provides utilities for generating combinations where only one or two bits differ
/// between consecutive elements
/// \tparam T[in]: integer type to represent combinations
template<class T>
class enumeration_min_change {
private:
    /// Current value in the enumeration
    T val = 0;

    /// Computes the next minimal change combination using inverse Gray code
    /// Alternative version optimized for speed with constant amortized time (CAT)
    /// \param x[in]: current inverse Gray code value
    /// \return next inverse Gray code value with minimal bit changes
    constexpr static inline ulong igc_next_minchange_comb(ulong x) noexcept {
        ulong gx = gray_code( x );
        ulong i = 2;
        do {
            ulong y = x + i;
            i <<= 1;
            ulong gy = gray_code( y );
            ulong r = gx ^ gy;
    
            // Check that change consists of exactly one bit
            // of the new and one bit of the old pattern:
            if ( is_pow_of_2( r & gy ) && is_pow_of_2( r & gx ) )  return y;
            // is_pow_of_2(x):=((x & -x) == x)  returns 1 also for x==0.
            // But this cannot happen for both tests at the same time
        } while ( 1 );
        return  0;  // not reached
    }

    /// Computes the next minimal change combination using inverse Gray code
    /// Uses the fact that the difference between successive values is the smallest
    /// possible power of 2. Efficient when CPU has a bitcount instruction.
    /// \param x[in]: current inverse Gray code value
    /// \param k[in]: bit-count of x (must be provided accurately)
    /// \return next inverse Gray code value with minimal bit changes
    constexpr static inline ulong igc_next_minchange_comb(ulong x, ulong k) {
        ulong y;
        ulong i = 2;
        do {
            y = x + i;
            i <<= 1;
        } while ( bit_count( gray_code(y) ) != k );
        return  y;
    }
    
    /// Computes the previous combination in minimal-change order
    /// Input must be the inverse Gray code of the current combination
    /// \param x[in]: current inverse Gray code value
    /// \param k[in]: bit-count of x
    /// \return previous inverse Gray code with minimal bit changes
    constexpr static inline ulong igc_prev_minchange_comb(ulong x, ulong k) noexcept {
        ulong y, i = 1;
        do {
            i <<= 1;
            y = x - i;
        } while ( bit_count( gray_code(y) ) != k );
        return  y;
    }
    
    /// Computes the last combination in the minimal-change sequence
    /// Used with igc_next_minchange_comb() to determine end of sequence
    ///
    /// Example (n=6)   c:=first_comb(n) == 111111
    ///
    ///   k:  f=first_seq(k)   c^(f>>1) == return
    ///   0:     ......        111111 (!) special case:  return 0==......
    ///   1:     .....1        111111
    ///   2:     ....1.        11111.
    ///   3:     ...1.1        1111.1
    ///   4:     ..1.1.        111.1.
    ///   5:     .1.1.1        11.1.1
    ///   6:     1.1.1.        1.1.1.
    ///
    /// \param k[in]: bit-count parameter
    /// \param n[in]: number of bits in the combination
    /// \return last combination in the minimal change sequence
    static inline ulong igc_last_comb(ulong k, ulong n) noexcept {
        if ( 0==k )  return 0;
    
    #if ( BITS_PER_LONG < 64 )
        const ulong f = 0xaaaaaaaaUL >> (BITS_PER_LONG-k);  // == first_sequency(k);
    #else
        const ulong f = 0xaaaaaaaaaaaaaaaaUL >> (BITS_PER_LONG-k);  // == first_sequency(k);
    #endif
        const ulong c =  ~0UL >> (BITS_PER_LONG-n);  // == first_comb(n);
        return c ^ (f>>1);
        // =^=  (by Doug Moore)
        //    return  ((1UL<<n) - 1) ^ (((1UL<<k) - 1) / 3);
    }
    
    /// Computes the next minimal change combination
    /// Demonstration function showing usage of igc_next_minchange_comb()
    ///
    /// Example with  k==3, n==5:
    ///      x       inverse_gray_code(x)
    ///    ..111       ..1.1 == first_sequency(k)
    ///    .11.1       .1..1
    ///    .111.       .1.11
    ///    .1.11       .11.1
    ///    11..1       1...1
    ///    11.1.       1..11
    ///    111..       1.111
    ///    1.1.1       11..1
    ///    1.11.       11.11
    ///    1..11       111.1 == igc_last_comb(k, n)
    ///
    /// \param x[in]: current combination
    /// \param last[in]: last combination (must be igc_last_comb(k, n))
    /// \return next combination in minimal change order or 0 if at end
    constexpr static inline ulong next_minchange_comb(ulong x, ulong last) noexcept {
        x = inverse_gray_code(x);
        if ( x==last )  return 0;
        x = igc_next_minchange_comb(x);
        return  gray_code(x);
    }

public:
    /// Returns the current combination and advances to the next
    /// Uses co-lexicographic ordering for minimal change
    /// \return current combination value
    [[nodiscard]] constexpr inline T next() noexcept {
        const T ret = val;
        val = next_colex_comb(val);
        return ret;
    } 

    /// Returns the current combination and moves to the previous
    /// Uses co-lexicographic ordering for minimal change
    /// \return current combination value
    [[nodiscard]] constexpr inline T prev() noexcept {
        const T ret = val;
        val = prev_colex_comb(val);
        return ret;
    } 
};
