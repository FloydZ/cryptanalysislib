#pragma once

template<class T>
class enumeration_min_change {
private:
    T val = 0;


    // Alternative version, faster.
    // Constant amortized time (CAT).
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

    /// Alternative version, uses the fact that the difference
    /// of two successive x is the smallest possible power of 2.
    /// Should be fast if the CPU has a bitcount instruction.
    /// k must be the bit-count of x
    /// Constant amortized time (CAT).
    /// Note: this version has 2 arguments.
    constexpr static inline ulong igc_next_minchange_comb(ulong x, ulong k) {
        ulong y;
        ulong i = 2;
        do {
            y = x + i;
            i <<= 1;
        } while ( bit_count( gray_code(y) ) != k );
        return  y;
    }
    
    /// Return the inverse Gray code of the previous combination in minimal-change order.
    /// Input must be the inverse Gray code of the current combination.
    /// Constant amortized time (CAT).
    /// With input==first the output is the last for n=BITS_PER_LONG
    constexpr static inline ulong igc_prev_minchange_comb(ulong x, ulong k) noexcept {
        ulong y, i = 1;
        do {
            i <<= 1;
            y = x - i;
        } while ( bit_count( gray_code(y) ) != k );
        return  y;
    }
    
    /// Return the (inverse Gray code of the) last combination
    /// as in igc_next_minchange_comb().
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
    
    /// Not efficient, just to explain the usage of igc_next_minchange_comb()
    /// Must have: last==igc_last_comb(k, n)
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
    constexpr static inline ulong next_minchange_comb(ulong x, ulong last) noexcept {
        x = inverse_gray_code(x);
        if ( x==last )  return 0;
        x = igc_next_minchange_comb(x);
        return  gray_code(x);
    }

public:
    constexpr inline T next() noexcept {
        const T ret = val;
        val = next_colex_comb(val);
        return ret;
    } 
    constexpr inline T prev() noexcept {
        const T ret = val;
        val = prev_colex_comb(val);
        return ret;
    } 
};
