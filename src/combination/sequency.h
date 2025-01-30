#pragma once

template<typename T>
class sequence {
    constexpr static size_t BITS = sizeof(T) * 8u;

    // Return the number of zero-one (or one-zero)
    //  transitions (sequency) of x.
    constexpr static inline T bit_sequency(T x) noexcept {
        return bit_count( gray_code(x) );
    }
    
    // Return the first (i.e. smallest) word with sequency k,
    // e.g.  00..00010101010 (seq 8)
    // e.g.  00..00101010101 (seq 9)
    // Must have:  0 <= k <= BITS_PER_LONG
    constexpr static inline T first_sequency(T k) noexcept {
        if ( k==0 )  return 0;
        const T m = 0xaaaaaaaaaaaaaaaaUL;
        return  m >> (BITS-k);
    
        // =^=
        //    return inverse_gray_code( first_comb(k) );
        
        // =^= (by Doug Moore)
        //    T s = ((1UL<<k) - 1);
        //    return  s ^ (s / 3);
    }
    
    // Return the last (i.e. biggest) word with sequency k.
    constexpr static inline T last_sequency(T k, T n=BITS) noexcept {
        T x = inverse_gray_code( last_comb(k, n) );
        return  x;
    }
    
    
    // Return next word with the same number
    // of zero-one transitions (sequency) as x.
    // The value of the lowest bit is conserved.
    //
    // Zero is returned when there is no further sequence.
    //
    // e.g.:
    //  ...1.1.1 ->
    //  ..11.1.1 ->
    //  ..1..1.1 ->
    //  ..1.11.1 ->
    //  ..1.1..1 ->
    //  ..1.1.11 ->
    //  .111.1.1 ->
    //  .11..1.1 ->
    //  .11.11.1 ->
    //  .11.1..1 ->
    //  .11.1.11 -> ...
    constexpr static inline T next_sequency(T x) noexcept {
        x = gray_code(x);
        x = next_colex_comb(x);
        x = inverse_gray_code(x);
        return x;
    }
    
    constexpr static inline T prev_sequency(T x) noexcept {
        x = gray_code(x);
        x = prev_colex_comb(x);
        x = inverse_gray_code(x);
        return x;
    }
    
    // Return word whose sequency is BITS_PER_LONG - s
    // where s is the sequency of x
    static inline T complement_sequency(T x) {
        return x ^ 0xaaaaaaaaaaaaaaaaUL;
    }
};
