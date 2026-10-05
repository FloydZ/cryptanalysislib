#pragma once

/// Class for working with sequency (number of bit transitions) in binary words
/// Provides utilities for generating, analyzing, and manipulating words based on their sequency
///  ...1.1.1 ->
///  ..11.1.1 ->
///  ..1..1.1 ->
///  ..1.11.1 ->
///  ..1.1..1 ->
///  ..1.1.11 ->
///  .111.1.1 ->
///  .11..1.1 ->
///  .11.11.1 ->
///  .11.1..1 ->
///  .11.1.11 -> ...
/// \tparam T[in]: integer type to represent binary words
template<typename T>
class sequence {
private:
    /// Number of bits in type T
    constexpr static size_t BITS = sizeof(T) * 8u;

public:
    /// Computes the number of zero-one (or one-zero) transitions (sequency) of x
    /// \param x[in]: binary word to analyze
    /// \return the sequency (number of bit transitions) of x
    [[nodiscard]] constexpr static inline T bit_sequency(T x) noexcept {
        return bit_count( gray_code(x) );
    }
    
    /// Returns the first (smallest) word with sequency k
    /// For example:
    /// - 00..00010101010 (sequency 8)
    /// - 00..00101010101 (sequency 9)
    /// 
    /// \param k[in]: desired sequency (must be 0 <= k <= BITS_PER_LONG)
    /// \return the first binary word with sequency k
    [[nodiscard]] constexpr static inline T first_sequency(T k) noexcept {
        if ( k==0 )  return 0;
        const T m = 0xaaaaaaaaaaaaaaaaUL;
        return  m >> (BITS-k);
    
        // Equivalent implementations:
        //    return inverse_gray_code( first_comb(k) );
        
        // Implementation by Doug Moore:
        //    T s = ((1UL<<k) - 1);
        //    return  s ^ (s / 3);
    }
    
    /// Returns the last (largest) word with sequency k
    /// \param k[in]: desired sequency
    /// \param n[in]: number of bits to consider (defaults to BITS)
    /// \return the last binary word with sequency k
    [[nodiscard]] constexpr static inline T last_sequency(const T k,
                                                          const T n=BITS) noexcept {
        T x = inverse_gray_code( last_comb(k, n) );
        return  x;
    }
    
    /// Returns the next word with the same number of zero-one transitions (sequency) as x
    /// The value of the lowest bit is conserved
    /// 
    /// Example sequence:
    ///  ...1.1.1 ->
    ///  ..11.1.1 ->
    ///  ..1..1.1 ->
    ///  ..1.11.1 ->
    ///  ..1.1..1 ->
    ///  ..1.1.11 ->
    ///  .111.1.1 ->
    ///  .11..1.1 ->
    ///  .11.11.1 ->
    ///  .11.1..1 ->
    ///  .11.1.11 -> ...
    ///
    /// \param x[in]: current binary word
    /// \return next word with the same sequency, or zero if there is no next word
    [[nodiscard]] constexpr static inline T next_sequency(T x) noexcept {
        x = gray_code(x);
        x = next_colex_comb(x);
        x = inverse_gray_code(x);
        return x;
    }
    
    /// Returns the previous word with the same number of zero-one transitions (sequency) as x
    /// \param x[in]: current binary word
    /// \return previous word with the same sequency, or zero if there is no previous word
    [[nodiscard]] constexpr static inline T prev_sequency(T x) noexcept {
        x = gray_code(x);
        x = prev_colex_comb(x);
        x = inverse_gray_code(x);
        return x;
    }
    
    /// Returns a word whose sequency is BITS_PER_LONG - s, where s is the sequency of x
    /// Effectively finds the sequency complement of x
    /// \param x[in]: binary word to find the sequency complement of
    /// \return word with complementary sequency
    [[nodiscard]] constexpr static inline T complement_sequency(const T x) noexcept {
        return x ^ 0xaaaaaaaaaaaaaaaaUL;
    }
};
