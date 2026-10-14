#pragma once

#include <cstddef>
#include <cstdint>

#include "combination/enumerator.h"

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
/// As an enumerator: all words of `n` bits with sequency `k`, from
/// `first_sequency(k)` to `last_sequency(k, n)` (`next_sequency` order).
template<typename T>
class sequence {
private:
    /// Number of bits in type T
    constexpr static size_t BITS = sizeof(T) * 8u;

    /// first and last word, the current word
    T first, last, x;

    /// \return the Gray code of x
    [[nodiscard]] constexpr static inline T gray_code(const T x) noexcept {
        return T(x ^ (x >> 1u));
    }

    /// \return the inverse Gray code of x (prefix parity from the top)
    [[nodiscard]] constexpr static inline T inverse_gray_code(T x) noexcept {
        for (uint32_t s = 1; s < BITS; s <<= 1u) { x ^= T(x >> s); }
        return x;
    }

    /// \return number of set bits
    [[nodiscard]] constexpr static inline T bit_count(const T x) noexcept {
        return T(__builtin_popcountll(uint64_t(x)));
    }

    /// \return the word with the lowest `k` bits set
    [[nodiscard]] constexpr static inline T first_comb(const T k) noexcept {
        if (k == 0) { return 0; }
        return T(T(~T(0)) >> (BITS - k));
    }

    /// \return the word with the bits [n-k, n) set
    [[nodiscard]] constexpr static inline T last_comb(const T k, const T n) noexcept {
        return T(first_comb(k) << (n - k));
    }

    /// \return the next larger word with the same number of bits (colex
    /// 	order), 0 if there is none
    [[nodiscard]] constexpr static inline T next_colex_comb(T x) noexcept {
        const T r = T(x & T(-x));
        x = T(x + r);
        if (x == 0) { return 0; }
        T z = T(x & T(-x));
        z = T(z - r);
        while (0 == (z & 1u)) { z >>= 1u; }
        return T(x | (z >> 1u));
    }

    /// \return the next smaller word with the same number of bits, 0 if
    /// 	there is none
    [[nodiscard]] constexpr static inline T prev_colex_comb(T x) noexcept {
        x = next_colex_comb(T(~x));
        if (x != 0) { x = T(~x); }
        return x;
    }

public:
    /// \param k[in]: sequency, 0 <= k <= n
    /// \param n[in]: number of bits, <= bits of T
    constexpr sequence(const T k, const T n = BITS) noexcept :
        first(first_sequency(k)), last(last_sequency(k, n)), x(first) {}

    /// \return the current word
    [[nodiscard]] constexpr inline T current() const noexcept {
        return x;
    }

    /// moves to the next word with the same sequency
    /// \return false if the current word is the last one (unchanged)
    constexpr inline bool next() noexcept {
        if (x == last) { return false; }
        x = next_sequency(x);
        return true;
    }

    /// moves to the previous word with the same sequency
    /// \return false if the current word is the first one (unchanged)
    constexpr inline bool previous() noexcept {
        if (x == first) { return false; }
        x = prev_sequency(x);
        return true;
    }

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
