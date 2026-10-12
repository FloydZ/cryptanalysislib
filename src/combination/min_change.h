#pragma once

#include <cstdint>

#include "algorithm/bits/popcount.h"

// port of the minimal-change combination order of the fxtbook (Jörg Arndt,
// comb/comb-minchange, bits/bitcombminchange.h)

/// Generates all `k`-subsets of {0, ..., n-1} (as bit masks with `k` set
/// bits) in minimal-change order: two consecutive combinations differ by
/// exactly one cleared and one set bit.
/// Usage:
/// \code
/// 	enumeration_min_change<uint64_t> e(n, k);
/// 	do { use(e.current()); } while (e.next());
/// \endcode
/// \tparam T[in]: integer type to represent combinations (n <= 64)
template<class T>
class enumeration_min_change {
private:
    constexpr static uint32_t BITS = 64;

    /// current combination
    uint64_t val = 0;
    /// last combination, in the inverse Gray code domain
    uint64_t last = 0;
    /// false once the last combination was passed
    bool valid = true;

public:
    /// \return the Gray code of `x`
    constexpr static inline uint64_t gray_code(const uint64_t x) noexcept {
        return x ^ (x >> 1u);
    }

    /// \return the inverse Gray code of `x`, i.e. gray_code(inverse_gray_code(x)) == x
    constexpr static inline uint64_t inverse_gray_code(uint64_t x) noexcept {
        x ^= x >> 1u;
        x ^= x >> 2u;
        x ^= x >> 4u;
        x ^= x >> 8u;
        x ^= x >> 16u;
        x ^= x >> 32u;
        return x;
    }

    /// \return true if `x` has at most one bit set (also for x == 0)
    constexpr static inline bool is_pow_of_2(const uint64_t x) noexcept {
        return (x & (~x + 1u)) == x;
    }

    /// \return number of set bits
    constexpr static inline uint64_t bit_count(const uint64_t x) noexcept {
        return cryptanalysislib::popcount::popcount(x);
    }

    /// \return the inverse Gray code of the first combination with `k` bits:
    /// 	..1010101 (k bits, the lowest one is set)
    constexpr static inline uint64_t first_sequency(const uint64_t k) noexcept {
        if (k == 0) { return 0; }
        return 0xaaaaaaaaaaaaaaaaull >> (BITS - k);
    }

    /// Computes the next minimal change combination using inverse Gray code
    /// Alternative version optimized for speed with constant amortized time (CAT)
    /// \param x[in]: current inverse Gray code value
    /// \return next inverse Gray code value with minimal bit changes
    constexpr static inline uint64_t igc_next_minchange_comb(const uint64_t x) noexcept {
        const uint64_t gx = gray_code(x);
        uint64_t i = 2;
        while (true) {
            const uint64_t y = x + i;
            i <<= 1u;
            const uint64_t gy = gray_code(y);
            const uint64_t r = gx ^ gy;

            // Check that change consists of exactly one bit
            // of the new and one bit of the old pattern:
            if (is_pow_of_2(r & gy) && is_pow_of_2(r & gx)) { return y; }
            // is_pow_of_2(x) returns true also for x == 0,
            // but this cannot happen for both tests at the same time
        }
    }

    /// Computes the next minimal change combination using inverse Gray code
    /// Uses the fact that the difference between successive values is the smallest
    /// possible power of 2. Efficient when CPU has a bitcount instruction.
    /// \param x[in]: current inverse Gray code value
    /// \param k[in]: bit-count of gray_code(x) (must be provided accurately)
    /// \return next inverse Gray code value with minimal bit changes
    constexpr static inline uint64_t igc_next_minchange_comb(const uint64_t x,
                                                             const uint64_t k) noexcept {
        uint64_t y;
        uint64_t i = 2;
        do {
            y = x + i;
            i <<= 1u;
        } while (bit_count(gray_code(y)) != k);
        return y;
    }

    /// Computes the previous combination in minimal-change order
    /// Input must be the inverse Gray code of the current combination
    /// \param x[in]: current inverse Gray code value
    /// \param k[in]: bit-count of gray_code(x)
    /// \return previous inverse Gray code with minimal bit changes
    constexpr static inline uint64_t igc_prev_minchange_comb(const uint64_t x,
                                                             const uint64_t k) noexcept {
        uint64_t y, i = 1;
        do {
            i <<= 1u;
            y = x - i;
        } while (bit_count(gray_code(y)) != k);
        return y;
    }

    /// Computes the last combination in the minimal-change sequence
    /// (in the inverse Gray code domain).
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
    constexpr static inline uint64_t igc_last_comb(const uint64_t k,
                                                   const uint64_t n) noexcept {
        if (0 == k) { return 0; }
        const uint64_t f = first_sequency(k);
        const uint64_t c = ~0ull >> (BITS - n);  // == first_comb(n);
        return c ^ (f >> 1u);
        // =^=  (by Doug Moore)
        //    return  ((1UL<<n) - 1) ^ (((1UL<<k) - 1) / 3);
    }

    /// Computes the next minimal change combination
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
    constexpr static inline uint64_t next_minchange_comb(uint64_t x,
                                                         const uint64_t last) noexcept {
        x = inverse_gray_code(x);
        if (x == last) { return 0; }
        x = igc_next_minchange_comb(x);
        return gray_code(x);
    }

    /// starts the enumeration of the `k`-subsets of {0, ..., n-1}
    /// \param n[in]: number of elements, <= 64
    /// \param k[in]: subset size, <= n
    constexpr enumeration_min_change(const uint32_t n,
                                     const uint32_t k) noexcept :
        val(gray_code(first_sequency(k))), last(igc_last_comb(k, n)) {
        static_assert(sizeof(T) <= 8);
    }

    /// \return the current combination
    [[nodiscard]] constexpr inline T current() const noexcept {
        return T(val);
    }

    /// advances to the next combination
    /// \return false if the current one was the last combination
    constexpr inline bool next() noexcept {
        if (!valid) { return false; }
        if (inverse_gray_code(val) == last) {
            valid = false;
            return false;
        }
        val = gray_code(igc_next_minchange_comb(inverse_gray_code(val)));
        return true;
    }
};
