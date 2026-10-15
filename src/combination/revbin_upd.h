#pragma once

#include <cstdint>

#include "algorithm/countr.h"
#include "combination/enumerator.h"

/// Class for efficiently updating bit-reversed values
/// Provides utilities for computing the next bit-reversed value
/// in a sequence without performing full reversal operations
/// As an enumerator: revbin(k, ldn) for k = 0, ..., n-1, where n = 2*h = 2**ldn.
/// \tparam T[in]: integer type for representing bit patterns
template<typename T>
class revbin_upd {
private:
    /// Maximum number of bits supported
    constexpr static inline uint32_t BITS = 64;

    /// Lookup table for updating bit-reversed values
    T utab[BITS];

    /// n - 1, the last index
    T last;
    /// the current index
    T k = 0;
    /// revbin(k, ldn)
    T r = 0;

    /// Initializes the lookup table used by revbin_tupd()
    /// \param h[in]: half the size of the domain (n/2, where n=2^ldn)
    constexpr inline void make_revbin_upd_tab(T h) {
        utab[0] = h;  // == n/2 == 1UL<<(ldn-1);
        for (T i=1; h!=0; ++i)
        {
            h >>= 1;
            utab[i] = utab[i-1] ^ h;
        }
    }
public:
    /// Constructor that initializes the lookup table
    /// \param h[in]: half the size of the domain (n/2, where n=2^ldn), >= 1
    explicit revbin_upd(T h) noexcept : last(T(2 * h - 1)) {
        make_revbin_upd_tab(h);
    }

    /// \return revbin(index(), ldn)
    [[nodiscard]] constexpr inline T current() const noexcept {
        return r;
    }

    /// \return the current index
    [[nodiscard]] constexpr inline T index() const noexcept {
        return k;
    }

    /// moves to revbin(k+1, ldn)
    /// \return false if the current index is n-1 (unchanged)
    constexpr inline bool next() noexcept {
        if (k == last) { return false; }
        r = revbin_tupd(r, k);
        k += 1;
        return true;
    }

    /// moves to revbin(k-1, ldn): k-1 differs from k in the bits up to the
    /// lowest set bit of k
    /// \return false if the current index is 0 (unchanged)
    constexpr inline bool previous() noexcept {
        if (k == 0) { return false; }
        r ^= utab[cryptanalysislib::countr_zero(k)];
        k -= 1;
        return true;
    }

    /// Updates a bit-reversed value directly without lookup table
    /// Let n=2**ldn and h=n/2.
    /// Then, with r == revbin(x, ldn) at entry, returns revbin(x+1, ldn)
    ///
    /// \param r[in]: current bit-reversed value
    /// \param h[in]: half the size of the domain (n/2, where n=2^ldn)
    /// \return the next bit-reversed value
    /// \note routine will hang if called with r as the all-ones word
    constexpr inline T revbin_upd_(T r, T h) {
        while ( ! ((r^=h) & h) )  h >>= 1;
        return  r;
    }

    /// Updates a bit-reversed value using the lookup table
    /// Let r==revbin(k, ldn) then returns revbin(k+1, ldn).
    ///
    /// \param r[in]: current bit-reversed value
    /// \param k[in]: current index whose bit-reversal is r
    /// \return the next bit-reversed value
    /// \note 1: need to call make_revbin_upd_tab(ldn) before usage
    ///          where ldn=log_2(n)
    /// \note 2: different argument structure than revbin_upd_()
    constexpr inline T revbin_tupd(T r, T k) {
        k = cryptanalysislib::countr_zero(T(~k));  // lowest zero idx
        r ^= utab[k];
        return r;
    }
};
