#pragma once 

#include <cstdint>

/// Class for efficiently updating bit-reversed values
/// Provides utilities for computing the next bit-reversed value
/// in a sequence without performing full reversal operations
/// \tparam T[in]: integer type for representing bit patterns
template<typename T>
class revbin_upd {
private:
    /// Maximum number of bits supported
    constexpr static inline uint32_t BITS = 64;
    
    /// Lookup table for updating bit-reversed values
    T utab[BITS];
    
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
    /// \param h[in]: half the size of the domain (n/2, where n=2^ldn)
    explicit revbin_upd(T h) noexcept {
        make_revbin_upd_tab(h);
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
        k = lowest_one_idx( ~k );  // lowest zero idx
        r ^= utab[k];
        return r;
    }
};
