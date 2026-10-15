#pragma once

#include <cstdint>

#include "combination/bit_subset.h"

/// Choose the method for calculating subset differences
/// METHOD1 uses ~U0 & U1 (default)
/// Alternative method uses (U0 ^ U1) & U1
#define BITSUBSET_GRAY_METHOD1

/// Class for generating subsets in Gray code order
/// Provides utilities to enumerate all subsets of a given set with minimal bit changes
/// between consecutive subsets
/// \tparam T[in]: integer type to represent subsets
template<typename T>
class bit_subset_gray_T {
protected:
    /// Finds the highest one bit in a number
    /// \param x[in]: value to analyze
    /// \return 1 << MSB(x), the power of 2 corresponding to the highest set bit
    constexpr inline T highest_one(const T x) {
        // NOTE: the highest one is at bit 63 - clz (was `1 << clz`)
        return T(1) << (63u - __builtin_clzll(uint64_t(x) | 1u));
    }

    /// Underlying subset generator
    bit_subset_T<T> S;
    
    /// Current subset in Gray code order
    T G;
    
    /// Highest bit in S.V; needed for the prev() method
    T H;

public:
    /// Constructor initializes the subset Gray code generator
    /// \param v[in]: full set to generate subsets for
    constexpr explicit bit_subset_gray_T(const T v) noexcept 
        : S(v), G(0), H(highest_one(v)) { ; }

    /// Destructor
    ~bit_subset_gray_T() { ; }

    /// Gets the current subset in Gray code order
    /// \return current subset
    constexpr T current() const noexcept {
        return G; 
    }

    /// Gets the full set being used for subset generation
    /// \return the full set
    constexpr T full_set() const noexcept { 
        return S.full_set(); 
    }

    /// Advances to the next subset in Gray code order
    /// Each successive subset differs from the previous by exactly one bit
    /// \return false if the current subset is the last one (unchanged)
    constexpr bool next() noexcept {
        const T U0 = S.current();
        if (!S.next()) { return false; }
        const T U1 = S.current();
#if defined BITSUBSET_GRAY_METHOD1
        // Method 1: find bits in U1 that are not in U0
        T X = ~U0 & U1;
#else
        // Alternative method: find new bits added in U1
        T X = (U0 ^ U1) & U1;
#endif
        G ^= X;
        return true;
    }

    /// Sets the generator to the first subset for a new full set
    /// \param v[in]: new full set to generate subsets for
    /// \return the first subset (empty set)
    constexpr T first(T v) noexcept {
        S.first(v);
        H = highest_one(v);
        G = 0;
        return G;
    }

    /// Sets the generator to the first subset (empty set)
    /// \return the first subset (empty set)
    constexpr T first() noexcept {
        S.first();
        G = 0;
        return G;
    }

    /// Moves to the previous subset in Gray code order
    /// \return false if the current subset is the first one (unchanged)
    constexpr bool previous() noexcept {
        const T U1 = S.current();
        if (!S.previous()) { return false; }
        const T U0 = S.current();
#if defined BITSUBSET_GRAY_METHOD1
        // Method 1: find bits in U1 that are not in U0
        T X = ~U0 & U1;
#else
        // Alternative method: find bits that were removed
        T X = (U0 ^ U1) & U1;
#endif
        G ^= X;
        return true;
    }

    /// Sets the generator to the last subset for a new full set
    /// \param v[in]: new full set to generate subsets for
    /// \return the last subset
    constexpr T last(T v) noexcept {
        S.last(v);
        H = highest_one(v);
        G = H;
        return G;
    }

    /// Sets the generator to the last subset
    /// \return the last subset
    constexpr T last() noexcept {
        S.last();
        G = H;
        return G;
    }
};

