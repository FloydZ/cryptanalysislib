#pragma once 

#include <cstdint>


#include <cstdlib>
#include <type_traits>

#include "algorithm/swap.h"

// Whether f[] is sorted wrt. bits b0,...,b0+z-1 where z is the number of bits
// set in m. m must contain a single run of bits starting at bit zero.
template<typename T>
    requires std::is_integral_v<T>
[[nodiscard]] constexpr bool is_counting_sorted(const T *f,
                                                const size_t n,
                                                const size_t b0,
                                                T m) noexcept {
    m <<= b0;
    for (uint64_t k=1; k<n; ++k) {
        uint64_t xm = (f[k-1] & m ) >> b0;
        uint64_t xp = (f[k] & m ) >> b0;
        if ( xm > xp )  return false;
    }

    return true;
}

// Write to g[] the array f[] sorted wrt. bits b0,...,b0+z-1
//  where z is the number of bits set in m.
// m must contain a single run of bits starting at bit zero.
template<typename T>
    requires std::is_integral_v<T>
constexpr void counting_sort_core(const T *__restrict__ f,
                                  const uint64_t n,
                                  T *__restrict__ g,
                                  const size_t b0,
                                  size_t m) noexcept {
    uint64_t nb = m + 1;
    m <<= b0;
    //ALLOCA(uint64_t, cv, nb);
    size_t *cv = (size_t *)calloc(nb, sizeof(size_t));
    // size_t cv[nb] = {0};

    // --- count:
    for (size_t k=0; k<n; ++k) {
        T x = (f[k] & m ) >> b0;
        ++cv[ x ];
    }

    // --- cumulative sums:
    for (size_t k=1; k<nb; ++k) { cv[k] += cv[k-1]; }

    // --- reorder:
    uint64_t k = n;
    // backwards ==> stable sort
    while ( k-- ) {
        T fk = f[k];
        T x = (fk & m) >> b0;
        --cv[x];
        uint64_t i = cv[x];
        g[i] = fk;
    }

    free(cv);
}

/// \param f[in/out]: vector to sort 
/// \param n[in]; length of the vector 
template<typename T>
    requires std::is_unsigned_v<T>
constexpr void radix_sort(T *f,
                          const size_t n) noexcept  {
    // Number of bits sorted with each step
    const size_t nb = 8;  
    const size_t tnb = sizeof(T) * 8;

    T *fi = f;
    T *g = new T[n];

    uint64_t m = (1UL<<nb) - 1;
    for (uint64_t b0=0;  b0<tnb; b0+=nb) {
        counting_sort_core(f, n, g, b0, m);
        cryptanalysislib::swap(f, g);
    }

    // result is actually in g[]
    if ( f!=fi ) {
        cryptanalysislib::swap(f, g);
        for (uint64_t k=0; k<n; ++k)  f[k] = g[k];
    }

    delete [] g;
}
