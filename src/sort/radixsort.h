#pragma once 


#include <cstdlib>
#include <type_traits>

// Whether f[] is sorted wrt. bits b0,...,b0+z-1 where z is the number of bits
// set in m. m must contain a single run of bits starting at bit zero.
template<typename T>
    requires std::is_integral_v<T>
[[nodiscard]] constexpr bool is_counting_sorted(const T *f,
                                                const size_t n,
                                                const size_t b0,
                                                T m) noexcept {
    m <<= b0;
    for (ulong k=1; k<n; ++k) {
        ulong xm = (f[k-1] & m ) >> b0;
        ulong xp = (f[k] & m ) >> b0;
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
                                  const ulong n,
                                  T *__restrict__ g,
                                  const size_t b0,
                                  size_t m) noexcept {
    ulong nb = m + 1;
    m <<= b0;
    //ALLOCA(ulong, cv, nb);
    size_t *cv = (size_t *)calloc(1, nb);
    // size_t cv[nb] = {0};

    // --- count:
    for (size_t k=0; k<n; ++k) {
        T x = (f[k] & m ) >> b0;
        ++cv[ x ];
    }

    // --- cumulative sums:
    for (size_t k=1; k<nb; ++k) { cv[k] += cv[k-1]; }

    // --- reorder:
    ulong k = n;
    // backwards ==> stable sort
    while ( k-- ) {
        T fk = f[k];
        T x = (fk & m) >> b0;
        --cv[x];
        ulong i = cv[x];
        g[i] = fk;
    }

    free(cv);
}

/// \param f[in/out]: vector to sort 
/// \param n[in]; length of the vector 
template<typename T>
    requires std::is_integral_v<T>
constexpr void radix_sort(T *f,
                          const size_t n) noexcept  {
    // Number of bits sorted with each step
    const size_t nb = 8;  
    const size_t tnb = sizeof(T) * 8;

    ulong *fi = f;
    ulong *g = new ulong[n];

    ulong m = (1UL<<nb) - 1;
    for (ulong b0=0;  b0<tnb; b0+=nb) {
        counting_sort_core(f, n, g, b0, m);
        swap2(f, g);
    }

    // result is actually in g[]
    if ( f!=fi ) {
        swap2(f, g);
        for (ulong k=0; k<n; ++k)  f[k] = g[k];
    }

    delete [] g;
}
