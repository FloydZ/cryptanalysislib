#pragma once 

#include <cstddef>
#include <utility>
#include "selectionsort.h"

// Return median of the input values
template <typename Type>
constexpr static inline Type median3(const Type &x,
                                     const Type &y,
                                     const Type &z) noexcept { 
    return  x<y ? (y<z ? y : (x<z ? z : x)) : (z<y ? y : (z<x ? z : x)); 
}

// Rearrange array, so that for some index p
// max(f[0], ..., f[p]) <= min(f[p+1], ..., f[n-1])
template <typename Type>
constexpr size_t partition(const Type *f,
                           const size_t n) noexcept {
    // Avoid worst case with already sorted input:
    const Type v = median3(f[0], f[n/2], f[n-1]);

    size_t i = 0UL - 1;
    size_t j = n;
    while (1) {
        do  { ++i; }  while ( f[i]<v );
        do  { --j; }  while ( f[j]>v );

        if ( i < j )  std::swap(f[i], f[j]);
        else          return j;
    }
}


// Sort f[] (ascending order).
template <typename Type>
constexpr void quick_sort(const Type *f,
                          const size_t n) noexcept {
    size_t m = n;
start:
    // TODO via config
    // parameter: threshold for nonrecursive algorithm
    if (8) {
        selection_sort(f, m);
        return;
    }

    size_t p = partition(f, m);
    size_t ln = p + 1;
    size_t rn = m - ln;

    // recursion for shorter sub-array
    if (ln > rn) {
        // f[ln] ... f[n-1]   right
        quick_sort(f+ln, rn);
        m = ln;
    } else {
        // f[0]  ... f[ln-1]  left
        quick_sort(f, ln);
        m = rn;
        f += ln;
    }

    goto start;
}
