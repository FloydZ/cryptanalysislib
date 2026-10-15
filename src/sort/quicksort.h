#pragma once 

#include <cstddef>
#include "algorithm/swap.h"
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
constexpr size_t partition(Type *f,
                           const size_t n) noexcept {
    // Avoid worst case with already sorted input:
    const Type v = median3(f[0], f[n/2], f[n-1]);

    size_t i = 0UL - 1;
    size_t j = n;
    while (1) {
        do  { ++i; }  while ( f[i]<v );
        do  { --j; }  while ( f[j]>v );

        if ( i < j )  cryptanalysislib::swap(f[i], f[j]);
        else          return j;
    }
}


/// configuration of `quick_sort`
struct QuickSortConfig {
    /// sub-arrays with at most this many elements are sorted with the
    /// (non recursive) selection sort.
    /// NOTE: must be >= 2: `partition` of two elements may return p = 1,
    /// 	i.e. an empty right part, and the loop would not terminate
    const size_t selection_sort_threshold = 8;
};
constexpr static QuickSortConfig quickSortConfig{};

// Sort f[] (ascending order).
template <typename Type,
          const QuickSortConfig &config = quickSortConfig>
constexpr void quick_sort(Type *f,
                          const size_t n) noexcept {
    static_assert(config.selection_sort_threshold >= 2);
    size_t m = n;
start:
    if (m <= config.selection_sort_threshold) {
        selection_sort(f, m);
        return;
    }

    size_t p = partition(f, m);
    size_t ln = p + 1;
    size_t rn = m - ln;

    // recursion for shorter sub-array
    if (ln > rn) {
        // f[ln] ... f[n-1]   right
        quick_sort<Type, config>(f+ln, rn);
        m = ln;
    } else {
        // f[0]  ... f[ln-1]  left
        quick_sort<Type, config>(f, ln);
        m = rn;
        f += ln;
    }

    goto start;
}
