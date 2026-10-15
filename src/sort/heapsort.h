#pragma once

#include <cstddef>
#include "container/heap.h"
#include "algorithm/swap.h"


// Sort x[] into ascending order.
// Uses a max-heap with capacity n: all elements are pushed, then popped
// (largest first) into x[n-1], x[n-2], ..., x[0].
template <typename T,
          class Heap=Heap<T>>
constexpr void heap_sort(T *x,
                         const size_t n) noexcept {
    if (n < 2) {
        return;
    }

    Heap heap(n);
    for (size_t i = 0; i < n; ++i) {
        heap.push(x[i]);
    }

    for (size_t i = n; i-- > 0; ) {
        heap.pop(x[i]);
    }
}

// Sort x[] into descending order.
template <typename Type>
constexpr void heap_sort_descending(Type *x,
                                    const size_t n) noexcept {
    heap_sort( x, n );
    // reverse x[]
    for (size_t i = 0, j = n; i + 1 < j; ++i) {
        --j;
        cryptanalysislib::swap(x[i], x[j]);
    }
}
