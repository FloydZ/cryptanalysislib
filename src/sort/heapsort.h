#pragma once

#include <cstddef>
#include "container/heap.h"


// Sort x[] into ascending order.
template <typename T,
          class Heap=Heap<T>>
constexpr void heap_sort(T *x,
                         const size_t n) noexcept {
    Heap heap(x, n);
    size_t m = n;
    // one-based for heapify()
    T *p = x - 1;  
    for (size_t k=m; k>1; --k) {
        swap2(p[1], p[k]);  // move largest element (p[1]) to end of array
        --m;                // remaining array has one element less
        heapify(p, n, 1);   // restore heap-property
    }
}

// Sort x[] into descending order.
template <typename Type>
constexpr void heap_sort_descending(Type *x,
                                    const size_t n) noexcept {
    heap_sort( x, n );
    reverse( x, n );
}
