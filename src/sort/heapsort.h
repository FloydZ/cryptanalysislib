#pragma once


#include "container/heap.h"


// Sort x[] into ascending order.
template <typename T,
          class Heap=cryptanalysislib::heap2<T>>
void heap_sort(T *x, ulong n) {
    Heap heap(x, n);
    // one-based for heapify()
    T *p = x - 1;  
    for (ulong k=n; k>1; --k) {
        swap2(p[1], p[k]);  // move largest element (p[1]) to end of array
        --n;                // remaining array has one element less
        heapify(p, n, 1);   // restore heap-property
    }
}

template <typename Type>
void heap_sort_descending(Type *x,
                          const size_t n) // Sort x[] into descending order.
{
    heap_sort( x, n );
    reverse( x, n );
}
