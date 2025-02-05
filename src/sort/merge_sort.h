#pragma once 

#include <cstddef>
#include "alloc/alloc.h"

// Merge the (sorted) arrays
//   A[] := f[0], f[1], ..., f[na-1]  and  B[] := f[na], f[na+1], ..., f[na+nb-1]
// into  t[] := t[0], t[1], ..., t[na+nb-1]  such that t[] is sorted.
// Must have: na >= 1 and nb >= 1
template<typename Type>
void merge(const Type * const __restrict__ f,
           size_t na,
           size_t nb,
           Type * const __restrict__ t) noexcept {
    const Type * const A = f;
    const Type * const B = f + na;
    size_t nt = na + nb;
    Type ta = A[--na],  tb = B[--nb];

    while ( true ) {
        if ( ta > tb ) { // copy ta
            t[--nt] = ta;
            if ( na==0 )  // A[] empty?
            {
                for (size_t j=0; j<=nb; ++j)  t[j] = B[j];  // copy rest of B[]
                return;
            }

            ta = A[--na];  // read next element of A[]
        } else { // copy tb
            t[--nt] = tb;
            if ( nb==0 ) {  // B[] empty?
                for (size_t j=0; j<=na; ++j)  t[j] = A[j];  // copy rest of A[]
                return;
            }

            tb = B[--nb];  // read next element of B[]
        }
    }
}

template <typename Type>
constexpr void merge_sort_rec(Type *f, 
                    const size_t n,
                    Type *t) noexcept {
    if ( n<8 ) {
        selection_sort(f, n);
        return;
    }

    const size_t na = n>>1;
    const size_t nb = n - na;

    merge_sort_rec(f, na, t);
    merge_sort_rec(f+na, nb, t);

    merge(f, na, nb, t);
    for (size_t j=0; j<n; ++j)  f[j] = t[j];   // copy back
}

/// \tparam
template <typename Type,
          class Allocator = cryptanalysislib::allocator<Type>>
constexpr void merge_sort(Type *f, size_t n, Type *tmp=nullptr) noexcept {
    Allocator allocator;
    Type *t = tmp;
    // if (tmp==nullptr)  t = new Type[n];
    if (tmp == nullptr) { allocator.allocate(n); }
    merge_sort_rec(f, n, t);
    // if (tmp==nullptr)  delete [] t;
    if (tmp == nullptr) { allocator.deallocate(tmp, n); }
}

template <typename Type>
constexpr void merge_sort_rec4(Type *f, size_t n, Type *t) noexcept {
    // threshold must be at least 8
    if(n < 8) {
        selection_sort(f, n);
        return;
    }

    // left and right half:
    const size_t na = n>>1;
    const size_t nb = n - na;

    // left quarters:
    const size_t na1 = na>>1;
    const size_t na2 = na - na1;
    merge_sort_rec4(f, na1, t);
    merge_sort_rec4(f+na1, na2, t);

    // right quarters:
    const size_t nb1 = nb>>1;
    const size_t nb2 = nb - nb1;
    merge_sort_rec4(f+na, nb1, t);
    merge_sort_rec4(f+na+nb1, nb2, t);

    // merge quarters (F-->T):
    merge(f, na1, na2, t);
    merge(f+na, nb1, nb2, t+na);

    // merge halves (T-->F):
    merge(t, na, nb, f);
}

template <typename Type,
          class Allocator = cryptanalysislib::allocator<Type>>
constexpr void merge_sort4(Type *f, size_t n, Type *tmp=nullptr) noexcept {
    Allocator allocator;
    Type *t = tmp;
    // if (tmp==nullptr)  t = new Type[n];
    if (tmp == nullptr) { allocator.allocate(n); }
    merge_sort_rec(f, n, t);
    // if (tmp==nullptr)  delete [] t;
    if (tmp == nullptr) { allocator.deallocate(tmp, n); }
}
