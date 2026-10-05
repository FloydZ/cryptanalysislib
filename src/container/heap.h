#pragma once

#include <vector>
#include <functional>
#include <cassert>


#include "alloc/alloc.h"

// An implementation of a binary heap.
template <class T, 
          class Comp = std::less<T>,
          class Allocator = cryptanalysislib::allocator<T>>
struct Heap2 {
private:
    std::vector<T, Allocator> q, loc;
    Comp op;
    
    /// \param i[in]:
    /// \param j[in]:
    /// \return i == j
    constexpr bool cmp(const size_t i,
                       const size_t j) noexcept { 
        return op(q[i], q[j]); 
    }

    /// \param i[in]:
    /// \param j[in]:
    constexpr void swp(const size_t i,
                       const size_t j) noexcept {
        swap(q[i], q[j]), swap(loc[q[i]], loc[q[j]]);
    }

public:
    Heap2() : op(Comp()) { }

    /// \param i[in]:
    void swim(size_t i) noexcept {
        for (size_t p; i; swp(i, p), i = p) {
            if (!cmp(i, p=(i-1)/2)) { 
                break; 
            }
        }
    }

    /// \param i[in]:
    void sink(T i) noexcept {
        for (T j; (j=2*i+1)<sz(q); swp(j, i), i=j) {
            if (j+1 < q.size() && cmp(j+1, j)) { ++j; }
            if (!cmp(j, i)) { 
                break; 
            }
        }
    }

    /// \param n[in]:
    void push(T n) noexcept {
        while (n >= sz(loc)) { 
            loc.pb(-1);
        }

        assert(loc[n] == -1);
        loc[n] = sz(q), q.pb(n);
        swim(sz(q) - 1);
    }

    T top() noexcept{ 
        assert(!empty()); 
        return q[0]; 
    }

    T pop() noexcept {
        T res = top();
        q[0] = q.back(), q.pop_back();
        loc[q[0]]=0, loc[res] = -1;
        sink(0); 
        return res;
    }

    void heapify() {
        for (int i=sz(q); --i; ) {
            if (cmp(i, (i-1)/2)) { 
                swp(i, (i-1)/2);
            }
        }
    }

    void update_key(int n) {
        assert(loc[n] != -1);
        swim(loc[n]), sink(loc[n]);
    }

    int size() { 
        return sz(q);
    }

    bool empty() { 
        return !size(); 
    }

    void clear() { 
        q.clear(), 
        loc.clear(); 
    }
};


template <typename T,
          class Comp = std::less<T>,
          class Allocator = cryptanalysislib::allocator<T>>
class Heap {
private:
    Allocator allocator;
    T *x;

public:
    Heap(const size_t n) noexcept {
        x = allocator.allocate(n);
    }

    /// Return 0 if x[] has heap property
    /// else index of node found to be greater than its parent.
    ulong test_heap(const ulong n) noexcept {
        const T *p = x - 1;  // make one-based
        for (ulong k=n; k>1; --k) {
            // parent(k)
            const size_t t = (k>>1);  
            // in {1, 2, ..., n}
            if ( p[t]<p[k] ) { 
                return k-1; 
            }
        }

        // has heap property
        return 0;  
    }
    
    /// Subject to the condition that the trees below the children of node
    /// k are heaps, move the element z[k] (down) until the tree below node 
    /// k is a heap.
    /// Data expected in z[1,2,...,n].
    void heapify(const ulong n, 
                 const ulong k) noexcept {
        // index of max of k, left(k), and right(k)
        size_t m = k;  
    
        // left(k);
        const size_t l = (k<<1);  
        // left child (exists and) greater than k
        if ((l <= n) && (x[l] > x[k])) { m = l; }
    
        // right(k);
        const size_t r = (k<<1) + 1;  
        // right child (exists and) greater than max(k,l)
        if ((r <= n) && (x[r] > x[m])) { m = r; } 
    
        if ( m != k ) { 
            // need to swap
            std::swap(x[k], x[m]);
            heapify(x, n, m);
        }
    }
    
    /// Reorder data to a heap.
    /// Data expected in x[0,1,...,n-1].
    void heap(const size_t n) noexcept {
        // make one-based
        T *z = x - 1;
        // max index such that node has at least one child
        ulong j = (n>>1);  
        while ( j > 0 ) {
            heapify(z, n, j);
            --j;
        }
        //    for (ulong j=(n>>1); j>0; --j)  heapify(z, n, j);
    }
    
    /// With x[] a heap of current size n
    /// and max size s (i.e. space for s elements allocated),
    /// insert t and restore heap-property.
    /// Return true if successful, else (i.e. if space exhausted) false.
    /// Complexity is O(log(n)).
    bool push(const T &t,
                ulong n,
                const ulong s) noexcept {
        if (n > s) {
            return false;
        }

        ++n;
        T *x1 = x - 1;  // make one-based
        ulong j = n;
        while ( j > 1 ) { // move towards root as needed
            ulong k = (j>>1);  // k==parent(j)
            if ( x1[k] >= t )  break;
            x1[j] = x1[k];
            j = k;
        }
        x1[j] = t;
        return true;
    }
    
    /// Return maximal element of heap and restore heap structure.
    /// Return value is undefined for 0==n.
    T extract_max(const size_t n) noexcept {
        T m = x[0];
        if ( 0 != n ) {
            T *x1 = x - 1;
            x1[1] = x1[n];
            heapify(x1, n-1, 1);
        }
        return m;
    }
};
