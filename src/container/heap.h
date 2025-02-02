#pragma once

#include <vector>
#include <functional>
#include <cassert>


#include "alloc/alloc.h"

// 2.6. Heap. An implementation of a binary heap.
template <class T, 
          class Comp = std::less<T>,
          class Allocator = cryptanalysislib::allocator>
struct heap {
private:
    std::vector<T, Allocator> q, loc; Comp op;

public:
    heap() : op(Comp()) {
    }

    bool cmp(int i, int j) { 
        return op(q[i], q[j]); 
    }

    void swp(int i, int j) {
        swap(q[i], q[j]), swap(loc[q[i]], loc[q[j]]);
    }

    void swim(int i) {
        for (int p; i; swp(i, p), i = p) {
            if (!cmp(i, p=(i-1)/2)) { 
                break; 
            }
        }
    }

    void sink(int i) {
        for (int j; (j=2*i+1)<sz(q); swp(j, i), i=j) {
            if (j+1 < sz(q) && cmp(j+1, j)) { ++j; }
            if (!cmp(j, i)) { break; }
        }
    }

    void push(int n) {
        while (n >= sz(loc)) { 
            loc.pb(-1);
        }
        assert(loc[n] == -1);
        loc[n] = sz(q), q.pb(n);
        swim(sz(q) - 1);
    }

    int top() { 
        assert(!empty()); 
        return q[0]; 
    }
    int pop() {
        int res = top();
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


template <typename Type>
class heap2{

    // Return 0 if x[] has heap property
    // else index of node found to be greater than its parent.
    ulong test_heap(const Type *x, const ulong n) {
        const Type *p = x - 1;  // make one-based
        for (ulong k=n; k>1; --k) {
            // parent(k)
            const size_t t = (k>>1);  
            // in {1, 2, ..., n}
            if ( p[t]<p[k] ) { return k-1; }
        }
        // has heap property
        return 0;  
    }
    
    // Subject to the condition that the trees below the children of node
    // k are heaps, move the element z[k] (down) until the tree below node k is a heap.
    // Data expected in z[1,2,...,n].
    void heapify(Type *z, 
                 const ulong n, 
                 const ulong k) {
        ulong m = k;  // index of max of k, left(k), and right(k)
    
        const ulong l = (k<<1);  // left(k);
        if ( (l <= n) && (z[l] > z[k]) )  m = l;  // left child (exists and) greater than k
    
        const ulong r = (k<<1) + 1;  // right(k);
        if ( (r <= n) && (z[r] > z[m]) )  m = r;  // right child (exists and) greater than max(k,l)
    
        if ( m != k )  // need to swap
        {
            swap2(z[k], z[m]);
            heapify(z, n, m);
        }
    }
    
    // Reorder data to a heap.
    // Data expected in x[0,1,...,n-1].
    void heap(Type *x,
              const ulong n) noexcept {
        Type *z = x - 1;   // make one-based
        ulong j = (n>>1);  // max index such that node has at least one child
        while ( j > 0 ) {
            heapify(z, n, j);
            --j;
        }
        //    for (ulong j=(n>>1); j>0; --j)  heapify(z, n, j);
    }
    
    // With x[] a heap of current size n
    // and max size s (i.e. space for s elements allocated),
    // insert t and restore heap-property.
    // Return true if successful, else (i.e. if space exhausted) false.
    // Complexity is O(log(n)).
    bool insert(Type *x,
                ulong n,
                ulong s,
                Type t) noexcept {
        if ( n > s )  return false;
        ++n;
        Type *x1 = x - 1;  // make one-based
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
    
    // Return maximal element of heap and restore heap structure.
    // Return value is undefined for 0==n.
    Type heap_extract_max(Type *x, 
                          ulong n) noexcept {
        Type m = x[0];
        if ( 0 != n ) {
            Type *x1 = x - 1;
            x1[1] = x1[n];
            --n;
            heapify(x1, n, 1);
        }
        return m;
    }
};
