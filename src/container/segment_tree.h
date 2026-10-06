#pragma once

#include <vector>
#include <cstdint>
#include <cstdlib>

#include "alloc/alloc.h"

/// Iterative segment tree over the positions [0, n) with point updates and
/// range queries. Every position starts with the value 0.
/// \tparam T value type
/// \tparam n number of positions
template <typename T,
          const size_t n = 1u << 20,
          class Allocator = cryptanalysislib::allocator<T>>
class SegmentTree {
    Allocator allocator;
    // the tree has 2n nodes, the leaves are t[n, 2n)
    T *t;

    SegmentTree(const SegmentTree&) = delete;
    SegmentTree& operator = (const SegmentTree&) = delete;

    // combine must be an associative function with identity 0!
    constexpr static T combine(const T l,
                               const T r) noexcept {
        // or max(l,r) etc
        return l+r;
    }

public:
    SegmentTree() noexcept {
        t = allocator.allocate(2 * n);
        for (size_t i = 0; i < 2 * n; i++) {
            t[i] = T(0);
        }
    }

    ~SegmentTree() noexcept {
        allocator.deallocate(t, 2 * n);
    }

    /// \return number of positions
    constexpr static size_t size() noexcept {
        return n;
    }

    /// set the value of position pos without updating the inner nodes;
    /// call build() afterwards
    void set(const size_t pos, const T v) noexcept {
        t[pos + n] = v;
    }

    /// recompute all inner nodes from the leaves
    void build() noexcept {
        for (size_t i = n; --i; ) {
            t[i] = combine(t[2 * i], t[2 * i + 1]);
        }
    }

    // set value v on position pos
    void update(const size_t pos, T v) noexcept {
        size_t i = pos;
        for (t[i+=n] = v; i /= 2; ) {
            t[i] = combine(t[2 * i], t[2 * i + 1]);
        }
    }

    // sum on interval [l, r)
    T query(const size_t left,
            const size_t right) const noexcept {
        size_t l = left, r = right;
        T resL = 0, resR = 0;
        for (l += n, r += n; l < r; l /= 2, r /= 2) {
            if (l & 1) { resL = combine(resL, t[l++]); }
            if (r & 1) { resR = combine(t[--r], resR); }
        }
        return combine(resL, resR);
    }
};


/// Segment tree with range-add updates, point assignments and range-min
/// queries over the positions [0, n).
template <typename T>
struct lazy_segment_tree {

    struct node {
        int l = 0, r = -1;
        T x{}, lazy{};
        // false for an empty range (identity of min)
        bool valid = false;

        node() {}
        node(int _l, int _r) : l(_l), r(_r) {}
        node(int _l, int _r, T _x) : l(_l), r(_r), x(_x), valid(true) {}
        node(const node &a, const node &b) : l(a.l), r(b.r) {
            if (a.valid && b.valid) {
                x = a.x < b.x ? a.x : b.x;
                valid = true;
            } else if (a.valid) {
                x = a.x;
                valid = true;
            } else if (b.valid) {
                x = b.x;
                valid = true;
            }
        }
        void update(const T v) { x = v; valid = true; }
        void range_update(const T v) { lazy = v; }
        void apply() {
            if (valid) {
                x += lazy;
            }
            lazy = T{};
        }
        void push(node &u) { u.lazy += lazy; }
    };

    int n = 0;
    std::vector<node> arr;
    lazy_segment_tree() { }

    lazy_segment_tree(const std::vector<T> &a) : n(static_cast<int>(a.size())), arr(4*a.size()) {
        if (n > 0) {
            mk(a,0,0,n-1);
        }
    }

    node mk(const std::vector<T> &a, int i, int l, int r) {
        int m = (l+r)/2;
        return arr[i] = l > r  ? node(l,r) :
                        l == r ? node(l,r,a[l]) :
        node(mk(a,2*i+1,l,m),mk(a,2*i+2,m+1,r));
    }

    /// a[at] = v
    node update(int at, const T v, int i=0) {
        propagate(i);
        int hl = arr[i].l, hr = arr[i].r;
        if (at < hl || hr < at) { return arr[i]; }
        if (hl == at && at == hr) {
            arr[i].update(v); return arr[i];
        }
        return arr[i] = node(update(at,v,2*i+1),update(at,v,2*i+2));
    }

    /// min(a[l..r]), both inclusive; read the result from `.x`
    node query(int l, int r, int i=0) {
        propagate(i);
        int hl = arr[i].l, hr = arr[i].r;
        if (r < hl || hr < l) { return node(hl,hr); }
        if (l <= hl && hr <= r) { return arr[i]; }
        return node(query(l,r,2*i+1),query(l,r,2*i+2));
    }

    /// a[l..r] += v, both inclusive
    node range_update(int l, int r, T v, int i=0) {
        propagate(i);
        int hl = arr[i].l, hr = arr[i].r;
        if (r < hl || hr < l) { return arr[i]; }
        if (l <= hl && hr <= r) {
            arr[i].range_update(v);
            propagate(i);
            return arr[i];
        }

        return arr[i] = node(range_update(l,r,v,2*i+1),
        range_update(l,r,v,2*i+2));
    }

    void propagate(int i) {
        if (arr[i].l < arr[i].r) {
            arr[i].push(arr[2*i+1]);
            arr[i].push(arr[2*i+2]);
        }
        arr[i].apply();
    }
};
