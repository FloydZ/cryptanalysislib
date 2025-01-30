#pragma once 

#include <vector>
#include <cstdint>
#include <cstdlib>


template <typename T>
class SegmentTree {
    constexpr static size_t n = 1 << 20;
    T t[2 * n];
    // combine must be an associative function!
    T combine(const T l, 
              const T r) noexcept { 
        // or max(l,r) etc
        return l+r; 
    } 

    // 
    void build() noexcept {
        for (int64_t i = n; --i; ) {
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


template <typename T>
struct lazy_segment_tree {

    struct node {
        int l, r, x, lazy;
        node() {}
        node(int _l, int _r) : l(_l), r(_r), x(INT_MAX), lazy(0) {}
        node(int _l, int _r, int _x) : node(_l,_r){x=_x;}
        node(node a,node b) : node(a.l,b.r){
            x = std::min(a.x, b.x);
        }
        void update(int v) { x = v; }
        void range_update(int v) { lazy = v; }
        void apply() { x += lazy; lazy = 0; }
        void push(node &u) { u.lazy += lazy; }
    };

    int n;
    std::vector<node> arr;
    lazy_segment_tree() { }

    lazy_segment_tree(const std::vector<T> &a) : n(sz(a)), arr(4*n) {
        mk(a,0,0,n-1); 
    }

    node mk(const std::vector<T> &a, int i, int l, int r) {
        int m = (l+r)/2;
        return arr[i] = l > r  ? node(l,r) : 
                        l == r ? node(l,r,a[l]) :
        node(mk(a,2*i+1,l,m),mk(a,2*i+2,m+1,r));
    }

    node update(int at, const T v, int i=0) {
        propagate(i);
        int hl = arr[i].l, hr = arr[i].r;
        if (at < hl || hr < at) { return arr[i]; }
        if (hl == at && at == hr) {
            arr[i].update(v); return arr[i]; 
        }
        return arr[i] = node(update(at,v,2*i+1),update(at,v,2*i+2));
    }

    node query(int l, int r, int i=0) {
        propagate(i);
        int hl = arr[i].l, hr = arr[i].r;
        if (r < hl || hr < l) { return node(hl,hr); }
        if (l <= hl && hr <= r) { return arr[i]; }
        return node(query(l,r,2*i+1),query(l,r,2*i+2));
    }

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
