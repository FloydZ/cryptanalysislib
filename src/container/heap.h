#pragma once

#include <vector>
#include <functional>
#include <cassert>

// 2.6. Heap. An implementation of a binary heap.
template <class T, 
          class Comp = std::less<T>>
struct heap {
    std::vector<T> q, loc; Comp op;
    heap() : op(Comp()) {}

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
