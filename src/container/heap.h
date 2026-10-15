#pragma once

#include <vector>
#include <functional>
#include <cassert>
#include <cstddef>
#include <cstdint>


#include "alloc/alloc.h"

/// An indexed binary heap over the keys 0, 1, 2, ...
/// Every key can be in the heap at most once; its position is tracked, so
/// `update_key` can restore the heap after the priority of a key changed.
/// `Comp(a, b) == true` means `a` is closer to the top (std::less: min-heap).
template <class T,
          class Comp = std::less<T>,
          class Allocator = cryptanalysislib::allocator<T>>
struct Heap2 {
private:
    /// marks a key that is currently not in the heap
    constexpr static size_t npos = ~size_t(0);

    std::vector<T, Allocator> q;
    std::vector<size_t> loc;
    Comp op;

    /// \param i[in]: position in the heap
    /// \param j[in]: position in the heap
    /// \return true if q[i] has a higher priority than q[j]
    constexpr bool cmp(const size_t i,
                       const size_t j) noexcept {
        return op(q[i], q[j]);
    }

    /// swaps the elements at the positions i and j and updates their locations
    /// \param i[in]: position in the heap
    /// \param j[in]: position in the heap
    constexpr void swp(const size_t i,
                       const size_t j) noexcept {
        const T t = q[i];
        q[i] = q[j];
        q[j] = t;
        loc[q[i]] = i;
        loc[q[j]] = j;
    }

public:
    Heap2() : op(Comp()) { }

    /// move the element at position i up
    void swim(size_t i) noexcept {
        while (i > 0) {
            const size_t p = (i - 1) / 2;
            if (!cmp(i, p)) {
                break;
            }
            swp(i, p);
            i = p;
        }
    }

    /// move the element at position i down
    void sink(size_t i) noexcept {
        for (size_t j; (j = 2*i + 1) < q.size(); i = j) {
            if (j+1 < q.size() && cmp(j+1, j)) { ++j; }
            if (!cmp(j, i)) {
                break;
            }
            swp(j, i);
        }
    }

    /// insert the key n, which must not be in the heap already
    void push(const T n) noexcept {
        while (n >= loc.size()) {
            loc.push_back(npos);
        }

        assert(loc[n] == npos);
        loc[n] = q.size();
        q.push_back(n);
        swim(q.size() - 1);
    }

    T top() noexcept{
        assert(!empty());
        return q[0];
    }

    /// remove and return the key with the highest priority
    T pop() noexcept {
        const T res = top();
        q[0] = q.back();
        q.pop_back();
        loc[res] = npos;
        if (!q.empty()) {
            loc[q[0]] = 0;
            sink(0);
        }
        return res;
    }

    /// restore the heap property for the whole array
    void heapify() {
        for (size_t i = q.size() / 2; i-- > 0; ) {
            sink(i);
        }
    }

    /// restore the heap after the priority of key n changed
    void update_key(const T n) {
        assert(n < loc.size() && loc[n] != npos);
        swim(loc[n]);
        sink(loc[n]);
    }

    size_t size() const noexcept {
        return q.size();
    }

    bool empty() const noexcept {
        return q.empty();
    }

    void clear() {
        q.clear();
        loc.clear();
    }
};


/// A binary heap with a fixed capacity.
/// `Comp(a, b) == true` means `a` is below `b` (std::less: max-heap).
template <typename T,
          class Comp = std::less<T>,
          class Allocator = cryptanalysislib::allocator<T>>
class Heap {
private:
    Allocator allocator;
    /// data, x[0] is the top of the heap
    T *x;
    /// current number of elements
    size_t n_;
    /// capacity
    size_t s_;
    Comp op;

    Heap(const Heap&) = delete;
    Heap& operator = (const Heap&) = delete;

    /// Subject to the condition that the trees below the children of node
    /// k are heaps, move the element t (placed at k) down until the tree
    /// below node k is a heap.
    void sift_down(size_t k, const T t) noexcept {
        while (true) {
            size_t c = 2*k + 1;
            if (c >= n_) {
                break;
            }
            if (c + 1 < n_ && op(x[c], x[c + 1])) {
                ++c;
            }
            if (!op(t, x[c])) {
                break;
            }
            x[k] = x[c];
            k = c;
        }
        x[k] = t;
    }

public:
    /// \param n[in]: capacity
    explicit Heap(const size_t n) noexcept : n_(0), s_(n), op(Comp()) {
        x = allocator.allocate(s_);
    }

    ~Heap() noexcept {
        allocator.deallocate(x, s_);
    }

    /// \return number of elements
    constexpr size_t size() const noexcept {
        return n_;
    }

    /// \return capacity
    constexpr size_t capacity() const noexcept {
        return s_;
    }

    constexpr bool empty() const noexcept {
        return n_ == 0;
    }

    /// Return 0 if x[] has heap property
    /// else index (one-based) of node found to be greater than its parent.
    uint64_t test_heap() const noexcept {
        for (uint64_t k = n_; k > 1; --k) {
            // parent(k), both one-based
            const size_t t = (k>>1);
            if (op(x[t - 1], x[k - 1])) {
                return k-1;
            }
        }

        // has heap property
        return 0;
    }

    /// Insert t and restore heap-property. Complexity is O(log(n)).
    /// \return number of elements after the insertion, 0 if the heap is full
    size_t push(const T &t) noexcept {
        if (n_ >= s_) {
            return 0;
        }

        // move towards root as needed
        size_t j = n_++;
        while (j > 0) {
            const size_t p = (j - 1) / 2;
            if (!op(x[p], t)) {
                break;
            }
            x[j] = x[p];
            j = p;
        }
        x[j] = t;
        return n_;
    }

    /// \return the top element. Undefined for an empty heap.
    T top() const noexcept {
        assert(n_ != 0);
        return x[0];
    }

    /// Remove the top element and restore the heap structure.
    /// \param z[out]: the removed element (untouched if the heap is empty)
    /// \return number of elements before the removal, 0 if the heap was empty
    size_t pop(T &z) noexcept {
        if (n_ == 0) {
            return 0;
        }

        const size_t ret = n_;
        z = x[0];
        --n_;
        if (n_ != 0) {
            sift_down(0, x[n_]);
        }
        return ret;
    }

    /// Remove all elements.
    constexpr void clear() noexcept {
        n_ = 0;
    }
};
