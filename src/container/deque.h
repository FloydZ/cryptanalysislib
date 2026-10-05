#pragma once 

#include <cstdlib>
#include "alloc/alloc.h"

struct DequeConfig {
    const size_t growSize = 0;
};
constexpr static DequeConfig dequeConfig;

// deque := double-ended queue
// Can grow dynamically
template <typename Type,
          typename Allocator = cryptanalysislib::allocator<Type>,
          const DequeConfig config = dequeConfig>
class Deque {
public:
    /// data (ring buffer)
    Type *x_;
    
    /// allocated size (# of elements)
    size_t s_;

    /// current number of entries in buffer
    size_t n_;

    /// position of first element in buffer
    size_t fpos_;

    /// insert_first() will write to (fpos-1)%n
    size_t lpos_;  // position of last element in buffer plus one
    // insert_last() will write to  lpos,  n==(lpos-fpos) (mod s)
    // entries are at [fpos, ..., lpos-1]  (range may be empty)
    
    /// 
    constexpr static bool gq_ = config.growSize;  

    Deque(const Deque&) = delete;
    Deque & operator = (const Deque&) = delete;

public:

    /// \param n[in]:
    explicit Deque(const size_t n) noexcept {
        s_ = n;
        x_ = Allocator::allocator(s_);
        n_ = 0;
        fpos_ = 0;
        lpos_ = 0;
    }

    ~Deque() noexcept { 
        Allocator::deallocate(x_);
    }

    /// \return current numbers of elements in the queue
    constexpr size_t size() const noexcept { 
        return n_; 
    }

    /// \return current capacity
    constexpr size_t capacity() const noexcept { 
        return s_; 
    }

    /// Return number of entries after insertion.
    /// Zero is returned on failure
    ///   (i.e. space exhausted and 0==gq_)
    size_t insert_first(const Type &z) noexcept {
        if ( n_ >= s_ ) {
            if ( 0==gq_ ) {
                // growing disabled
                return 0;  
            }
            grow();
        }

        --fpos_;
        if ( fpos_ == -1UL ) { fpos_ = s_ - 1; }
        x_[fpos_] = z;
        ++n_;
        return  n_;
    }

    /// Return number of entries after insertion.
    /// Zero is returned on failure
    ///   (i.e. space exhausted and 0==gq_)
    size_t insert_last(const Type &z) noexcept {
        if ( n_ >= s_ ) {
            // growing disabled
            if constexpr (0 == gq_) { 
                return 0;
            }
            grow();
        }

        x_[lpos_] = z;
        ++lpos_;
        if ( lpos_>=s_ )  lpos_ = 0;
        ++n_;
        return  n_;
    }

    //// Return number of elements before extract.
    //// Return 0 if extract on empty deque was attempted.
    size_t extract_first(const Type &z) noexcept {
        if ( 0==n_ )  return 0;
        z = x_[fpos_];
        ++fpos_;
        if ( fpos_ >= s_ )  fpos_ = 0;
        --n_;
        return  n_ + 1;
    }

    // Return number of elements before extract.
    // Return 0 if extract on empty deque was attempted.
    size_t extract_last(Type & z) noexcept {
        if ( 0==n_ )  return 0;
        --lpos_;
        if ( lpos_ == -1UL )  lpos_ = s_ - 1;
        z = x_[lpos_];
        --n_;
        return  n_ + 1;
    }

    // Read (but don't remove) first entry.
    // Return number of elements (i.e. on error return zero).
    size_t read_first(Type & z) const noexcept {
        if ( 0==n_ )  return 0;
        z = x_[fpos_];
        return n_;
    }

    // Read (but don't remove) last entry.
    // Return number of elements (i.e. on error return zero).
    size_t read_last(Type & z) const noexcept {
        return  read(n_-1, z);  // ok for n_==0
    }

    // Read entry k (that is, [(fpos_ + k)%s_]).
    // Return 0 if k>=n_ else return k+1
    size_t read(const size_t k,
                Type & z) const noexcept {
        if ( k>=n_ )  return 0;
        size_t j = fpos_ + k;
        if ( j>=s_ )  j -= s_;
        z = x_[j];
        return  k + 1;
    }

private:

    // Reverse order of array f.
    constexpr inline void reverse(Type *f,
                                  const size_t n) noexcept {
        if ( n >= 2 ) {
            for (size_t k=0, i=n-1;  k<i;  ++k, --i) {
                swap2(f[k], f[i]);
            }
        }
    }

    // Rotate towards element #0
    // Shift is taken modulo n
    constexpr void rotate_left(Type *f, 
                               size_t n,
                               size_t s) noexcept {
        if ( n <= 1 ) { return; } // nothing to do
        if ( s >= n ) { s %= n; }
        if ( s == 0 ) { return; } // nothing to do
    
        reverse(f,   s);
        reverse(f+s, n-s);
        reverse(f,   n);
    }

    /// 
    void grow() noexcept {
        size_t ns = s_ + gq_;  // new size
        // Move read-position to zero:
        rotate_left(x_, s_, fpos_);
        Type *a = Allocator::allocate(ns);
        cryptanalysislib::memcpy(a, x_, s_);
        Allocator::deallocate(x_);
        x_ = a;

        // x_ = ReAlloc<Type>(x_, ns, s_);
        fpos_ = 0;
        lpos_ = n_;
        s_ = ns;
    }
};
