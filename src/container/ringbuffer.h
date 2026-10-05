#pragma once

#include <cstddef>
#include "alloc/alloc.h"

// Implementation of a ring buffer
template <typename Type,
          typename Allocator = cryptanalysislib::allocator<Type>>
class RingBuffer {
private:
    Allocator allocator;

    // data (ring buffer)
    Type *x_;   
    
    // allocated size (# of elements)
    size_t s_;   
    
    // current number of entries in buffer
    size_t n_;   
    
    // next position to write in buffer
    size_t wpos_;  
    
    // first position to read in buffer
    size_t fpos_;  

    RingBuffer(const RingBuffer&) = delete;
    RingBuffer & operator = (const RingBuffer&) = delete;

public:
    explicit RingBuffer(size_t n) noexcept {
        s_ = n;
        x_ = allocator.allocator(s_);
        // x_ = new Type[s_];
        n_ = 0;
        wpos_ = 0;
        fpos_ = 0;
    }

    ~RingBuffer()  { 
        // delete [] x_; 
        allocator.deallocate(x_, s_);
    }

    [[nodiscard]] constexpr size_t size() const noexcept {
        return n_; 
    }

    [[nodiscard]] constexpr size_t capacity() const noexcept {
        return s_; 
    }

    constexpr void insert(const Type &z) noexcept {
        x_[wpos_] = z;
        if ( ++wpos_>=s_ )  wpos_ = 0;
        if ( n_ < s_ )  ++n_;
        else  fpos_ =  wpos_;
    }

    // Read entry k (that is, [(fpos_ + k)%s_]).
    // Return 0 if k>=n, else return k+1.
    constexpr size_t read(size_t k, Type &z)  const noexcept  {
        if ( k>=n_ )  return 0;
        size_t j = fpos_ + k;
        if ( j>=s_ )  j -= s_;
        z = x_[j];
        return  k + 1;
    }
};

