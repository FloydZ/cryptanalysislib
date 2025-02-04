#pragma once

#include <stack>
#include <cstddef>
#include "alloc/alloc.h"

struct StackConfig {
    const size_t growSize = 0;
};
constexpr static StackConfig stackConfig;

/// \tparam Type
/// \tparam Allocator
/// \tparam config
template <class Type,
          class Allocator = cryptanalysislib::allocator<Type>,
          const StackConfig config = stackConfig>
class Stack {
private:
    Allocator allocator;
    // data
    Type  *x_;

    // size
    size_t  s_;

    // stack pointer (position of next write), top entry @ p-1
    size_t  p_;

    /// grow by gq elements if necessary, 0 for "never grow"
    constexpr static size_t gq_ = config.growSize;  

public:
    /// \param n[in]: 
    explicit Stack(const size_t n) noexcept {
        s_ = n;
        x_ = allocator.allocate(s_);
        // x_ = (Type *)std::malloc( s_ * sizeof(Type) );
        p_ = 0;  
    }

    ~Stack() noexcept {
        allocator.deallocate(x_, s_);
        // std::free( x_ );
    }

private:
    Stack(const Stack&) = delete;
    Stack& operator = (const Stack&) = delete;

public:
    // Return number of entries.
    constexpr size_t size() const noexcept { 
        return p_;
    }
   
    /// \return capacity
    constexpr size_t capacity() const noexcept { 
        return p_;
    }

    /// Add element z on top of stack.
    /// Return size of stack, zero on stack overflow.
    /// If gq_ is nonzero the stack grows if needed.
    size_t push(const Type & z) noexcept {
        if ( p_ >= s_ ) {
            if constexpr ( 0 == gq_ ) {
                return 0;
            }
            grow();
        }

        x_[p_] = z;
        ++p_;

        return  s_;
    }

    /// Retrieve top entry and remove it.
    /// Return number of entries before removing element.
    /// If empty return zero and leave z undefined.
    size_t pop(Type &z) noexcept {
        const size_t ret = p_;
        if ( 0 != p_ )  { --p_;  z = x_[p_]; }
        return  ret;
    }

    /// Drop top entry.
    /// Return number of entries before removing element.
    /// If empty return zero.
    size_t pop() noexcept {
        const size_t ret = p_;
        if ( 0 != p_ )  --p_;
        return  ret;
    }

    // Modify top entry.
    // Return number of entries.
    // If empty return zero and do nothing.
    size_t poke(const Type z) noexcept {
        if ( 0 != p_ ) { x_[p_-1] = z; }
        return p_;
    }

    // Read top entry, without removing it.
    // Return number of entries.
    // If empty return zero and leave z undefined.
    size_t peek(Type &z) const noexcept {
        if ( 0 != p_ )  z = x_[p_-1];
        return p_;
    }

    // Read entry x[j], without removing anything.
    // Return number of entries.
    // If j is out of range return zero and leave z undefined.
    size_t peek_at(const size_t j,
                   Type &z) const noexcept {
        if ( j >= p_ )  return 0;
        z = x_[j];
        return p_;
    }

private:
    void grow() noexcept {
        if constexpr (gq_ != 0) {
            const size_t ns = s_ + gq_;  // new size
            // TODO x_ = ReAlloc<Type>(x_, ns, s_);
            s_ = ns;
        }
    }
};
