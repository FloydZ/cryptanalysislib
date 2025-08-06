#pragma once 

#include <stdint.h>
#include <stdlib.h>

/// Class for generating and manipulating Run Length Limited (RLL) and Fibonacci words
/// RLL words have restrictions on the maximum number of consecutive identical bits
/// The RLL words are enumerated in lexicographic order,
/// while the Fibonacci words are in a minimal change order (Gray code)
///
/// Example sequence:
/// 01001001001010
/// 01001001001011
/// 01001001001100
/// 01001001001101
/// 01001001010010
/// 01001001010011
/// 01001001010100
/// 01001001010101
/// 01001001010110
/// 01001001011001
/// 01001001011010
/// 01001001011011
/// 01001001100100
/// 01001001100101
/// 01001001100110
/// 01001001101001
/// 01001001101010
/// 01001001101011
/// 01001001101100
/// \tparam T[in]: integer type to represent the words
template<typename T>
class bit_rll2 {
private:
    /// Number of bits in type T
    constexpr static size_t BITS = sizeof(T) * 8u;
    
    /// Current RLL word value
    T w_;

public:
    /// Default constructor initializes to the first RLL word
    constexpr bit_rll2() noexcept { 
        first(); 
    }

    /// Sets the word to the first value in the RLL sequence
    /// Initializes with a pattern that has the maximum allowed run length
    constexpr void first() noexcept {
        w_ = 1;
        T s = 3;  // max run length + 1
        while (s <= BITS ) {
            w_ |= w_ << s;
            s <<= 1;
        }
    }

    /// Sets the word to the last value in the RLL sequence
    constexpr void last() noexcept {
        first();
        w_ <<= 2;  // shift by max run length
        w_ = ~w_;
    }

    /// Sets the word to the middle value in the sequence
    /// RLL word corresponding to all-zero Fibonacci word
    constexpr void middle() noexcept {
        w_ = 0xaaaaaaaaaaaaaaaaUL;
    }

private:
    /// Performs a single step in the RLL sequence
    /// \param x[in]: input value to process
    /// \return the new RLL word after the step
    [[nodiscard]] constexpr T step(T x) noexcept {
        x |= ( (x>>1) & (x>>2) ); // max run length 2
        // ==> Gray code with max 1 successive one (Fibonacci words)

        // x |= ( (x>>1) & (x>>2) & (x>>3) ); // max run length 3
        // ==> Gray code with max 2 successive ones

        // x |= ( (x>>1) & (x>>2) & (x>>4) ); // max run length 4
        // ==> Gray code with max 3 successive ones

        x ^= (x+1);
        w_ ^= x;
        return w_;
    }

public:
    /// Advances to the next RLL word in the sequence
    /// \return the next RLL word
    [[nodiscard]] constexpr T next() noexcept { 
        return step( w_ ); 
    }
    
    /// Moves to the previous RLL word in the sequence
    /// \return the previous RLL word
    [[nodiscard]] constexpr T prev() noexcept { 
        return step( ~w_ ); 
    }

    /// Gets the current RLL word (in lexicographic order)
    /// \return the current RLL word
    [[nodiscard]] constexpr T data() const noexcept { 
        return w_; 
    }

    /// Converts the current RLL word to a Fibonacci word (in Gray code order)
    /// \return the Fibonacci word corresponding to the current RLL word
    [[nodiscard]] constexpr T fib() const noexcept {
        return  ~( w_ ^ (w_ >> 1) ); 
    }

    /// Advances to the next RLL word and returns the corresponding Fibonacci word
    /// \return the next Fibonacci word
    [[nodiscard]] constexpr T next_fib() noexcept {
        next(); 
        return fib(); 
    }
    
    /// Moves to the previous RLL word and returns the corresponding Fibonacci word
    /// \return the previous Fibonacci word
    [[nodiscard]] constexpr T prev_fib() noexcept { 
        prev(); 
        return fib(); 
    }
};
