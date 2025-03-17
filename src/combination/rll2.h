#pragma once 

// Run length limited (RLL) words and Fibonacci words.
// The RLL words are in lexicographic order,
// the Fibonacci words are in a minimal change order (Gray code).
// 01001001001010
// 01001001001011
// 01001001001100
// 01001001001101
// 01001001010010
// 01001001010011
// 01001001010100
// 01001001010101
// 01001001010110
// 01001001011001
// 01001001011010
// 01001001011011
// 01001001100100
// 01001001100101
// 01001001100110
// 01001001101001
// 01001001101010
// 01001001101011
// 01001001101100
template<typename T>
class bit_rll2 {
private:
    constexpr static size_t BITS = sizeof(T) * 8u;
    T w_;  // RLL-word

public:
    constexpr bit_rll2() noexcept { first(); }

    constexpr void first() noexcept {
        w_ = 1;
        T s = 3;  // max run length + 1
        while (s <= BITS ) {
            w_ |= w_ << s;
            s <<= 1;
        }
    }

    constexpr void last() noexcept {
        first();
        w_ <<= 2;  // shift by max run length
        w_ = ~w_;
    }

    // RLL word corresponding to all-zero Fibonacci word
    constexpr void middle() noexcept {
        w_ = 0xaaaaaaaaaaaaaaaaUL;
    }

private:
    constexpr T step(T x) {
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
    constexpr T next() noexcept { return step( w_ ); }
    constexpr T prev() noexcept { return step( ~w_ ); }

    // RLL word (lexicographic order)
    constexpr T data() const noexcept 
    { return w_; }

    // Fibonacci word (Gray code)
    constexpr T fib() const noexcept 
    { return  ~( w_ ^ (w_ >> 1) ); }

    constexpr T next_fib() noexcept { next(); return fib(); }
    constexpr T prev_fib() noexcept { prev(); return fib(); }
};
