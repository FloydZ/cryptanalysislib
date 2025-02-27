#pragma once

template<class T,
         const uint32_t n,
         const uint32_t k>
class enumeration_colex {
private:
    T val = first_comb();

    constexpr static size_t BITS = sizeof(T) * 8;
    static_assert(n <= BITS);
    static_assert(k <= n);
public:

	/// Return the first combination of (i.e. smallest word with) k bits,
	/// i.e.  00..001111..1 (k low bits set)
	/// Must have:  0 <= k <= BITS_PER_LONG
	constexpr static inline T first_comb() noexcept {
		if (k == 0) return 0;// shift with BITS_PER_LONG is undefined
		return ~0UL >> (BITS- k);
	}
	
    /// Return the first combination of (i.e. smallest word with) k bits,
	/// i.e.  00..001111..1 (k low bits set)
	/// Must have:  0 <= k <= BITS_PER_LONG
	constexpr static inline T first_comb(const T k_) noexcept {
		if (k == 0) return 0;// shift with BITS_PER_LONG is undefined
		return ~0UL >> (BITS- k_);
	}


	/// Return the last combination of (biggest n-bit word with) k bits
	/// i.e.  1111..100..00 (k high bits set)
	/// Must have:  0 <= k <= n <= BITS_PER_LONG
	constexpr static inline T last_comb() noexcept {
		return first_comb(k) << (n - k);
	}

	/// Return smallest integer greater than x with the same number of bits set.
	///
	/// colex order (5 over 3):
	///   set        word    set reversed (sorted!)
	///  0  1  2    ..111    2  1  0
	///  0  1  3    .1.11    3  1  0
	///  0  2  3    .11.1    3  2  0
	///  1  2  3    .111.    3  2  1
	///  0  1  4    1..11    4  1  0
	///  0  2  4    1.1.1    4  2  0
	///  1  2  4    1.11.    4  2  1
	///  0  3  4    11..1    4  3  0
	///  1  3  4    11.1.    4  3  1
	///  2  3  4    111..    4  3  2
	///
	///  Examples:
	///    000001 -> 000010 -> 000100 -> 001000 -> 010000 -> 100000
	///    000011 -> 000101 -> 000110 -> 001001 -> 001010 -> 001100 -> 010001 -> ...
	///    000111 -> 001011 -> 001101 -> 001110 -> 010011 -> 010101 -> 010110 -> ...
	///
	///  Special cases:
	///    0 -> 0
	///    all bits on the high side (i.e. last combination) -> 0
	///.
	/// based on code by Doug Moore / Glenn Rhoads
	/// note: might want to use bitscan near end
	constexpr static inline T next(T x) noexcept {
		T r = x & -x;// lowest set bit
		x += r;          // replace lowest block by a one left to it

		if (0 == x) return 0;// input was last combination

		T z = x & -x;// first zero beyond lowest block
		z -= r;          // lowest block  (cf. lowest_block())

		while (0 == (z & 1)) { z >>= 1; }// move block to low end of word
		return x | (z >> 1);             // need one bit less of low block
	}

	// Inverse of next_colex_comb()
	constexpr static inline T prev(T x) noexcept {
		x = next_colex_comb(~x);
		if (0 != x) x = ~x;
		return x;
	}

public:
    constexpr enumeration_colex() noexcept {};

	///
    constexpr inline T next() noexcept {
        const T ret = val;
        val = next(val);
        return ret;
    }

	///
    constexpr inline T prev() noexcept {
        const T ret = val;
        val = prev(val);
        return ret;
    } 
};
