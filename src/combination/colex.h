#pragma once

template<class T>
class colex {
private:
    constexpr static size_t BITS = sizeof(T) * 8;
public:
	/// Return the first combination of (i.e. smallest word with) k bits,
	/// i.e.  00..001111..1 (k low bits set)
	/// Must have:  0 <= k <= BITS_PER_LONG
	static inline T first_comb(const T k){
		if (k == 0) return 0;// shift with BITS_PER_LONG is undefined
		return ~0UL >> (BITS- k);
	}


	/// Return the last combination of (biggest n-bit word with) k bits
	/// i.e.  1111..100..00 (k high bits set)
	/// Must have:  0 <= k <= n <= BITS_PER_LONG
	static inline T last_comb(T k,
                              const T n = BITS) {
		//    if ( BITS_PER_LONG == k )  return  ~0UL;
		//    else return  ((1UL<<k)-1) << (n - k);
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
	static inline T next_colex_comb(T x) {
		T r = x & -x;// lowest set bit
		x += r;          // replace lowest block by a one left to it

		if (0 == x) return 0;// input was last combination

		T z = x & -x;// first zero beyond lowest block
		z -= r;          // lowest block  (cf. lowest_block())

		while (0 == (z & 1)) { z >>= 1; }// move block to low end of word
		return x | (z >> 1);             // need one bit less of low block
	}

	// Inverse of next_colex_comb()
	static inline T prev_colex_comb(T x) {
		x = next_colex_comb(~x);
		if (0 != x) x = ~x;
		return x;
	}
};
