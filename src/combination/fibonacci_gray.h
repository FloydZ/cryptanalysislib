#pragma once 
#include <cstdint>


/// Fibonacci Gray code with binary words.
/// Example (n = 5)
/// 10000
/// 10001
/// 10101
/// 10100
/// 00100
/// 00101
/// 00001
/// 00000
/// 00010
/// 01010
/// 01000
/// 01001
template<typename T,
          const uint32_t n>
class bit_fibgray {
private:
    T x_;  // current Fibonacci word
    T k_;  // aux
    T fw_, lw_;  // first and last Fibonacci word in Gray code
    T mw_;  // max(fw_, lw_)

	// binary --> radix(-2)
	static inline constexpr T bin2neg(T x) noexcept {
		// mask in radix 2 is ...10101010
		const T m = 0xaaaaaaaaaaaaaaaaUL;
		x += m;
		x ^= m;
		return  x;
	}

	// radix(-2) --> binary
	// inverse of bin2neg()
	constexpr inline T neg2bin(T x) noexcept {
		const T m = 0xaaaaaaaaaaaaaaaaUL;
		x ^= m;
		x -= m;
		return  x;
	}

	// inverse of gray_code()
	// note: the returned value contains at each bit position
	// the parity of all bits of the input left from it (incl. itself)
	//
	constexpr static inline T inverse_gray_code(T x) noexcept {
		x ^= x>>1;  // gray ** 1
		x ^= x>>2;  // gray ** 2
		x ^= x>>4;  // gray ** 4
		x ^= x>>8;  // gray ** 8
		x ^= x>>16;  // gray ** 16
		// here: x = gray**31(input)
		// note: the statements can be reordered at will
		x ^= x>>32;  // for 64bit words
		return  x;
	}


public:
    explicit bit_fibgray() noexcept {
        fw_ = 0;
        for (T m=(1UL<<(n-1)); m!=0; m>>=3)  fw_ |= m;
        lw_ = fw_ >> 1;
        if ( 0==(n&1) )  { T t=fw_; fw_=lw_; lw_=t; }  // swap first/last
        mw_ = ( lw_>fw_ ? lw_ : fw_ );
        x_ = fw_;

        k_ = inverse_gray_code(fw_);
        k_ = neg2bin(k_);
    }

    ~bit_fibgray()  { ; }

    constexpr inline T data() const noexcept { return x_; }

    // Return next word in Gray code.
    // Return ~0 if current word is the last one.
    constexpr T next() noexcept {
        if ( x_ == lw_ )  return ~0UL;

        T s = n;  // shift
        while(1) {
            --s;
            T c = 1 | (mw_ >> s);  // possible difference for negbin word
            T i = k_ - c;
            T x = bin2neg(i);
            x ^= (x>>1);

			// is_fibrep(x)
            if ( 0==(x&(x>>1))) {
                k_ = i;
                x_ = x;
                return x;
            }
        }
    }
};
