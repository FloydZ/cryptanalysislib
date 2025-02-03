#pragma once 

template<typename T>
class enumeration_reverse_gray_code {
private:
    T val = 1;
    
    /// Return the reversed Gray code of x.
    /// ('bit-wise derivative modulo 2 towards high bits').
    /// Also: multiplication by x+1 as binary polynomial.
    /// Also: return x^2+x in binary normal basis.
    /// rev_gray_code(x) == revbin( gray_code( revbin(x) ) )
    constexpr static inline T rev_gray_code(const T x) noexcept {
        return  x ^ (x<<1);
    }

    // Inverse of rev_gray_code()
    // Note: the returned value contains at each bit position the parity
    //   of all bits of the input right from it (incl. itself).
    // Also: division by x+1 as powers series over GF(2)
    // Also: return solution of z = x^2+x in binary normal basis.
    // Note: the statements can be reordered at will.
    // inverse_rev_gray_code(x) == revbin( inverse_gray_code( revbin(x) ) )
    constexpr static inline T inverse_rev_gray_code(T x) noexcept {
        // use: rev_gray ** BITSPERLONG == id:
        x ^= x<<1;  // rev_gray ** 1
        x ^= x<<2;  // rev_gray ** 2
        if constexpr (sizeof(T) == 1) { x ^= x<<4;  } // rev_gray ** 4
        if constexpr (sizeof(T) == 2) { x ^= x<<8;  } // rev_gray ** 8
        if constexpr (sizeof(T) == 4) { x ^= x<<16; } // rev_gray ** 16
        // here: x = rev_gray**31(input)
        if constexpr (sizeof(T) == 8) { x ^= x<<32; }  // for 64bit words
        return  x;
    }
public:

    constexpr enumeration_reverse_gray_code() noexcept {};
    constexpr inline T next() noexcept {
        const T ret = val;
        val = rev_gray_code(val);
        return ret;
    } 
    constexpr inline T prev() noexcept {
        const T ret = val;
        val = inverse_rev_gray_code(val);
        return ret;
    } 
};
