#pragma once 

/// Class providing utilities for working with Fibonacci representations of numbers
/// Fibonacci representation: a binary representation where no two adjacent 1s appear
class fibrep {

/// Converts a binary number to its Fibonacci representation
/// Limitation: the first Fibonacci number greater than b must be representable as ulong
/// 32 bit:  b < 2971215073=F(47) [F(48)=4807526976 > 2^32]
/// 64 bit:  b < 12200160415121876738=F(93) [F(94) > 2^64]
/// \param b[in]: binary number to convert
/// \return Fibonacci representation of b
constexpr static inline ulong bin2fibrep(ulong b) noexcept {
    ulong f0 = 1,  f1 = 1,  s = 1;
    while ( f1 <= b )  { ulong t = f0+f1;  f0=f1;  f1=t;  s<<=1; }
    ulong f = 0;
    while ( b )
    {
        s >>= 1;
        if ( b>=f0 )  { b -= f0;  f^=s; }
        { ulong t = f1 - f0;  f1=f0;  f0=t; }
    }
    return f;
}

/// Converts a Fibonacci representation to its binary value
/// Inverse of bin2fibrep()
/// \param f[in]: Fibonacci representation to convert
/// \return binary value of the Fibonacci representation
constexpr static inline ulong fibrep2bin(ulong f) noexcept {
    ulong f0 = 1,  f1 = 1;
    ulong b = 0;
    while ( f )
    {
        if ( f & 1 )   b += f1;
        { ulong t = f0 + f1;  f0=f1;  f1=t; }
        f >>= 1;
    }
    return b;
}

/// Computes the Fibonacci representation of n+1 given the Fibonacci representation of n
/// \param x[in]: Fibonacci representation of n
/// \return Fibonacci representation of n+1
constexpr static inline ulong next_fibrep(ulong x) noexcept {
    // From the Python code by Falk Hueffner in https://oeis.org/A003714
    const ulong y = ~(x >> 1);
    x -= y;
    x &= y;
    return x;
}

/// Computes the Fibonacci representation of n-1 given the Fibonacci representation of n
/// \param x[in]: Fibonacci representation of n
/// \return Fibonacci representation of n-1
constexpr static inline ulong prev_fibrep(ulong x) noexcept {
    // 2 examples:                   //  ex. 1             //  ex.2
    //                               // x == [*]0 100000   // x == [*]0 10000
    const ulong y = x & -x;          // y == [0]0 100000   // y == [0]0 10000
    x ^= y;                          // x == [*]0 000000   // x == [*]0 00000
    ulong m = 0x5555555555555555UL;  // m == ...01010101
    if ( m & y )  m >>= 1;           // m == ...01010101   // m == ...0101010
    m &= (y-1);                      // m == [0]0 010101   // m == [0]0 01010
    x ^= m;                          // x == [*]0 010101   // x == [*]0 01010
    return x;
}


/// Checks if a number is a valid Fibonacci representation
/// A valid Fibonacci representation does not contain two adjacent ones
/// \param f[in]: number to check
/// \return true if f is a valid Fibonacci representation, false otherwise
constexpr static inline bool is_fibrep(ulong f) noexcept  {
    return  ( 0 == (f & (f>>1)) );
}

};
