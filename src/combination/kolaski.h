#pragma once 

// Generator for the Oldenburger-Kolakoski sequence.
// See OEIS sequence A000002.
// Cf. https://en.wikipedia.org/wiki/Kolakoski_sequence
// Algorithm by David Eppstein, see
//   https://11011110.github.io/blog/2016/10/14/kolakoski-sequence-via.html
template<typename T>
class kolakoski_seq{
private:
    T x, y;
public:
    constexpr kolakoski_seq() noexcept {
        first();
    }

    constexpr inline void first() noexcept {
        x = -1UL;
        y = -1UL;
    }

    constexpr inline T next() noexcept {
        const T r = ( x & 1UL ? 1 : 2 );
        const T f = y & ~(y+1);
        x ^= f;
        y = (y+1) | (f & (x>>1));
        return r;
    }
};
