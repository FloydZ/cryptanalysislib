#pragma once

template<typename T>
class parenthesis_word {

    // Return whether x is a valid paren word.
    //
    // Binary words < 16, those that are valid
    //  'paren words' are marked with 'P':
    //  .... P   [empty string]
    //  ...1 P   ()
    //  ..1.
    //  ..11 P   (())
    //  .1..
    //  .1.1 P   ()()
    //  .11.
    //  .111 P   ((()))
    //  1...
    //  1..1
    //  1.1.
    //  1.11 P   (()())
    //  11..
    //  11.1 P   ()(())
    //  111.
    //  1111 P   (((())))
    constexpr static inline bool is_parenword(T x) noexcept {
        long s = 0;
        for (T j=0; x!=0; ++j)
        {
            s += ( x&1 ? +1 : -1 );
            if ( s < 0 )  return false;  // invalid word
            x >>= 1;
        }
        return  true;
    }
    
    // Fill paren string corresponding to x into str.
    //
    // Binary words < 32 that are valid
    //  'paren words' together with paren-string:
    //  .....   [empty string]
    //  ....1   ()
    //  ...11   (())
    //  ..1.1   ()()
    //  ..111   ((()))
    //  .1.11   (()())
    //  .11.1   ()(())
    //  .1111   (((())))
    //  1..11   (())()
    //  1.1.1   ()()()
    //  1.111   ((()()))
    //  11.11   (()(()))
    //  111.1   ()((()))
    //  11111   ((((()))))
    // Note 1: lower bits in word (right end) correspond
    //         to the begin of string (left end).
    // Note 2: Word is extended with zeros (to the left) when necessary,
    //         so the length of str must be >= 1 + 2*(number of set bits)
    static inline void parenword2str(T x, char *str) {
        int s = 0;
        T j = 0;
        for (j=0; x!=0; ++j)
        {
            s += ( x&1 ? +1 : -1 );
            // if ( s<0 )  {"Invalid word"}
            str[j] = ")("[x&1];
            x >>= 1;
        }
        while ( s-- > 0 )  str[j++] = ')';  // finish string
        str[j] = 0;  // terminate string
    }
    
    // Return least binary word corresponding to n pairs of parens.
    // Example, n=5:  .....11111   ((((()))))
    static inline T first_parenword(T n) {
        return first_comb(n);
    }
    
    // Return biggest binary word corresponding to n pairs of parens.
    // Must have: 1 <= n <= BITS_PER_LONG/2.
    // Example, n=5:  .1.1.1.1.1   ()()()()()
    static inline T last_parenword(T n) {
        return  0x5555555555555555UL >> (BITS_PER_LONG-2*n);
    }
    
    
    // Next (colex order) binary word that is a paren word.
    // With n=4 and starting with first_parenword(n)
    //  one gets the following sequence:
    //  .....1111   (((())))
    //  ....1.111   ((()()))
    //  ....11.11   (()(()))
    //  ....111.1   ()((()))
    //  ...1..111   ((())())
    //  ...1.1.11   (()()())
    //  ...1.11.1   ()(()())
    //  ...11..11   (())(())
    //  ...11.1.1   ()()(())
    //  ..1...111   ((()))()
    //  ..1..1.11   (()())()
    //  ..1..11.1   ()(())()
    //  ..1.1..11   (())()()
    //  ..1.1.1.1   ()()()()
    //  .........    [zero]
    static inline T next_parenword(T x) {
        if ( x & 2 )  // Easy case, move highest bit of lowest block to the left:
        {
            T b = lowest_zero(x);
            x ^= b;
            x ^= (b>>1);
            return x;
        }
        else
        {
            const T m0 = -1UL/3;
            T t = x ^ m0;               // XOR t, x, m0;
            if ( (t&x)==0 )  return 0;      // current is last
            T u = (t-1) ^ t;            // SUBU u, t, 1;  XOR u, t, u;
            T v = x | u;                // OR v, x, u;
            T y = bit_count( u & m0 );  // SADD y, u, m0;
            T w = v + 1;                // ADDU w, v, 1;
            t = v & ~w;                     // ANDN t, v, w;
            y = t >> y;                     // SRU y, t, y;
            y += w;                         // ADDU y, w, y;
            return y;
        }
    }
};
