#pragma once

/// Class for enumerating parenthesis-like structures in binary representation
/// Provides utilities for generating, validating, and manipulating binary words
/// that correspond to valid parenthesis expressions
/// \tparam T[in]: integer type to represent parenthesis words
template<typename T>
class enumeration_parenthesis {
private:
    /// Number of bits in type T
    constexpr static size_t BITS = sizeof(T) * 8;
    
    /// Current value in the enumeration
    T val = first_parenword();

public:
    /// Determines if a binary word represents a valid parenthesis structure
    /// 
    /// Binary words < 16, those that are valid
    /// 'paren words' are marked with 'P':
    ///  .... P   [empty string]
    ///  ...1 P   ()
    ///  ..1.
    ///  ..11 P   (())
    ///  .1..
    ///  .1.1 P   ()()
    ///  .11.
    ///  .111 P   ((()))
    ///  1...
    ///  1..1
    ///  1.1.
    ///  1.11 P   (()())
    ///  11..
    ///  11.1 P   ()(())
    ///  111.
    ///  1111 P   (((())))
    ///
    /// \param x[in]: binary word to check
    /// \return true if x represents a valid parenthesis structure, false otherwise
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
    
    /// Converts a binary parenthesis word to a string representation
    /// 
    /// Binary words < 32 that are valid
    /// 'paren words' together with paren-string:
    ///  .....   [empty string]
    ///  ....1   ()
    ///  ...11   (())
    ///  ..1.1   ()()
    ///  ..111   ((()))
    ///  .1.11   (()())
    ///  .11.1   ()(())
    ///  .1111   (((())))
    ///  1..11   (())()
    ///  1.1.1   ()()()
    ///  1.111   ((()()))
    ///  11.11   (()(()))
    ///  111.1   ()((()))
    ///  11111   ((((()))))
    /// 
    /// Note 1: lower bits in word (right end) correspond
    ///         to the begin of string (left end).
    /// Note 2: Word is extended with zeros (to the left) when necessary,
    ///         so the length of str must be >= 1 + 2*(number of set bits)
    ///
    /// \param str[out]: output buffer for the string representation
    /// \param x[in]: binary word to convert
    constexpr static inline void parenword2str(char *str, 
                                               T x) noexcept {
        int s = 0;
        T j = 0;
        for (j=0; x!=0; ++j) {
            s += ( x&1 ? +1 : -1 );
            // if ( s<0 )  {"Invalid word"}
            str[j] = ")("[x&1];
            x >>= 1;
        }
        while ( s-- > 0 )  str[j++] = ')';  // finish string
        str[j] = 0;  // terminate string
    }
    
    /// Returns the first (smallest) binary word corresponding to n pairs of parentheses
    /// Example, n=5:  .....11111   ((((()))))
    /// \param n[in]: number of parenthesis pairs
    /// \return binary word representing the first valid parenthesis structure with n pairs
    constexpr static inline T first_parenword(const T n) noexcept {
        return first_comb(n);
    }
    
    /// Returns the last (largest) binary word corresponding to n pairs of parentheses
    /// Must have: 1 <= n <= BITS_PER_LONG/2
    /// Example, n=5:  .1.1.1.1.1   ()()()()()
    /// \param n[in]: number of parenthesis pairs
    /// \return binary word representing the last valid parenthesis structure with n pairs
    constexpr static inline T last_parenword(const T n) noexcept {
        return  0x5555555555555555UL >> (BITS-2*n);
    }
    
    /// Computes the next parenthesis word in colex (co-lexicographic) order
    /// With n=4 and starting with first_parenword(n),
    /// the sequence is:
    ///  .....1111   (((())))
    ///  ....1.111   ((()()))
    ///  ....11.11   (()(()))
    ///  ....111.1   ()((()))
    ///  ...1..111   ((())())
    ///  ...1.1.11   (()()())
    ///  ...1.11.1   ()(()())
    ///  ...11..11   (())(())
    ///  ...11.1.1   ()()(())
    ///  ..1...111   ((()))()
    ///  ..1..1.11   (()())()
    ///  ..1..11.1   ()(())()
    ///  ..1.1..11   (())()()
    ///  ..1.1.1.1   ()()()()
    ///  .........    [zero]
    ///
    /// \param x[in]: current parenthesis word
    /// \return next valid parenthesis word, or 0 if x is the last
    [[nodiscard]] constexpr static inline T next(T x) noexcept {
        if (x & 2) {
            // Easy case, move highest bit of lowest block to the left:
            T b = lowest_zero(x);
            x ^= b;
            x ^= (b>>1);
            return x;
        } else {
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

    /// Returns the current parenthesis word and advances to the next
    /// \return current parenthesis word
    [[nodiscard]] constexpr inline T next() noexcept {
        const T ret = val;
        val = next(val);
        return ret;
    }
};
