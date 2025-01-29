#ifndef CRYPTANALYSISLIB_COMBINATION_LEXICOGRAPHIC_H
#define CRYPTANALYSISLIB_COMBINATION_LEXICOGRAPHIC_H

#include <cstdint>
#include <cstdlib>

class Combinations_Fq_Lexicographic {
	/// length to enumerate
	const uint32_t n;

	/// max value to enumerate
	const uint32_t q;


	Combinations_Fq_Lexicographic(const uint32_t n, const uint32_t q) :
	n(n), q(q) {}

	/// \param stack array of size
	/// \param sp current stack pointer. Do not think about this
	/// \param k max weight to enumerate
	/// \return the number of subsequent numbers which have hamming weight < k.
	uint64_t restricted_zeta_update_stack(uint32_t *stack, uint32_t *sp, const uint64_t k) {
		if (stack[*sp] == k) {
			uint32_t i = *sp + 1;

			// walk up
			while (stack[i] == k)
				i += 1;

			// update
			stack[i] += 1;
			const uint32_t val = stack[i];
			const uint32_t altered_sp = i;

			// walk down
			while (i > 0) {
				stack[i - 1] = val;
				i -= 1;
			}

			// fix up stack pointer
			*sp = 0;
			return (1ull << (altered_sp + 1)) - 1;
		} else {
			stack[*sp] += 1;
			return 1ull;
		}
	}
public:
};

/// lexicographic enumeration of p error position <= n
/// so it can be that certain posiiton occur twice within the error
/// \tparam n max size of each index
/// \tparam p number of error position
/// \tparam q (unused, needed for api compability)
template<const uint32_t n,
         const uint32_t p,
         const uint32_t q = 2>
class enumerate_t {
	using T = uint16_t;
	T idx[16] = {0};

	static_assert(q>=2);
	static_assert(p<=4);
	static_assert(n > p);

public:
	[[nodiscard]] constexpr size_t list_size() const noexcept {
		size_t ret = 1;
		for (uint32_t i = 0; i < p; i++) {
			ret	*= n;
		}
		return ret;
	}

	template<typename F>
	constexpr inline void enumerate(F &&f) noexcept {
		if constexpr (p == 0) {
			// catch for prange
			return;
		} else if constexpr (p == 1) {
			return enumerate1(idx, f);
		} else if constexpr (p == 2) {
			return enumerate2(idx, f);
		} else if constexpr (p == 3) {
			return enumerate3(idx, f);
		}
	}

	/// \tparam F
	/// \param idx
	/// \param f
	template<typename F>
	constexpr static inline void enumerate1(T *idx,
										    F &&f) noexcept {
		for (idx[0] = 0; idx[0] < n; ++idx[0]) {
			f(idx);
		}
	}

	/// \tparam F
	/// \param idx
	/// \param f
	template<typename F>
	constexpr static inline void enumerate2(T *idx, F &&f) noexcept {
		for (idx[0] = 0; idx[0] < n; ++idx[0]) {
			for (idx[1] = idx[0] + 1; idx[1] < n; ++idx[1]) {
				f(idx);
			}
		}
	}

	/// \tparam F
	/// \param idx
	/// \param f
	template<typename F>
	constexpr static inline void enumerate3(T *idx,
											F &&f) noexcept {
		for (idx[0] = 0; idx[0] < n; ++idx[0]) {
			for (idx[1] = idx[0] + 1; idx[1] < n; ++idx[1]) {
				for (idx[2] = idx[1] + 1; idx[2] < n; ++idx[2]) {
					f(idx);
				}
			}
		}
	}

	/// \tparam F
	/// \param idx
	/// \param f
	template<typename F>
	constexpr static inline void enumerate4(T *idx,
											F &&f) noexcept {
		for (idx[0] = 0; idx[0] < n; ++idx[0]) {
			for (idx[1] = idx[0] + 1; idx[1] < n; ++idx[1]) {
				for (idx[2] = idx[1] + 1; idx[2] < n; ++idx[2]) {
					for (idx[3] = idx[2] + 1; idx[3] < n; ++idx[3]) {
						f(idx);
					}
				}
			}
		}
	}

	/// enumerating <= p
	/// \tparam F
	/// \param idx
	/// \param f
	template<typename F>
	constexpr static inline void enumeratep(T *idx,
											F &&f) noexcept {
		for (uint32_t i = 0; i < p; i++) { idx[i] = 0; }
		while (true) {
			for (idx[0] = 0; idx[0] < n; idx[0]++) {
				f(idx);
			}

			uint32_t nsp = 0;
			while(idx[nsp] == (n-1)) {
				if (nsp == p-1) { goto __exit; }
				idx[nsp++] = 0;
				idx[nsp]++;
				f(idx);
			}
		}

		__exit:
		return;
	}
};

template<typename T>
class BinaryLexicographic {
    // Return next word in subset-lexrev order.
    // Start with a one-bit word at position n-1 to
    // generate 2**n subsets of length n.
    // E.g., for n==4 the subsets are
    //     word   subset of {0,1,2,3}
    //     1...   {0}
    //     11..   {0, 1}
    //     111.   {0, 1, 2}
    //     1111   {0, 1, 2, 3}
    //     11.1   {0, 1, 3}
    //     1.1.   {0, 2}
    //     1.11   {0, 2, 3}
    //     1..1   {0, 3}
    //     .1..   {1}
    //     .11.   {1, 2}
    //     .111   {1, 2, 3}
    //     .1.1   {1, 3}
    //     ..1.   {2}
    //     ..11   {2, 3}
    //     ...1   {3}
    //     ....   {}
    // Note (1): the first element of the subset corresponds
    // to the highest set bit.  When interpreting the binary
    // words via "bit(n)==element n" (as usual), the order
    // would be:
    //     1...           { 3 }
    //     11..        { 2, 3 }
    //     111.     { 1, 2, 3 }
    //     1111  { 0, 1, 2, 3 }
    //     11.1     { 0, 2, 3 }
    //     1.1.        { 1, 3 }
    //     1.11     { 0, 1, 3 }
    //     1..1        { 0, 3 }
    //     .1..           { 2 }
    //     .11.        { 1, 2 }
    //     .111     { 0, 1, 2 }
    //     .1.1        { 0, 2 }
    //     ..1.           { 1 }
    //     ..11        { 0, 1 }
    //     ...1           { 0 }
    // Note (2): the lex order for the delta sets would simply
    // be the counting order (of the words or reversed words
    // depending on the interpretation as explained above).
    constexpr static inline T next_lexrev(T x) {
        T x0 = x & -x;  // lowest one
        if (1 != x0) {  // easy case: set bit right of lowest one
            x0 >>= 1;
            x ^= x0;
            return  x;
        } else  {
            // lowest one at word end
            x ^= 1;  // clear lowest one
            x0 = x & -x;  // new lowest one ...
            x0 >>= 1;  x -= x0;  // ... is moved one to the right
            return  x;
        }
    }
    
    // Return previous word in subset-lexrev order.
    // Start with zero and use 2**n calls
    // generate 2**n subsets of length n.
    // E.g., for n==4:
    //  ....  =  0
    //  ...1  =  1
    //  ..11  =  3
    //  ..1.  =  2
    //  .1.1  =  5
    //  .111  =  7
    //  .11.  =  6
    //  .1..  =  4
    //  1..1  =  9
    //  1.11  = 11
    //  1.1.  = 10
    //  11.1  = 13
    //  1111  = 15
    //  111.  = 14
    //  11..  = 12
    //  1...  =  8
    static inline T prev_lexrev(T x) {
        T x0 = x & -x;  // lowest one
        if ( x & (x0<<1) ) {
            // easy case: next higher bit is set
            x ^= x0;  // clear lowest one
            return x;
        } else {
            x += x0;  // move lowest one to the left
            x |= 1;   // set rightmost bit
            return x;
        }
    }
    
    //   k:  negidx2lexrev(k)
    //   0:  .....
    //   1:  ....1
    //   2:  ...11
    //   3:  ...1.
    //   4:  ..1.1
    //   5:  ..111
    //   6:  ..11.
    //   7:  ..1..
    //   8:  .1..1
    //   9:  .1.11
    //  10:  .1.1.
    //  11:  .11.1
    //  12:  .1111
    //  13:  .111.
    //  14:  .11..
    //  15:  .1...
    //  16:  1...1
    static inline T negidx2lexrev(size_t k) noexcept {
        T z = 0;
        // T h = highest_one(k);
        T h = 64 - __builtin_clzll(k);
        while ( k )
        {
            while ( 0 == (h & k) )  h >>= 1;
            z ^= h;
            ++k;
            k &= h - 1;
        }
    
        return  z;
    }
    
    // inverse of negidx2lexrev()
    static inline size_t lexrev2negidx(T x) {
        if ( 0==x )  return 0;
        T h = x & -x;  // lowest one
        T r = (h-1);
        while ( x^=h ) {
            r += (h-1);
            h = x & -x;  // next higher one
        }
        r += h;  // highest one
        return  r;
    
    //    if ( 0==x )  return 0;
    //    T h = highest_one(x);
    //    T r = h - 1;
    //    x ^= h;
    //    if ( x )   r += lexrev2negidx(x);
    //    else       r += r + 1;
    //    return r;
    
    //    T r = 0;
    //    T h = highest_one(x);
    //    while ( x )
    //    {
    //        while ( 0==(h&x) )  h >>= 1;
    //        r ^= h;
    //        x = next_lexrev(x);
    //        if ( 0==x )  return r;
    //        while ( 0==(h&x) )  h >>= 1;
    //        x ^= h;
    //    }__builtin__builtin_clzll_clzll
    //    return  r;
    
    //    T r = 0;
    //    while ( x )
    //    {
    //        T h = highest_one(x);
    //        r ^= h;
    //        x = next_lexrev(x);
    //        h = highest_one(x);
    //        x ^= h;
    //    }
    //    return  r;
    }
    // -------------------------
    
    
    
    // Return whether x is a fixed point in the prev_lexrev() - sequence
    static inline bool is_lexrev_fixed_point(T x) {
        if ( x & 1 )  return  (1==x);
    
        T w = __builtin_popcountll(x);
        if ( w != (w & -w) )  return  false;
        if ( 0==x )  return  true;
        return  0 != ( (x & -x) & w );
    }
};
#endif//CRYPTANALYSISLIB_COMBINATION_LEXICOGRAPHIC_H
