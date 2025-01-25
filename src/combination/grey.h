#pragma once 

#include <cstdint>
#include "bit_subset.h"

template<class T>
class enumeration_gray {
private:
    T val = 0; //first_comb();
    constexpr inline void word_gray(T *f, ulong n) noexcept {
        for (ulong k=0;  k<n-1;  ++k)  f[k] ^= f[k+1];
    }
    // -------------------------
    
    constexpr inline void inverse_word_gray(T *f, ulong n) noexcept {
        ulong x = 0,  k = n;
        while ( k-- )  { x ^= f[k];  f[k] = x; }
    }
    
    // result is identical to
    //   for (ulong k=0; k<x; ++k)  word_gray(f, n);
    // Work <= n/2
    void word_gray_pow(T *f, ulong n, ulong x) {
        for (uint32_t s=1; s<n; s*=2) {
            if ( x & 1 ) {
                // word_gray ** s:
                for (uint32_t k=0, j=k+s;  j<n;  ++k, ++j)  f[k] ^= f[j];
            }
            x >>= 1;
        }
    }
    
    void word_rev_gray(T *f, ulong n) {
        for (uint32_t k=n-1; 0!=k; --k)  f[k] ^= f[k-1];
    }
    
    void inverse_word_rev_gray(T *f, ulong n) {
        uint32_t x = 0;
        for (uint32_t k=0;  k<n; ++k)  { 
            x ^= f[k];
            f[k] = x; 
        }
    }
    
    /// result is identical to
    ///   for (ulong k=0; k<x; ++k)  word_rev_gray(f, n);
    /// work <= n/2
    void word_rev_gray_pow(T *f, ulong n, ulong x) {
        x &= (n-1);  // modulo n
        for (uint32_t s=1; s<n; s*=2) {
            if ( x & 1) {
                // word_rev_gray ** s:
                for (uint32_t k=n-1, j=k-s;  k>=s;  --k, --j)  f[k] ^= f[j];
            }
            x >>= 1;
        }
    }
};


#define BITSUBSET_GRAY_METHOD1// un/define to choose method (default:=defined)

template<typename T>
class bit_subset_gray_T {
protected:
    /// \return 1<<MSB(x)
	constexpr inline T highest_one(const T x) {
		return T(1) << __builtin_clzll(x | 1);
	}

	bit_subset_T<T> S;
	T G;// subsets in Gray code order
	T H;// highest bit in S.V;  needed for the prev() method

public:
	constexpr explicit bit_subset_gray_T(const T v) noexcept 
	    : S(v), G(0), H(highest_one(v)) { ; }

	~bit_subset_gray_T() { ; }

    /// \return
	constexpr T current() const noexcept {
        return G; 
    }

    /// \return
	constexpr T full_set() const noexcept { 
        return S.full_set(); 
    }

    /// \return
	constexpr T next() noexcept {
		T U0 = S.current();
		if (U0 == S.full_set()) return first();
		T U1 = S.next();
#if defined BITSUBSET_GRAY_METHOD1
		T X = ~U0 & U1;
#else
		T X = (U0 ^ U1) & U1;
#endif
		G ^= X;
		return G;
	}

    /// \return
	constexpr T first(T v) noexcept {
		S.first(v);
		H = highest_one(v);
		G = 0;
		return G;
	}

    /// \return
	constexpr T first() noexcept {
		S.first();
		G = 0;
		return G;
	}

    /// \return
	constexpr T prev() noexcept {
		T U1 = S.current();
		if (U1 == 0) return last();
		T U0 = S.prev();
#if defined BITSUBSET_GRAY_METHOD1
		T X = ~U0 & U1;
#else
		T X = (U0 ^ U1) & U1;
#endif
		G ^= X;
		return G;
	}

    /// \return
	constexpr T last(T v) noexcept {
		S.last(v);
		H = highest_one(v);
		G = H;
		return G;
	}

    /// \return
	constexpr T last() noexcept {
		S.last();
		G = H;
		return G;
	}
};
