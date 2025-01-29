#pragma once

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

