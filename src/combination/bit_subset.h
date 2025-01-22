#pragma once


// Generate all all subsets of bits of  a given word.
//
// E.g., for the word ('.' printed for unset bits)
//   ...11.1.
// these words are produced by subsequent next()-calls:
//   ......1.
//   ....1...
//   ....1.1.
//   ...1....
//   ...1..1.
//   ...11...
//   ...11.1.
//   ........
template<typename T = uint64_t>
class bit_subset {
protected:
	T U;// current subset
	T V;// the full set

public:
	explicit bit_subset(T v) : U(0), V(v) { ; }
	~bit_subset() { ; }

    /// \return
	constexpr inline T current() const noexcept {
		return U;
	}
    
    /// \return
	constexpr inline T full_set() const noexcept {
		return V;
	}

    /// \return
	constexpr inline T next() noexcept {
		U = (U - V) & V;
		return U;
	}

    /// \return
	constexpr inline T prev() noexcept {
		U = (U - 1) & V;
		return U;
	}

    /// \return
	constexpr inline T first(const T v) noexcept {
		V = v;
		U = 0;
		return U;
	}

    /// \return
	constexpr inline T first() noexcept {
		first(V);
		return U;
	}

    /// \return
	constexpr inline T last(T v) noexcept {
		V = v;
		U = v;
		return U;
	}

    /// \return
	constexpr inline T last() noexcept {
		last(V);
		return U;
	}

    /// \return
	constexpr inline void set(T u) noexcept {
		U = u & V;
	}

    /// \return
	constexpr inline T complement() noexcept {
		U ^= V;
		return U;
	}

    /// \return
	constexpr inline T next_all_blocks() noexcept {
		U = (U - V + ~V) & V;
		return U;
	}

    /// \return
	constexpr inline T prev_all_blocks() noexcept {
		U = (U + V) & V;
		return U;
	}

    /// \return
	constexpr inline T negate_all_blocks() noexcept {
		complement();
		return next_all_blocks();
	}

    /// \return
	constexpr inline T set_all_blocks_one() noexcept {
		U = (-V + ~V) & V;
		return U;
	}

    /// \return
	constexpr inline T set_right_block_borders() noexcept {
		U |= ((-V + ~V) & V);
		return U;
	}

    /// \return
	constexpr inline T shift_left() {
		U = ((U << 1) + ~V) & V;
		return U;
	}

    /// \return
	constexpr inline T shift_left_fill() {
		shift_left();
		next();
		return U;
	}

    /// \return
	constexpr inline T shift_left_blocks() {
		U = (U << 1) & V;
		return U;
	}

    /// \return
	constexpr inline T shift_left_blocks_fill() {
		shift_left_blocks();
		U |= ((-V + ~V) & V);
		return U;
	}

    /// \return
	constexpr inline T rev_gray_code() {
		U ^= ((U << 1) + ~V);
		U &= V;
		return U;
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

	bit_subset<T> S;
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

	constexpr T next() noexcept {
    /// \return
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
// -------------------------
