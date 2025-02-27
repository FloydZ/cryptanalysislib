#pragma once

#include <cstdint>

// Generate all subsets of bits of  a given word.
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
class bit_subset_T {
protected:
	T U;// current subset
	const T V;// the full set

public:
	explicit bit_subset_T(T v) : U(0), V(v) { ; }
	~bit_subset_T() { ; }

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
	constexpr inline T shift_left() noexcept {
		U = ((U << 1) + ~V) & V;
		return U;
	}

    /// \return
	constexpr inline T shift_left_fill() noexcept {
		shift_left();
		next();
		return U;
	}

    /// \return
	constexpr inline T shift_left_blocks() noexcept {
		U = (U << 1) & V;
		return U;
	}

    /// \return
	constexpr inline T shift_left_blocks_fill() noexcept {
		shift_left_blocks();
		U |= ((-V + ~V) & V);
		return U;
	}

    /// \return
	constexpr inline T rev_gray_code() noexcept {
		U ^= ((U << 1) + ~V);
		U &= V;
		return U;
	}
};