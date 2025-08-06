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
/// Class for generating all subsets of bits in a given word
/// \tparam T[in]: type of the word, defaults to uint64_t
template<typename T = uint64_t>
class bit_subset_T {
protected:
	T U;// current subset
	const T V;// the full set

public:
	/// Constructor initializing with a full set
	/// \param v[in]: the full set that defines the universe
	explicit bit_subset_T(T v) : U(0), V(v) { ; }
	/// Destructor
	~bit_subset_T() { ; }

    /// Returns the current subset
    /// \return the current subset
	constexpr inline T current() const noexcept {
		return U;
	}
    
    /// Returns the full set
    /// \return the full set that defines the universe
	[[nodiscard]] constexpr inline T full_set() const noexcept {
		return V;
	}

    /// Advances to the next subset in the enumeration
    /// \return the next subset
	[[nodiscard]] constexpr inline T next() noexcept {
		U = (U - V) & V;
		return U;
	}

    /// Returns to the previous subset in the enumeration
    /// \return the previous subset
	[[nodiscard]] constexpr inline T prev() noexcept {
		U = (U - 1) & V;
		return U;
	}

    /// Sets the current subset to the given value, intersected with the full set
    /// \param u[in]: the subset to set
	constexpr inline void set(T u) noexcept {
		U = u & V;
	}

    /// Computes the complement of the current subset within the full set
    /// \return the complement of the current subset
	[[nodiscard]] constexpr inline T complement() noexcept {
		U ^= V;
		return U;
	}

    /// Advances to the next subset considering all blocks
    /// \return the next subset with all blocks
	[[nodiscard]] constexpr inline T next_all_blocks() noexcept {
		U = (U - V + ~V) & V;
		return U;
	}

    /// Returns to the previous subset considering all blocks
    /// \return the previous subset with all blocks
	[[nodiscard]] constexpr inline T prev_all_blocks() noexcept {
		U = (U + V) & V;
		return U;
	}

    /// Negates all blocks in the current subset
    /// \return the subset with all blocks negated
	[[nodiscard]] constexpr inline T negate_all_blocks() noexcept {
		complement();
		return next_all_blocks();
	}

    /// Sets all blocks in the subset to one
    /// \return the subset with all blocks set to one
	[[nodiscard]] constexpr inline T set_all_blocks_one() noexcept {
		U = (-V + ~V) & V;
		return U;
	}

    /// Sets the right borders of all blocks
    /// \return the subset with right block borders set
	[[nodiscard]] constexpr inline T set_right_block_borders() noexcept {
		U |= ((-V + ~V) & V);
		return U;
	}

    /// Shifts the current subset to the left
    /// \return the subset after left shift
	[[nodiscard]] constexpr inline T shift_left() noexcept {
		U = ((U << 1) + ~V) & V;
		return U;
	}

    /// Shifts the current subset to the left and fills empty positions
    /// \return the subset after left shift with filling
	[[nodiscard]] constexpr inline T shift_left_fill() noexcept {
		shift_left();
		next();
		return U;
	}

    /// Shifts the blocks of the current subset to the left
    /// \return the subset after left block shift
	[[nodiscard]] constexpr inline T shift_left_blocks() noexcept {
		U = (U << 1) & V;
		return U;
	}

    /// Shifts the blocks of the current subset to the left and fills empty positions
    /// \return the subset after left block shift with filling
	[[nodiscard]] constexpr inline T shift_left_blocks_fill() noexcept {
		shift_left_blocks();
		U |= ((-V + ~V) & V);
		return U;
	}

    /// Applies a reverse Gray code transformation to the current subset
    /// \return the subset after reverse Gray code transformation
	[[nodiscard]] constexpr inline T rev_gray_code() noexcept {
		U ^= ((U << 1) + ~V);
		U &= V;
		return U;
	}
};
