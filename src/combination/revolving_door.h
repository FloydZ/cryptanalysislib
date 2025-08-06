#pragma once 

#include <stdint.h>
#include <stdlib.h>

#include "algorithm/max.h"
#include "algorithm/min.h"

/// Class for generating combinations in minimal-change order using the revolving-door algorithm
/// Implements Algorithm R, "revolving-door combinations", from Knuth's TAOCP 4A/1, pp.363
/// Based on W. H. Payne, F. M. Ives: "Combination Generators",
/// ACM Transactions on Mathematical Software (TOMS), vol.5, no.2, pp.163-172, (June-1979)
///
/// The revolving-door algorithm generates combinations where each successive
/// combination differs from the previous by exactly one element being removed and
/// one element being added.
///
/// k1 = bit position to be cleared (element removed) 
/// k2 = bit position to be set (element added)
///
/// Example output for (10, 2) showing the delta set and which bits change (k1,k2):
///     11........ 0 1
///     .11....... 0 2
///     1.1....... 1 0
///     ..11...... 0 3
///     .1.1...... 2 1
///     1..1...... 1 0
///     ...11..... 0 4
///     ..1.1..... 3 2
///     .1..1..... 2 1
///     1...1..... 1 0
///     ....11.... 0 5
///     ...1.1.... 4 3
///     ..1..1.... 3 2
///     .1...1.... 2 1
///     1....1.... 1 0
///     .....11... 0 6
///     ....1.1... 5 4
///     ...1..1... 4 3
///     ..1...1... 3 2
///     .1....1... 2 1
///     1.....1... 1 0
///     ......11.. 0 7
///     .....1.1.. 6 5
///     ....1..1.. 5 4
///     ...1...1.. 4 3
///     ..1....1.. 3 2
///     .1.....1.. 2 1
///     1......1.. 1 0
///     .......11. 0 8
///     ......1.1. 7 6
///     .....1..1. 6 5
///     ....1...1. 5 4
///     ...1....1. 4 3
///     ..1.....1. 3 2
///     .1......1. 2 1
///     1.......1. 1 0
///     ........11 0 9
///     .......1.1 8 7
///     ......1..1 7 6
///     .....1...1 6 5
///     ....1....1 5 4
///     ...1.....1 4 3
///     ..1......1 3 2
///     .1.......1 2 1
///     1........1 1 0
///
/// Usage example:
///     combination_revdoor c(10, 4);
///     uint32_t k1 = 1, k2 = 2;
///     do {
///     	c.print_deltaset();
///     	std::cout << " " << k1 << " " << k2 << std::endl;
///     } while (c.next(&k1, &k2));
///     return;
class combination_revdoor {
private:
    /// Integer type used for combinations
    using T = uint32_t;

    /// Array storing the delta set representation
	T *c_;
	
	/// n_ is the total number of elements, k_ is the subset size
	/// Must have: n>=1, 1<=k<=n
	T n_, k_;

	/// Deleted copy constructor to prevent unintended copies
	combination_revdoor(const combination_revdoor&) = delete;
	
	/// Deleted assignment operator to prevent unintended copies
	combination_revdoor & operator = (const combination_revdoor&) = delete;

public:
	/// Constructor for the revolving-door combination generator
	/// \param n[in]: total number of elements (must be >= 1)
	/// \param k[in]: subset size (must be 1 <= k <= n)
	explicit combination_revdoor(const T n,
                                 const T k) noexcept {
		n_ = n;  // (n ? n : 1);
		k_ = k;
		c_ = new T[k_+1];  // incl. sentinel
		first();
	}

	/// Destructor frees allocated memory
	~combination_revdoor() noexcept { delete [] c_; }

	/// Sets the combination to the first one in the sequence
	constexpr void first() noexcept {
		for (T j=0; j<k_; ++j) { c_[j] = j; }
		c_[k_] = n_;  // sentinel
	}

    /// Gets the current combination data
    /// \return pointer to the current combination array
	constexpr const T* data() const noexcept { return c_; }

    /// TODO constexpr not working as there are gotos.
	/// Advances to the next combination in the revolving-door sequence
	/// \param k1[out]: bit position to be cleared (element removed)
	/// \param k2[out]: bit position to be set (element added)
	/// \return true if a next combination exists, false if at the end
	/* constexpr */ bool next(uint32_t *k1,
                              uint32_t *k2 ) noexcept{
		T j = 1;
		// R3: [Easy case?]
		// odd k (try to increase)
		if ( k_ & 1 ) {
			const T c = c_[0] + 1;
			if ( c < c_[1] )  {
				*k1 = c_[0];
				c_[0] = c;
				*k2 = c;
				return true;
			} else { goto R4; }
		} else {
			// even k (try to decrease)
			const T c = c_[0];
			if ( c )  {
				*k1 = cryptanalysislib::max(c_[0], c-1);
				*k2 = cryptanalysislib::min(c_[0], c-1);
				c_[0] = c-1;
				return true;
			} else {
				goto R5;
			}
		}

	R4:  // R4: [Try to decrease]
		if ( j==k_ )  return false;
		if ( c_[j] > j ) {
			*k1 = c_[j];
			*k2 = j-1;
			c_[j] = c_[j-1];
			c_[j-1] = j-1;
			return true;
		}
		++j;

	R5:  // R5: [Try to increase]
		if ( j==k_ ) { return false; }

		{
			T c = c_[j] + 1;
			// can read sentinel
			if ( c < c_[j+1] ) {
				*k1 = c_[j-1];
				*k2 = c;
				c_[j-1] = c - 1;
				c_[j] = c;
				return true;
			}
		}
		++j;
		goto R4;
	}

    /// Prints the current combination as a delta set
    /// A delta set representation shows a 1 at each position that belongs to the set
    /// Example: x[]=[0,1,3,4,8] is printed as "11.11...1"
    /// \param bla[in]: optional prefix string to print before the delta set
	void print_deltaset(const char *bla=nullptr) const
	{ print_set_as_deltaset(bla, c_, k_, n_); }

private:
    /// Helper function to print a set as a delta set representation
    /// \param bla[in]: optional prefix string to print
    /// \param x[in]: array containing the elements of the set
    /// \param n[in]: number of elements in the set x
    /// \param N[in]: total number of elements in the universe
    /// \param c01[in]: optional character array for representing 0s and 1s
    void print_set_as_deltaset(const char *bla, 
                               const T *x,
                               T n, 
                               T N, 
                               const char *c01=0) const {
    	static const char n01[] = {'.', '1'};
    	if ( bla )  std::cout << bla;
    
    	const char *d = ( nullptr==c01 ?  n01 : c01 );
    
    	T j = 0;
    	for (T k=0; k<n; ++k) {
    		for (  ; j<x[k]; ++j)  std::cout << d[0];
    		std::cout << d[1];
    		++j;
    	}
    
    	while ( j++ < N )  std::cout << d[0];
    }
};
