#pragma once 

/// TODO allocator
/// Combinations in a minimal-change order.
/// Algorithm R, "revolving-door combinations", TAOCP 4A/1, pp.363.
///  W. H. Payne, F. M. Ives: "Combination Generators",
///  ACM Transactions on Mathematical Software (TOMS),
///  vol.5, no.2, pp.163-172, (June-1979).
///
/// k1 = bit which is cleared 
/// k2 = bit which is set
/// Output(10, 2): k1k2
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
/// Example Code:
///     combination_revdoor c(10, 4);
///     uint32_t k1 = 1, k2 = 2;
///     do {
///     	c.print_deltaset();
///     	std::cout << " " << k1 << " " << k2 << std::endl;
///     } while (c.next(&k1, &k2));
///     return;
class combination_revdoor {
private:
    using T = uint32_t;

	T *c_;  // delta set
	T n_, k_;  // (n choose k)  n>=1,  1<=k<=n

	combination_revdoor(const combination_revdoor&) = delete;
	combination_revdoor & operator = (const combination_revdoor&) = delete;

public:
	// Must have:  1 <= k <= n
	explicit combination_revdoor(const T n,
                                 const T k) noexcept {
		n_ = n;  // (n ? n : 1);
		k_ = k;
		c_ = new T[k_+1];  // incl. sentinel
		first();
	}

	~combination_revdoor() noexcept { delete [] c_; }

	constexpr void first() noexcept {
		for (T j=0; j<k_; ++j) { c_[j] = j; }
		c_[k_] = n_;  // sentinel
	}

    /// \return 
	constexpr const T* data() const noexcept { return c_; }

	/// \param k1[out]: bit-position to be cleared
	/// \param k2[out]: bit-position to be set
	/// \return
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
				*k1 = std::max(c_[0], c-1);
				*k2 = std::min(c_[0],c-1);
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

    // Print x[0,..,n-1], a subset of {0,1,...,N-1} as delta set,
    // n is the number of elements in the set.
    // Example:  x[]=[0,1,3,4,8]  ==> "11.11...1"
	void print_deltaset(const char *bla=nullptr)  const
	{ print_set_as_deltaset(bla, c_, k_, n_); }

private:
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
