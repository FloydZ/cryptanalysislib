#ifndef CRYPTANALYSISLIB_TERNARY_H
#define CRYPTANALYSISLIB_TERNARY_H

#include "combination/chase.h"


// TODO replace with fq combinariont
class CombinationsMeta {
	/// Base Class: Dont use it normally.
	/// This class provides some helpers nothing more.

protected:
	template<class T>
	static inline uint64_t ith_value_left_zero_position(const T *value, const uint32_t len, const uint64_t i, const uint64_t start = 0) {
		uint64_t count = 0;
		for (uint64_t j = 0; j < len; ++j) {
			if (value[j] == 0)
				count += 1;

			if (count == (i+start+1))
				return j;
		}

		return -1;
	}

	// same as the function above. Only counting from right.
	template<class T>
	static inline uint64_t ith_value_right_zero_position(const T *value, const uint32_t len, const uint64_t i, const uint64_t start = 0) {
		uint64_t count = 0;
		for (uint64_t j = len; j > 0; --j) {
			if (value[j - 1] == 0)
				count += 1;

			if (count == (i+start+1))
				return j-1;
		}

		return -1;
	}

	template<class T>
	static void apply_bool_vector2(T *e_out, const T *e,
								   const uint16_t len_e,
								   const uint8_t *v, const uint32_t len_v,
								   const uint64_t start = 0, const bool mode = false) {
		for (int j = 0; j < len_v; ++j) {
			// is a bit set?
			if (int(v[j]) == 1) {
				// get the position of the i-th bit which is zero.
				// IMPORTANT we want the ith position in 'e'
				uint64_t pos;
				if (mode) {
					pos = ith_value_right_zero_position(e, len_e, j, start);
				} else {
					pos = ith_value_left_zero_position(e, len_e, j, start);
				}

				e_out[pos] = -1;
			}
		}
	}
};


template<class T>
class Combinations_Chase : CombinationsMeta {
	/*
	 *  generate a sequence of all bit vectors with length n and k bits set by only changing two bits.
	 * 	idea taken from https://stackoverflow.com/questions/36451090/permutations-of-binary-number-by-swapping-two-bits-not-lexicographically
	 * 	algorithm 3
	 */
	Combinations_Chase() : n(0), k(0), start(0) {};

protected:
	// update round variables.
	void left_round(const int b) {
		ASSERT(b < two_changes_binary_o.size());
		two_changes_binary_o[b] = two_changes_binary_o[b-1] + two_changes_binary_d[b-1] * (two_changes_binary_p[b-1]%2 ? two_changes_binary_n[b-1]-1 : two_changes_binary_p[b-1]+1);
		two_changes_binary_d[b] = two_changes_binary_d[b-1] * (two_changes_binary_p[b-1]%2 ? -1 : 1);
		two_changes_binary_n[b] = two_changes_binary_n[b-1] - two_changes_binary_p[b-1] - 1;
		two_changes_binary_p[b] = 0;
	}

	// write one bit `bit`.
	uint64_t left_write(T *A, const int b, const int bit){
		ASSERT(b < two_changes_binary_o.size());
		uint64_t ret = two_changes_binary_o[b] + two_changes_binary_p[b] * two_changes_binary_d[b];
		A[ret] = bit;
		return ret;
	}

public:
	Combinations_Chase(const uint64_t n, const uint64_t k, const uint64_t start = 0) :
	                                                                                   n(n), k(k), start(start) {
		ASSERT(k < (n-start) && "Wrong k size");
	};

	// IMPORTANT clear the input befor calling this function.
	void left_init(T *A) {
		two_changes_binary_o.clear();
		two_changes_binary_d.clear();
		two_changes_binary_n.clear();
		two_changes_binary_p.clear();

		two_changes_binary_o.resize(n);
		two_changes_binary_d.resize(n);
		two_changes_binary_n.resize(n);
		two_changes_binary_p.resize(n);
		two_changes_binary_b = 0;

		two_changes_binary_o[0] = 0;
		two_changes_binary_d[0] = 1;
		two_changes_binary_n[0] = n;
		two_changes_binary_p[0] = 0;
	}

	// pos1 pos of the new zero; pos2 = position of the new 1 both zero based
	uint64_t left_step(T *A, bool init = false) {
		if (!init) { // cleanup of the previous round
			do {
				left_write(A, two_changes_binary_b, 0);
			} while (++two_changes_binary_p[two_changes_binary_b] > (two_changes_binary_n[two_changes_binary_b] + two_changes_binary_b - k) && two_changes_binary_b--);
		}

		if (two_changes_binary_p[0] > n-k)
			return 0;

		// this is the bit which will be set to one.
		left_write(A, two_changes_binary_b, 1);

		while (++two_changes_binary_b < k) {
			left_round(two_changes_binary_b);
			left_write(A, two_changes_binary_b, 1);
		}

		if (two_changes_binary_p[0] > n-k)
			return 0;

		two_changes_binary_b = k-1;
		return 1;
	}


	// IMPORTANT clear the input before calling this function.
	void right_init(T *A) {
		std::cout << "not implemented\n";
		left_init(A);
	}

	// pos1 pos of the new zero; pos2 = position of the new 1 both zero based
	uint64_t right_step(T *A, bool init = false) {
		std::cout << "not implemented\n";
		return 0;
	}

	static void diff(const T *p, const T *p_old, const uint32_t limbs, uint32_t *pos1, uint32_t *pos2) {
		uint8_t sols = 0;                       // solution counter. Should be at most 2 if Chase generation is used.
		uint32_t* sol_ptr[2] = {pos1, pos2};    // easy access to the solution array

		for (int i = 0; i < limbs; ++i) {
			// get the diff of the current limb
			T x = p[i] - p_old[i];
			if (x != T(0)) {
				*sol_ptr[sols++] = i;
			}

			// early exit.
			if(sols == 2)
				break;
		}
	}

private:
	std::vector<uint64_t> two_changes_binary_o;     // offset from the left most position
	std::vector<int64_t>  two_changes_binary_d;     // direction set bit is moving
	std::vector<uint64_t> two_changes_binary_n;     // length of current part of the sequence
	std::vector<uint64_t> two_changes_binary_p;     // current position of the bit in the current part
	uint64_t two_changes_binary_b = 0;              // how many permutations already processed in the current window

	const uint64_t n;                               // length of the array
	const uint64_t k;                               // number of bits set
	const uint64_t start;                           // offset position (not implemented.)
};


template<class T>
class Combinations_Chase_Ternary : CombinationsMeta {
	/*
	 *
	 */
	Combinations_Chase_Ternary() : cce(0,0,0), cc(0,0,0),
								   n(0), e1(0), em1(0), start(0), e_with_ones(nullptr), v(nullptr){};
public:
	///
	/// \param n 		length of the array
	/// \param e1 		# ones
	/// \param em1 		# minus ones (or what ever you use to represent ternary values)
	/// \param start	offset to start
	Combinations_Chase_Ternary(const uint64_t n, const uint64_t e1, const uint64_t em1, const uint64_t start = 0) :
			cce(n, e1, start), cc(n-e1-start, em1, 0), n(n), e1(e1), em1(em1), start(start) {
		ASSERT(e1 < (n-start) && em1 < (n-start-e1)&& "Wrong size");

		// make sure everything is initialised to zero
		e_with_ones = (T *)calloc(n-start, sizeof(T));
		v = (uint8_t *)calloc(n-e1-start, sizeof(uint8_t));
	};

	// dont forget to free allocated data.
	~Combinations_Chase_Ternary() { free(e_with_ones); free(v); }

	void left_init(T *e) {
		cce.left_init(e_with_ones);
		cc.left_init(v);

		// we need to to the first step as an init.
		cce.left_step(e_with_ones, true);
		cc.left_step(v, true);

		std::copy(e_with_ones,e_with_ones + n, e);
		apply_bool_vector2(e, e_with_ones, n-start, v, n-start-e1, false);
	}

	uint64_t left_step(T *e) {
		uint64_t i = cc.left_step(v);
		if (i == 0) {
			i =  cce.left_step(e_with_ones);

			// dont forget do re init the `inner` miuns one loop.
			cc.left_init(v);
			cc.left_step(v, true);
		}

		std::copy(e_with_ones,e_with_ones + n, e);
		apply_bool_vector2(e, e_with_ones, n-start, v, n-start-e1, false);

		return i;
	}

	void right_init(T *e) {
		cce.right_init(e_with_ones);
		cc.right_init(v);

		// we need to to the first step as an init.
		cce.right_step(e_with_ones, true);
		cc.right_step(v, true);

		std::copy(e_with_ones,e_with_ones + n, e);
		apply_bool_vector2(e, e_with_ones, n-start, v, n-start-e1, true);
	}

	uint64_t right_step(T *e) {
		uint64_t i = cc.right_step(v);
		if (i == 0) {
			i =  cce.right_step(e_with_ones);

			// dont forget do re init the `inner` miuns one loop.
			cc.right_init(v);
			cc.right_step(v, true);
		}

		std::copy(e_with_ones,e_with_ones + n, e);
		apply_bool_vector2(e, e_with_ones, n-start, v, n-start-e1, true);

		return i;
	}

private:
	// some local variables
	const uint64_t n;       // length of the vector
	const uint64_t e1;      // #of ones
	const uint64_t em1;     // #of minus ones
	const uint64_t start;   // offset from one site

	T *e_with_ones;
	Combinations_Chase<T> cce;      // outer `loop` == keep track of the ones

	uint8_t *v;                     // helper to keep track of the minus ones.
	Combinations_Chase<uint8_t> cc;
};

template<class TernaryRow, class ChangeList>
class Combinations_Chase_TernaryRow {
	using CCB = Combinations_Chase_Binary<>;
	/// in comparison to `Combinations_Chase_Ternary` this class will enumerate 0,1,2 and NOT 0,1,-1
	CCB cc1;
	CCB cc2;

	uint32_t sym;
	const uint32_t n, nr1, nr2, limbs;
	std::vector<uint64_t> ones, ones_tmp;
	std::vector<uint64_t> twos, twos_tmp;
	Combinations_Chase_TernaryRow() : cc1(1,1,0), cc2(1,1,0) {};

	size_t resize(const uint32_t n) {
		return (n+63)/64;
	}

public:
	Combinations_Chase_TernaryRow(const uint32_t n, const uint32_t nr1, const uint32_t nr2) :
			cc1(n, nr1,0), cc2(n-nr1, nr2,0), n(n), nr1(nr1), nr2(nr2), limbs(resize(n)) {
		ASSERT(nr1+nr2 <= n);
		//ASSERT(nr2 > 0);
		//ASSERT(nr1 > 0);
		twos.resize(resize(n));
		ones.resize(resize(n));
	};

	// magic function doing all the ternary magic.
	void ternary_traverse(TernaryRow &row, const uint32_t us,
						  const uint16_t pos1, const uint16_t pos2,
						  uint16_t *rpos1, uint16_t *rpos2) {
		uint32_t ctr = 0, j = 0;
		int32_t sign = 0;

		for (; j < n; ++j) {
			if (row.get(j) == (3 - us)) continue;
			if (ctr == pos1) break;
			ctr += 1;
		}

		*rpos1 = j;
		if (row.get(j) == 0) {
			// we are moving left now.
			sign = 1;
			row.set(us, j);
		} else {
			row.set(0, j);
		}

		j += 1;
		ctr += 1;

		for (; j < n; ++j) {
			if (row.get(j) == (3 - us)) continue;
			if (ctr == pos2) break;
			ctr += 1;
		}

		*rpos2 = j;
		if (sign) {
			row.set(0, j);
			std::swap(*rpos1, *rpos2);
		} else {
			row.set(us, j);
		}
	}

	// translates relative position into new absolute positions and sets pos1 and pos2 to us
	void ternary_set(TernaryRow &row, const uint32_t us, const uint16_t pos1) {
		uint32_t ctr = 0, j = 0;
		for (; j < n; ++j) {
			if (row.get(j) == (3 - us)) continue;
			if (ctr == pos1) break;
			ctr += 1;
		}

		row.set(us, j);
	}

	void left_init(TernaryRow &row) {
		cc1.left_init(ones.data());
		cc2.left_init(twos.data());
		cc1.left_step(ones.data(), true);
		cc2.left_step(twos.data(), true);

		row.zero();
		for (uint32_t i = 0; i < nr1; ++i) {
			row.set(1, i);
		}
		for (uint32_t i = nr1; i < nr1+nr2; ++i) {
			row.set(2, i);
		}
	}

	// returns      <  0: if a one was changed
	//              >  0: if a two was change
	//              == 0: if finished
	int64_t left_step(TernaryRow &row, ChangeList &cl) {
		twos_tmp = twos;
		int64_t i = cc2.left_step(twos.data());
		uint16_t pos1, pos2, rpos1=0, rpos2=0;
		if (i == 0) {
			ones_tmp = ones;
			i =  cc1.left_step(ones.data());
			if (i == 0)
				return 0;

			i = -i;

			// dont forget do re init the twos
			cc2.left_init(twos.data());
			cc2.left_step(twos.data(), true);

			// make one step with the ones.
			CCB::diff(ones.data(), ones_tmp.data(), limbs, &pos1, &pos2);

			// reset the row
			row.zero();
			for (uint32_t j = 0; j < n; ++j) { // copy ones array into row
				row.set(CCB::GET_BIT(ones.data(), j), j);
			}

			// some coping of the ptrs
			cl.first = pos1; cl.second = pos2;
			rpos1 = pos1; rpos2 = pos2;

			// if at the current position is a "one" we need to swap the rpos values
			if (CCB::GET_BIT(ones.data(), rpos1)) {
				std::swap(rpos1, rpos2);
			}


			// set the twos in the row
			for (uint32_t j = 0; j < nr2; ++j) {
				ternary_set(row, 2, j);
			}

		} else {
			CCB::diff(twos.data(), twos_tmp.data(),
					  limbs, &pos1, &pos2);
			ternary_traverse(row, 2, pos1, pos2, &rpos1, &rpos2);
		}

		//std::cout << row <<  " " << pos1 << ":" << pos2 << ", " << rpos1 << ":" << rpos2 << " out\n";

		// next update the change list.
		cl.first = rpos1; cl.second = rpos2;
		return i;
	}

	void left_single_init(TernaryRow &row, const uint32_t sym_) {
		cc1.left_init(ones.data());
		cc1.left_step(ones.data(), true);
		sym = sym_;
		row.zero();
		for (uint32_t i = 0; i < nr1; ++i) {
			row.set(sym, i);
		}
	}

	int64_t left_single_step(TernaryRow &row, ChangeList &cl) {
		ones_tmp = ones;
		int64_t i = cc1.left_step(ones.data());
		if (i == 0)
			return 0;

		CCB::diff(ones.data(), ones_tmp.data(),
				  limbs, &cl.first, &cl.second);

		if (row.get(cl.first) == 0) {
			row.set(sym, cl.first);
			row.set(0, cl.second);
			std::swap(cl.first, cl.second);
		} else {
			row.set(0, cl.first);
			row.set(sym, cl.second);
		}

		//std::cout << row <<  " " << cl.first << ":" << cl.second << " out\n";
		return i;
	}
};

#endif//CRYPTANALYSISLIB_TERNARY_H
