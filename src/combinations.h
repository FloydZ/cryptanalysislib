#ifndef SMALLSECRETLWE_COMBINATIONS_H
#define SMALLSECRETLWE_COMBINATIONS_H

#include <iostream>
#include <vector>
#include <functional>

#include "m4ri/m4ri.h"

#include "element.h"

class CombinationsIndex {
private:
	const uint32_t n,k,offset;
	uint32_t j;
	int32_t *a = nullptr;
public:

	/// generate the indeces of permutations for n over k.
	/// if offset is set, this value is added to each permutation index.
	/// EXAMPLE:
	///		generate(P, 20, 3, 0) returns:
	///			[[0,1,2], [0,1,3], ...]
	/// \param P 	Output array. __MUST__ be of size binomial(n,k)*k*sizeof(uin32_t)
	/// \param n 	Number of coordinates
	/// \param k 	from which to choose k
	/// \param offset
	static void generate(uint32_t *P, const uint32_t n, const uint32_t k, const uint32_t offset = 0) {
		int i = 0, j;
		unsigned int ctr = 0;

		// Helper Array.
		int32_t *a = (int32_t *)malloc(sizeof(int32_t)*k);
		for (uint32_t j = 0; j < k; ++j) {
			a[j] = 0;
		}

		while(true) {
			for (j = i+1; j < k; ++j) {
				a[j] = a[j-1]+1;
			}

			i=j-1;

			// here yield
			for (uint32_t j = 0; j < k; ++j) {
				P[ctr*k + j] = a[j] + offset;
			}
			ctr += 1;
			while (a[i] == int32_t(i + n - k)) { i-=1; }

			// Exit condition.
			if (i == -1)
				break;

			a[i] += 1;
		}

		free(a);
	}

	~CombinationsIndex() { free(a); };

	CombinationsIndex(uint16_t *P,  const uint32_t n, const uint32_t k, const uint32_t offset = 0) : n(n), k(k), offset(offset), j(0) {
		a = (int32_t *)malloc(sizeof(int32_t)*k);
		for (j = 0; j < k; ++j) {
			P[j] = a[j] = j;
		}
		 // do the first half step.
		for (j = 1; j < k; ++j) {
			a[j] = a[j-1]+1;
		}
	}

	int next(uint16_t *P) {
		for (uint32_t i = 0; i < k; ++i) {
			P[i] = offset + a[i];
		}

		int i = k - 1;
		while (a[i] == int32_t(i + n - k)) { i -= 1; }

		// Exit condition.
		if (i == -1)
			return 0;

		a[i] += 1;
		return 1;
	}
};

class CombinationsMeta {
	/// Base Class: Dont use it normaly. This class provides some helpers nothing more.

protected:
	template<class T>
	static void _swap(T *e, uint64_t left, uint64_t right) {
		auto tmp = e[left];
		e[left] = e[right];
		e[right] = tmp;
	}


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
/*
 * Some Ideas to this class.
 * - Currently this file only depend on the ASSERT macro, M4Ri and Element class
 *
 */
template<class Element>
class Combinations : CombinationsMeta {
	/// Base Class: Dont use it normally. This class provides some helpers nothing more.
	/// little doc:
	///		- right means: the ones/minus_ones will be initializes on the right side of the gicen vector/Element.
	///						and after invocation of a 'step'-function this ones/minus_ones will be shifted to the left.
	///		- left means:  the exact opposite.
protected:

	static void _swap(Element &e, uint64_t left, uint64_t right) {
		auto tmp = e.get_value(left);
		e.get_value().data()[left] = e.get_value(right);
		e.get_value().data()[right] = tmp;
		// std::cout << e << "\n";
	}

	static void _swap_only_on_diff(Element &e, uint64_t left, uint64_t right) {
		if (e.get_value().data()[left] != e.get_value().data()[right]) {
			auto tmp = e.get_value().data()[left];
			e.get_value().data()[left] = e.get_value().data()[right];
			e.get_value().data()[right] = tmp;
		}
	}


	/// given e with only '1' and a binary vector 'v' this function will do the following:
	///			e: [ 0 0 1 0 0 ] 	v: [ 1 1 0 ] -> e_out: [ -1 -1 1 0 0 ]
	///			e: [ 0 0 1 0 0 ] 	v: [ 0 1 1 ] -> e_out: [  0 -1 1 -1 0]
	///
	/// \param e_out Out Array/Element to apply the minus ones
	/// \param e
	/// \param v
	/// \param len 		length of v
	/// \param start
	/// \param mode 	false: 	start counting from left,
	///					true:	start counting from right
	static void apply_bool_vector(Element &e_out, const Element &e,
	                              const uint8_t *v, const uint32_t len,
	                              const uint64_t start = 0, const bool mode = false) {
		uint64_t pos;
		for (int j = 0; j < len; ++j) {
			// is a bit set?
			if (v[j]) {
				// get the position of the i-th bit which is zero.
				// IMPORTANT we want the ith position in 'e'
				if (mode)
					pos = e.ith_value_right_zero_position(j, start);
				else
					pos = e.ith_value_left_zero_position(j, start);
				e_out.get_value().data()[pos] = -1;
			}
		}
	}

	static void apply_bool_vector(Element &e_out, const Element &e,
	                              const std::vector<uint8_t> &v,
	                              const uint64_t start = 0, const bool mode = false) {
		apply_bool_vector(e_out, e, v.data(), v.size(), start, mode);
	}
};

class Combinations_Chase_VV_Binary : CombinationsMeta {
	/*
	 *  generate a sequence of all bit vectors with length n and k bits set by only changing two bits.
	 * 	code/idea taken from Valentin Vasseur implementation.
	 *
	 *  Generate a Chase's sequence: the binary representation of a combination and
	 *  its successor only differ by two bits that are either consecutive of
	 *  separated by only one position.
	 *
	 *  See exercise 45 of Knuth's The art of computer programming volume 4A.
	 */

	// disable the standard constructor.
	Combinations_Chase_VV_Binary() : n(0), t(0), offset(0) {};

public:
	static void chase2(size_t n, size_t t, uint16_t *combinations, uint16_t *diff) {
		size_t N = 0;
		uint16_t diff_pos = 0;
		//uint16_t diff_len;
		int32_t x;
		uint16_t *c = (uint16_t*)malloc((t + 2) * sizeof(uint16_t));
		uint16_t *z = (uint16_t*)malloc((t + 2) * sizeof(uint16_t));
		for (size_t j = 1; j <= t + 1; ++j) {
			z[j] = 0;
		}
		for (size_t j = 1; j <= t + 1; ++j) {
			c[j] = n - t - 1 + j;
		}
		/* r is the least subscript with c[r] >= r. */
		size_t r = 1;
		size_t j;

		while (true) {
			for (size_t i = 1; i <= t; ++i) {
				combinations[i - 1 + N * t] = c[i];
			}
			diff[N] = diff_pos;// + (diff_len - 1);// * (n - 1);
			++N;
			j = r;

			novisit:
			if (z[j]) {
				x = c[j] + 2;
				if (x < z[j]) {
					diff_pos = c[j];
					//diff_len = 2;
					c[j] = x;
				} else if (x == z[j] && z[j + 1]) {
					diff_pos = c[j];
					//diff_len = 2 - (c[j + 1] % 2);
					c[j] = x - (c[j + 1] % 2);
				} else {
					z[j] = 0;
					++j;
					if (j <= t)
						goto novisit;
					else
						goto exit;
				}
				if (c[1] > 0) {
					r = 1;
				} else {
					r = j - 1;
				}
			} else {
				x = c[j] + (c[j] % 2) - 2;
				if (x >= (int32_t)j) {
					diff_pos = x;
					//diff_len = 2 - (c[j] % 2);
					c[j] = x;
					r = 1;
				} else if (c[j] == j) {
					diff_pos = j - 1;
					//diff_len = 1;
					c[j] = j - 1;
					z[j] = c[j + 1] - ((c[j + 1] + 1) % 2);
					r = j;
				} else if (c[j] < j) {
					diff_pos = c[j];
					//diff_len = j - c[j];
					c[j] = j;
					z[j] = c[j + 1] - ((c[j + 1] + 1) % 2);
					r = (j > 2) ? j - 1 : 1;
				} else {
					diff_pos = x;
					//diff_len = 2 - (c[j] % 2);
					c[j] = x;
					r = j;
				}
			}
		}

		exit:
		free(c);
		free(z);
	}

	int next(uint16_t *P) {
		for (size_t i = 1; i <= t; ++i) {
			P[i - 1] = c[i] + offset;
		}
		//diff[N] = diff_pos;// + (diff_len - 1);// * (n - 1);
		++N;
		j = r;

		novisit:
		if (z[j]) {
			x = c[j] + 2;
			if (x < z[j]) {
				diff_pos = c[j];
				diff_len = 2;
				c[j] = x;
			} else if (x == z[j] && z[j + 1]) {
				diff_pos = c[j];
				diff_len = 2 - (c[j + 1] % 2);
				c[j] = x - (c[j + 1] % 2);
			} else {
				z[j] = 0;
				++j;
				if (j <= t)
					goto novisit;
				else
					return 0; //goto exit;
			}
			if (c[1] > 0) {
				r = 1;
			} else {
				r = j - 1;
			}
		} else {
			x = c[j] + (c[j] % 2) - 2;
			if (x >= (int32_t)j) {
				diff_pos = x;
				diff_len = 2 - (c[j] % 2);
				c[j] = x;
				r = 1;
			} else if (c[j] == j) {
				diff_pos = j - 1;
				diff_len = 1;
				c[j] = j - 1;
				z[j] = c[j + 1] - ((c[j + 1] + 1) % 2);
				r = j;
			} else if (c[j] < j) {
				diff_pos = c[j];
				diff_len = j - c[j];
				c[j] = j;
				z[j] = c[j + 1] - ((c[j + 1] + 1) % 2);
				r = (j > 2) ? j - 1 : 1;
			} else {
				diff_pos = x;
				diff_len = 2 - (c[j] % 2);
				c[j] = x;
				r = j;
			}
		}

		return 1;
	}

public:
	~Combinations_Chase_VV_Binary() {
		free(c);
		free(z);
	}
	Combinations_Chase_VV_Binary(const uint64_t n, const uint64_t t, const uint64_t offset = 0) :
			n(n), t(t), offset(offset) {
		c = (uint16_t *)malloc((t + 2) * sizeof(uint16_t));
		z = (uint16_t *)malloc((t + 2) * sizeof(uint16_t));

		for (size_t j = 1; j <= t + 1; ++j) {
			z[j] = 0;
		}
		for (size_t j = 1; j <= t + 1; ++j) {
			c[j] = n - t - 1 + j;
		}
	};

	const uint64_t n;                               // length of the array
	const uint64_t t;                               // number of bits set
	const uint64_t offset;                           // offset position (not implemented.)
	uint16_t *c = nullptr;
	uint16_t *z = nullptr;

	size_t N = 0;
	uint16_t diff_pos = 0;
	uint16_t diff_len = 0;
	int32_t x;


	/* r is the least subscript with c[r] >= r. */
	size_t r = 1;
	size_t j;
};


template<class T>
class Combinations_Chase2 : CombinationsMeta {
	// Implementation taken from Valentinos Code
	// Call it with:
	//      	Combinations_Chase2<DecodingContainer> cm{k+l, p, 0};

	// disable the standard constructor.
	Combinations_Chase2() : n(0), k(0), start(0) {};

public:
	void table(std::vector<T> &table, std::vector<std::pair<uint64_t, uint64_t>> &diff){
		const uint64_t size = bc(n, k);
		uint64_t pos1 = 0, pos2 = 0, ctr = 0;

		table.resize(size);
		diff.resize(size);
		T current, tmp;

		current.zero(); tmp.zero();

		for (int j = n-k; j < n; ++j) {
			current[j] = true;
		}

		do {
			table[ctr] = current;
			diff[ctr]  = std::pair<uint64_t, uint64_t>{pos1, pos2};

			ctr += 1;
		} while (next(current, &pos1, &pos2) != 0);

		// make sure correctly precomputed everything
		ASSERT(ctr == size);
	}

	int next(T &cc, uint64_t *pos1, uint64_t *pos2) {
		bool found_r = false;
		int j;
		for (j = r; !w[j]; j++) {
			int b = a[j] + 1;
			int n = a[j + 1];
			if (b < (w[j + 1] ? n - (2 - (n & 1u)) : n)) {
				if ((b & 1u) == 0 && b + 1 < n) b++;
				cc.clear_bit(a[j]);
				*pos1 = a[j];
				a[j] = b;
				cc.set_bit(a[j]);
				*pos2 = a[j];

				if (!found_r) r = j > 1 ? j - 1 : 0;
				return 1;
			}
			w[j] = a[j] - 1 >= j;
			if (w[j] && !found_r) {
				r = j;
				found_r = true;
			}
		}
		int b = a[j] - 1;
		if ((b & 1u) != 0 && b - 1 >= j)
			b--;

		*pos1 = a[j];
		cc.clear_bit(a[j]);
		a[j] = b;
		cc.set_bit(a[j]);
		*pos2 = a[j];

		w[j] = b - 1 >= j;
		if (!found_r)
			r = j;

		return (a[k] == n);
	}

	void left_init(T &cc) {
		for (int i = 0; i < k; ++i) {
			cc[i] = true;
		}
	}

	Combinations_Chase2(const uint64_t n, const uint64_t k, const uint64_t start = 0) :
			n(n), k(k), start(start) {
		ASSERT(k < (n - start) && "Wrong k size");
		a.resize(k+1), 0;
		w.resize(k+1, true);

		for (int j = 0; j < k + 1; j++) {
			a[j] = n - (k - j);
			w[j] = true;
		}
	};

	const uint64_t n;
	const uint64_t k;
	const uint64_t start;
	std::vector<int> a;
	std::vector<bool> w;

	int r = 0;
};

class Combinations_Chase_Binary2 : CombinationsMeta {
	// disable the standard constructor.
	Combinations_Chase_Binary2() : n(0), k(0), start(0) {};

public:
	int next(word *cc, uint64_t *pos1) {
		bool found_r = false;
		int j;
		for (j = r; !w[j]; j++) {
			int b = a[j] + 1;
			int n = a[j + 1];
			if (b < (w[j + 1] ? n - (2 - (n & 1u)) : n)) {
				if ((b & 1u) == 0 && b + 1 < n) b++;

				__M4RI_WRITE_BIT(cc[(start + a[j]) / m4ri_radix], (start + a[j]) % m4ri_radix, 0);
				*pos1 = start+a[j];
				a[j] = b;
				__M4RI_WRITE_BIT(cc[(start + a[j]) / m4ri_radix], (start + a[j]) % m4ri_radix, 1);
				//*pos2 = start+a[j];

				if (!found_r)
					r = j > 1 ? j - 1 : 0;
				return 1;
			}
			w[j] = a[j] - 1 >= j;
			if (w[j] && !found_r) {
				r = j;
				found_r = true;
			}
		}
		int b = a[j] - 1;
		if ((b & 1u) != 0 && b - 1 >= j)
			b--;

		*pos1 = found_r ? start +b : start+a[j];
		__M4RI_WRITE_BIT(cc[(start + a[j]) / m4ri_radix], (start + a[j]) % m4ri_radix, 0);
		a[j] = b;
		__M4RI_WRITE_BIT(cc[(start + a[j]) / m4ri_radix], ( start + a[j]) % m4ri_radix, 1);
		//*pos2 = start+a[j];

		w[j] = b - 1 >= j;
		if (!found_r)
			r = j;
		return (a[k] == n);
	}

	Combinations_Chase_Binary2(const uint32_t n, const uint32_t k, const uint32_t start = 0) :
			n(n), k(k), start(start) {
		//ASSERT(k < (n - start) && "Wrong k size");
		a.resize(k+1);
		w.resize(k+1, true);

		for (uint32_t j = 0; j < k + 1; j++) {
			a[j] = n - (k - j);
			w[j] = true;
		}
	};

	void init(word *p) {
		for (uint32_t j = start+n-k; j < start+n; ++j) {
			__M4RI_WRITE_BIT(p[j / m4ri_radix], j % m4ri_radix, 1);
		}
	}

	const uint32_t n;
	const uint32_t k;
	const uint32_t start;
	std::vector<int> a;
	std::vector<bool> w;

	int r = 0;
};

template<class T>
class Combinations_Lexicographic : protected CombinationsMeta {
	/*
	 * source: https://stackoverflow.com/questions/27755687/what-is-the-fastest-algorithm-to-computer-all-permutations-of-a-binary-number-wi
	 * generate lexicographically next bit-permutation of a n bit vector with weight k by gospers hack.
	 * So this does NOT generate the next vector by only changing two bits.
	 *
	 * The class `T` needs to implement:
	 *      - [ ]
	 */
private:
	// disable the default constructor
	Combinations_Lexicographic() : n(0), k(0), start(0) {};

public:
	Combinations_Lexicographic(const uint64_t n, const uint64_t k, const uint64_t start = 0) :
						n(n), k(k), start(start) {ASSERT(k < (n-start) && "Wrong k size"); };


	void right_init(T *e) const {
		for (uint64_t i = start; i < n-k; ++i) {
			e[i] = T(0);
		}
		for (uint64_t i = n-k; i < n; ++i) {
			e[i] = T(1);
		}
	}

	uint64_t right_step(T *e) const{
		uint64_t i = n-1;
		for (; i > start ; i--) {
			if (e[i] && !e[i-1]) break;
		}

		if (i == start)
			return 0;

		// Step 2 swap
		_swap<T>(e, i, i-1);

		// step 3 move up
		uint64_t l = 0;
		for (uint64_t j = i+1; j < i + 1 + ((n-i)/2); ++j) {
			_swap<T>(e, n-l-1, j);
			l += 1;
		}

		return i;
	}


	void left_init(T *e) const {
		for (uint64_t i = start; i < start+k; ++i) {
			e[i] = 1;
		}
		for (uint64_t i = start+k; i < n; ++i) {
			e[i] = 0;
		}
	}

	uint64_t left_step(T *e) const {
		uint64_t i = start;
		for (; i < n-1 ; i++) {
			if (e[i] && !e[i+1]) break;
		}

		if (i == n-1)
			return 0;

		// Step 2 swap
		_swap(e, i, i+1);

		// step 3 move up
		uint64_t l = start;
		if (i != 0) {
			for (uint64_t j = i - 1; j > start-1 + ((i - 1) / 2); --j) {
				_swap(e, l, j);
				l += 1;
			}
		}

		return n-i;
	}

protected:
	const uint64_t n;
	const uint64_t k;
	const uint64_t start;
};



class Combinations_Lexicographic_Binary : CombinationsMeta {
	/*
	 * Use it like this:
	  	mzd_t *p = mzd_init(1, G_n);
	 	Combinations_Lexicographic_M4RI c{G_n, G_k};
		c.left_init(p);
		do {
			// do stuff with it.
		}while(c.left_step(p) != 0);

	 * if you implement and use the class as shown above wou will get in each iteration a element with weight omega in
	 * lexigraphical ordering. Which means with more than just two changed bits.
	 */
private:
	// disable the default constructor
	Combinations_Lexicographic_Binary() : n(0), k(0), start(0)  {};

	// increments the first row by one, by interpreting this row as a bigint.
	// currently unused
	static uint64_t mpz_d_inc_ctz(const mzd_t *const c) {
		ASSERT(c->nrows == 1);
		__uint128_t t;      // This is actually only valid if M4RI uses 64Bit limbs
		word *a = (word *)malloc(c->width*sizeof(word));

		// a = c+1
		t = (__uint128_t)c->rows[0][0] + 1;
		a[0] = (uint64_t) t;
		t >>= m4ri_radix;
		int x = 0;
		while(t > 0 && x < c->width){
			x += 1;
			t = (__uint128_t)c->rows[0][x] + t;
			a[x] = (uint64_t)t;
			t >>= m4ri_radix;
		}

		// now count trailing zero.
		int ret = 0;
		int i = 0;
		for (; i < c->width; i++) {
			if (a[i] == 0)
				continue;

			ret = __builtin_ctzll(a[i]);
			break;
		}

		ret += i*m4ri_radix;
		free(a);
		return ret;
	}

protected:
	Combinations_Lexicographic_Binary(const uint64_t n, const uint64_t k, const uint64_t start = 0) :
			n(n), k(k), start(start) {};

	// we need these little helpers, because M4RI does not implement any row access functions, only ones for matrices.
	inline static void write_bit(word *row, const uint16_t pos, const BIT bit) { __M4RI_WRITE_BIT(row[pos/m4ri_radix], pos%m4ri_radix, bit); }
	inline static BIT get_bit(const word *row, const uint16_t pos) { return __M4RI_GET_BIT(row[pos/m4ri_radix], pos%m4ri_radix); }
	static inline void SWAP_BITS(word *row, const int pos1, const int pos2) {
		BIT p1 = get_bit(row, pos1);
		BIT p2 = get_bit(row, pos2);
		write_bit(row, pos1, p2);
		write_bit(row, pos2, p1);
	}

public:
	// we can not abstract this more, because otherwise we coulnt work on bit but only on limbs.
	// binary version for m4ri
	void left_init(word *v) const {
		for (uint64_t i = 0; i < n; ++i) { write_bit(v, i, false); }
		for (uint64_t i = start; i < start+k; ++i) { write_bit(v, i, true); }
	}
	uint64_t left_step(word *v) const {
		uint64_t i = start;
		for (; i < n-1 ; i++) {
			if (get_bit(v, i) && !get_bit(v, i+1)) break;
		}

		if (i == n-1)
			return 0;

		// Step 2 swap
		SWAP_BITS(v, i, i+1);
		//std::swap(v[i], v[i+1]);

		// step 3 move down
		uint64_t l = start;
		if (i != 0) {
			for (uint64_t j = i-1; j > start-1 + ((i-1)/2); --j) {
				//std::swap(v[k], v[j]);
				SWAP_BITS(v, l, j);

				l += 1;
			}
		}

		// correct the output
		return n-i;
	}

private:
	const uint64_t n;
	const uint64_t k;
	const uint64_t start;
};

class Combinations_Lexicographic_M4RI : Combinations_Lexicographic_Binary {
	/*
	 * The same as the class `Combinations_Lexicographic_Binary` but specified/specialised for a m4ri
	 * matrix with only one row. It does not inherit from
	 */
private:
	// disable the default constructor
	Combinations_Lexicographic_M4RI() : Combinations_Lexicographic_Binary(0, 0, 0) {};

public:
	Combinations_Lexicographic_M4RI(const uint64_t n, const uint64_t k, const uint64_t start = 0) :
			Combinations_Lexicographic_Binary(n, k, start) {};

	// binary version for 1xn matrices from m4ri
	void left_init(mzd_t *A) {
		ASSERT(A->nrows == 1 && A->ncols > 0);
		Combinations_Lexicographic_Binary::left_init(A->rows[0]);
	}

	uint64_t left_step(mzd_t *A){
		ASSERT(A->nrows == 1 && A->ncols > 0);
		return Combinations_Lexicographic_Binary::left_step(A->rows[0]);
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

template<typename Word=uint64_t>
class Combinations_Chase_Binary {
	/*
	 *  generate a sequence of all bit vectors with length n and k bits set by only changing two bits.
	 * 	idea taken from https://stackoverflow.com/questions/36451090/permutations-of-binary-number-by-swapping-two-bits-not-lexicographically
	 * 	algorithm 3
	 */
	// data for the two functions: 'two_changes_binary_left_init', 'two_changes_binary_left_init'
	std::vector<uint64_t> two_changes_binary_o;      // offset from the left most position
	std::vector<int64_t>  two_changes_binary_d;      // direction set bit is moving
	std::vector<uint64_t> two_changes_binary_n;      // length of current part of the sequence
	std::vector<uint64_t> two_changes_binary_p;      // current position of the bit in the current part
	uint64_t two_changes_binary_b = 0;      // how many permutations already processed

	constexpr static uint32_t RADIX = sizeof(Word)*8;

	inline void left_round(const uint64_t b) {
		ASSERT(b < two_changes_binary_o.size());

		two_changes_binary_o[b] = two_changes_binary_o[b-1] + two_changes_binary_d[b-1] *
		                                                      (two_changes_binary_p[b-1]%2 ? two_changes_binary_n[b-1]-1 : two_changes_binary_p[b-1]+1);
		two_changes_binary_d[b] = two_changes_binary_d[b-1] * (two_changes_binary_p[b-1]%2 ? -1 : 1);
		two_changes_binary_n[b] = two_changes_binary_n[b-1] - two_changes_binary_p[b-1] - 1;
		two_changes_binary_p[b] = 0;
	}

	inline uint64_t left_write(Word *A, const uint32_t b, const int bit){
		ASSERT(b < two_changes_binary_o.size());
		uint64_t ret = start + two_changes_binary_o[b] + two_changes_binary_p[b] * two_changes_binary_d[b];
		WRITE_BIT(A, ret, bit);
		return ret;
	}

	// disable the normal standard constructor,
	Combinations_Chase_Binary() : two_changes_binary_b(0), n(0), k(0), start(0) {};
protected:
	// make them protected so all child classes can access them
	const uint64_t n;
	const uint64_t k;
	const uint64_t start;

public:
	// we need these little helpers, because M4RI does not implement any row access functions, only ones for matrices.
	static inline void WRITE_BIT(Word *v, const uint64_t i, const BIT b) { __M4RI_WRITE_BIT(v[i/RADIX], i%RADIX, b); }
	static inline BIT GET_BIT(const Word *v, const uint64_t i) { return __M4RI_GET_BIT(v[i/RADIX], i%RADIX); }

	Combinations_Chase_Binary(const uint64_t n, const uint64_t k, const uint64_t start = 0) :
			two_changes_binary_b(0),  n(n-start), k(k), start(start) {
		// ASSERT(k > 0);
	};


	// REMINDER: Make sure to set A on all limbs on zero.
	void left_init(Word *A) {
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

	uint64_t left_step(Word *A, bool init = false) {
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

	// xors the two matrices limb per limb
	static void diff(const Word *p, const Word *p_old, const uint32_t limbs, uint16_t *pos1, uint16_t *pos2) {
		uint8_t sols = 0;                       // solution counter. Should be at most 2 if Chase generation is used.
		uint32_t sol;                           // bit position of the change
		uint16_t* sol_ptr[2] = {pos1, pos2};    // easy access to the solution array

		for (uint32_t i = 0; i < limbs; ++i) {
			// get the diff of the current limb
			Word x = p[i] ^ p_old[i];
			// loop as long we found ones in the limb. (Maybe we have two ones in on limb)
			while (x != 0 && sols < 2) {
				sol = ffsll(x)-1;
				// clear the bit
				x ^= (uint64_t (1) << sol);

				// now check if the bit was already set in p_old. If so we know the new zero pos (pos1).
				// if not we know that the bit was set in p. So we know pos2. Na we ignore this now. Doesnt matter
				// where the zero or the one is.
				const uint64_t pos = i*RADIX + sol;
				*(sol_ptr[sols]) = pos;
				sols += 1;
			}

			// early exit.
			if(sols == 2)
				break;
		}
	}
};

// implemented from: https://stackoverflow.com/questions/22650522/how-to-generate-chases-sequence
//class Combinations_Chase_Binary2 {}
class Combinations_Chase_M4RI : Combinations_Chase_Binary<uint64_t> {
	// disable the normal standard constructor,
	Combinations_Chase_M4RI() : Combinations_Chase_Binary(0, 0, 0) {};
public:
	/*
	 * Example of usage:
	        Combinations_Chase_M4RI c{G_n, k};
			c.left_init(p);
			uint64_t rt = c.left_step(p, true);
			while(rt != 0) {
				print_matrix("p", p);
				rt = c.left_step(p);
			}
	 */
	Combinations_Chase_M4RI(const uint64_t n, const uint64_t k, const uint64_t start = 0) :
			Combinations_Chase_Binary(n, k, start) {};

	void left_init(mzd_t *A) {
		ASSERT(A->nrows == 1 && A->ncols > 0);
		Combinations_Chase_Binary::left_init(A->rows[0]);
		for (int i = 0; i < A->width; i++)
			A->rows[0][i] = 0;
	}

	uint64_t left_step(mzd_t *A, bool init = false) {
		ASSERT(A->nrows == 1 && A->ncols > 0);
		return Combinations_Chase_Binary::left_step(A->rows[0], init);
	}

	// xors the two matrices limb per limb
	static void diff(const mzd_t *p, const mzd_t *p_old, const uint32_t limbs, uint16_t *pos1, uint16_t *pos2) {
		ASSERT(p->nrows == 1 && p->ncols > 0 && p_old->nrows == 1 && p_old->ncols > 0);
		Combinations_Chase_Binary::diff(p->rows[0], p_old->rows[0], limbs, pos1, pos2);
	}

	// this function automatically resizes the vector `v`.
	// step 1) iterate over all elements with length n with weight k, while only changing two bits between two element.
	// step 2) save the diffs in a vector
	uint64_t generate_diff_list(std::vector<std::pair<uint64_t, uint64_t>> &v) {
		uint64_t rt, counter = 0;
		uint16_t pos1, pos2;

		mzd_t *p = mzd_init(1, n);
		mzd_t *p_old = mzd_init(1, n);
		const uint32_t limbs = p->width;

		// performance boost. Hopefully
		v.resize(bc(n, k));

		// init
		Combinations_Chase_Binary::left_init(p->rows[0]);
		Combinations_Chase_Binary::left_step(p->rows[0], true);
		mzd_copy(p_old, p);

		// do also the second step
		rt = Combinations_Chase_Binary::left_step(p->rows[0]);

		while(rt != 0) {
			diff(p, p_old, limbs, &pos1, &pos2);
			auto b = std::pair<uint64_t, uint64_t> (pos1, pos2);
			v[counter++] = b;

			// just for debugging:
			// print_matrix("p_old", p_old);
			// print_matrix("p", p);
			// std::cout << b.first << " " << b.second << "\n";

			mzd_copy(p_old, p);
			rt = Combinations_Chase_Binary::left_step(p->rows[0]);
		}


		mzd_free(p);
		mzd_free(p_old);
		return 0;
	}
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

#endif //SMALLSECRETLWE_COMBINATIONS_H
