#ifndef CRYPTANALYSISLIB_COMBINATION_CHASE_H
#define CRYPTANALYSISLIB_COMBINATION_CHASE_H

#include <cstdint>
#include <vector>
#include "helper.h"

/// functions/fields an enumerator must implement
/// \tparam Enumerator
template<typename Enumerator>
concept EnumeratorAble = requires(Enumerator e) {
	e.step();
};

/// needed to compute the list size before initializing the enumerator
/// \tparam n length to enumerate
/// \tparam q base field size
/// \tparam w hamming weight to enumerate: NOTE: can be zero, for Prange
/// \return list size
template<const uint32_t n, const uint32_t q, const uint32_t w>
constexpr size_t compute_combinations_fq_chase_list_size() {
	static_assert(n > w);
	static_assert(q > 1);
	size_t size=1;
	for (uint64_t i = 0; i < w; i++) {
		size *= (q-1);
	}

	// just make sure that we do not return zero.
	return std::max(size, size_t(1ull)) * (bc(n, w) - 1);
}

/// This class enumerates a normal chase sequence of weight w. This means
/// all elements are enumerated which only differ in at most two positions.
/// On top of this sequence, a grey code will enumerated to enumerated over
/// all elementes in Fq
/// \tparam n length to enumerate
/// \tparam q base field size
/// \tparam w hamming weight to enumerate
template<uint32_t n, const uint32_t q, const uint32_t w>
class Combinations_Fq_Chase {
//private:
public:
	/// max value to enumerate
	constexpr static uint32_t qm1 = q-1;

	/// \return  number of elements in the gray code
	constexpr static size_t compute_gray_size() noexcept {
		uint64_t sum = 1;

		for (uint64_t i = 0; i < w; i++) {
			sum *= qm1;
		}

		// just make sure that we do not return zero.
		return std::max(sum, uint64_t(1ull));
	}

	constexpr static size_t chase_size = bc(n, w) - 1;
	constexpr static size_t gray_size = compute_gray_size();
	
	constexpr static size_t LIST_SIZE = chase_size * gray_size;

	/// NOTE: this is the output container for `mixed_radix_grey`
	std::array<uint32_t, n> a;

	/// stuff for the mixed radix enumeration
	std::array<uint32_t, n + 1> f;
	std::array<uint32_t, n> s; // sentinel

	/// stuff for the chase sequence
	std::array<int, w + 1> chase_w;
	std::array<int, w + 1> chase_a;
	/// NOTE: probably unused
	/// \return 1 if the sequence is still valid
	/// 	    0 if the sequence is finished
	constexpr bool mixed_radix_grey() {
		while (true) {
			uint32_t j = f[0];
			f[0] = 0;
			if (j == n) {
				return false;
			}

			a[j] = (a[j] + 1) % q;

			if (a[j] == s[j]) {
				s[j] = (s[j]-1 + q) % q;
				f[j] = f[j+1];
				f[j+1] = j + 1;
			}
		}

		return true;
	}

	/// NOTE: this enumerates on a length w NOT on length n
	/// NOTE: you have to free this stuff yourself
	/// NOTE: the input must be of size sum_bc(w, w)*q**w
	/// NOTE: returns 0 on success
	/// NOTE: only enumerates to q-1
	constexpr int changelist_mixed_radix_grey(uint16_t *ret) noexcept {
		uint32_t j = 0;
		size_t ctr = 0;

		while (ctr < gray_size) {
			j = f[0];
			f[0] = 0;
			ret[ctr++] = j;
			if (j == w)
				return 0;

			a[j] = (a[j] + 1) % (qm1);

			if (a[j] == s[j]) {
				s[j] = (s[j]-1 + qm1) % qm1;
				f[j] = f[j+1];
				f[j+1] = j + 1;
			}
		}

		return 0;
	}

	/// \param r helper value, init with 0
	/// \param jj returns the change position
	constexpr void chase(int *r, int *jj) noexcept {
		bool found_r = false;
		int j;
		for (j = *r; !chase_w[j]; j++) {
			int b = chase_a[j] + 1;
			int nn = chase_a[j + 1];
			if (b < (chase_w[j + 1] ? nn - (2 - (nn & 1u)) : nn)) {
				if ((b & 1) == 0 && b + 1 < nn) {
					b++;
				}

				chase_a[j] = b;
				if (!found_r) {
					*r = (int)(j > 1 ? j - 1 : 0);
				}

				*jj = j;
				return;
			}

			chase_w[j] = chase_a[j] - 1 >= j;
			if (chase_w[j] && !found_r) {
				*r = (int)j;
				found_r = true;
			}
		}

		int b = (int)chase_a[j] - 1;
		if ((b & 1) != 0 && b - 1 >= j) {
			b--;
		}

		chase_a[j] = b;
		chase_w[j] = b - 1 >= j;
		if (!found_r) {
			*r = j;
		}

		*jj = j;
	}

	/// NOTE: this function inverts the output of the chase sequence
	/// \param ret
	constexpr void changelist_chase(std::pair<uint16_t,uint16_t> *ret) {
		int r = 0, j = 0;

		uint32_t tmp[w+1] = {0};
		for (uint32_t i = 0; i < w; ++i) {
			tmp[i] = n-chase_a[i]-1;
		}

		for (uint32_t i = 0; i < chase_size; ++i) {
			//print_chase_state(r, j);
			chase(&r, &j);
			ASSERT(j < w);

			ret[i].first = j;
			ret[i].second = n-chase_a[j]-1;

			tmp[j] = n-chase_a[j]-1;
		}
	}

public:
	///
	/// \param n length
	/// \param q field size
	/// \param w max hamming weight to enumerate
	constexpr Combinations_Fq_Chase() noexcept {
		/// init the restricted gray code
		for (uint32_t i = 0; i < n; ++i) {
			a[i] = 0;
			f[i] = i;
			s[i] = qm1-1;
		}
		f[n] = n;

		/// init the chase sequence
		for (uint32_t i = 0; i < w + 1; ++i) {
			chase_a[i] = n - (w - i);
			chase_w[i] = true;
		}
	}

	/// return TODO
	constexpr uint32_t step() {
		return 1;
	}

};

///
/// \tparam T
template<typename T=uint64_t>
requires std::is_integral_v<T>
class Combinations_Chase_Binary {
	/*
	 *  generate a sequence of all bit vectors with length n and k bits set by only changing two bits.
	 * 	idea taken from https://stackoverflow.com/questions/36451090/permutations-of-binary-number-by-swapping-two-bits-not-lexicographically
	 * 	algorithm 3
	 *
	 *  This is the data struct which is used in the BJMM/MMT algorithm.
	 */
	// data for the two functions: 'two_changes_binary_left_init', 'two_changes_binary_left_init'
	std::vector<uint64_t> two_changes_binary_o;      // offset from the left most position
	std::vector<int64_t>  two_changes_binary_d;      // direction set bit is moving
	std::vector<uint64_t> two_changes_binary_n;      // length of current part of the sequence
	std::vector<uint64_t> two_changes_binary_p;      // current position of the bit in the current part
	uint64_t two_changes_binary_b = 0;      // how many permutations already processed

	// number of bits in one limb
	constexpr static uint32_t RADIX = sizeof(T) * 8;

	//
	inline void left_round(const uint64_t b) {
		ASSERT(b < two_changes_binary_o.size());

		two_changes_binary_o[b] = two_changes_binary_o[b-1] + two_changes_binary_d[b-1] *
															  (two_changes_binary_p[b-1]%2 ? two_changes_binary_n[b-1]-1 : two_changes_binary_p[b-1]+1);
		two_changes_binary_d[b] = two_changes_binary_d[b-1] * (two_changes_binary_p[b-1]%2 ? -1 : 1);
		two_changes_binary_n[b] = two_changes_binary_n[b-1] - two_changes_binary_p[b-1] - 1;
		two_changes_binary_p[b] = 0;
	}

	//
	inline uint64_t left_write(T *A, const uint32_t b, const int bit){
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
	constexpr static inline void WRITE_BIT(T *v, const size_t i, const uint64_t b) {
		v[i/RADIX] = ((v[i/RADIX] & ~(1ull << (i%RADIX))) | (T(b) << (i%RADIX)));
	}
	constexpr static inline uint64_t GET_BIT(const T *v, const size_t i) {
		return __M4RI_GET_BIT(v[i / RADIX], i % RADIX);
	}

	Combinations_Chase_Binary(const uint64_t n, const uint64_t k, const uint64_t start = 0) :
			two_changes_binary_b(0),  n(n-start), k(k), start(start) {
	};


	// REMINDER: Make sure to set A on all limbs on zero.
	void left_init(T *A) {
		(void) A;
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

	/// computes one step in the cache sequence
	/// \param A  input array = limbs needed to represent 'n' bits.
	/// \param init if True: if the first element in the cache sequence is going to be generated,
	/// 			  False: else
	/// \return false if the end of the sequence is reached
	///         true if the end of the sequence is NOT reached
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

	/// This functions simply xors together two given rows `p` and `p_old` and finds the two positions where they differ
	/// \param p newly generated chase element. Output from `left_step`
	/// \param p_old  last generated element from the chase sequence.
	/// \param limbs Number of limbs of type T needed to represent a element of the chase sequence
	/// \param pos1 output: first bit position where 'p` and `p_old` differs
	/// \param pos2 output: second bit position
	static void diff(const T *p, const T *p_old, const uint32_t limbs, uint16_t *pos1, uint16_t *pos2) {
		uint8_t sols = 0;                       // solution counter. Should be at most 2 if Chase generation is used.
		uint32_t sol;                           // bit position of the change
		uint16_t* sol_ptr[2] = {pos1, pos2};    // easy access to the solution array

		for (uint32_t i = 0; i < limbs; ++i) {
			// get the diff of the current limb
			T x = p[i] ^ p_old[i];
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

#endif//CRYPTANALYSISLIB_CHASE_H
