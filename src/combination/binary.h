#ifndef CRYPTANALYSISLIB_BINARY_H
#define CRYPTANALYSISLIB_BINARY_H

#include <cstdint>
#include <concepts>
#include <vector>

#include "helper.h"

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
		// TODO this is currently disabled, because ASSERT(k > 0);
	};


	// REMINDER: Make sure to set A on all limbs on zero.
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
#endif//CRYPTANALYSISLIB_BINARY_H
