#ifndef CRYPTANALYSISLIB_COMBINATION_CHASE_H
#define CRYPTANALYSISLIB_COMBINATION_CHASE_H

#include <cstdint>
#include <vector>

#include "helper.h"
#include "math/bc.h"

/// functions/fields an enumerator must implement
/// \tparam Enumerator
template<typename Enumerator>
concept EnumeratorAble = requires(Enumerator e) {
	e.step();
};

/// \return  number of elements in the gray code
template<const uint32_t w, const uint32_t q>
constexpr static size_t compute_gray_size() noexcept {
	uint64_t sum = 1;
	for (uint64_t i = 0; i < w; i++) { sum *= q - 1; }

	// just make sure that we do not return zero.
	return std::max(size_t(sum), size_t(1ul));
}

/// needed to compute the list size before initializing the enumerator
/// \tparam n length to enumerate
/// \tparam q base field size
/// \tparam w hamming weight to enumerate: NOTE: can be zero, for Prange
/// \return list size
template<const uint32_t n, const uint32_t q, const uint32_t w>
constexpr size_t compute_combinations_fq_chase_list_size() {
	static_assert(n > w);
	static_assert(q > 1);
	size_t size = compute_gray_size<w, q>();

	// just make sure that we do not return zero.
	return std::max(size, size_t(1ul)) * bc(n, w);
}

/// \tparam T base limb type of the input const_array.
/// 			Most likely always `uint64_t`
/// \tparam n number of positions to enumerate
/// \tparam w weight to enumerate
/// \tparam start offset to start the enumeration process (in bits)
template<typename T,
         const uint32_t n,
         const uint32_t w,
         const uint32_t start = 0>
#if __cplusplus > 201709L
    requires std::is_integral_v<T>
#endif
class Combinations_Binary_Chase {
	/*
	 *  generate a sequence of all bit vectors with length n and k bits set by only changing two bits.
	 * 	idea taken from https://stackoverflow.com/questions/36451090/permutations-of-binary-number-by-swapping-two-bits-not-lexicographically
	 * 	algorithm 3
	 *
	 *  This is the data struct which is used in the BJMM/MMT algorithm.
	 */
	// data for the two functions: 'two_changes_binary_left_init', 'two_changes_binary_left_init'
	std::array<uint64_t, n> two_changes_binary_o;// offset from the left most position
	std::array<int64_t, n> two_changes_binary_d; // direction set bit is moving
	std::array<uint64_t, n> two_changes_binary_n;// length of current part of the sequence
	std::array<uint64_t, n> two_changes_binary_p;// current position of the bit in the current part
	uint64_t two_changes_binary_b = 0;           // how many permutations already processed
	bool init = true;                            // if set to true the next call to `left_step`
	                                             // will init the given pointer

	// number of bits in one limb
	constexpr static uint32_t RADIX = sizeof(T) * 8;


	/// computes a single round of the chase sequence
	/// \param b
	inline void left_round(const uint64_t b) noexcept {
		ASSERT(b < two_changes_binary_o.size());

		two_changes_binary_o[b] = two_changes_binary_o[b - 1] + two_changes_binary_d[b - 1] *
		                                                                (two_changes_binary_p[b - 1] % 2 ? two_changes_binary_n[b - 1] - 1 : two_changes_binary_p[b - 1] + 1);
		two_changes_binary_d[b] = two_changes_binary_d[b - 1] * (two_changes_binary_p[b - 1] % 2 ? -1 : 1);
		two_changes_binary_n[b] = two_changes_binary_n[b - 1] - two_changes_binary_p[b - 1] - 1;
		two_changes_binary_p[b] = 0;
	}

	/// \return
	template<const bool write = true>
	inline uint64_t left_write(T *A, const uint32_t b, const int bit) noexcept {
		ASSERT(b < two_changes_binary_o.size());
		uint64_t ret = start + two_changes_binary_o[b] + two_changes_binary_p[b] * two_changes_binary_d[b];
		if constexpr (write) { WRITE_BIT(A, ret, bit); }
		return ret;
	}

public:
	/// max length of the sequence
	constexpr static size_t chase_size = bc(n, w);

	// we need these little helpers, because M4RI does not implement any row access functions, only ones for matrices.
	constexpr static inline void WRITE_BIT(T *v, 
			const size_t i, const uint64_t b) noexcept {
		v[i / RADIX] = ((v[i / RADIX] & ~(1ull << (i % RADIX))) | (T(b) << (i % RADIX)));
	}

	///
	constexpr Combinations_Binary_Chase() noexcept {
		static_assert(n > w);
		reset();
	};

	/// resets the
	void reset() noexcept {
		two_changes_binary_o.fill(0);
		two_changes_binary_d.fill(0);
		two_changes_binary_n.fill(0);
		two_changes_binary_p.fill(0);

		two_changes_binary_b = 0;

		two_changes_binary_d[0] = 1;
		two_changes_binary_n[0] = n;

		init = true;
	}

	/// computes one step in the cache sequence
	/// \param A  input const_array = limbs needed to represent 'n' bits.
	/// \param init if True: if the first element in the cache sequence is going to be generated,
	/// 			  False: else
	/// \return false if the end of the sequence is reached
	///         true if the end of the sequence is NOT reached
	template<bool write = true>
	bool left_step(T *A, uint16_t *pos1, uint16_t *pos2) noexcept {
		uint16_t pos = 0;
		if (!init) {// cleanup of the previous round
			do {
				pos = left_write<write>(A, two_changes_binary_b, 0);
			} while (++two_changes_binary_p[two_changes_binary_b] > (two_changes_binary_n[two_changes_binary_b] + two_changes_binary_b - w) && two_changes_binary_b--);
		}

		init = false;
		*pos1 = pos;
		if (two_changes_binary_p[0] > n - w) {
			return false;
		}

		// this is the bit which will be set to one.
		pos = left_write<write>(A, two_changes_binary_b, 1);
		while (++two_changes_binary_b < w) {
			left_round(two_changes_binary_b);
			left_write<write>(A, two_changes_binary_b, 1);
			pos += two_changes_binary_d[two_changes_binary_b];
			if (pos == *pos1)
				pos += -two_changes_binary_d[two_changes_binary_b];
		}

		*pos2 = pos;
		if (two_changes_binary_p[0] > n - w) {
			return false;
		}

		two_changes_binary_b = w - 1ul;
		return true;
	}

	/// \param ret input/output const_array containing ``
	/// \return nothing
	template<bool write = true>
	constexpr void changelist(std::pair<uint16_t, uint16_t> *ret, const size_t listsize = 0) {
		const size_t size = listsize == 0 ? chase_size : listsize;

		left_step<write>(NULL, &ret[0].first, &ret[0].second);
		for (uint32_t i = 0; i < size; ++i) {
			bool c = left_step<write>(nullptr, &ret[i].first, &ret[i].second);
			ASSERT(c == (i != size - 1u));
			ASSERT(ret[i].first < n);
			ASSERT(ret[i].second < n);
		}
	}

	/// NOTE: old function, only for testing.
	/// NOTE: normally you shouldnt use this function.
	/// This functions simply xors together two given rows `p` and `p_old` and finds the two positions where they differ
	/// \param p newly generated chase element. Output from `left_step`
	/// \param p_old  last generated element from the chase sequence.
	/// \param limbs Number of limbs of type T needed to represent a element of the chase sequence
	/// \param pos1 output: first bit position where 'p` and `p_old` differs
	/// \param pos2 output: second bit position
	static void __diff(const T *p,
	                   const T *p_old,
	                   const uint32_t limbs,
	                   uint16_t *pos1,
	                   uint16_t *pos2) noexcept {
		uint8_t sols = 0;                   // solution counter. Should be at most 2 if Chase generation is used.
		uint32_t sol;                       // bit position of the change
		uint16_t *sol_ptr[2] = {pos1, pos2};// easy access to the solution const_array

		for (uint32_t i = 0; i < limbs; ++i) {
			// get the diff of the current limb
			T x = p[i] ^ p_old[i];
			// loop as long we found ones in the limb. (Maybe we have two ones in on limb)
			while (x != 0 && sols < 2) {
				sol = ffsll(x) - 1;
				// clear the bit
				x ^= (uint64_t(1) << sol);

				// now check if the bit was already set in p_old. If so we know the new zero pos (pos1).
				// if not we know that the bit was set in p. So we know pos2. Na we ignore this now. Doesnt matter
				// where the zero or the one is.
				const uint64_t pos = i * RADIX + sol;
				*(sol_ptr[sols]) = pos;
				sols += 1;
			}

			// early exit.
			if (sols == 2) {
				break;
			}
		}
	}
};

/// This class enumerates a normal chase sequence of weight w. This means
/// all elements are enumerated which only differ in at most two positions.
/// On top of this sequence, a grey code will enumerated to enumerated over
/// all elementes in Fq
/// \tparam n length to enumerate
/// \tparam q base field size
/// \tparam w hamming weight to enumerate
template<uint32_t n, const uint32_t q, const uint32_t w>
class Combinations_Fq_Chase {
	/// max value to enumerate
	constexpr static uint32_t qm1 = q - 1;

	/// NOTE: this is the output container for `mixed_radix_grey`
	std::array<uint32_t, n> a;

	/// stuff for the mixed radix enumeration
	std::array<uint32_t, n + 1> f;
	std::array<uint32_t, n> s;// sentinel

	/// stuff for the chase sequence
	Combinations_Binary_Chase<uint64_t, n, w> chase;

public:
	constexpr static size_t chase_size = bc(n, w);

	// the -1u is needed, because we already enumerated the 1...10...0 vector
	// which is the first vector
	constexpr static size_t gray_size = compute_gray_size<w, q>() - 1u;

	// So here things become tricky. As we substracted one, we need to add one
	constexpr static size_t LIST_SIZE = chase_size * (gray_size + 1u);

	/// NOTE: this enumerates on a length w NOT on length n
	/// NOTE: you have to free this stuff yourself
	/// NOTE: the input must be of size sum_bc(w, w)*q**w
	/// NOTE: returns 0 on success
	/// NOTE: only enumerates to q-1
	constexpr int changelist_mixed_radix_grey(uint16_t *ret) noexcept {
		uint32_t j;
		size_t ctr = 0;

		while (ctr < gray_size) {
			j = f[0];
			f[0] = 0;
			ret[ctr++] = j;
			if (j == w) {
				return 0;
			}

			a[j] = (a[j] + 1) % (qm1);

			if (a[j] == s[j]) {
				s[j] = (s[j] - 1 + qm1) % qm1;
				f[j] = f[j + 1];
				f[j + 1] = j + 1;
			}
		}

		return 0;
	}

	/// \param ret input/output const_array containing ``
	/// \return nothing
	template<bool write = true>
	constexpr void changelist_chase(std::pair<uint16_t, uint16_t> *ret) {
		chase.template changelist<write>(ret);
	}

	///
	/// \param n length
	/// \param q field size
	/// \param w max hamming weight to enumerate
	constexpr Combinations_Fq_Chase() noexcept {
		/// init the restricted gray code
		for (uint32_t i = 0; i < n; ++i) {
			a[i] = 0;
			f[i] = i;
			s[i] = qm1 - 1;
		}
		f[n] = n;
	}
};

template<const uint32_t n, const uint32_t p>
void next3(uint32_t *c1, uint32_t *c2) {
	static_assert(n > p);
	static_assert(p > 0);

	static bool jumps[p] = {false};
	// last_pos[0] = slow ctr
	// last_pos[1] = middle ctr
	static uint32_t last_pos[p] = {0};
	static uint32_t cp = 1;

	/// jump the middle/fast ctr back to the slow
	for (uint32_t i = 0; i < p; i++) {
		if (jumps[i]) {
			*c1 = n - 1u;
			*c2 = last_pos[cp] + 1u;
			jumps[i] = false;
			return;
		}
	}

	/// step with slow/middle ctr
	for (int32_t i = (p - 1); i >= 0; i--) {
		if (*c2 == (n - 1u)) {
			*c1 = last_pos[cp];
			*c2 = last_pos[cp] + 1u;
			last_pos[cp] += 1;

			jumps[cp] = true;
			cp -= 1u * (last_pos[cp] == (n - 1u - cp));
			return;
		}
	}


	*c1 = *c2;
	*c2 += 1u;
	return;
}

// only generated the change list
//https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGIM9KuADJ4DJgAcj4ARpjEIADMAOykAA6oCoRODB7evv7SaRmOAiFhkSwxcUm2mPbFDEIETMQEOT5%2BATV1WY3NBKUR0bEJyQpNLW15nWN9A%2BWVIwCUtqhexMjsHObxocjeWADUJvFuyGP4gsfYJhoAgje3BJgsKQZPx6cCYwdeoQTxZgA%2BgQDgxSAc0Axvr9BADgQcUlcHn9QZhVAQzBAYf8gSCAFTILjg7Fw/HIMyLI6JKx3G4AThmjmQByiqE8B20PhSR3iABEDlQxEpjjT7hoGU0mT8/qSDgYxoDCjz%2BRoRQ91eKAPTajlcg4EBCYAVMb4OYgspjIADW%2BtQ%2BsNBwUtFQAHcNXS8FQDhBOS9KSZqe76QSuMrQQcALQHLheNW08XB8lh%2BUERXpI6WaOx%2BKi%2Bn033c478wW0YU5oPi4iYAhrBjRuNiukB3kaxvazWOp7cl2EBD2o1O10QgjEd2e70EszKovehgRmOLf2B%2BNN8Uh5Mm1OFBt5tdJmcptMKDMWLM7hPiw9K6wzrjnxv5vUzkdeTDn1d0qs14h1u/llfNq29LtgcaKEBCAj4PU7rauOEDrkWM4QHOC4HEuuZapq9JfrWByqv%2BbZYYkLbxo2CF8gck7vpOJ63tmGGftWuF/rmxGtiiTxjBAkLfCiUQGOh7o8U8qgpOaKIMHWt4aPhDHCWiYkHCihYUWY94PAcmlKYIw7mjOsnuiSuIIqG%2BngikU63u%2B6B2gGDH0mE6JqScknmVcEDmAAbCkRIZt5FINlpQW6bRFEsUGxEHC6CB0Ea3EjjybgzvxTCLIFWk4T%2BJkngizmsbyHDLLQnAAKy8H4HBaKQqCcElljWI6qzrEa2w8KQBCaIVyxWiAJUaPonCSOVnXVZwvAKCA/UdZVhWkHAsBIGgLyxWQFDcagy30HEwBcACfB0E8xATRAUQjVEoTNAAnpwbXncwxCXQA8lE2iYA4N28EtbCCI9DC0NdM2kFgUReMAbhiKWH1A88hjAOIgP4FWDh4AAbpgE2A2ib1eE8UN/LUI20HgUTEFdHhYCNI54CwH3LFQBjAAoABqeCYC6j0pIwUP8IIIhiOwUgyIIigqOogO6ESBhGCgN6WPoxMTZAyyoCk9QYxGj3xJGLBMCjqja7rqhTrwqBo8QxB4FgisQMsdhvfULgMO4njtHowShIMFTDEShSZAIkx%2BD76R%2BwwcxDHERJ28jAi9BMLt5JHtT2z04z9B78ze7YqcB3oMwtGHXsR7bTUbBIRWlcNgM1RwByqAAHJ5EaeZIBzAMgzK7QAdFOEC4IQJAZvEXCLLw01aIupA9X1A0cENpAVVV1fjZN7WdRPxUcGYleL2Nq8zRPZsZM4khAA
template<const uint32_t n, const uint32_t p>
void next2(uint32_t *c1, uint32_t *c2) {
	static bool jump = false;
	static uint32_t last_pos = 0;

	/// jump the fast ctr back to the slow
	if (jump) {
		*c1 = n - 1u;
		*c2 = last_pos + 1u;
		jump = false;
		return;
	}

	/// step with the slow ctr
	if (*c2 == (n - 1u)) {
		*c1 = last_pos;
		*c2 = last_pos + 1u;
		last_pos += 1;

		jump = true;
		return;
	}


	*c1 = *c2;
	*c2 += 1u;
	return;
}

///
template<const uint32_t n, const uint32_t p>
void next1(uint32_t *c1, uint32_t *c2) {
	*c1 += 1;
	*c2 += 1;
}

///
template<const uint32_t n, const uint32_t p>
void next_chase(uint32_t *c1, uint32_t *c2) {
	static_assert(p > 0);
	static_assert(n > p);
	if constexpr (p == 1) {
		next1<n, p>(c1, c2);
	} else if constexpr (p == 2) {
		next2<n, p>(c1, c2);
	} else if constexpr (p == 3) {
		next3<n, p>(c1, c2);
	}
}

#endif//CRYPTANALYSISLIB_CHASE_H
