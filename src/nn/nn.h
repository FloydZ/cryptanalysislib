#ifndef SMALLSECRETLWE_NN_H
#define SMALLSECRETLWE_NN_H

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <array>
#include <cassert>
#include <iostream>

#include "random.h"
#include "helper.h"
#include "simd/simd.h"

/// unrolled bruteforce step.
/// stack: uint64_t[64]
/// a1-a8, b1-b7: __m256i
#define BRUTEFORCE256_32_8x8_STEP(stack, a1, a2, a3, a4, a5, a6, a7, a8, b1, b2, b3, b4, b5, b6, b7, b8)    \
	stack[ 0] = (uint8_t)compare_256_32(a1, b1); \
	stack[ 1] = (uint8_t)compare_256_32(a1, b2); \
	stack[ 2] = (uint8_t)compare_256_32(a1, b3); \
	stack[ 3] = (uint8_t)compare_256_32(a1, b4); \
	stack[ 4] = (uint8_t)compare_256_32(a1, b5); \
	stack[ 5] = (uint8_t)compare_256_32(a1, b6); \
	stack[ 6] = (uint8_t)compare_256_32(a1, b7); \
	stack[ 7] = (uint8_t)compare_256_32(a1, b8); \
	stack[ 8] = (uint8_t)compare_256_32(a2, b1); \
	stack[ 9] = (uint8_t)compare_256_32(a2, b2); \
	stack[10] = (uint8_t)compare_256_32(a2, b3); \
	stack[11] = (uint8_t)compare_256_32(a2, b4); \
	stack[12] = (uint8_t)compare_256_32(a2, b5); \
	stack[13] = (uint8_t)compare_256_32(a2, b6); \
	stack[14] = (uint8_t)compare_256_32(a2, b7); \
	stack[15] = (uint8_t)compare_256_32(a2, b8); \
	stack[16] = (uint8_t)compare_256_32(a3, b1); \
	stack[17] = (uint8_t)compare_256_32(a3, b2); \
	stack[18] = (uint8_t)compare_256_32(a3, b3); \
	stack[19] = (uint8_t)compare_256_32(a3, b4); \
	stack[20] = (uint8_t)compare_256_32(a3, b5); \
	stack[21] = (uint8_t)compare_256_32(a3, b6); \
	stack[22] = (uint8_t)compare_256_32(a3, b7); \
	stack[23] = (uint8_t)compare_256_32(a3, b8); \
	stack[24] = (uint8_t)compare_256_32(a4, b1); \
	stack[25] = (uint8_t)compare_256_32(a4, b2); \
	stack[26] = (uint8_t)compare_256_32(a4, b3); \
	stack[27] = (uint8_t)compare_256_32(a4, b4); \
	stack[28] = (uint8_t)compare_256_32(a4, b5); \
	stack[29] = (uint8_t)compare_256_32(a4, b6); \
	stack[30] = (uint8_t)compare_256_32(a4, b7); \
	stack[31] = (uint8_t)compare_256_32(a4, b8); \
	stack[32] = (uint8_t)compare_256_32(a5, b1); \
	stack[33] = (uint8_t)compare_256_32(a5, b2); \
	stack[34] = (uint8_t)compare_256_32(a5, b3); \
	stack[35] = (uint8_t)compare_256_32(a5, b4); \
	stack[36] = (uint8_t)compare_256_32(a5, b5); \
	stack[37] = (uint8_t)compare_256_32(a5, b6); \
	stack[38] = (uint8_t)compare_256_32(a5, b7); \
	stack[39] = (uint8_t)compare_256_32(a5, b8); \
	stack[40] = (uint8_t)compare_256_32(a6, b1); \
	stack[41] = (uint8_t)compare_256_32(a6, b2); \
	stack[42] = (uint8_t)compare_256_32(a6, b3); \
	stack[43] = (uint8_t)compare_256_32(a6, b4); \
	stack[44] = (uint8_t)compare_256_32(a6, b5); \
	stack[45] = (uint8_t)compare_256_32(a6, b6); \
	stack[46] = (uint8_t)compare_256_32(a6, b7); \
	stack[47] = (uint8_t)compare_256_32(a6, b8); \
	stack[48] = (uint8_t)compare_256_32(a7, b1); \
	stack[49] = (uint8_t)compare_256_32(a7, b2); \
	stack[50] = (uint8_t)compare_256_32(a7, b3); \
	stack[51] = (uint8_t)compare_256_32(a7, b4); \
	stack[52] = (uint8_t)compare_256_32(a7, b5); \
	stack[53] = (uint8_t)compare_256_32(a7, b6); \
	stack[54] = (uint8_t)compare_256_32(a7, b7); \
	stack[55] = (uint8_t)compare_256_32(a7, b8); \
	stack[56] = (uint8_t)compare_256_32(a8, b1); \
	stack[57] = (uint8_t)compare_256_32(a8, b2); \
	stack[58] = (uint8_t)compare_256_32(a8, b3); \
	stack[59] = (uint8_t)compare_256_32(a8, b4); \
	stack[60] = (uint8_t)compare_256_32(a8, b5); \
	stack[61] = (uint8_t)compare_256_32(a8, b6); \
	stack[62] = (uint8_t)compare_256_32(a8, b7); \
	stack[63] = (uint8_t)compare_256_32(a8, b8)


#if 0
///
#define BRUTEFORCE256_64_4x4_STEP(stack, a0, a1, a2, a3, b0, b1, b2, b3)    \
	stack[ 0] = (uint8_t)compare_256_64(a0, b0); 	\
	stack[ 1] = (uint8_t)compare_256_64(a0, b1); 	\
	stack[ 2] = (uint8_t)compare_256_64(a0, b2); 	\
	stack[ 3] = (uint8_t)compare_256_64(a0, b3); 	\
	stack[ 4] = (uint8_t)compare_256_64(a1, b0); 	\
	stack[ 5] = (uint8_t)compare_256_64(a1, b1); 	\
	stack[ 6] = (uint8_t)compare_256_64(a1, b2); 	\
	stack[ 7] = (uint8_t)compare_256_64(a1, b3); 	\
	stack[ 8] = (uint8_t)compare_256_64(a2, b0); 	\
	stack[ 9] = (uint8_t)compare_256_64(a2, b1); 	\
	stack[10] = (uint8_t)compare_256_64(a2, b2); 	\
	stack[11] = (uint8_t)compare_256_64(a2, b3); 	\
	stack[12] = (uint8_t)compare_256_64(a3, b0); 	\
	stack[13] = (uint8_t)compare_256_64(a3, b1); 	\
	stack[14] = (uint8_t)compare_256_64(a3, b2); 	\
	stack[15] = (uint8_t)compare_256_64(a3, b3); 	\
	b0 = _mm256_permute4x64_epi64(b0, 0b00111001); 	\
	b1 = _mm256_permute4x64_epi64(b1, 0b00111001); 	\
	b2 = _mm256_permute4x64_epi64(b2, 0b00111001); 	\
	b3 = _mm256_permute4x64_epi64(b3, 0b00111001); 	\
	stack[16] = (uint8_t)compare_256_64(a0, b0); 	\
	stack[17] = (uint8_t)compare_256_64(a0, b1); 	\
	stack[18] = (uint8_t)compare_256_64(a0, b2); 	\
	stack[19] = (uint8_t)compare_256_64(a0, b3); 	\
	stack[20] = (uint8_t)compare_256_64(a1, b0); 	\
	stack[21] = (uint8_t)compare_256_64(a1, b1); 	\
	stack[22] = (uint8_t)compare_256_64(a1, b2); 	\
	stack[23] = (uint8_t)compare_256_64(a1, b3); 	\
	stack[24] = (uint8_t)compare_256_64(a2, b0); 	\
	stack[25] = (uint8_t)compare_256_64(a2, b1); 	\
	stack[26] = (uint8_t)compare_256_64(a2, b2); 	\
	stack[27] = (uint8_t)compare_256_64(a2, b3); 	\
	stack[28] = (uint8_t)compare_256_64(a3, b0); 	\
	stack[29] = (uint8_t)compare_256_64(a3, b1); 	\
	stack[30] = (uint8_t)compare_256_64(a3, b2); 	\
	stack[31] = (uint8_t)compare_256_64(a3, b3); 	\
	b0 = _mm256_permute4x64_epi64(b0, 0b00111001); 	\
	b1 = _mm256_permute4x64_epi64(b1, 0b00111001); 	\
	b2 = _mm256_permute4x64_epi64(b2, 0b00111001); 	\
	b3 = _mm256_permute4x64_epi64(b3, 0b00111001); 	\
	stack[32] = (uint8_t)compare_256_64(a0, b0); 	\
	stack[33] = (uint8_t)compare_256_64(a0, b1); 	\
	stack[34] = (uint8_t)compare_256_64(a0, b2); 	\
	stack[35] = (uint8_t)compare_256_64(a0, b3); 	\
	stack[36] = (uint8_t)compare_256_64(a1, b0); 	\
	stack[37] = (uint8_t)compare_256_64(a1, b1); 	\
	stack[38] = (uint8_t)compare_256_64(a1, b2); 	\
	stack[39] = (uint8_t)compare_256_64(a1, b3); 	\
	stack[40] = (uint8_t)compare_256_64(a2, b0); 	\
	stack[41] = (uint8_t)compare_256_64(a2, b1); 	\
	stack[42] = (uint8_t)compare_256_64(a2, b2); 	\
	stack[43] = (uint8_t)compare_256_64(a2, b3); 	\
	stack[44] = (uint8_t)compare_256_64(a3, b0); 	\
	stack[45] = (uint8_t)compare_256_64(a3, b1); 	\
	stack[46] = (uint8_t)compare_256_64(a3, b2); 	\
	stack[47] = (uint8_t)compare_256_64(a3, b3); 	\
	b0 = _mm256_permute4x64_epi64(b0, 0b00111001); 	\
	b1 = _mm256_permute4x64_epi64(b1, 0b00111001); 	\
	b2 = _mm256_permute4x64_epi64(b2, 0b00111001); 	\
	b3 = _mm256_permute4x64_epi64(b3, 0b00111001); 	\
	stack[48] = (uint8_t)compare_256_64(a0, b0); 	\
	stack[49] = (uint8_t)compare_256_64(a0, b1); 	\
	stack[50] = (uint8_t)compare_256_64(a0, b2); 	\
	stack[51] = (uint8_t)compare_256_64(a0, b3); 	\
	stack[52] = (uint8_t)compare_256_64(a1, b0); 	\
	stack[53] = (uint8_t)compare_256_64(a1, b1); 	\
	stack[54] = (uint8_t)compare_256_64(a1, b2); 	\
	stack[55] = (uint8_t)compare_256_64(a1, b3); 	\
	stack[56] = (uint8_t)compare_256_64(a2, b0); 	\
	stack[57] = (uint8_t)compare_256_64(a2, b1); 	\
	stack[58] = (uint8_t)compare_256_64(a2, b2); 	\
	stack[59] = (uint8_t)compare_256_64(a2, b3); 	\
	stack[60] = (uint8_t)compare_256_64(a3, b0); 	\
	stack[61] = (uint8_t)compare_256_64(a3, b1); 	\
	stack[62] = (uint8_t)compare_256_64(a3, b2); 	\
	stack[63] = (uint8_t)compare_256_64(a3, b3);
#endif


/// configuration for the following algorithm: https://arxiv.org/abs/2102.02597
class NN_Config {
private:
	// disable the normal constructor
	NN_Config() : n(0), r(0), N(0), k(0), d(0), dk(0), dk_bruteforce_weight(0),
				  dk_bruteforce_size(0),
				  LIST_SIZE(0), epsilon(0), BRUTEFORCE_THRESHOLD(0) {}
public:
	const uint32_t 	n, 						// length of the input vectors
	r,						// number of limbs to separate n on (number of levels)
	N,  					// number of leaves per leve
	k, 						// size of each limb
	d, 						// weight difference to match on the golden solution
	dk,						// weight difference to match on each limb
	dk_bruteforce_weight,   // max. weight different to allow on each limb during the bruteforce step
	dk_bruteforce_size,     // number of bits to check `dk_bruteforce_weight` on, should be 32/64
	LIST_SIZE, 				// list size on scale
	epsilon,
			BRUTEFORCE_THRESHOLD;   // max. number of elements to be in both lists, until we switch to the bruteforce
	constexpr NN_Config(const uint32_t n,
						const uint32_t r,
						const uint32_t N,
						const uint32_t k,
						const uint32_t ls,
						const uint32_t dk,
						const uint32_t d,
						const uint32_t epsilon,
						const uint32_t bf,
						const uint32_t dk_bruteforce_weight=0,
						const uint32_t dk_bruteforce_size=0) noexcept :
			n(n), r(r), N(N), k(k), d(d), dk(dk), dk_bruteforce_weight(dk_bruteforce_weight),
			dk_bruteforce_size(dk_bruteforce_size),
			LIST_SIZE(ls), epsilon(epsilon), BRUTEFORCE_THRESHOLD(bf) {};

	///
	/// helper function, only printing the internal parameters
	void print() const noexcept {
		std::cout
				<< "n: " << n
				<< ",r: " << r
				<< ",N " << N
				<< ",k " << k
				<< ",|L|: " << LIST_SIZE
				<< ", dk: " << dk
				<< ", dk_bruteforce_size: " << dk_bruteforce_size
				<< ", dk_bruteforce_weight: " << dk_bruteforce_weight
				<< ", d: " << d
				<< ", e: " << epsilon
				<< ", bf: " << BRUTEFORCE_THRESHOLD
				<< ", k: " << n/r
				<< "\n";
	}
};


template<const NN_Config &config>
class NN {
public:
	constexpr static size_t n = config.n;
	constexpr static size_t r = config.r;
	constexpr static size_t N = config.N;
	constexpr static size_t LIST_SIZE = config.LIST_SIZE;
	constexpr static uint64_t k_ = n / r;
	constexpr static uint64_t k = config.k;
	constexpr static uint32_t dk = config.dk;
	constexpr static uint32_t d = config.d;
	constexpr static uint64_t epsilon = config.epsilon;
	constexpr static uint64_t BRUTEFORCE_THRESHHOLD = config.BRUTEFORCE_THRESHOLD;

	/// Additional  performance tuning parameters
	constexpr static uint32_t dk_bruteforce_size = config.dk_bruteforce_size;
	constexpr static uint32_t dk_bruteforce_weight = config.dk_bruteforce_weight;


	/// only exact matching in the bruteforce step, if set to `true`
	/// This means that only EXACTLY equal elements are searched for
	constexpr static bool EXACT = false;

	/// in each step of the NN search only the exact weight dk is accepted
	constexpr static bool NN_EQUAL = false;
	/// in each step of the NN search all elements <= dk are accepted
	constexpr static bool NN_LOWER = true;
	/// in each step of the NN search all elements dk-epsilon <= wt <= dk+epsilon are accepted
	constexpr static bool NN_BOUNDS = false;

	/// Base types
	using T = uint64_t;// NOTE do not change.
	constexpr static size_t T_BITSIZE = sizeof(T) * 8;
	constexpr static size_t ELEMENT_NR_LIMBS = (n + T_BITSIZE - 1) / T_BITSIZE;
	using Element = T[ELEMENT_NR_LIMBS];

	/// TODO must be passed as an argument
	/// The Probability that a element will end up in the subsequent list.
	constexpr static bool USE_REARRANGE = false;
	constexpr static double survive_prob = 0.025;
	constexpr static uint32_t BUCKET_SIZE = 1024;
	alignas(32) uint64_t LB[BUCKET_SIZE * ELEMENT_NR_LIMBS];
	alignas(32) uint64_t RB[BUCKET_SIZE * ELEMENT_NR_LIMBS];

	// instance
	alignas(64) Element *L1 = nullptr,
	                    *L2 = nullptr;

	// solution
	size_t solution_l = 0, solution_r = 0, solutions_nr = 0;
	std::vector<std::pair<size_t, size_t>> solutions;

	~NN() noexcept {
		/// probably its ok to assert some stuff here
		static_assert(k <= n);
		// TODO static_assert(dk_bruteforce_size >= dk_bruteforce_weight);

		if (L1) { free(L1); }
		if (L2) { free(L2); }
	}

	/// transposes the two lists into the two buckets
	void transpose(const size_t list_size) {
		if constexpr (!USE_REARRANGE) {
			ASSERT(false);
		}

		ASSERT(list_size <= BUCKET_SIZE);
		for (size_t i = 0; i < list_size; i++) {
			for (uint32_t j = 0; j < ELEMENT_NR_LIMBS; j++) {
				LB[i + j * list_size] = L1[i][j];
				RB[i + j * list_size] = L2[i][j];
			}
		}
	}

	/// choose a random element e1 s.t. \wt(e1) < d, and set e2 = 0
	/// \param e1 output value
	/// \param e2 output value
	static void generate_golden_element_simple(Element &e1, Element &e2) noexcept {
		constexpr T mask = n % T_BITSIZE == 0 ? T(-1) : ((1ul << n % T_BITSIZE) - 1ul);
		while (true) {
			uint32_t wt = 0;
			for (uint32_t i = 0; i < ELEMENT_NR_LIMBS - 1; i++) {
				e1[i] = fastrandombytes_uint64();
				e2[i] = 0;
				wt += __builtin_popcount(e1[i]);
			}

			e1[ELEMENT_NR_LIMBS - 1] = fastrandombytes_uint64() & mask;
			e2[ELEMENT_NR_LIMBS - 1] = 0;
			wt += __builtin_popcount(e1[ELEMENT_NR_LIMBS - 1]);

			if (wt < d) {
				return;
			}
		}
	}

	/// chooses e1 completely random and e2 a weight d vector. finally e2 = e2 ^ e1
	/// \param e1 input/output
	/// \param e2 input/output
	static void generate_golden_element(Element &e1, Element &e2) noexcept {
		constexpr T mask = n % T_BITSIZE == 0 ? T(-1) : ((1ul << n % T_BITSIZE) - 1ul);
		static_assert(n > d);
		static_assert(64 > d);

		// clear e1, e2
		for (uint32_t i = 0; i < ELEMENT_NR_LIMBS; ++i) {
			e1[i] = 0;
			e2[i] = 0;
		}

		// choose e2;
		e2[0] = (1ull << d) - 1ull;
		for (uint32_t i = 0; i < d; ++i) {
			const uint32_t pos = fastrandombytes_uint64() % (n - i - 1);

			const uint32_t from_limb = 0;
			const uint32_t from_pos = i;
			const T from_mask = 1u << from_pos;

			const uint32_t to_limb = pos / T_BITSIZE;
			const uint32_t to_pos = pos % T_BITSIZE;
			const T to_mask = 1u << to_pos;

			const T from_read = (e2[from_limb] & from_mask) >> from_pos;
			const T to_read = (e2[to_limb] & to_mask) >> to_pos;
			e2[to_limb] ^= (-from_read ^ e2[to_limb]) & (1ul << to_pos);
			e2[from_limb] ^= (-to_read ^ e2[from_limb]) & (1ul << from_pos);
		}

		uint32_t wt = 0;
		for (uint32_t i = 0; i < ELEMENT_NR_LIMBS - 1; i++) {
			e1[i] = fastrandombytes_uint64();
			e2[i] ^= e1[i];
			wt += __builtin_popcountll(e1[i] ^ e2[i]);
		}

		e1[ELEMENT_NR_LIMBS - 1] = fastrandombytes_uint64() & mask;
		e2[ELEMENT_NR_LIMBS - 1] ^= e1[ELEMENT_NR_LIMBS - 1];
		wt += __builtin_popcountll(e1[ELEMENT_NR_LIMBS - 1] ^ e2[ELEMENT_NR_LIMBS - 1]);
		ASSERT(wt == d);
	}

	/// simply chooses an uniform random element
	/// \param e
	static void generate_random_element(Element &e) noexcept {
		constexpr T mask = n % T_BITSIZE == 0 ? T(-1) : ((1ul << n % T_BITSIZE) - 1ul);
		for (uint32_t i = 0; i < ELEMENT_NR_LIMBS - 1; i++) {
			e[i] = fastrandombytes_uint64();
		}

		e[ELEMENT_NR_LIMBS - 1] = fastrandombytes_uint64() & mask;
	}

	/// generate a random list
	/// \param L
	static void generate_random_lists(Element *L) noexcept {
		for (size_t i = 0; i < LIST_SIZE; i++) {
			generate_random_element(L[i]);
		}
	}

	/// generate
	/// \param insert_sol
	void generate_special_instance(bool insert_sol = true, bool create_zero = true) noexcept {
		constexpr size_t list_size = (ELEMENT_NR_LIMBS * LIST_SIZE * sizeof(T));
		L1 = (Element *) aligned_alloc(64, list_size);
		L2 = (Element *) aligned_alloc(64, list_size);
		ASSERT(L1);
		ASSERT(L2);
		if (create_zero) {
			memset(L1, 0, list_size);
			memset(L2, 0, list_size);
		}
	}

	/// generate a random instance, just for testing and debugging
	/// \param insert_sol if false, no solution will inserted, this is just for quick testing/benchmarking
	void generate_random_instance(bool insert_sol = true) noexcept {
		constexpr size_t list_size = (ELEMENT_NR_LIMBS * LIST_SIZE * sizeof(T));
		L1 = (Element *) aligned_alloc(4096, list_size);
		L2 = (Element *) aligned_alloc(4096, list_size);
		ASSERT(L1);
		ASSERT(L2);

		generate_random_lists(L1);
		generate_random_lists(L2);

		/// fill the stuff with randomness, I'm to lazy for tail management
		if constexpr (USE_REARRANGE) {
			for (size_t i = 0; i < BUCKET_SIZE; ++i) {
				LB[i] = fastrandombytes_uint64();
				RB[i] = fastrandombytes_uint64();
			}
		}

		/// only insert the solution if wanted.
		if (!insert_sol)
			return;

		// generate solution:
		solution_l = 0;//fastrandombytes_uint64() % LIST_SIZE;
		solution_r = 0;//fastrandombytes_uint64() % LIST_SIZE;
		//std::cout << "sols at: " << solution_l << " " << solution_r << "\n";

		if constexpr (EXACT) {
			Element sol;
			generate_random_element(sol);

			// inject the solution
			for (uint32_t i = 0; i < ELEMENT_NR_LIMBS; ++i) {
				L1[solution_l][i] = sol[i];
				L2[solution_r][i] = sol[i];
			}
		} else {
			Element sol1, sol2;
			generate_golden_element(sol1, sol2);
			// inject the solution
			for (uint32_t i = 0; i < ELEMENT_NR_LIMBS; ++i) {
				L1[solution_l][i] = sol1[i];
				L2[solution_r][i] = sol2[i];
			}
		}
	}

	/// adds `li` and `lr` to the solutions list.
	/// an additional final check for correctness is done.
	void found_solution(const size_t li,
	                    const size_t lr) noexcept {
		ASSERT(li < LIST_SIZE);
		ASSERT(lr < LIST_SIZE);
#ifdef DEBUG
		uint32_t wt = 0;
		for (uint32_t i = 0; i < ELEMENT_NR_LIMBS; i++) {
			wt += __builtin_popcountll(L1[li][i] ^ L2[lr][i]);
		}

		ASSERT(wt <= d);
#endif

		//std::cout << solutions_nr << "\n";
		solutions.resize(solutions_nr + 1);
#ifdef ENABLE_BENCHMARK
		solutions[0] = std::pair<size_t, size_t>{li, lr};
#else
		solutions[solutions_nr++] = std::pair<size_t, size_t>{li, lr};
#endif
	}

	// checks whether all submitted solutions are correct
	bool all_solutions_correct() const noexcept {
		if (solutions_nr == 0)
			return false;

		for (uint32_t i = 0; i < solutions_nr; i++) {
			bool equal = true;
			if constexpr (EXACT) {
				for (uint32_t j = 0; j < ELEMENT_NR_LIMBS; j++) {
					equal &= L1[solutions[i].first][j] == L2[solutions[i].second][j];
				}
			} else {
				uint32_t wt = 0;
				for (uint32_t j = 0; j < ELEMENT_NR_LIMBS; j++) {
					wt += __builtin_popcountll(L1[solutions[i].first][j] ^ L2[solutions[i].second][j]);
				}

				equal = wt <= d;
			}

			if (!equal)
				return false;
		}

		return true;
	}

	/// checks whether a,b are a solution or not
	/// NOTE: upper bound `d` is inclusive
	bool compare_u32(const uint32_t a, const uint32_t b) const noexcept {
		if constexpr (EXACT) {
			return a == b;
		} else {
			return uint32_t(__builtin_popcount(a ^ b)) <= d;
		}
	}

	/// checks whether a,b are a solution or not
	/// NOTE: upper bound `d` is inclusive
	/// \param a first value
	/// \param b second value
	/// \return
	bool compare_u64(const uint64_t a, const uint64_t b) const noexcept {
		if constexpr (EXACT) {
			return a == b;
		} else {
			return uint32_t(__builtin_popcountll(a ^ b)) <= d;
		}
	}

	/// compares the limbs from the given pointer on.
	/// \param a first pointer
	/// \param b second pointer
	/// \return true if the limbs following the pointer are within distance `d`
	///			false else
	template<const uint32_t s = 0>
	bool compare_u64_ptr(const uint64_t *a, const uint64_t *b) const noexcept {
		if constexpr (!USE_REARRANGE) {
			ASSERT((T) a < (T) (L1 + LIST_SIZE));
			ASSERT((T) b < (T) (L2 + LIST_SIZE));
		}
		if constexpr (EXACT) {
			for (uint32_t i = s; i < ELEMENT_NR_LIMBS; i++) {
				if (a[i] != b[i])
					return false;
			}

			return true;
		} else {
			constexpr T mask = n % T_BITSIZE == 0 ? 0 : ~((1ul << n % T_BITSIZE) - 1ul);
			uint32_t wt = 0;
			for (uint32_t i = s; i < ELEMENT_NR_LIMBS; i++) {
				wt += __builtin_popcountll(a[i] ^ b[i]);
			}

			ASSERT(!(a[ELEMENT_NR_LIMBS - 1] & mask));
			ASSERT(!(b[ELEMENT_NR_LIMBS - 1] & mask));

			// TODO maybe some flag?
			return wt <= d;
			//return wt == d;
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: without avx2
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_32(const size_t e1,
	                   const size_t e2) noexcept {
		ASSERT(n <= 32);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// limb position to compare on, basically the column to compare on.
		constexpr uint32_t limb_pos = 0;

		for (size_t i = s1; i < e1; ++i) {
			for (size_t j = s2; j < s2 + e2; ++j) {
				if (compare_u32(L1[i][limb_pos], L2[j][limb_pos])) {
					found_solution(i, j);
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: without avx2
	/// \param s1 start index of list 1
	/// \param e1 end index of list 1
	/// \param s2 start index list 2
	/// \param e2 end index list 2
	void bruteforce_64(const size_t e1,
	                   const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// limb position to compare on, basically the column to compare on.
		constexpr uint32_t limb_pos = 0;

		for (size_t i = s1; i < e1; ++i) {
			for (size_t j = s2; j < e2; ++j) {
				if constexpr (EXACT) {
					if (L1[i][limb_pos] == L2[j][limb_pos]) {
						found_solution(i, j);
					}
				} else {
					if (__builtin_popcountll(L1[i][limb_pos] ^ L2[j][limb_pos]) <= d) {
						found_solution(i, j);
					}
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: without avx2
	/// NOTE: internally the comparison is done on 32 bits
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_96(const size_t e1,
	                   const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		// we need 32bit limbs
		using TT = uint32_t;// NOTE do not change.
		using Element2 = T[3];
		Element2 *LL1 = (Element2 *) L1;
		Element2 *LL2 = (Element2 *) L2;
		for (size_t i = s1; i < e1; i++) {
			for (size_t j = s2; j < e2; j++) {
				if constexpr (EXACT) {
					const uint32_t t = (LL1[i][0] == LL2[j][0]) + (LL1[i][1] == LL2[j][1]) + (LL1[i][1] == LL2[j][1]);
					if (t == 3) {
						found_solution(i, j);
					}
				} else {
					const uint32_t t = (__builtin_popcountll(LL1[i][0] ^ LL2[j][0]) <= d) +
					                   (__builtin_popcountll(LL1[i][1] ^ LL2[j][1]) <= d) +
					                   (__builtin_popcountll(LL1[i][2] ^ LL2[j][2]) <= d);
					if (t == 3) {
						found_solution(i, j);
					}
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: without avx2
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_128(const size_t e1,
	                    const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		for (size_t i = s1; i < e1; i++) {
			for (size_t j = s2; j < e2; j++) {
				if (compare_u64_ptr(L1[i], L2[j])) {
					found_solution(i, j);
				}
			}
		}
	}


	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: without avx2
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_256(const size_t e1,
	                    const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		for (size_t i = s1; i < e1; i++) {
			for (size_t j = s2; j < e2; j++) {
				if (compare_u64_ptr(L1[i], L2[j])) {
					found_solution(i, j);
				}
			}
		}
	}

	///
	/// \tparam limb
	/// \param e1
	/// \param z
	/// \param L
	/// \return
	template<const uint32_t limb>
	size_t sort_nn_on32(const size_t e1,
	                    const uint64_t z,
	                    Element *__restrict__ L) const noexcept {
		//TODO
		return 0;
	}

	///
	/// \tparam limb
	/// \param e1
	/// \param z
	/// \param L
	/// \return
	template<const uint32_t limb>
	size_t sort_nn_on64(const size_t e1,
	                    const uint64_t z,
	                    Element *__restrict__ L) const noexcept {
		//TODO
		return 0;
	}

	/// runs the Esser, Kübler, Zweydinger NN on a the two lists
	/// dont call ths function normally.
	/// \tparam level current level of the
	/// \param e1 end of list L1
	/// \param e2 end of list L2
	template<const uint32_t level>
	void nn_internal(const size_t e1,
	                 const size_t e2) noexcept {
		// TODO
	}

	/// core entry function for the implementation of the Esser, Kuebler, Zweydinger NN algorithm
	/// \param e1 size of the left list
	/// \param e2 size of the right list
	constexpr void run(const size_t e1 = LIST_SIZE,
	                   const size_t e2 = LIST_SIZE) noexcept {
		constexpr size_t P = 1;//n;//256ull*256ull*256ull*256ull;

		for (size_t i = 0; i < P * N; ++i) {
			if constexpr (32 < n and n <= 256) {
#ifdef USE_AVX2
				//TODO avx2_nn_internal<r>(e1, e2);
#else
				nn_internal<r>(e1, e2);
#endif
			} else {
				ASSERT(false);
			}
			if (solutions_nr > 0) {
				break;
			}
		}
	}

	///
	/// \param e1
	/// \param e2
	/// \return
	constexpr void nn(const size_t e1 = LIST_SIZE,
	                  const size_t e2 = LIST_SIZE) noexcept {
		run(e1, e2);
	}

	///////////////////////////// SIMD \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\


	/// \tparam exact if set to true: a simple equality check is done
	/// \param in1 first input
	/// \param in2 second input
	/// \return compresses equality check:
	///			[bit0 = in1.v32[0] == in2.v32[0],
	/// 						....,
	/// 		 bit7 = in1.v32[7] == in2.v32[7]]
	template<const bool exact = false>
	int compare_256_32(const uint32x8_t in1,
	                   const uint32x8_t in2) const noexcept {
		if constexpr (exact) {
			return uint32x8_t::cmp(in1, in2);
		}

		const uint32x8_t pop = uint32x8_t::popcnt(in1 ^ in2);

		if constexpr (dk_bruteforce_weight > 0) {
			if constexpr (EXACT) {
				const uint32x8_t weight = uint32x8_t::set1(dk_bruteforce_weight);
				return weight == pop;
			} else {
				const uint32x8_t weight = uint32x8_t::set1(dk_bruteforce_weight + 1);
				return weight > pop;
			}

			// just to make sure that the compiler will not compiler the
			// following code
			return 0;
		}

		if constexpr (EXACT) {
			const uint32x8_t weight = uint32x8_t::set1(d);
			return weight == pop;
		} else {
			const uint32x8_t weight = uint32x8_t::set1(d + 1);
			return weight > pop;
		}
	}

	/// \tparam exact if set to true: a simple equality check is done
	/// \param in1 first input
	/// \param in2 second input
	/// \return compresses equality check:
	///			[bit0 = in1.v32[0] == in2.v32[0],
	/// 						....,
	/// 		 bit7 = in1.v32[7] == in2.v32[7]]
	template<const bool exact = false>
	int compare_256_64(const uint64x4_t in1,
	                   const uint64x4_t in2) const noexcept {
		if constexpr (exact) {
			return uint64x4_t::cmp(in1, in2);
		}

		const uint64x4_t pop = uint64x4_t::popcnt(in1 ^ in2);

		if constexpr (dk_bruteforce_weight > 0) {
			if constexpr (EXACT) {
				constexpr uint64x4_t weight = uint64x4_t::set1(dk_bruteforce_weight);
				return weight == pop;
			} else {
				constexpr uint64x4_t weight = uint64x4_t::set1(dk_bruteforce_weight + 1);
				return weight > pop;
			}

			// just to make sure that the compiler will not compiler the
			// following code
			return 0;
		}

		if constexpr (EXACT) {
			constexpr uint64x4_t weight = uint64x4_t::set1(d);
			return weight == pop;
		} else {
			constexpr uint64x4_t weight = uint64x4_t::set1(d + 1);
			return weight > pop;
		}
	}


	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: only compares a single 32 bit column of the list. But its
	///			still possible
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 23...43) is impossible.
	/// NOTE: uses avx2
	/// NOTE: only a single 32bit element is compared.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_simd_32(const size_t e1,
							const size_t e2) noexcept {
		ASSERT(n <= 32);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);
		ASSERT(d < 16);

		/// difference of the memory location in the right list
		const uint32x8_t loadr = uint32x8_t::setr(0, 1, 2, 3, 4, 5, 6, 7);

		for (size_t i = s1; i < e1; ++i) {
			// NOTE: implicit typecast because T = uint64
			const uint32x8_t li = uint32x8_t::set1(L1[i][0]);

			/// NOTE: only possible because L2 is a continuous memory block
			const uint32_t *ptr_r = (uint32_t *)L2;

			for (size_t j = s2; j < s2+(e2+7)/8; ++j, ptr_r += 16) {
				const uint32x8_t ri = uint32x8_t::template gather<8>((const int *)ptr_r, loadr);
				const int m = compare_256_32(li, ri);

				if (m) {
					const size_t jprime = j*8 + __builtin_ctz(m);
					if (compare_u64_ptr((T *)(L1 + i), (T *)(L2 + jprime))) {
						//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
						found_solution(i, jprime);
					}
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only compares a single 64 bit column of the list. But its
	///			still possible
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_simd_64(const size_t e1,
							const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(n <= 64);
		ASSERT(n > 32);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// strive is the number of bytes to the next element
		constexpr uint32_t stride = 8;

		for (size_t i = s1; i < e1; ++i) {
			const uint64x4_t li = uint64x4_t::set1(L1[i][0]);

			/// NOTE: only possible because L2 is a continuous memory block
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 4) {
				const uint64x4_t ri = uint64x4_t::load(ptr_r);
				const int m = compare_256_64(li, ri);

				if (m) {
					const size_t jprime = j*4+__builtin_ctz(m);

					if (compare_u64_ptr((T *)(L1 + i), (T *)(L2 + jprime))) {
						//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
						found_solution(i, jprime);
					}
				}
			}
		}
	}

	/// NOTE: in comparison to the other version `bruteforce_avx2_64` this implementation
	///			assumes that the elements to compare are fully compared on all n variables
	///  		e.g. ELEMENT_NR_LIMBS == 1
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_simd_64_1x1(const size_t e1,
								const size_t e2) noexcept {
		ASSERT(ELEMENT_NR_LIMBS == 1);
		ASSERT(n <= 64);
		ASSERT(n > 32);

		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);


		for (size_t i = s1; i < e1; ++i) {
			const uint64x4_t li = uint64x4_t::set1(L1[i][0]);

			/// NOTE: only possible because L2 is a continuous memory block
			uint64x4_t *ptr_r = (uint64x4_t *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 1) {
				const uint64x4_t ri = uint64x4_t::load(ptr_r);
				const int m = compare_256_64(li, ri);

				if (m) {
					const size_t jprime = j*4 + __builtin_ctz(m);

					if (compare_u64_ptr((T *)(L1 + i), (T *)(L2 + jprime))) {
						found_solution(i, jprime);
					}
				}
			}
		}
	}


	/// NOTE: in comparison to the other version `bruteforce_avx2_64` this implementation
	///			assumes that the elements to compare are fully compared on all n variables
	///  		e.g. ELEMENT_NR_LIMBS == 1
	/// NOTE: compared to `bruteforce_avx2_64_1x1` this unrolls `u` elementes in the left
	///			list and `v` elements on the right.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<const uint32_t u, const uint32_t v>
	void bruteforce_simd_64_uxv(const size_t e1,
								const size_t e2) noexcept {
		ASSERT(ELEMENT_NR_LIMBS == 1);
		ASSERT(n <= 64);
		ASSERT(n >= 33);

		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		uint64x4_t lii[u], rii[v];

		for (size_t i = s1; i < e1; i += u) {

			#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				lii[j] = uint64x4_t::set1(L1[i + j][0]);
			}

			/// NOTE: only possible because L2 is a continuous memory block
			uint64x4_t *ptr_r = (uint64x4_t *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; j += v, ptr_r += v) {

				#pragma unroll
				for (uint32_t s = 0; s < v; ++s) {
					rii[s] = uint64x4_t::load(ptr_r + s);
				}

				#pragma unroll
				for (uint32_t a1 = 0; a1 < u; ++a1) {
					const uint64x4_t tmp1 = lii[a1];

					#pragma unroll
					for (uint32_t a2 = 0; a2 < v; ++a2) {
						const uint64x4_t tmp2 = rii[a2];
						const int m = compare_256_64(tmp1, tmp2);

						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m);
							const size_t iprime = i + a1;

							if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
								found_solution(iprime, jprime);
							}
						} // if
					} // for v
				} // for u
			} // for right list
		} // for left list
	}


	/// NOTE: in comparison to the other version `bruteforce_avx2_64` this implementation
	///			assumes that the elements to compare are fully compared on all n variables
	///  		e.g. ELEMENT_NR_LIMBS == 1
	/// NOTE: compared to `bruteforce_avx2_64_1x1` this unrolls `u` elements in the left
	///			list and `v` elements on the right.
	/// NOTE: compared to `bruteforce_avx2_64_uxv` this function is not only comparing 1
	///			element of the left list with u elements from the right. Side
	///			Internally the loop is unrolled to compare u*4 elements to v on the right
	/// NOTE: assumes the input list to of length multiple of 16
	/// 	 	if this is not the case, there will be oob reads
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<const uint32_t u, const uint32_t v>
	void bruteforce_simd_64_uxv_shuffle(const size_t e1,
										const size_t e2) noexcept {
		ASSERT(ELEMENT_NR_LIMBS == 1);
		ASSERT(n <= 64);
		ASSERT(n >= 33);

		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		uint64x4_t lii[u], rii[v];
		auto *ptr_l = (uint64x4_t *)L1;

		for (size_t i = s1; i < s1 + (e1+3)/4; i += u, ptr_l += u) {

			#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				lii[j] = uint64x4_t::load(ptr_l + j);
			}

			/// NOTE: only possible because L2 is a continuous memory block
			auto *ptr_r = (uint64x4_t *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; j += v, ptr_r += v) {

				#pragma unroll
				for (uint32_t s = 0; s < v; ++s) {
					rii[s] = uint64x4_t::load(ptr_r + s);
				}

				#pragma unroll
				for (uint32_t a1 = 0; a1 < u; ++a1) {
					const uint64x4_t tmp1 = lii[a1];

					#pragma unroll
					for (uint32_t a2 = 0; a2 < v; ++a2) {
						uint64x4_t tmp2 = rii[a2];
						int m = compare_256_64(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m);
							const size_t iprime = i*4 + a1*4 + __builtin_ctz(m);
							if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}

						tmp2 = uint64x4_t::template permute<0b10010011>(tmp2);
						m = compare_256_64(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m) + 3;
							const size_t iprime = i*4 + a1*4 + __builtin_ctz(m);
							if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}

						tmp2 = uint64x4_t::template permute<0b10010011>(tmp2);
						m = compare_256_64(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m) + 2;
							const size_t iprime = i*4 + a1*4+ __builtin_ctz(m);
							if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}

						tmp2 = uint64x4_t::template permute<0b10010011>(tmp2);
						m = compare_256_64(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m) + 1;
							const size_t iprime = i*4 + a1*4 + __builtin_ctz(m);
							if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}
					}
				}
			}
		}
	}


	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_simd_128(const size_t e1,
							 const size_t e2) noexcept {

		ASSERT(n <= 128);
		ASSERT(n > 64);
		ASSERT(2 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// difference of the memory location in the right list
		const uint32x4_t loadr1 = uint32x4_t::setr(       (2ull << 32u), (4ul) | (6ull << 32u));
		const uint32x4_t loadr2 = uint32x4_t::setr(1ull | (3ull << 32u), (5ul) | (7ull << 32u));

		for (size_t i = s1; i < e1; ++i) {
			const uint64x4_t li1 = uint64x4_t::set1(L1[i][0]);
			const uint64x4_t li2 = uint64x4_t::set1(L1[i][1]);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 8) {
				const auto ri = uint64x4_t::template gather<8>((const long long int *)ptr_r, loadr1);
				const int m1 = compare_256_64(li1, ri);

				if (m1) {
					const auto ri2 = uint64x4_t::template gather<8>((const long long int *)ptr_r, loadr2);
					const int m2 = compare_256_64(li2, ri2);

					if (m2) {
						const size_t jprime = j*4;

						if (compare_u64_ptr((T *)(L1 + i), (T *)(L2 + jprime))) {
							//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
							found_solution(i, jprime + __builtin_ctz(m1));
						} // if solution
					} // if m2
				} // if m1
			} // right for loop
		} // left for loop
	}

	/// TODO explain
	/// \tparam u
	/// \tparam v
	/// \param mask
	/// \param m1
	/// \param round
	/// \param i
	/// \param j
	template<const uint32_t u, const uint32_t v>
	void bruteforce_simd_32_2_uxv_helper(uint32_t mask,
										 const uint8_t *__restrict__ m1,
										 const uint32_t round,
										 const size_t i,
										 const size_t j) noexcept {
		while (mask > 0) {
			const uint32_t ctz = __builtin_ctz(mask);

			const uint32_t test_i = ctz / v;
			const uint32_t test_j = ctz % u;

			const uint32_t inner_i2 = __builtin_ctz(m1[ctz]);
			const uint32_t inner_j2 = inner_i2;

			const int32_t off_l = test_i*8 + inner_i2;
			const int32_t off_r = test_j*8 + ((8+inner_j2-round)%8);


			const T *test_tl = ((T *)L1) + i*2 + off_l*2;
			const T *test_tr = ((T *)L2) + j*2 + off_r*2;
			if (compare_u64_ptr<0>(test_tl, test_tr)) {
				found_solution(i + off_l, j + off_r);
			}

			mask ^= 1u << ctz;
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: checks weight d on the first 2 limbs. Then direct checking.
	/// NOTE: unrolls the left loop by u and the right by v
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<const uint32_t u, const uint32_t v>
	void bruteforce_simd_128_32_2_uxv(const size_t e1,
									  const size_t e2,
	                                  const size_t s1=0,
	                                  const size_t s2=0) noexcept {
		static_assert(u <= 8);
		static_assert(v <= 8);
		ASSERT(n <= 128);
		ASSERT(n > 64);
		ASSERT(2 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// some constants
		constexpr uint8x32_t zero = uint8x32_t::set1(0);
		constexpr uint32x8_t shuffl = uint32x8_t::setr(7, 0, 1, 2, 3, 4, 5, 6);
		constexpr uint32x8_t loadr = uint32x8_t::setr(0, 4, 8, 12, 16, 20, 24, 28);
		constexpr size_t ptr_ctr_l = u*8,
		                 ptr_ctr_r = v*8;
		constexpr size_t ptr_inner_ctr_l = 8*4,
		                 ptr_inner_ctr_r = 8*4;

		/// container for the unrolling
		uint32x8_t lii_1[u], rii_1[v], lii_2[u], rii_2[v];

		/// container for the solutions masks
		constexpr uint32_t size_m1 = std::max(u*v, 32u);
		alignas(32) uint8_t m1[size_m1] = {0}; /// NOTE: init with 0 is important
		uint32x8_t *m1_256 = (uint32x8_t *)m1;

		uint32_t *ptr_l = (uint32_t *)L1;
		for (size_t i = s1; i < s1 + e1; i += ptr_ctr_l, ptr_l += ptr_ctr_l*4) {

			#pragma unroll
			for (uint32_t s = 0; s < u; ++s) {
				lii_1[s] = uint32x8_t::template gather<4>((const int *)(ptr_l + s*ptr_inner_ctr_l + 0), loadr);
				lii_2[s] = uint32x8_t::template gather<4>((const int *)(ptr_l + s*ptr_inner_ctr_l + 1), loadr);
			}

			uint32_t *ptr_r = (uint32_t *)L2;
			for (size_t j = s2; j < s2 + e2; j += ptr_ctr_r, ptr_r += ptr_ctr_r*4) {

				// load the fi
				#pragma unroll
				for (uint32_t s = 0; s < v; ++s) {
					rii_1[s] = uint32x8_t::template gather<4>((const int *)(ptr_r + s*ptr_inner_ctr_r + 0), loadr);
					rii_2[s] = uint32x8_t::template gather<4>((const int *)(ptr_r + s*ptr_inner_ctr_r + 1), loadr);
				}

				/// Do the 8x8 shuffle
				#pragma unroll
				for (uint32_t l = 0; l < 8; ++l) {
					if (l > 0) {
						// shuffle the right side
						#pragma unroll
						for (uint32_t s2s = 0; s2s < v; ++s2s) {
							rii_1[s2s] = uint32x8_t::permute(rii_1[s2s], shuffl);
							rii_2[s2s] = uint32x8_t::permute(rii_2[s2s], shuffl);
						}
					}

					// compare the first limb
					#pragma unroll
					for (uint32_t f1 = 0; f1 < u; ++f1) {
						#pragma  unroll
						for (uint32_t f2 = 0; f2 < v; ++f2) {
							m1[f1*u + f2] = compare_256_32(lii_1[f1], rii_1[f2]);
						}
					}

					// early exit
					uint32_t mask = zero < uint8x32_t::load(m1);
					if (unlikely(mask == 0)) {
						continue;
					}

					// second limb
					#pragma unroll
					for (uint32_t f1 = 0; f1 < u; ++f1) {
						#pragma unroll
						for (uint32_t f2 = 0; f2 < v; ++f2) {
							m1[f1*u + f2] &= compare_256_32(lii_2[f1], rii_2[f2]);
						}
					}


					// early exit from the second limb computations
					mask = zero < uint8x32_t::load(m1);
					if (likely(mask == 0)) {
						continue;
					}

					// maybe write back a solution
					bruteforce_simd_32_2_uxv_helper<u,v>(mask, m1, l, i, j);
				} // 8x8 shuffle
			} // j: enumerate right side
		} // i: enumerate left side
	} // end func

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 4.
	/// NOTE: this check every element on the left against 4 on the right
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_simd_256(const size_t e1,
							 const size_t e2,
	                         const size_t s1=0,
	                         const size_t s2=0) noexcept {

		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// difference of the memory location in the right list
		constexpr uint32x4_t loadr1 = uint32x4_t::setr(       (4ull << 32u), ( 8ul) | (12ull << 32u));
		constexpr uint32x4_t loadr2 = uint32x4_t::setr(1ull | (5ull << 32u), ( 9ul) | (13ull << 32u));
		constexpr uint32x4_t loadr3 = uint32x4_t::setr(2ull | (6ull << 32u), (10ul) | (14ull << 32u));
		constexpr uint32x4_t loadr4 = uint32x4_t::setr(3ull | (7ull << 32u), (11ul) | (15ull << 32u));

		for (size_t i = s1; i < e1; ++i) {
			const uint64x4_t li1 = uint64x4_t::set1(L1[i][0]);
			const uint64x4_t li2 = uint64x4_t::set1(L1[i][1]);
			const uint64x4_t li3 = uint64x4_t::set1(L1[i][2]);
			const uint64x4_t li4 = uint64x4_t::set1(L1[i][3]);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 16) {
				const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *)ptr_r, loadr1);
				const int m1 = compare_256_64(li1, ri);

				if (m1) {
					const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *)ptr_r, loadr2);
					const int m1 = compare_256_64(li2, ri);

					if (m1) {
						const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *)ptr_r, loadr3);
						const int m1 = compare_256_64(li3, ri);

						if (m1) {
							const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *)ptr_r, loadr4);
							const int m1 = compare_256_64(li4, ri);
							if (m1) {
								const size_t jprime = j*4 + __builtin_ctz(m1);
								if (compare_u64_ptr((T *)(L1 + i), (T *)(L2 + jprime))) {
									//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
									found_solution(i, jprime);
								}
							}
						}
					}
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 4.
	/// NOTE: additional to 'bruteforce_avx2_256' this functions unrolls `u` elements of
	///		the left list.
	/// NOTE: i think this approach is stupid. It checks each limb if wt < d.
	///			This is only good to find false ones, but not correct ones
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<uint32_t u>
	void bruteforce_simd_256_ux4(const size_t e1,
								 const size_t e2) noexcept {
		static_assert(u > 0, "");
		static_assert(u <= 8, "");

		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// difference of the memory location in the right list
		constexpr uint32x4_t loadr1 = uint32x4_t::setr(       (4ull << 32u), ( 8ul) | (12ull << 32u));
		constexpr uint32x4_t loadr2 = uint32x4_t::setr(1ull | (5ull << 32u), ( 9ul) | (13ull << 32u));
		constexpr uint32x4_t loadr3 = uint32x4_t::setr(2ull | (6ull << 32u), (10ul) | (14ull << 32u));
		constexpr uint32x4_t loadr4 = uint32x4_t::setr(3ull | (7ull << 32u), (11ul) | (15ull << 32u));

		alignas(128) uint64x4_t li[u*4u];
		alignas(32) uint32_t m1s[8] = {0}; // this clearing is important

		/// allowed weight to match on
		uint32_t m1s_tmp = 0;
		const uint32_t m1s_mask = 1u << 31;

		for (size_t i = s1; i < s1 + e1; i += u) {
			/// Example u = 2
			/// li[0] = L[0][0]
			/// li[1] = L[1][0]
			/// li[2] = L[0][1]
			/// ...
			//#pragma unroll
			for (uint32_t ui = 0; ui < u; ui++) {
				li[ui + 0*u] = uint64x4_t::set1(L1[i + ui][0]);
				li[ui + 1*u] = uint64x4_t::set1(L1[i + ui][1]);
				li[ui + 2*u] = uint64x4_t::set1(L1[i + ui][2]);
				li[ui + 3*u] = uint64x4_t::set1(L1[i + ui][3]);
			}


			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 16) {
				//#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *)ptr_r, loadr1);
					const uint32_t tmp  = compare_256_64(li[0 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *)ptr_r, loadr2);
					const uint32_t tmp  = compare_256_64(li[1 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}

				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *)ptr_r, loadr3);
					const uint32_t tmp  = compare_256_64(li[2 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *)ptr_r, loadr4);
					const uint32_t tmp  = compare_256_64(li[3 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp) {
					ASSERT(__builtin_popcount(m1s_tmp) == 1);
					const uint32_t m1s_ctz = __builtin_ctz(m1s_tmp);
					const uint32_t bla = __builtin_ctz(m1s[m1s_ctz]);
					const size_t iprime = i + m1s_ctz;
					const size_t jprime = j*4 + bla;

					if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
						found_solution(iprime, jprime);
					}
				}
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 8.
	/// NOTE: additional to 'bruteforce_avx2_256' this functions unrolls `u` elements of
	///		the left list.
	///	NOTE: compared to `bruteforce_avx2_256_ux4` this function compares on 32 bit
	/// NOTE only made for extremely low weight.
	/// BUG: can only find one solution at the time.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<uint32_t u>
	void bruteforce_simd_256_32_ux8(const size_t e1,
									const size_t e2,
	                                const size_t s1=0,
	                                const size_t s2=0) noexcept {
		static_assert(u > 0, "");
		static_assert(u <= 8, "");

		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// NOTE: limit arbitrary but needed for correctness
		ASSERT(d < 7);

		/// difference of the memory location in the right list
		constexpr uint32x8_t loadr1    = uint32x8_t::setr(0u, 8u, 16u, 24u, 32u, 40u, 48u, 56u);
		constexpr uint32x8_t loadr_add = uint32x8_t::set1(1u);
		uint32x8_t loadr;

		alignas(32) uint32x8_t li[u*8u];
		alignas(32) uint32_t m1s[8] = {0}; // this clearing is important

		/// allowed weight to match on
		uint32_t m1s_tmp = 0;
		const uint32_t m1s_mask = 1u << 31;

		for (size_t i = s1; i < s1 + e1; i += u) {
			// Example u = 2
			// li[0] = L[0][0]
			// li[1] = L[1][0]
			// li[2] = L[0][1]
			// ...
			#pragma unroll
			for (uint32_t ui = 0; ui < u; ui++) {
				#pragma unroll
				for (uint32_t uii = 0; uii < 8; uii++) {
					const uint32_t tmp = ((uint32_t *)L1[i + ui])[uii];
					li[ui + uii*u] = uint32x8_t::set1(tmp);
				}
			}


			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			uint32_t *ptr_r = (uint32_t *)L2;

			for (size_t j = s2; j < s2+(e2+7)/8; ++j, ptr_r += 64) {
				loadr = loadr1;
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *)ptr_r, loadr);
					const uint32_t tmp  = compare_256_32(li[0 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *)ptr_r, loadr);
					const uint32_t tmp  = compare_256_32(li[1 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *)ptr_r, loadr);
					const uint32_t tmp  = compare_256_32(li[2 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *)ptr_r, loadr);
					const uint32_t tmp  = compare_256_32(li[3 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *)ptr_r, loadr);
					const uint32_t tmp  = compare_256_32(li[4 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *)ptr_r, loadr);
					const uint32_t tmp  = compare_256_32(li[5 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *)ptr_r, loadr);
					const uint32_t tmp  = compare_256_32(li[6 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *)ptr_r, loadr);
					const uint32_t tmp  = compare_256_32(li[7 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp) {
					// TODO limitation. Welche?
					ASSERT(__builtin_popcount(m1s_tmp) == 1);
					const uint32_t m1s_ctz = __builtin_ctz(m1s_tmp);
					const uint32_t bla = __builtin_ctz(m1s[m1s_ctz]);
					const size_t iprime = i + m1s_ctz;
					const size_t jprime = j*8 + bla;
					//std::cout << L1[iprime][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
					if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
						found_solution(iprime, jprime);
					}
				}
			}
		}
	}

	/// TODO explain
	/// \tparam off
	/// \tparam rotation
	/// \param m1sx
	/// \param m1s
	/// \param ptr_l
	/// \param ptr_r
	/// \param i
	/// \param j
	template<const uint32_t off, const uint32_t rotation>
	void bruteforce_avx2_256_32_8x8_helper(uint32_t m1sx,
										   const uint8_t *m1s,
										   const uint32_t *ptr_l,
										   const uint32_t *ptr_r,
										   const size_t i, const size_t j) noexcept {
		static_assert(rotation < 8);
		static_assert(off%32 == 0);

		while (m1sx > 0) {
			const uint32_t ctz1 = __builtin_ctz(m1sx);
			const uint32_t ctz = off + ctz1;
			const uint32_t m1sc = m1s[ctz];
			const uint32_t m1sc_ctz = __builtin_ctz(m1sc);

			const uint32_t test_j = ctz % 8;
			const uint32_t test_i = ctz / 8;

			// NOTE: the signed is important
			const int32_t off_l = test_i * 8 + m1sc_ctz;
			const int32_t off_r = test_j * 8 + (8-rotation +m1sc_ctz)%8;

			const uint64_t *test_tl = (uint64_t *) (ptr_l + off_l*8);
			const uint64_t *test_tr = (uint64_t *) (ptr_r + off_r*8);
			if (compare_u64_ptr(test_tl, test_tr)) {
				found_solution(i + off_l, j + off_r);
			}

			m1sx ^= 1u << ctz1;
		}
	}

	/// NOTE: this is hyper optimized for the case if there is only one solution with extremely low weight.
	/// \param e1 size of the list L1
	/// \param e2 size of the list L2
	void bruteforce_simd_256_32_8x8(const size_t e1,
									const size_t e2,
	                                const size_t s1=0,
									const size_t s2=0) noexcept {
		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);
		ASSERT(d < 16);

		uint32_t *ptr_l = (uint32_t *)L1;

		/// difference of the memory location in the right list
		constexpr uint32x8_t loadr1 = uint32x8_t::setr(0, 8, 16, 24, 32, 40, 48, 56);
		constexpr uint32x8_t shuffl = uint32x8_t::setr(7, 0, 1, 2, 3, 4, 5, 6);

		alignas(32) uint8_t m1s[64];

		/// helper to detect zeros
		const uint8x32_t zero = uint8x32_t::set1(0);

		for (size_t i = s1; i < s1 + e1; i += 64, ptr_l += 512) {
			const uint32x8_t l1 = uint32x8_t::template gather<4>((const int *)(ptr_l +   0), loadr1);
			const uint32x8_t l2 = uint32x8_t::template gather<4>((const int *)(ptr_l +  64), loadr1);
			const uint32x8_t l3 = uint32x8_t::template gather<4>((const int *)(ptr_l + 128), loadr1);
			const uint32x8_t l4 = uint32x8_t::template gather<4>((const int *)(ptr_l + 192), loadr1);
			const uint32x8_t l5 = uint32x8_t::template gather<4>((const int *)(ptr_l + 256), loadr1);
			const uint32x8_t l6 = uint32x8_t::template gather<4>((const int *)(ptr_l + 320), loadr1);
			const uint32x8_t l7 = uint32x8_t::template gather<4>((const int *)(ptr_l + 384), loadr1);
			const uint32x8_t l8 = uint32x8_t::template gather<4>((const int *)(ptr_l + 448), loadr1);

			uint32_t *ptr_r = (uint32_t *)L2;
			for (size_t j = s1; j < s2 + e2; j += 64, ptr_r += 512) {
				uint32x8_t r1 = uint32x8_t::template gather<4>((const int *)(ptr_r +   0), loadr1);
				uint32x8_t r2 = uint32x8_t::template gather<4>((const int *)(ptr_r +  64), loadr1);
				uint32x8_t r3 = uint32x8_t::template gather<4>((const int *)(ptr_r + 128), loadr1);
				uint32x8_t r4 = uint32x8_t::template gather<4>((const int *)(ptr_r + 192), loadr1);
				uint32x8_t r5 = uint32x8_t::template gather<4>((const int *)(ptr_r + 256), loadr1);
				uint32x8_t r6 = uint32x8_t::template gather<4>((const int *)(ptr_r + 320), loadr1);
				uint32x8_t r7 = uint32x8_t::template gather<4>((const int *)(ptr_r + 384), loadr1);
				uint32x8_t r8 = uint32x8_t::template gather<4>((const int *)(ptr_r + 448), loadr1);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				uint32_t m1s1 = zero != uint8x32_t::load(m1s +  0);
				uint32_t m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 0>(m1s2, m1s, ptr_l, ptr_r, i, j); }


				r1 = uint32x8_t::permute(r1, shuffl);
				r2 = uint32x8_t::permute(r2, shuffl);
				r3 = uint32x8_t::permute(r3, shuffl);
				r4 = uint32x8_t::permute(r4, shuffl);
				r5 = uint32x8_t::permute(r5, shuffl);
				r6 = uint32x8_t::permute(r6, shuffl);
				r7 = uint32x8_t::permute(r7, shuffl);
				r8 = uint32x8_t::permute(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = zero != uint8x32_t::load(m1s +  0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 1>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 1>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = uint32x8_t::permute(r1, shuffl);
				r2 = uint32x8_t::permute(r2, shuffl);
				r3 = uint32x8_t::permute(r3, shuffl);
				r4 = uint32x8_t::permute(r4, shuffl);
				r5 = uint32x8_t::permute(r5, shuffl);
				r6 = uint32x8_t::permute(r6, shuffl);
				r7 = uint32x8_t::permute(r7, shuffl);
				r8 = uint32x8_t::permute(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = zero != uint8x32_t::load(m1s +  0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 2>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 2>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = uint32x8_t::permute(r1, shuffl);
				r2 = uint32x8_t::permute(r2, shuffl);
				r3 = uint32x8_t::permute(r3, shuffl);
				r4 = uint32x8_t::permute(r4, shuffl);
				r5 = uint32x8_t::permute(r5, shuffl);
				r6 = uint32x8_t::permute(r6, shuffl);
				r7 = uint32x8_t::permute(r7, shuffl);
				r8 = uint32x8_t::permute(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = zero != uint8x32_t::load(m1s +  0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 3>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 3>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = uint32x8_t::permute(r1, shuffl);
				r2 = uint32x8_t::permute(r2, shuffl);
				r3 = uint32x8_t::permute(r3, shuffl);
				r4 = uint32x8_t::permute(r4, shuffl);
				r5 = uint32x8_t::permute(r5, shuffl);
				r6 = uint32x8_t::permute(r6, shuffl);
				r7 = uint32x8_t::permute(r7, shuffl);
				r8 = uint32x8_t::permute(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = zero != uint8x32_t::load(m1s +  0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 4>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 4>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = uint32x8_t::permute(r1, shuffl);
				r2 = uint32x8_t::permute(r2, shuffl);
				r3 = uint32x8_t::permute(r3, shuffl);
				r4 = uint32x8_t::permute(r4, shuffl);
				r5 = uint32x8_t::permute(r5, shuffl);
				r6 = uint32x8_t::permute(r6, shuffl);
				r7 = uint32x8_t::permute(r7, shuffl);
				r8 = uint32x8_t::permute(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = zero != uint8x32_t::load(m1s +  0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 5>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 5>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = uint32x8_t::permute(r1, shuffl);
				r2 = uint32x8_t::permute(r2, shuffl);
				r3 = uint32x8_t::permute(r3, shuffl);
				r4 = uint32x8_t::permute(r4, shuffl);
				r5 = uint32x8_t::permute(r5, shuffl);
				r6 = uint32x8_t::permute(r6, shuffl);
				r7 = uint32x8_t::permute(r7, shuffl);
				r8 = uint32x8_t::permute(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = zero != uint8x32_t::load(m1s +  0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 6>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 6>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = uint32x8_t::permute(r1, shuffl);
				r2 = uint32x8_t::permute(r2, shuffl);
				r3 = uint32x8_t::permute(r3, shuffl);
				r4 = uint32x8_t::permute(r4, shuffl);
				r5 = uint32x8_t::permute(r5, shuffl);
				r6 = uint32x8_t::permute(r6, shuffl);
				r7 = uint32x8_t::permute(r7, shuffl);
				r8 = uint32x8_t::permute(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = zero != uint8x32_t::load(m1s +  0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 7>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 7>(m1s2, m1s, ptr_l, ptr_r, i, j); }
			}
		}
	}

	/// TODO explain
	/// \tparam off
	/// \param m1sx
	/// \param m1s
	/// \param ptr_l
	/// \param ptr_r
	/// \param i
	/// \param j
	template<const uint32_t off>
	void bruteforce_avx2_256_64_4x4_helper(uint32_t m1sx,
										   const uint8_t *__restrict__ m1s,
										   const uint64_t *__restrict__ ptr_l,
										   const uint64_t *__restrict__ ptr_r,
										   const size_t i, const size_t j) noexcept {
		while (m1sx > 0) {
			const uint32_t ctz1 = __builtin_ctz(m1sx);
			const uint32_t ctz = off + ctz1;
			const uint32_t m1sc = m1s[ctz];
			const uint32_t inner_ctz = __builtin_ctz(m1sc);

			const uint32_t test_j = ctz % 4;
			const uint32_t test_inner = (inner_ctz + (ctz / 16)) % 4;
			const uint32_t test_i = (ctz % 16)/4;

			const uint32_t off_l = test_i * 4 + inner_ctz;
			const uint32_t off_r = test_j * 4 + test_inner;

			const T *test_tl = ptr_l + off_l*4;
			const T *test_tr = ptr_r + off_r*4;
			if (compare_u64_ptr(test_tl, test_tr)) {
				found_solution(i + off_l, j + off_r);
			}

			m1sx ^= 1u << ctz1;
		}
	}

	/// TODO explain
	/// \param stack
	/// \param a0
	/// \param a1
	/// \param a2
	/// \param a3
	/// \param b0
	/// \param b1
	/// \param b2
	/// \param b3
	void BRUTEFORCE256_64_4x4_STEP2(uint8_t* stack,
									uint64x4_t a0, uint64x4_t a1, uint64x4_t a2, uint64x4_t a3,
									uint64x4_t b0, uint64x4_t b1, uint64x4_t b2, uint64x4_t b3) noexcept {
		stack[0] = (uint8_t) compare_256_64(a0, b0);
		stack[1] = (uint8_t) compare_256_64(a0, b1);
		stack[2] = (uint8_t) compare_256_64(a0, b2);
		stack[3] = (uint8_t) compare_256_64(a0, b3);
		stack[4] = (uint8_t) compare_256_64(a1, b0);
		stack[5] = (uint8_t) compare_256_64(a1, b1);
		stack[6] = (uint8_t) compare_256_64(a1, b2);
		stack[7] = (uint8_t) compare_256_64(a1, b3);
		stack[8] = (uint8_t) compare_256_64(a2, b0);
		stack[9] = (uint8_t) compare_256_64(a2, b1);
		stack[10] = (uint8_t) compare_256_64(a2, b2);
		stack[11] = (uint8_t) compare_256_64(a2, b3);
		stack[12] = (uint8_t) compare_256_64(a3, b0);
		stack[13] = (uint8_t) compare_256_64(a3, b1);
		stack[14] = (uint8_t) compare_256_64(a3, b2);
		stack[15] = (uint8_t) compare_256_64(a3, b3);
		b0 = uint64x4_t::template permute<0b00111001>(b0);
		b1 = uint64x4_t::template permute<0b00111001>(b1);
		b2 = uint64x4_t::template permute<0b00111001>(b2);
		b3 = uint64x4_t::template permute<0b00111001>(b3);
		stack[16] = (uint8_t) compare_256_64(a0, b0);
		stack[17] = (uint8_t) compare_256_64(a0, b1);
		stack[18] = (uint8_t) compare_256_64(a0, b2);
		stack[19] = (uint8_t) compare_256_64(a0, b3);
		stack[20] = (uint8_t) compare_256_64(a1, b0);
		stack[21] = (uint8_t) compare_256_64(a1, b1);
		stack[22] = (uint8_t) compare_256_64(a1, b2);
		stack[23] = (uint8_t) compare_256_64(a1, b3);
		stack[24] = (uint8_t) compare_256_64(a2, b0);
		stack[25] = (uint8_t) compare_256_64(a2, b1);
		stack[26] = (uint8_t) compare_256_64(a2, b2);
		stack[27] = (uint8_t) compare_256_64(a2, b3);
		stack[28] = (uint8_t) compare_256_64(a3, b0);
		stack[29] = (uint8_t) compare_256_64(a3, b1);
		stack[30] = (uint8_t) compare_256_64(a3, b2);
		stack[31] = (uint8_t) compare_256_64(a3, b3);
		b0 = uint64x4_t::template permute<0b00111001>(b0);
		b1 = uint64x4_t::template permute<0b00111001>(b1);
		b2 = uint64x4_t::template permute<0b00111001>(b2);
		b3 = uint64x4_t::template permute<0b00111001>(b3);
		stack[32] = (uint8_t) compare_256_64(a0, b0);
		stack[33] = (uint8_t) compare_256_64(a0, b1);
		stack[34] = (uint8_t) compare_256_64(a0, b2);
		stack[35] = (uint8_t) compare_256_64(a0, b3);
		stack[36] = (uint8_t) compare_256_64(a1, b0);
		stack[37] = (uint8_t) compare_256_64(a1, b1);
		stack[38] = (uint8_t) compare_256_64(a1, b2);
		stack[39] = (uint8_t) compare_256_64(a1, b3);
		stack[40] = (uint8_t) compare_256_64(a2, b0);
		stack[41] = (uint8_t) compare_256_64(a2, b1);
		stack[42] = (uint8_t) compare_256_64(a2, b2);
		stack[43] = (uint8_t) compare_256_64(a2, b3);
		stack[44] = (uint8_t) compare_256_64(a3, b0);
		stack[45] = (uint8_t) compare_256_64(a3, b1);
		stack[46] = (uint8_t) compare_256_64(a3, b2);
		stack[47] = (uint8_t) compare_256_64(a3, b3);
		b0 = uint64x4_t::template permute<0b00111001>(b0);
		b1 = uint64x4_t::template permute<0b00111001>(b1);
		b2 = uint64x4_t::template permute<0b00111001>(b2);
		b3 = uint64x4_t::template permute<0b00111001>(b3);
		stack[48] = (uint8_t) compare_256_64(a0, b0);
		stack[49] = (uint8_t) compare_256_64(a0, b1);
		stack[50] = (uint8_t) compare_256_64(a0, b2);
		stack[51] = (uint8_t) compare_256_64(a0, b3);
		stack[52] = (uint8_t) compare_256_64(a1, b0);
		stack[53] = (uint8_t) compare_256_64(a1, b1);
		stack[54] = (uint8_t) compare_256_64(a1, b2);
		stack[55] = (uint8_t) compare_256_64(a1, b3);
		stack[56] = (uint8_t) compare_256_64(a2, b0);
		stack[57] = (uint8_t) compare_256_64(a2, b1);
		stack[58] = (uint8_t) compare_256_64(a2, b2);
		stack[59] = (uint8_t) compare_256_64(a2, b3);
		stack[60] = (uint8_t) compare_256_64(a3, b0);
		stack[61] = (uint8_t) compare_256_64(a3, b1);
		stack[62] = (uint8_t) compare_256_64(a3, b2);
		stack[63] = (uint8_t) compare_256_64(a3, b3);
	}

	/// NOTE: this is hyper optimized for the case if there is only one solution.
	/// NOTE: uses avx2
	/// NOTE: hardcoded unroll parameter of 4
	/// NOTE: only checks the first limb. If this passes the weight check all others are checked
	/// 		within `check_solution`
	/// \param e1 end index of list 1
	/// \param e2 end index of list 2
	void bruteforce_simd_256_64_4x4(const size_t e1,
									const size_t e2,
	                                const size_t s1=0,
	                                const size_t s2=0) noexcept {
		ASSERT(n <= 256);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);
		ASSERT(dk < 32);

		/// NOTE is already aligned
		T *ptr_l = (T *)L1;

		/// difference of the memory location in the right list
		const uint32x4_t loadr1 = uint32x4_t::setr(       (4ull << 32u), ( 8ul) | (12ull << 32u));
		alignas(32) uint8_t m1s[64];

		/// allowed weight to match on
		const uint8x32_t zero = uint8x32_t::set1(0);

		for (size_t i = s1; i < s1 + e1; i += 16, ptr_l += 64) {
			const uint64x4_t l1 = uint64x4_t::template gather<8>((const long long int *)(ptr_l +  0), loadr1);
			const uint64x4_t l2 = uint64x4_t::template gather<8>((const long long int *)(ptr_l + 16), loadr1);
			const uint64x4_t l3 = uint64x4_t::template gather<8>((const long long int *)(ptr_l + 32), loadr1);
			const uint64x4_t l4 = uint64x4_t::template gather<8>((const long long int *)(ptr_l + 48), loadr1);

			/// reset right list pointer
			T *ptr_r = (T *)L2;

			#pragma unroll 4
			for (size_t j = s1; j < s2 + e2; j += 16, ptr_r += 64) {
				uint64x4_t r1 = uint64x4_t::template gather<8>((const long long int *)(ptr_r +  0), loadr1);
				uint64x4_t r2 = uint64x4_t::template gather<8>((const long long int *)(ptr_r + 16), loadr1);
				uint64x4_t r3 = uint64x4_t::template gather<8>((const long long int *)(ptr_r + 32), loadr1);
				uint64x4_t r4 = uint64x4_t::template gather<8>((const long long int *)(ptr_r + 48), loadr1);

				BRUTEFORCE256_64_4x4_STEP2(m1s, l1, l2, l3, l4, r1, r2, r3, r4);
				uint32_t m1s1 = zero < uint8x32_t::load(m1s +  0);
				uint32_t m1s2 = zero < uint8x32_t::load(m1s + 32);

				if (m1s1 != 0) { bruteforce_avx2_256_64_4x4_helper< 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_64_4x4_helper<32>(m1s2, m1s, ptr_l, ptr_r, i, j); }
			}
		}
	}



	template<const uint32_t off, const uint32_t bucket_size>
	void bruteforce_avx2_256_64_4x4_rearrange_helper(uint32_t m1sx,
													 const uint8_t *__restrict__ m1s,
													 const uint64_t *__restrict__ ptr_l,
													 const uint64_t *__restrict__ ptr_r,
													 const size_t i, const size_t j) noexcept {
		while (m1sx > 0) {
			const uint32_t ctz1 = __builtin_ctz(m1sx);
			const uint32_t ctz = off + ctz1;
			const uint32_t m1sc = m1s[ctz];
			const uint32_t inner_ctz = __builtin_ctz(m1sc);

			const uint32_t test_j = ctz % 4;
			const uint32_t test_inner = (inner_ctz + (ctz / 16)) % 4;
			const uint32_t test_i = (ctz % 16)/4;

			const uint32_t off_l = test_i * 4 + inner_ctz;
			const uint32_t off_r = test_j * 4 + test_inner;
			ASSERT(off_l < 16);
			ASSERT(off_r < 16);

			uint32_t wt = 0;
			for (uint32_t s = 0; s < ELEMENT_NR_LIMBS; s++) {
				const T t1 = ptr_l[off_l + s*bucket_size];
				const T t2 = ptr_r[off_r + s*bucket_size];
				wt += __builtin_popcountll(t1 ^ t2);
			}

			ASSERT(wt);
			if (wt <= d) {
				/// TODO tell the thing it needs to get the solutions from the buckets
				solutions.resize(solutions_nr + 1);
				solutions[solutions_nr++] = std::pair<size_t, size_t>{i + off_l, j + off_r};
				//found_solution(i + off_l, j + off_r);
			}

			m1sx ^= 1u << ctz1;
		}
	}
	/// NOTE: this is hyper optimized for the case if there is only one solution.
	///
	/// \param e1 end index of list 1
	/// \param e2 end index of list 2
	template<const uint32_t bucket_size>
	void bruteforce_asm_256_64_4x4_rearrange(const size_t e1,
	                                         const size_t e2,
	                                         const size_t s1,
	                                         const size_t s2) noexcept {
		ASSERT(e1 <= bucket_size);
		ASSERT(e2 <= bucket_size);
		ASSERT(n <= 256);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);
		ASSERT(dk < 32);

		/// NOTE is already aligned
		T *ptr_l = (T *)LB;

		/// difference of the memory location in the right list
		alignas(32) uint8_t m1s[64];

		/// allowed weight to match on
		constexpr uint8x32_t zero = uint8x32_t::set1(0);

		size_t i = s1;
		#pragma unroll 2
		for (; i < s1 + e1; i += 16, ptr_l += 16) {
			const uint32x8_t l1 = uint32x8_t::template load<true>(ptr_l +  0);
			const uint32x8_t l2 = uint32x8_t::template load<true>(ptr_l +  4);
			const uint32x8_t l3 = uint32x8_t::template load<true>(ptr_l +  8);
			const uint32x8_t l4 = uint32x8_t::template load<true>(ptr_l + 12);

			/// reset right list pointer
			T *ptr_r = (T *)RB;

			#pragma unroll 4
			for (size_t j = s1; j < s2 + e2; j += 16, ptr_r += 16) {
				uint32x8_t r1 = uint32x8_t::template load<true>(ptr_r +  0);
				uint32x8_t r2 = uint32x8_t::template load<true>(ptr_r +  4);
				uint32x8_t r3 = uint32x8_t::template load<true>(ptr_r +  8);
				uint32x8_t r4 = uint32x8_t::template load<true>(ptr_r + 12);

				BRUTEFORCE256_64_4x4_STEP2(m1s, l1, l2, l3, l4, r1, r2, r3, r4);
				uint32_t m1s1 = zero < uint8x32_t::load(m1s +  0u);
				uint32_t m1s2 = zero < uint8x32_t::load(m1s + 32u);

				if (m1s1 != 0) { bruteforce_avx2_256_64_4x4_rearrange_helper< 0, bucket_size>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_64_4x4_rearrange_helper<32, bucket_size>(m1s2, m1s, ptr_l, ptr_r, i, j); }
			}
		}
	}

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 4.
	/// NOTE: in comparison to `bruteforce_avx_256` this implementation used no `gather`
	///		instruction, but rather direct (aligned) loads.
	/// NOTE: only works if exact matching is active
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_256_v2(const size_t e1,
								const size_t e2,
	                            const size_t s1,
	                            const size_t s2) noexcept {
		ASSERT(EXACT);
		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// allowed weight to match on
		const uint64x4_t zero = uint64x4_t::set1(0);
		uint64x4_t *ptr_l = (uint64x4_t *)L1;

		for (size_t i = s1; i < e1; ++i, ptr_l += 1) {
			const uint64x4_t li1 = uint64x4_t::load(ptr_l);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			uint64x4_t *ptr_r = (uint64x4_t *)L2;

			for (size_t j = s2; j < s2+e2; ++j, ptr_r += 1) {
				const uint64x4_t ri = uint64x4_t::load(ptr_r);
				const uint64x4_t tmp1 = li1 ^ ri;
				if (zero == tmp1) {
					found_solution(i, j);
				} // if solution found
			} // right list
		} // left list
	}
};


#endif //SMALLSECRETLWE_NN_H
