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

#ifdef USE_AVX2
#import "popcount/avx2.h"

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
	alignas(64) Element *L1 = nullptr, *L2 = nullptr;

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

			if (wt < d)
				return;
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
		ASSERT(wt = d);
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
		solution_l = fastrandombytes_uint64() % LIST_SIZE;
		solution_r = fastrandombytes_uint64() % LIST_SIZE;
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
			for (size_t j = s2; j < s2+e2; ++j) {
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
		using TT = uint32_t; // NOTE do not change.
		using Element2 = T[3];
		Element2 *LL1 = (Element2 *)L1;
		Element2 *LL2 = (Element2 *)L2;
		for (size_t i = s1; i < e1; i++) {
			for (size_t j = s2; j < e2; j++) {
				if constexpr (EXACT) {
					const uint32_t t = (LL1[i][0] == LL2[j][0]) + (LL1[i][1] == LL2[j][1]) + (LL1[i][1] == LL2[j][1]);
					if (t == 3) {
						found_solution(i, j);
					}
				} else {
					const uint32_t t =  (__builtin_popcountll(LL1[i][0] ^ LL2[j][0]) <= d) +
										(__builtin_popcountll(LL1[i][1] ^ LL2[j][1]) <= d) +
										(__builtin_popcountll(LL1[i][2] ^ LL2[j][2]) <= d);
					if (t == 3){
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
	constexpr void run(const size_t e1=LIST_SIZE,
	                   const size_t e2=LIST_SIZE) noexcept {
		constexpr size_t P = 1;//n;//256ull*256ull*256ull*256ull;

		for (size_t i = 0; i < P*N; ++i) {
			if constexpr(32 < n and n <= 256) {
#ifdef USE_AVX2
				avx2_nn_internal<r>(e1, e2);
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

	constexpr void nn(const size_t e1=LIST_SIZE,
	                  const size_t e2=LIST_SIZE) noexcept {
		run(e1, e2);
	}
#ifdef USE_AVX2
	#include "avx2.h"
#endif
};


#endif //SMALLSECRETLWE_NN_H
