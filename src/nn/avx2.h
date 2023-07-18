#ifndef NN_CODE_WINDOWED_AVX2_H
#define NN_CODE_WINDOWED_AVX2_H

#include <cstdint>
#include <cstdlib>
#include <vector>
#include <array>
#include <cassert>
#include <iostream>
#include <immintrin.h>

#include "random.h"
#include "helper.h"

/// TODO merge those global macros with `helper.h`
///
#define LOAD_ALIGNED
#ifdef LOAD_ALIGNED
#define LOAD256(x) _mm256_lddqu_si256(x)
#else
#define LOAD256(x) _mm256_load_si256(x)
#endif

///
#define STORE_ALIGNED
#ifdef STORE_ALIGNED
#define STORE256(ptr, x) _mm256_store_si256(ptr, x)
#else
#define STORE256(ptr, x) _mm256_storeu_si256(ptr, x)
#endif


/// TODO rename
class WindowedAVX2_Config {
private:
	// disable the normal constructor
	WindowedAVX2_Config() : n(0), r(0), N(0), k(0), d(0), dk(0), dk_bruteforce_weight(0),
							dk_bruteforce_size(0),
							LIST_SIZE(0), epsilon(0), BRUTEFORCE_THRESHOLD(0) {}
public:
	const uint32_t 	n, 						// length of the input vectors
		  			r,						// number of limbs to seperate n on (number of levels)
					N,  					// number of leaves per leve
					k, 						// size of each limb
					d, 						// weight difference to match on the golden solution
					dk,						// weight difference to match on each limb
					dk_bruteforce_weight,   // max. weight differenve to allow on each limb during the bruteforce step
					dk_bruteforce_size,     // number of bits to check `dk_bruteforce_weight` on, should be 32/64
					LIST_SIZE, 				// list size on scale
					epsilon, 
					BRUTEFORCE_THRESHOLD;   // max. number of elements to be in both lists, until we switch to the bruteforce
	constexpr WindowedAVX2_Config(const uint32_t n,
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
			<< ",N: " << N
			<< ",k: " << k
			<< ",|L|: " << LIST_SIZE
			<< ", dk: " << dk
			<< ", dk_bruteforce_size: " << dk_bruteforce_size
			<< ", dk_bruteforce_weight: " << dk_bruteforce_weight
			<< ", d: " << d
			<< ", e: " << epsilon
			<< ", bf: " << BRUTEFORCE_THRESHOLD
			<< "\n";
	}
};

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


template<const WindowedAVX2_Config &config>
class WindowedAVX2 {
public:
	constexpr static size_t n = config.n;
	constexpr static size_t r = config.r;
	constexpr static size_t N = config.N;
	constexpr static size_t LIST_SIZE = config.LIST_SIZE;
	constexpr static uint64_t k_ = n/r;
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
	using T = uint64_t; // NOTE do not change.
	constexpr static size_t T_BITSIZE = sizeof(T) * 8;
	constexpr static size_t ELEMENT_NR_LIMBS = (n + T_BITSIZE - 1) / T_BITSIZE;
	using Element = T[ELEMENT_NR_LIMBS];

	/// TODO must be passed as an argument
	constexpr static bool USE_REARRANGE = false;

	/// The Probability that a element will end up in the subsequent list.
	constexpr static double survive_prob = 0.025;
	constexpr static uint32_t BUCKET_SIZE = 1024;
	alignas(32) uint64_t LB[BUCKET_SIZE * ELEMENT_NR_LIMBS];
	alignas(32) uint64_t RB[BUCKET_SIZE * ELEMENT_NR_LIMBS];

	// if a solution found stop all further computations
	constexpr static bool EARLY_EXIT = false;
	// instance
	alignas(64) Element *L1 = nullptr, *L2 = nullptr;

	// solution
	size_t solution_l = 0, solution_r = 0, solutions_nr = 0;
	std::vector<std::pair<size_t, size_t>> solutions;

	~WindowedAVX2() noexcept {
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
				LB[i + j*list_size] = L1[i][j];
				RB[i + j*list_size] = L2[i][j];
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
			wt += __builtin_popcount(e1[ELEMENT_NR_LIMBS-1]);

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
		wt += __builtin_popcountll(e1[ELEMENT_NR_LIMBS-1] ^ e2[ELEMENT_NR_LIMBS-1]);
		ASSERT(wt = d);
	}

	/// simply chooses an uniform random element
	/// \param e
	static void generate_random_element(Element &e) noexcept {
		constexpr T mask = n%T_BITSIZE == 0 ? T(-1) : ((1ul << n%T_BITSIZE) - 1ul);
		for (uint32_t i = 0; i < ELEMENT_NR_LIMBS-1; i++) {
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
	void generate_special_instance(bool insert_sol=true, bool create_zero=true) noexcept {
		constexpr size_t list_size = (ELEMENT_NR_LIMBS * LIST_SIZE * sizeof(T));
		L1 = (Element *)aligned_alloc(64, list_size);
		L2 = (Element *)aligned_alloc(64, list_size);

		// reset number of solutions
		solutions_nr = 0;

		ASSERT(L1);
		ASSERT(L2);
		if (create_zero) {
			memset(L1, 0, list_size);
			memset(L2, 0, list_size);
		}
	}

	/// generate a random instance, just for testing and debugging
	/// \param insert_sol if false, no solution will inserted, this is just for quick testing/benchmarking
	void generate_random_instance(bool insert_sol=true) noexcept {
		constexpr size_t list_size = (ELEMENT_NR_LIMBS * LIST_SIZE * sizeof(T));
		L1 = (Element *)aligned_alloc(4096, list_size);
		L2 = (Element *)aligned_alloc(4096, list_size);
		ASSERT(L1);
		ASSERT(L2);

		// reset number of solutions
		solutions_nr = 0;

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

	/// TODO explain
	/// \param mask
	/// \return
	__m256i bit_mask_64(const uint64_t mask) {
		ASSERT(mask < (1u<<8u));

		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
		expanded_mask *= 0xFFU;
		// the identity shuffle for vpermps, packed to one index per byte
		const uint64_t identity_indices = 0x0706050403020100;
		uint64_t wanted_indices = identity_indices & expanded_mask;
		const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
		const __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);
		return shufmask;
	}

	/// returns a permutation that shuffles down a mask on 32bit limbs
	/// e.g INPUT: 0b10000001
	///				<-    256    ->
	///    OUTPUT: [  0 ,..., 7, 0]
	///		   MSB <-32->
	/// 		   <-  8 limbs   ->
	/// to apply the resulting __m256i permutation use:
	///			const uint64_t shuffle = 0b1000001;
	/// 		const __m256i permuted_data = _mm256_permutevar8x32_ps(data, shuffle);
	/// \param mask bit mask. Must be smaller than 2**8
	/// \return the permutation
	__m256i shuffle_down_32(const uint64_t mask) const noexcept {
		// make sure only sane inputs make it.
		ASSERT(mask < (1u<<8u));

		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
		// mask |= mask<<1 | mask<<2 | ... | mask<<7;
		expanded_mask *= 0xFFU;
		// ABC... -> AAAAAAAABBBBBBBBCCCCCCCC...: replicate each bit to fill its byte

		// the identity shuffle for vpermps, packed to one index per byte
		const uint64_t identity_indices = 0x0706050403020100;
		uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

		const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
		const __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);
		return shufmask;
	}

	/// returns a permutation that shuffles down a mask on 32bit limbs
	/// e.g INPUT: 0b1001
	///				<-     256   ->
	///    OUTPUT: [  0  , 0, 3, 0]
	///		   MSB <-64->
	/// 		   <-  4 limbs   ->
	/// to apply the resulting __m256i permutation use:
	///			const uint64_t shuffle = 0b1000001;
	/// 		const __m256i permuted_data = _mm256_permutevar4x64_pd(data, shuffle);
	/// \param mask bit mask. Must be smaller than 2**4
	/// \return the permutation
	__m256i shuffle_down_64(const uint64_t mask) const noexcept {
		// make sure only sane inputs make it.
		ASSERT(mask < (1u<<4u));

		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);  
		// mask |= mask<<1 | mask<<2 | ... | mask<<7;
		expanded_mask *= 0xFFU;  
		// ABC... -> AAAAAAAABBBBBBBBCCCCCCCC...: replicate each bit to fill its byte
		
		// the identity shuffle for vpermps, packed to one index per byte
		const uint64_t identity_indices = 0x0706050403020100;    
		uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

		const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
		const __m256i shufmask = _mm256_cvtepu8_epi64(bytevec);
		return shufmask;
	}

	/// pretty much the same as `shuffle_down_64` but accepts permutation mask bigger than 2**4 up to
	/// 2**8, meaning this function returns 2 permutations for at most 2 * 4  uint64_t limbs.
	/// \param higher: output parameterm, contains the higher/last 4 permutations
	/// \param lower:  output parameter, contain the lower/first 4 permutations
	/// \param mask: input parameter
	void shuffle_down_2_64(__m256i &higher, __m256i &lower, const uint64_t mask) const noexcept {
		// make sure only sane inputs make it.
		ASSERT(mask < (1u<<8u));

		/// see the description of this magic in `shuffle_down_64`
		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
		expanded_mask *= 0xFFU;
		const uint64_t identity_indices = 0x0302010003020100;
		uint64_t wanted_indices = _pext_u64(identity_indices, expanded_mask);

		const __m128i bytevec1 = _mm_cvtsi32_si128(uint16_t(wanted_indices));
		const __m128i bytevec2 = _mm_cvtsi32_si128(uint16_t(wanted_indices >> 16));
		lower = _mm256_cvtepu8_epi64(bytevec1);
		higher = _mm256_cvtepu8_epi64(bytevec2);
	}

	/// same as shuffle up, but instead a compressed array is expanded according to mask
	/// EXAMPLE: INPUT: 0b10100001
	/// 		<-      256        ->
	/// OUTPUT: [  2  , 0, 1, ..., 0]
	///			<-32->
	///			<-    8  limbs     ->
	/// USAGE:
	///			const uint64_t shuffle = 0b1000001;
	/// 		const __m256i permuted_data = _mm256_permutevar8x32_ps(data, shuffle);
	/// \param mask
	/// \return
	__m256i shuffle_up_32(const uint64_t mask) const noexcept {
		ASSERT(mask < (1u<<8u));

		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
		expanded_mask *= 0xFFU;
		const uint64_t identity_indices = 0x0706050403020100;
		uint64_t wanted_indices = _pdep_u64(identity_indices, expanded_mask);

		const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
		const __m256i shufmask = _mm256_cvtepu8_epi32(bytevec);
		return shufmask;
	}

	/// same as shuffle up, but instead a compressed array is expanded according to mask
	/// EXAMPLE: INPUT: 0b1010
	/// 		<-     256    ->
	/// OUTPUT: [  1  , 0, 0, 0]
	///			<-64->
	///			<-   4 limbs  ->
	/// USAGE:
	///			const uint64_t shuffle = 0b1000001;
	/// 		const __m256i permuted_data = _mm256_permutevar4x64_pd(data, shuffle);
	/// \param mask
	/// \return
	__m256i shuffle_up_64(const uint64_t mask) const noexcept {
		ASSERT(mask < (1u<<4u));

		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
		expanded_mask *= 0xFFU;
		const uint64_t identity_indices = 0x03020100;
		uint64_t wanted_indices = _pdep_u64(identity_indices, expanded_mask);

		const __m128i bytevec = _mm_cvtsi64_si128(wanted_indices);
		const __m256i shufmask = _mm256_cvtepu8_epi64(bytevec);
		return shufmask;
	}

	/// similar to `shuffle_up_64`, but instead it can shuffle up to 8 64bit
	///	limbs in parallel. Therefore it needs to return 2 __m256i
	/// \param mask
	void shuffle_up_2_64(__m256i &higher, __m256i &lower, const uint64_t mask) const noexcept {
		ASSERT(mask < (1u<<8u));

		uint64_t expanded_mask = _pdep_u64(mask, 0x0101010101010101);
		expanded_mask *= 0xFFU;
		const uint64_t identity_indices = 0x03020100;
		uint64_t wanted_indices1 = _pdep_u64(identity_indices, expanded_mask & ((1ul << 32u) - 1));
		uint64_t wanted_indices2 = _pdep_u64(identity_indices, expanded_mask >> 32u);

		const __m128i bytevec1 = _mm_cvtsi32_si128(wanted_indices1);
		const __m128i bytevec2 = _mm_cvtsi32_si128(wanted_indices2);
		lower = _mm256_cvtepu8_epi64(bytevec1);
		higher = _mm256_cvtepu8_epi64(bytevec2);
	}

	alignas(32) const __m256i avx_nn_k_mask = k < 32 ? _mm256_set1_epi32((uint32_t)(1ull << (k%32)) - 1ull) :
	                                          k < 64 ? _mm256_set1_epi64x((1ull << (k%64)) - 1ull) :
	                                                   _mm256_set1_epi32(0);

	/// AVX2 popcount, which computes the popcount of each of the 32bit limbs
	/// seperatly.
	__m256i popcount_avx2_32(const __m256i vec) const noexcept {
		const __m256i low_mask = _mm256_set1_epi8(0x0f);
		const __m256i lookup = _mm256_setr_epi8(
				/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
				/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
				/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
				/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,
				/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
				/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
				/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
				/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
		);
		const __m256i lo  = _mm256_and_si256(vec, low_mask);
		const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
		const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
		const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
		__m256i local = _mm256_setzero_si256();
		local = _mm256_add_epi8(local, popcnt1);
		local = _mm256_add_epi8(local, popcnt2);

		// not the best
		const __m256i mask = _mm256_set1_epi32(0xff);
		__m256i ret = _mm256_and_si256(local, mask);
		ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local,  8), mask));
		ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local, 16), mask));
		ret = _mm256_add_epi8(ret, _mm256_and_si256(_mm256_srli_epi32(local, 24), mask));
		return ret;
	}

	__m256i popcount_avx2_32_old(const __m256i vec) const noexcept {
		const __m256i low_mask = _mm256_set1_epi8(0x0f);
		const __m256i lookup = _mm256_setr_epi8(
				/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
				/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
				/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
				/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,
				/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
				/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
				/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
				/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
		);
		const __m256i lo  = _mm256_and_si256(vec, low_mask);
		const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
		const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
		const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
		__m256i local = _mm256_add_epi8(popcnt1, popcnt2);

		const __m256i mask1 = _mm256_set1_epi64x(uint32_t(-1ul));
		const __m256i mask2 = _mm256_set1_epi64x(uint64_t(uint32_t(-1ul)) << 32u);
		const __m256i zero = _mm256_setzero_si256();
		const __m256i c1 =_mm256_sad_epu8 (local&mask1, zero);
		const __m256i c2 = _mm256_slli_epi64(_mm256_sad_epu8 (local&mask2, zero), 32);
		const __m256i ret = c1 ^ c2;
		return ret;
	}

	/// special popcount which popcounts on 4 * 64 bit limbs in parallel
	__m256i popcount_avx2_64(const __m256i vec) const noexcept {
		const __m256i low_mask = _mm256_set1_epi8(0x0f);
		const __m256i lookup = _mm256_setr_epi8(
				/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
				/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
				/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
				/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,
				/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
				/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
				/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
				/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
		);
	    const __m256i lo  = _mm256_and_si256(vec, low_mask);
	    const __m256i hi  = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
	    const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
	    const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
	    const __m256i local = _mm256_add_epi8(popcnt2, popcnt1);
		const __m256i ret =_mm256_sad_epu8 (local, _mm256_setzero_si256());
		return ret;
	}

	/// first implementation of the avx popcount.
	/// DO NOT USE IT, its slower
	__m256i popcount_avx2_64_old(const __m256i vec) const noexcept {
		const __m256i low_mask = _mm256_set1_epi8(0x0f);
		const __m256i lookup = _mm256_setr_epi8(
				/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
				/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
				/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
				/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4,
				/* 0 */ 0, /* 1 */ 1, /* 2 */ 1, /* 3 */ 2,
				/* 4 */ 1, /* 5 */ 2, /* 6 */ 2, /* 7 */ 3,
				/* 8 */ 1, /* 9 */ 2, /* a */ 2, /* b */ 3,
				/* c */ 2, /* d */ 3, /* e */ 3, /* f */ 4
		);
		const __m256i lo = _mm256_and_si256(vec, low_mask);
		const __m256i hi = _mm256_and_si256(_mm256_srli_epi16(vec, 4), low_mask);
		const __m256i popcnt1 = _mm256_shuffle_epi8(lookup, lo);
		const __m256i popcnt2 = _mm256_shuffle_epi8(lookup, hi);
		__m256i local = _mm256_setzero_si256();
		local = _mm256_add_epi8(local, popcnt1);
		local = _mm256_add_epi8(local, popcnt2);

		const __m256i mask2 = _mm256_set1_epi64x(0xff);
		__m256i ret;

		ret = _mm256_add_epi8(local, _mm256_srli_epi32(local, 8));
		ret = _mm256_add_epi8(ret, _mm256_srli_epi32(ret, 16));
		ret = _mm256_add_epi8(ret, _mm256_srli_epi64(ret, 32));
		ret = _mm256_and_si256(ret, mask2);
		return ret;
	}



	/// Source:https://github.com/WojciechMula/sse-popcount/blob/master/popcnt-avx2-harley-seal.cpp
	/// Implementation of the harley-Seal algorithm
	__m256i popcount_avx2_64_old_v2(const __m256i v) const noexcept {
		const __m256i m1 = _mm256_set1_epi8(0x55);
		const __m256i m2 = _mm256_set1_epi8(0x33);
		const __m256i m4 = _mm256_set1_epi8(0x0F);
		const __m256i t1 = _mm256_sub_epi8(v,       (_mm256_srli_epi16(v,  1) & m1));
		const __m256i t2 = _mm256_add_epi8(t1 & m2, (_mm256_srli_epi16(t1, 2) & m2));
		const __m256i t3 = _mm256_add_epi8(t2, _mm256_srli_epi16(t2, 4)) & m4;
		return _mm256_sad_epu8(t3, _mm256_setzero_si256());
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
		if constexpr(EXACT) {
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
		if constexpr(EXACT) {
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
	template <const uint32_t s=0>
	bool compare_u64_ptr(const uint64_t *a, const uint64_t *b) const noexcept {
		if constexpr (!USE_REARRANGE) {
			ASSERT((T) a < (T) (L1 + LIST_SIZE));
			ASSERT((T) b < (T) (L2 + LIST_SIZE));
		}
		if constexpr(EXACT) {
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

	///
	/// NOTE: upper bound `d` is inclusive
	/// \param in
	/// \return
	template<const bool exact=false>
	int compare_256_32(const __m256i in1, const __m256i in2) const noexcept {
		if constexpr(EXACT) {
			const __m256i tmp2 = _mm256_cmpeq_epi32(in1, in2);
			return _mm256_movemask_ps((__m256) tmp2);
		}

		const __m256i tmp1 = _mm256_xor_si256(in1, in2);
#ifdef USE_AVX512
		const __m256i pop = _mm256_popcnt_epi32(tmp1);
#else
		const __m256i pop = popcount_avx2_32(tmp1);
#endif

		if constexpr(dk_bruteforce_weight > 0) {
			if constexpr (EXACT) {
				const __m256i avx_exact_weight32 = _mm256_set1_epi32(dk_bruteforce_weight);
				const __m256i tmp2 = _mm256_cmpeq_epi32(avx_exact_weight32, pop);
				return _mm256_movemask_ps((__m256) tmp2);
			} else  {
				const __m256i avx_weight32 = _mm256_set1_epi32(dk_bruteforce_weight+1);
				const __m256i tmp2 = _mm256_cmpgt_epi32(avx_weight32, pop);
				return _mm256_movemask_ps((__m256) tmp2);
			}

			// just to make sure that the compiler will not compiler the
			// following code
			return 0;
		}

		if constexpr (EXACT) {
			const __m256i avx_exact_weight32 = _mm256_set1_epi32(d);
			const __m256i tmp2 = _mm256_cmpeq_epi32(avx_exact_weight32, pop);
			return _mm256_movemask_ps((__m256) tmp2);
		} else  {
			const __m256i avx_weight32 = _mm256_set1_epi32(d+1);
			const __m256i tmp2 = _mm256_cmpgt_epi32(avx_weight32, pop);
			return _mm256_movemask_ps((__m256) tmp2);
		}
	}

	///
	/// NOTE: upper bound `d` is inclusive
	/// \param in
	/// \return
	template<bool exact=false>
	int compare_256_64(const __m256i in1, const __m256i in2) const noexcept {
		if constexpr(EXACT) {
			const __m256i tmp2 = _mm256_cmpeq_epi32(in1, in2);
			return _mm256_movemask_ps((__m256) tmp2);
		}

		const __m256i tmp1 = _mm256_xor_si256(in1, in2);
#ifdef USE_AVX512
		const __m256i pop = _mm256_popcnt_epi64(tmp1);
#else
		const __m256i pop = popcount_avx2_64(tmp1);
#endif

		/// in the special case where we want to match on a different weight to speed up
		/// the computation. This makes only sense if `dk_bruteforce_weight` < dk.
		if constexpr(dk_bruteforce_weight > 0) {
			if constexpr (EXACT) {
				const __m256i avx_exact_weight64 = _mm256_set1_epi64x(dk_bruteforce_weight);
#ifdef USE_AVX512
				return _mm256_cmp_epi64_mask(avx_exact_weight64, pop, 0);
#else
				const __m256i tmp2 = _mm256_cmpeq_epi64(avx_exact_weight64, pop);
				return _mm256_movemask_ps((__m256) tmp2);
#endif
			} else  {
				const __m256i avx_weight64 = _mm256_set1_epi64x(dk_bruteforce_weight+1);
#ifdef USE_AVX512
				return _mm256_cmp_epi64_mask(avx_weight64, pop, 6);
#else
				const __m256i tmp2 = _mm256_cmpgt_epi64(avx_weight64, pop);
				return _mm256_movemask_pd((__m256) tmp2);
#endif
			}

			// just to make sure that the compiler will not compiler the
			// following code
			return 0;
		}

		if constexpr (EXACT) {
			const __m256i avx_exact_weight64 = _mm256_set1_epi64x(d);
#ifdef USE_AVX512
			return _mm256_cmp_epi64_mask(avx_exact_weight64, pop, 0);
#else
			const __m256i tmp2 = _mm256_cmpeq_epi64(avx_exact_weight64, pop);
			return _mm256_movemask_pd((__m256d) tmp2);
#endif
		} else {
			const __m256i avx_weight64 = _mm256_set1_epi64x(d+1);
#ifdef USE_AVX512
			return _mm256_cmp_epi64_mask(avx_weight64, pop, 5);
#else
			const __m256i tmp2 = _mm256_cmpgt_epi64(avx_weight64, pop);
			return _mm256_movemask_pd((__m256d) tmp2);
#endif
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
	/// NOTE: only compares a single 32 bit column of the list. But its
	///			still possible
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 23...43) is impossible.
	/// NOTE: uses avx2
	/// NOTE: only a single 32bit element is compared.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_32(const size_t e1,
	                        const size_t e2) noexcept {
		ASSERT(n <= 32);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);
		ASSERT(d < 16);

		/// difference of the memory location in the right list
		const __m256i loadr = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

		for (size_t i = s1; i < e1; ++i) {
			// NOTE: implicit typecast because T = uint64
			const __m256i li = _mm256_set1_epi32(L1[i][0]);

			/// NOTE: only possible because L2 is a continuous memory block
			const uint32_t *ptr_r = (uint32_t *)L2;

			for (size_t j = s2; j < s2+(e2+7)/8; ++j, ptr_r += 16) {
				const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 8);
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

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only compares a single 64 bit column of the list. But its
	///			still possible
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_avx2_64(const size_t e1,
	                        const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(n <= 64);
		ASSERT(n > 32);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// strive is the number of bytes to the next element
		constexpr uint32_t stride = 8;

		/// difference of the memory location in the right list
		const __m128i loadr = {(1ull << 32u),  (2ul) | (3ull << 32u)};

		for (size_t i = s1; i < e1; ++i) {
			const __m256i li = _mm256_set1_epi64x(L1[i][0]);

			/// NOTE: only possible because L2 is a continuous memory block
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 4) {
				const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr, stride);
				const int m = compare_256_64<true>(li, ri);

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
	void bruteforce_avx2_64_1x1(const size_t e1,
	                            const size_t e2) noexcept {
		ASSERT(ELEMENT_NR_LIMBS == 1);
		ASSERT(n <= 64);
		ASSERT(n > 32);

		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);


		for (size_t i = s1; i < e1; ++i) {
			const __m256i li = _mm256_set1_epi64x(L1[i][0]);

			/// NOTE: only possible because L2 is a continuous memory block
			__m256i *ptr_r = (__m256i *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 1) {
				const __m256i ri = LOAD256(ptr_r);
				const int m = compare_256_64<true>(li, ri);

				if (m) {
					const size_t jprime = j*4 + __builtin_ctz(m);

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
	/// NOTE: compared to `bruteforce_avx2_64_1x1` this unrolls `u` elementes in the left
	///			list and `v` elements on the right.
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<const uint32_t u, const uint32_t v>
	void bruteforce_avx2_64_uxv(const size_t e1,
	                            const size_t e2) noexcept {
		ASSERT(ELEMENT_NR_LIMBS == 1);
		ASSERT(n <= 64);
		ASSERT(n >= 33);

		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		__m256i lii[u], rii[v];

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);

		for (size_t i = s1; i < e1; i += u) {

			#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				lii[j] = _mm256_set1_epi64x(L1[i + j][0]);
			}

			/// NOTE: only possible because L2 is a continuous memory block
			__m256i *ptr_r = (__m256i *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; j += v, ptr_r += v) {

				#pragma unroll
				for (uint32_t s = 0; s < v; ++s) {
					rii[s] = LOAD256(ptr_r + s);
				}

				#pragma unroll
				for (uint32_t a1 = 0; a1 < u; ++a1) {
					const __m256i tmp1 = lii[a1];
					#pragma unroll
					for (uint32_t a2 = 0; a2 < v; ++a2) {
						const __m256i tmp2 = rii[a2];
						const int m = compare_256_64<true>(tmp1, tmp2);

						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m);
							const size_t iprime = i + a1;

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

	/// NOTE: in comparison to the other version `bruteforce_avx2_64` this implementation
	///			assumes that the elements to compare are fully compared on all n variables
	///  		e.g. ELEMENT_NR_LIMBS == 1
	/// NOTE: compared to `bruteforce_avx2_64_1x1` this unrolls `u` elementes in the left
	///			list and `v` elements on the right.
	/// NOTE: compared to `bruteforce_avx2_64_uxv` this function is not only comparing 1
	///			element of the left list with u elements from the right. Side
	///			Internally the loop is unrolled to compare u*4 elements to v on the right
	/// NOTE: assumes the input list to of length multiple of 16
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	template<const uint32_t u, const uint32_t v>
	void bruteforce_avx2_64_uxv_shuffle(const size_t e1,
	                                    const size_t e2) noexcept {
		ASSERT(ELEMENT_NR_LIMBS == 1);
		ASSERT(n <= 64);
		ASSERT(n >= 33);

		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		__m256i lii[u], rii[v];
		__m256i *ptr_l = (__m256i *)L1;

		for (size_t i = s1; i < s1 + (e1+3)/4; i += u, ptr_l += u) {

			#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				lii[j] = LOAD256(ptr_l + j);
			}

			/// NOTE: only possible because L2 is a continuous memory block
			__m256i *ptr_r = (__m256i *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; j += v, ptr_r += v) {

				#pragma unroll
				for (uint32_t s = 0; s < v; ++s) {
					rii[s] = LOAD256(ptr_r + s);
				}

				#pragma unroll
				for (uint32_t a1 = 0; a1 < u; ++a1) {
					const __m256i tmp1 = lii[a1];
					#pragma unroll
					for (uint32_t a2 = 0; a2 < v; ++a2) {
						__m256i tmp2 = rii[a2];
						int m = compare_256_64<true>(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m);
							const size_t iprime = i*4 + a1*4 + __builtin_ctz(m);
							if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						m = compare_256_64<true>(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m) + 3;
							const size_t iprime = i*4 + a1*4 + __builtin_ctz(m);
							if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						m = compare_256_64<true>(tmp1, tmp2);
						if (m) {
							const size_t jprime = j*4 + a2*4 + __builtin_ctz(m) + 2;
							const size_t iprime = i*4 + a1*4+ __builtin_ctz(m);
							if (compare_u64_ptr((T *)(L1 + iprime), (T *)(L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}

						tmp2 = _mm256_permute4x64_epi64(tmp2, 0b10010011);
						m = compare_256_64<true>(tmp1, tmp2);
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
	void bruteforce_avx2_128(const size_t e1,
	                         const size_t e2) noexcept {

		ASSERT(n <= 128);
		ASSERT(n > 64);
		ASSERT(2 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// difference of the memory location in the right list
		const __m128i loadr1 = {       (2ull << 32u), (4ul) | (6ull << 32u)};
		const __m128i loadr2 = {1ull | (3ull << 32u), (5ul) | (7ull << 32u)};

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);

		for (size_t i = s1; i < e1; ++i) {
			const __m256i li1 = _mm256_set1_epi64x(L1[i][0]);
			const __m256i li2 = _mm256_set1_epi64x(L1[i][1]);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 8) {
				const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr1, 8);
				const int m1 = compare_256_64(li1, ri);

				if (m1) {
					const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr2, 8);
					const int m1 = compare_256_64(li2, ri);

					if (m1) {
						const size_t jprime = j*4;

						if (compare_u64_ptr((T *)(L1 + i), (T *)(L2 + jprime))) {
							//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
							found_solution(i, jprime + __builtin_ctz(m1));
						}
					}
				}
			}
		}
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
	void bruteforce_avx2_128_32_2_uxv_helper(uint32_t mask,
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
	void bruteforce_avx2_128_32_2_uxv(const size_t e1,
									  const size_t e2) noexcept {
		static_assert(u <= 8);
		static_assert(v <= 8);
		ASSERT(n <= 128);
		ASSERT(n > 64);
		ASSERT(2 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// some constants
		const __m256i zero = _mm256_set1_epi32(0);
		const __m256i shuffl = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
		const __m256i loadr = _mm256_setr_epi32(0, 4, 8, 12, 16, 20, 24, 28);
		constexpr size_t ptr_ctr_l = u*8, ptr_ctr_r = v*8;
		constexpr size_t ptr_inner_ctr_l = 8*4, ptr_inner_ctr_r = 8*4;

		/// container for the unrolling
		__m256i lii_1[u], rii_1[v], lii_2[u], rii_2[v];

		/// container for the solutions masks
		constexpr uint32_t size_m1 = std::max(u*v, 32u);
		alignas(32) uint8_t m1[size_m1] = {0}; /// NOTE: init with 0 is important
		__m256i *m1_256 = (__m256i *)m1;

		uint32_t *ptr_l = (uint32_t *)L1;
		for (size_t i = s1; i < s1 + e1; i += ptr_ctr_l, ptr_l += ptr_ctr_l*4) {

			#pragma unroll
			for (uint32_t s = 0; s < u; ++s) {
				lii_1[s] = _mm256_i32gather_epi32((const int *)(ptr_l + s*ptr_inner_ctr_l + 0), loadr, 4);
				lii_2[s] = _mm256_i32gather_epi32((const int *)(ptr_l + s*ptr_inner_ctr_l + 1), loadr, 4);
			}

			uint32_t *ptr_r = (uint32_t *)L2;
			for (size_t j = s2; j < s2 + e2; j += ptr_ctr_r, ptr_r += ptr_ctr_r*4) {

				// load the fi
				#pragma unroll
				for (uint32_t s = 0; s < v; ++s) {
					rii_1[s] = _mm256_i32gather_epi32((const int *)(ptr_r + s*ptr_inner_ctr_r + 0), loadr, 4);
					rii_2[s] = _mm256_i32gather_epi32((const int *)(ptr_r + s*ptr_inner_ctr_r + 1), loadr, 4);
				}

				/// Do the 8x8 shuffle
				#pragma unroll
				for (uint32_t l = 0; l < 8; ++l) {
					if (l > 0) {
						// shuffle the right side
						#pragma unroll
						for (uint32_t s2s = 0; s2s < v; ++s2s) {
							rii_1[s2s] = _mm256_permutevar8x32_epi32(rii_1[s2s], shuffl);
							rii_2[s2s] = _mm256_permutevar8x32_epi32(rii_2[s2s], shuffl);
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
					uint32_t mask = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)m1), zero));
					if (unlikely(mask == 0)) {
						continue;
					}

					//// second limb
					#pragma unroll
					for (uint32_t f1 = 0; f1 < u; ++f1) {
						#pragma unroll
						for (uint32_t f2 = 0; f2 < v; ++f2) {
							//if (m1[f1*u + f2]) {
								m1[f1*u + f2] &= compare_256_32(lii_2[f1], rii_2[f2]);
							//}
						}
					}


					// early exit from the second limb computations
					mask = _mm256_movemask_epi8(_mm256_cmpgt_epi8(LOAD256((__m256i *)m1), zero));
					if (likely(mask == 0)) {
						continue;
					}

					// maybe write back a solution
					bruteforce_avx2_128_32_2_uxv_helper<u,v>(mask, m1, l, i, j);
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
	void bruteforce_avx2_256(const size_t e1,
	                         const size_t e2) noexcept {

		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// difference of the memory location in the right list
		const __m128i loadr1 = {       (4ull << 32u), ( 8ul) | (12ull << 32u)};
		const __m128i loadr2 = {1ull | (5ull << 32u), ( 9ul) | (13ull << 32u)};
		const __m128i loadr3 = {2ull | (6ull << 32u), (10ul) | (14ull << 32u)};
		const __m128i loadr4 = {3ull | (7ull << 32u), (11ul) | (15ull << 32u)};

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);

		for (size_t i = s1; i < e1; ++i) {
			const __m256i li1 = _mm256_set1_epi64x(L1[i][0]);
			const __m256i li2 = _mm256_set1_epi64x(L1[i][1]);
			const __m256i li3 = _mm256_set1_epi64x(L1[i][2]);
			const __m256i li4 = _mm256_set1_epi64x(L1[i][3]);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 16) {
				const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr1, 8);
				const int m1 = compare_256_64(li1, ri);

				if (m1) {
					const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr2, 8);
					const int m1 = compare_256_64(li2, ri);

					if (m1) {
						const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr3, 8);
						const int m1 = compare_256_64(li3, ri);

						if (m1) {
							const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr4, 8);
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
	void bruteforce_avx2_256_ux4(const size_t e1,
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
		const __m128i loadr1 = {       (4ull << 32u), ( 8ul) | (12ull << 32u)};
		const __m128i loadr2 = {1ull | (5ull << 32u), ( 9ul) | (13ull << 32u)};
		const __m128i loadr3 = {2ull | (6ull << 32u), (10ul) | (14ull << 32u)};
		const __m128i loadr4 = {3ull | (7ull << 32u), (11ul) | (15ull << 32u)};

		alignas(32) __m256i li[u*4u];
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
				li[ui + 0*u] = _mm256_set1_epi64x(L1[i + ui][0]);
				li[ui + 1*u] = _mm256_set1_epi64x(L1[i + ui][1]);
				li[ui + 2*u] = _mm256_set1_epi64x(L1[i + ui][2]);
				li[ui + 3*u] = _mm256_set1_epi64x(L1[i + ui][3]);
			}


			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *)L2;

			for (size_t j = s2; j < s2+(e2+3)/4; ++j, ptr_r += 16) {
				//#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr1, 8);
					const uint32_t tmp  = compare_256_64(li[0 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr2, 8);
					const uint32_t tmp  = compare_256_64(li[1 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr3, 8);
					const uint32_t tmp  = compare_256_64(li[2 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi64((const long long int *)ptr_r, loadr4, 8);
					const uint32_t tmp  = compare_256_64(li[3 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
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
	void bruteforce_avx2_256_32_ux8(const size_t e1,
	                                const size_t e2) noexcept {
		static_assert(u > 0, "");
		static_assert(u <= 8, "");

		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// NOTE: limit arbitrary but needed for correctness
		ASSERT(d < 7);

		/// difference of the memory location in the right list
		const __m256i loadr1 = _mm256_setr_epi32(0, 8, 16, 24, 32, 40, 48, 56);
		const __m256i loadr_add = _mm256_set1_epi32(1);
		__m256i loadr;

		alignas(32) __m256i li[u*8u];
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
			#pragma unroll
			for (uint32_t ui = 0; ui < u; ui++) {
				#pragma unroll
				for (uint32_t uii = 0; uii < 8; uii++) {
					const uint32_t tmp = ((uint32_t *)L1[i + ui])[uii];
					li[ui + uii*u] = _mm256_set1_epi32(tmp);
				}
			}


			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			uint32_t *ptr_r = (uint32_t *)L2;

			for (size_t j = s2; j < s2+(e2+7)/8; ++j, ptr_r += 64) {
				loadr = loadr1;
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[0 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[1 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[2 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[3 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[4 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[5 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[6 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = _mm256_add_epi32(loadr, loadr_add);
				#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const __m256i ri = _mm256_i32gather_epi32((const int *)ptr_r, loadr, 4);
					const uint32_t tmp  = compare_256_32(li[7 * u + mi], ri);
					m1s[mi] = tmp ? tmp^m1s_mask : 0;
				}

				m1s_tmp = _mm256_movemask_ps((__m256) LOAD256((__m256i *)m1s));
				if (m1s_tmp) {
					// TODO limitation.
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

	///
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
	void bruteforce_avx2_256_32_8x8(const size_t e1,
	                                const size_t e2) noexcept {
		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		//ASSERT(e1 >= 64);
		//ASSERT(e2 >= 64);

		ASSERT(d < 16);

		uint32_t *ptr_l = (uint32_t *)L1;

		/// difference of the memory location in the right list
		const __m256i loadr1 = _mm256_setr_epi32(0, 8, 16, 24, 32, 40, 48, 56);
		const __m256i shuffl = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);

		alignas(32) uint8_t m1s[64];

		/// helper to detect zeros
		const __m256i zero = _mm256_set1_epi32(0);

		for (size_t i = s1; i < s1 + e1; i += 64, ptr_l += 512) {
			const __m256i l1 = _mm256_i32gather_epi32((const int *)(ptr_l +   0), loadr1, 4);
			const __m256i l2 = _mm256_i32gather_epi32((const int *)(ptr_l +  64), loadr1, 4);
			const __m256i l3 = _mm256_i32gather_epi32((const int *)(ptr_l + 128), loadr1, 4);
			const __m256i l4 = _mm256_i32gather_epi32((const int *)(ptr_l + 192), loadr1, 4);
			const __m256i l5 = _mm256_i32gather_epi32((const int *)(ptr_l + 256), loadr1, 4);
			const __m256i l6 = _mm256_i32gather_epi32((const int *)(ptr_l + 320), loadr1, 4);
			const __m256i l7 = _mm256_i32gather_epi32((const int *)(ptr_l + 384), loadr1, 4);
			const __m256i l8 = _mm256_i32gather_epi32((const int *)(ptr_l + 448), loadr1, 4);

			uint32_t *ptr_r = (uint32_t *)L2;
			for (size_t j = s1; j < s2 + e2; j += 64, ptr_r += 512) {
				__m256i r1 = _mm256_i32gather_epi32((const int *)(ptr_r +   0), loadr1, 4);
				__m256i r2 = _mm256_i32gather_epi32((const int *)(ptr_r +  64), loadr1, 4);
				__m256i r3 = _mm256_i32gather_epi32((const int *)(ptr_r + 128), loadr1, 4);
				__m256i r4 = _mm256_i32gather_epi32((const int *)(ptr_r + 192), loadr1, 4);
				__m256i r5 = _mm256_i32gather_epi32((const int *)(ptr_r + 256), loadr1, 4);
				__m256i r6 = _mm256_i32gather_epi32((const int *)(ptr_r + 320), loadr1, 4);
				__m256i r7 = _mm256_i32gather_epi32((const int *)(ptr_r + 384), loadr1, 4);
				__m256i r8 = _mm256_i32gather_epi32((const int *)(ptr_r + 448), loadr1, 4);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				uint32_t m1s1 = _mm256_movemask_epi8(~(_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero)));
				uint32_t m1s2 = _mm256_movemask_epi8(~(_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero)));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 0>(m1s2, m1s, ptr_l, ptr_r, i, j); }


				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 1>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 1>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 2>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 2>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 3>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 3>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 4>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 4>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 5>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 5>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 6>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 6>(m1s2, m1s, ptr_l, ptr_r, i, j); }

				r1 = _mm256_permutevar8x32_epi32(r1, shuffl);
				r2 = _mm256_permutevar8x32_epi32(r2, shuffl);
				r3 = _mm256_permutevar8x32_epi32(r3, shuffl);
				r4 = _mm256_permutevar8x32_epi32(r4, shuffl);
				r5 = _mm256_permutevar8x32_epi32(r5, shuffl);
				r6 = _mm256_permutevar8x32_epi32(r6, shuffl);
				r7 = _mm256_permutevar8x32_epi32(r7, shuffl);
				r8 = _mm256_permutevar8x32_epi32(r8, shuffl);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				m1s1 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s +  0)), zero));
				m1s2 = _mm256_movemask_epi8(~_mm256_cmpeq_epi8(LOAD256((__m256i *)(m1s + 32)), zero));
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper< 0, 7>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 7>(m1s2, m1s, ptr_l, ptr_r, i, j); }
			}
		}
	}
	
	///
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

	///
	/// \param stack
	/// \param a0
	/// \param a1
	/// \param a2
	/// \param a3
	/// \param b0
	/// \param b1
	/// \param b2
	/// \param b3
	void BRUTEFORCE256_64_4x4_STEP2(uint8_t* stack, __m256i a0,__m256i a1,__m256i a2,__m256i a3,__m256i b0,__m256i b1,__m256i b2,__m256i b3) {
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
		b0 = _mm256_permute4x64_epi64(b0, 0b00111001);
		b1 = _mm256_permute4x64_epi64(b1, 0b00111001);
		b2 = _mm256_permute4x64_epi64(b2, 0b00111001);
		b3 = _mm256_permute4x64_epi64(b3, 0b00111001);
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
		b0 = _mm256_permute4x64_epi64(b0, 0b00111001);
		b1 = _mm256_permute4x64_epi64(b1, 0b00111001);
		b2 = _mm256_permute4x64_epi64(b2, 0b00111001);
		b3 = _mm256_permute4x64_epi64(b3, 0b00111001);
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
		b0 = _mm256_permute4x64_epi64(b0, 0b00111001);
		b1 = _mm256_permute4x64_epi64(b1, 0b00111001);
		b2 = _mm256_permute4x64_epi64(b2, 0b00111001);
		b3 = _mm256_permute4x64_epi64(b3, 0b00111001);
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
	void bruteforce_avx2_256_64_4x4(const size_t e1,
	                                const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;

		ASSERT(n <= 256);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);
		ASSERT(dk < 32);

		/// NOTE is already aligned
		T *ptr_l = (T *)L1;

		/// difference of the memory location in the right list
		const __m128i loadr1 = {       (4ull << 32u), ( 8ul) | (12ull << 32u)};
		alignas(32) uint8_t m1s[64];

		/// allowed weight to match on
		const __m256i zero = _mm256_set1_epi32(0);

		for (size_t i = s1; i < s1 + e1; i += 16, ptr_l += 64) {
			const __m256i l1 = _mm256_i32gather_epi64((const long long int *)(ptr_l +  0), loadr1, 8);
			const __m256i l2 = _mm256_i32gather_epi64((const long long int *)(ptr_l + 16), loadr1, 8);
			const __m256i l3 = _mm256_i32gather_epi64((const long long int *)(ptr_l + 32), loadr1, 8);
			const __m256i l4 = _mm256_i32gather_epi64((const long long int *)(ptr_l + 48), loadr1, 8);

			/// reset right list pointer
			T *ptr_r = (T *)L2;

			#pragma unroll 4
			for (size_t j = s1; j < s2 + e2; j += 16, ptr_r += 64) {
				__m256i r1 = _mm256_i32gather_epi64((const long long int *)(ptr_r +  0), loadr1, 8);
				__m256i r2 = _mm256_i32gather_epi64((const long long int *)(ptr_r + 16), loadr1, 8);
				__m256i r3 = _mm256_i32gather_epi64((const long long int *)(ptr_r + 32), loadr1, 8);
				__m256i r4 = _mm256_i32gather_epi64((const long long int *)(ptr_r + 48), loadr1, 8);

				BRUTEFORCE256_64_4x4_STEP2(m1s, l1, l2, l3, l4, r1, r2, r3, r4);
				uint32_t m1s1 = _mm256_movemask_epi8((_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s +  0)), zero)));
				uint32_t m1s2 = _mm256_movemask_epi8((_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s + 32)), zero)));

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
	void bruteforce_avx2_256_64_4x4_rearrange(const size_t e1,
									          const size_t e2) noexcept {
		constexpr size_t s1 = 0, s2 = 0;
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
		const __m256i zero = _mm256_set1_epi32(0);

		size_t i = s1;
        #pragma unroll 2
		for (; i < s1 + e1; i += 16, ptr_l += 16) {
			const __m256i l1 = _mm256_load_si256((const __m256i *)(ptr_l +  0));
			const __m256i l2 = _mm256_load_si256((const __m256i *)(ptr_l +  4));
			const __m256i l3 = _mm256_load_si256((const __m256i *)(ptr_l +  8));
			const __m256i l4 = _mm256_load_si256((const __m256i *)(ptr_l + 12));

			/// reset right list pointer
			T *ptr_r = (T *)RB;

			#pragma unroll 4
			for (size_t j = s1; j < s2 + e2; j += 16, ptr_r += 16) {
				__m256i r1 = _mm256_load_si256((const __m256i *)(ptr_r +  0));
				__m256i r2 = _mm256_load_si256((const __m256i *)(ptr_r +  4));
				__m256i r3 = _mm256_load_si256((const __m256i *)(ptr_r +  8));
				__m256i r4 = _mm256_load_si256((const __m256i *)(ptr_r + 12));

				BRUTEFORCE256_64_4x4_STEP2(m1s, l1, l2, l3, l4, r1, r2, r3, r4);
				uint32_t m1s1 = _mm256_movemask_epi8((_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s +  0)), zero)));
				uint32_t m1s2 = _mm256_movemask_epi8((_mm256_cmpgt_epi8(LOAD256((__m256i *)(m1s + 32)), zero)));

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
	                            const size_t e2) noexcept {
		ASSERT(EXACT);
		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// allowed weight to match on
		const __m256i weight = _mm256_setr_epi64x(0, 0, 0, 0);
		__m256i *ptr_l = (__m256i *)L1;

		for (size_t i = s1; i < e1; ++i, ptr_l += 1) {
			const __m256i li1 = LOAD256(ptr_l);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			__m256i *ptr_r = (__m256i *)L2;

			for (size_t j = s2; j < s2+e2; ++j, ptr_r += 1) {
				const __m256i ri = LOAD256(ptr_r);
				const __m256i tmp1 = _mm256_xor_si256(li1, ri);
				const __m256i tmp2 = _mm256_cmpeq_epi64(tmp1, weight);
				const int m1 = _mm256_movemask_pd((__m256d) tmp2);

				if (m1) {
					//std::cout << L1[i][0] << " " << L2[j][0] << " " << L2[j+1][0] << " " << L2[j-1][0] << "\n";
					found_solution(i, j);
				}
			}
		}
	}



	/// mother of all bruteforce algorithms. This is a selector function,
	/// which tries to select heuristically the best subroutine
	/// \param e1
	/// \param e2
	void bruteforce(const size_t e1,
	                const size_t e2) noexcept {
		if constexpr (32 < n and n <= 64) {
			bruteforce_avx2_64_uxv<4, 4>(e1, e2);
		} else if constexpr (64 < n and n <= 128){
			//bruteforce_128(e1, e2);
			bruteforce_avx2_128_32_2_uxv<4, 4>(e1, e2);
			//bruteforce_avx2_128(e1, e2);

			// actualy also work
			// bruteforce_avx2_256_32_8x8(e1, e2);
		} else if constexpr (128 < n and n <= 256) {
			// TODO optimal value
			if (e1 < 10 && e2 < 10) {
				bruteforce_256(e1, e2);
				//bruteforce_avx2_256(e1, e2);
				//bruteforce_avx2_256_64_4x4(e1, e2);
				return;
			}
			
			// in the low weight case with have better implementations
			if constexpr (d < 16) {
				// bruteforce_avx2_256_32_ux8<8>(e1, e2);
				//bruteforce_avx2_256_32_8x8(e1, e2);
				bruteforce_avx2_256_64_4x4(e1, e2);
				return;
			}

			// generic best implementations for every weight

			//bruteforce_256(e1, e2);
			bruteforce_avx2_256_64_4x4(e1, e2);
		} else {
			ASSERT(false);
		}
	}

	/// NOTE: uses avx2 instructions
	/// NOTE: currently not correct
	/// \param gt_mask comparison mask. From this mask we can deduct the TODO
	/// \param ctr
	/// \return
	size_t swap_avx_64(const __m256i gt_mask, const size_t ctr,
			Element *__restrict__ ptr, Element *__restrict__ L) const noexcept {
		ASSERT(n <= 64);
		ASSERT(n > 32);

		// extract the int bit mask
		const int wt = _mm256_movemask_ps((__m256) gt_mask);

		// make sure only sane inputs make it.
		ASSERT(wt < (1u<<8u));

		/// see the description of this magic in `shuffle_down_64`
		const uint64_t expanded_mask_org = _pdep_u64(wt, 0b0101010101010101);
		const uint64_t expanded_mask = expanded_mask_org * 0b11;
		constexpr uint64_t identity_indices = 0b11100100;
		constexpr uint64_t identity_mask = (1u << 8u) - 1u;

		if (wt & 0b1111) {
			const uint64_t indices_down = _pext_u64(identity_indices, expanded_mask & identity_mask);
			const uint64_t indices_up   = _pdep_u64(identity_indices, expanded_mask & identity_mask);

			//const __m256i lower_down = _mm256_cvtepu8_epi64(_mm_cvtsi32_si128(uint16_t(wanted_indices_down)));
			//const __m256i tmp_down   = _mm256_i64gather_epi64(((uint64_t *) ptr), lower_down, 8);

			// store up
			const __m256i tmp_up1    = _mm256_lddqu_si256((__m256i *) (L + ctr));
			// TODO probably this needs to be done with a loop up table.
			//const __m256i tmp_up     = _mm256_permute4x64_epi64(tmp_up1, indices_up);

			// gt mask still wrong
			//_mm256_maskstore_epi64((long long *)ptr, gt_mask, tmp_up);
		}

		if (wt > 0b0001111) {
			const uint64_t wanted_indices_down2 = _pext_u64(identity_indices, expanded_mask >> 8u);
			const uint64_t wanted_indices_up2 = _pdep_u64(identity_indices, expanded_mask >> 8u);
		}

		return ctr + __builtin_popcount(wt);
	}

	/// copies `from` to `to` if the bit in `gt_mask` is set.
	/// \param wt int mask selecting the upper elements to shift down
	/// \param to
	/// \param from
	/// \return
	template<const uint32_t limit>
	inline size_t swap(int wt, Element *__restrict__ to, 
				Element *__restrict__ from) const noexcept {
		ASSERT(wt < (1u<<limit));

		uint32_t nctr = 0;

		#pragma unroll
		for (uint32_t i = 0; i < limit; ++i) {
			if (wt & 1u) {
				std::swap(to[nctr++], from[i]);
			}

			wt >>= 1u;
		}

		return nctr;
	}

	/// same as `swap` with the difference that this functions is swapping
	/// based on the count trailing bits functions.
	/// \param wt bit mask selecting the `from` elements
	/// \param to swp to
	/// \param from
	/// \return number of elements swapped
	inline size_t swap_ctz(uint32_t wt,
	                       Element *__restrict__ to,
	                       Element *__restrict__ from) const noexcept {
		uint32_t nctr = 0;
		const uint32_t bit_limit = __builtin_popcount(wt);
		for (uint32_t i = 0; i < bit_limit; ++i) {
			const uint32_t pos = __builtin_ctz(wt);
			std::swap(to[i], from[pos]);

			// clear the set bit.
			wt ^= 1u << pos;
		}

		return bit_limit;
	}

	///
	/// \tparam bucket_size
	/// \param wt
	/// \param to
	/// \param from
	/// \return
	template<const uint32_t bucket_size>
	inline size_t swap_ctz_rearrange(uint32_t wt,
						   			T *__restrict__ to,
						   			Element *__restrict__ from) const noexcept {
		if constexpr (!USE_REARRANGE) {
			ASSERT(false);
			return 0;
		}

		const uint32_t bit_limit = __builtin_popcount(wt);
		for (uint32_t i = 0; i < bit_limit; ++i) {
			const uint32_t pos = __builtin_ctz(wt);

			#pragma unroll
			for(uint32_t j = 0; j < ELEMENT_NR_LIMBS; j++) {
				ASSERT(i + j*bucket_size < (ELEMENT_NR_LIMBS * bucket_size));
				to[i + j*bucket_size] = from[pos][j];
			}

			// clear the set bit.
			wt ^= 1u << pos;
		}

		return bit_limit;
	}

	/// executes the comparison operator in the NN subroutine on 32 bit limbs
	/// \param tmp input
	/// \return a mask // TODO
	inline uint32_t compare_nn_on32(const __m256i tmp) const noexcept {
		static_assert(NN_EQUAL + NN_LOWER + NN_BOUNDS == 1);
		if constexpr (NN_EQUAL) {
			static_assert(epsilon == 0);
			const __m256i avx_nn_weight32 = _mm256_set1_epi32 (dk+NN_LOWER);
			const __m256i gt_mask = _mm256_cmpeq_epi32(avx_nn_weight32, tmp);
			return _mm256_movemask_ps((__m256) gt_mask);
		}

		if constexpr (NN_LOWER) {
			static_assert(epsilon == 0);
			const __m256i avx_nn_weight32 = _mm256_set1_epi32 (dk+NN_LOWER);
			const __m256i gt_mask = _mm256_cmpgt_epi32(avx_nn_weight32, tmp);
			return _mm256_movemask_ps((__m256) gt_mask);
		}

		if constexpr (NN_BOUNDS) {
			static_assert(epsilon > 0);
			static_assert(epsilon < dk);

			const __m256i avx_nn_weight32 = _mm256_set1_epi32 (dk+NN_LOWER+epsilon);
			const __m256i avx_nn_weight_lower32 = _mm256_set1_epi32(dk-epsilon);

			const __m256i lt_mask = _mm256_cmpgt_epi32(avx_nn_weight_lower32, tmp);
			const __m256i gt_mask = _mm256_cmpgt_epi32(avx_nn_weight32, tmp);
			return _mm256_movemask_ps((__m256) _mm256_and_si256(lt_mask, gt_mask));
		}
	}

	/// executes the comparison operator in the NN subroutine
	/// \param tmp
	/// \return
	inline uint32_t compare_nn_on64(const __m256i tmp) const noexcept {
		static_assert(NN_EQUAL + NN_LOWER + NN_BOUNDS == 1);
		if constexpr (NN_EQUAL) {
			static_assert(epsilon == 0);
			const __m256i avx_nn_weight64 = _mm256_set1_epi64x(dk+NN_LOWER+epsilon);
			const __m256i gt_mask = _mm256_cmpeq_epi64(avx_nn_weight64, tmp);
			return _mm256_movemask_pd((__m256d) gt_mask);
		}

		if constexpr (NN_LOWER) {
			static_assert(epsilon == 0);
			const __m256i avx_nn_weight64 = _mm256_set1_epi64x(dk+NN_LOWER+epsilon);
			const __m256i gt_mask = _mm256_cmpgt_epi64(avx_nn_weight64, tmp);
			return _mm256_movemask_pd((__m256d) gt_mask);
		}

		if constexpr (NN_BOUNDS) {
			static_assert(epsilon > 0);
			static_assert(epsilon < dk);
			const __m256i avx_nn_weight64 = _mm256_set1_epi64x(dk+NN_LOWER+epsilon);
			const __m256i avx_nn_weight_lower64 = _mm256_set1_epi64x(dk-epsilon);

			const __m256i lt_mask = _mm256_cmpgt_epi64(avx_nn_weight_lower64, tmp);
			const __m256i gt_mask = _mm256_cmpgt_epi64(avx_nn_weight64, tmp);
			return _mm256_movemask_pd((__m256d) _mm256_and_si256(lt_mask, gt_mask));
		}
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint32
	/// \param: e1 end index
	/// \param: random value z
	template<const uint32_t limb>
	size_t avx2_sort_nn_on32_simple(const size_t e1,
									const uint32_t z,
							        Element *__restrict__ L) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 32);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		const size_t s1 = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi32(z);
		alignas(32) const __m256i offset = _mm256_setr_epi32(0*enl, 1*enl, 2*enl, 3*enl,
		                                                     4*enl, 5*enl, 6*enl, 7*enl);

		// NR of partial solutions found
		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*4);
		Element *org_ptr = L;
	
		for (size_t i = s1; i < (e1+7)/8; i++, ptr += 8, org_ptr += 8) {
			const __m256i ptr_tmp = _mm256_i32gather_epi32(ptr, offset, 8);
			__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			if constexpr (k < 32) {
				tmp = _mm256_and_si256(tmp, avx_nn_k_mask);
			}
			const __m256i tmp_pop = popcount_avx2_32(tmp);
			const uint32_t wt = compare_nn_on32(tmp_pop);
			ASSERT(wt < (1ul << 8));

			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap<8>(wt, L + ctr, org_ptr);
			}
		}
	
		return ctr;
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint32_t
	/// NOTE: fixxed unrolling factor by 4
	/// \param: e1 end index
	/// \param: random value z
	template<const uint32_t limb>
	size_t avx2_sort_nn_on32(const size_t e1,
	                         const uint32_t z,
	                         Element *__restrict__ L) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 32);


		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi32(z);
		alignas(32) const __m256i offset = _mm256_setr_epi32(0*enl, 1*enl, 2*enl, 3*enl,
		                                                     4*enl, 5*enl, 6*enl, 7*enl);

		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*4);
		Element *org_ptr = L;

		constexpr uint32_t u = 4;
		constexpr uint32_t off = u*8;
		for (; i+off <= e1; i += off, ptr += off, org_ptr += off) {
			__m256i ptr_tmp0 = _mm256_i32gather_epi32(ptr +  0, offset, 8);
			ptr_tmp0 = _mm256_xor_si256(ptr_tmp0, z256);
			if constexpr (k < 32) { ptr_tmp0 = _mm256_and_si256(ptr_tmp0, avx_nn_k_mask); }
			__m256i ptr_tmp1 = _mm256_i32gather_epi32(ptr +  8, offset, 8);
			ptr_tmp1 = _mm256_xor_si256(ptr_tmp1, z256);
			if constexpr (k < 32) { ptr_tmp1 = _mm256_and_si256(ptr_tmp1, avx_nn_k_mask); }
			__m256i ptr_tmp2 = _mm256_i32gather_epi32(ptr + 16, offset, 8);
			ptr_tmp2 = _mm256_xor_si256(ptr_tmp2, z256);
			if constexpr (k < 32) { ptr_tmp2 = _mm256_and_si256(ptr_tmp2, avx_nn_k_mask); }
			__m256i ptr_tmp3 = _mm256_i32gather_epi32(ptr + 24, offset, 8);
			ptr_tmp3 = _mm256_xor_si256(ptr_tmp3, z256);
			if constexpr (k < 32) { ptr_tmp3 = _mm256_and_si256(ptr_tmp3, avx_nn_k_mask); }

			uint32_t wt = 0;
			__m256i tmp_pop = popcount_avx2_32(ptr_tmp0);
			wt = compare_nn_on32(tmp_pop);

			tmp_pop = popcount_avx2_32(ptr_tmp1);
			wt ^= compare_nn_on32(tmp_pop) << 8u;

			tmp_pop = popcount_avx2_32(ptr_tmp2);
			wt ^= compare_nn_on32(tmp_pop) << 16u;

			tmp_pop = popcount_avx2_32(ptr_tmp3);
			wt ^= compare_nn_on32(tmp_pop) << 24u;

			if (wt) {
				ctr += swap_ctz(wt, L + ctr, org_ptr);
			}
		}

		// tail work
		// #pragma unroll 4
		for (; i+8 < e1+7; i+=8, ptr += 8, org_ptr += 8) {
			const __m256i ptr_tmp = _mm256_i32gather_epi32(ptr, offset, 8);
			__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			if constexpr (k < 32) {
				tmp = _mm256_and_si256(tmp, avx_nn_k_mask);
			}
			const __m256i tmp_pop = popcount_avx2_32(tmp);
			const int wt = compare_nn_on32(tmp_pop);

			ASSERT(wt < 1u << 8u);
			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap<8>(wt, L + ctr, org_ptr);
			}
		}

		return ctr;
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint32_t
	/// NOTE: fixxed unrolling factor by 4
	/// \param: e1 end index
	/// \param: random value z
	template<const uint32_t limb, const uint32_t bucket_size>
	size_t avx2_sort_nn_on32_rearrange(const size_t e1,
							 		   const uint32_t z,
							 		   Element *__restrict__ L,
	                                   T *__restrict__ B) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 32);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi32(z);
		alignas(32) const __m256i offset = _mm256_setr_epi32(0*enl, 1*enl, 2*enl, 3*enl,
															 4*enl, 5*enl, 6*enl, 7*enl);

		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*4);
		Element *org_ptr = L;

		constexpr uint32_t u = 4;
		constexpr uint32_t off = u*8;
		for (; i+off <= e1; i += off, ptr += off, org_ptr += off) {
			__m256i ptr_tmp0 = _mm256_i32gather_epi32(ptr +  0, offset, 8);
			ptr_tmp0 = _mm256_xor_si256(ptr_tmp0, z256);
			if constexpr (k < 32) { ptr_tmp0 = _mm256_and_si256(ptr_tmp0, avx_nn_k_mask); }
			__m256i ptr_tmp1 = _mm256_i32gather_epi32(ptr +  8, offset, 8);
			ptr_tmp1 = _mm256_xor_si256(ptr_tmp1, z256);
			if constexpr (k < 32) { ptr_tmp1 = _mm256_and_si256(ptr_tmp1, avx_nn_k_mask); }
			__m256i ptr_tmp2 = _mm256_i32gather_epi32(ptr + 16, offset, 8);
			ptr_tmp2 = _mm256_xor_si256(ptr_tmp2, z256);
			if constexpr (k < 32) { ptr_tmp2 = _mm256_and_si256(ptr_tmp2, avx_nn_k_mask); }
			__m256i ptr_tmp3 = _mm256_i32gather_epi32(ptr + 24, offset, 8);
			ptr_tmp3 = _mm256_xor_si256(ptr_tmp3, z256);
			if constexpr (k < 32) { ptr_tmp3 = _mm256_and_si256(ptr_tmp3, avx_nn_k_mask); }

			uint32_t wt = 0;
			__m256i tmp_pop = popcount_avx2_32(ptr_tmp0);
			wt = compare_nn_on32(tmp_pop);

			tmp_pop = popcount_avx2_32(ptr_tmp1);
			wt ^= compare_nn_on32(tmp_pop) << 8u;

			tmp_pop = popcount_avx2_32(ptr_tmp2);
			wt ^= compare_nn_on32(tmp_pop) << 16u;

			tmp_pop = popcount_avx2_32(ptr_tmp3);
			wt ^= compare_nn_on32(tmp_pop) << 24u;

			if (wt) {
				ctr += swap_ctz_rearrange<bucket_size>(wt, B + ctr, org_ptr);
			}
		}

		// tail work
		// #pragma unroll 4
		for (; i+8 < e1+7; i+=8, ptr += 8, org_ptr += 8) {
			const __m256i ptr_tmp = _mm256_i32gather_epi32(ptr, offset, 8);
			__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			if constexpr (k < 32) {
				tmp = _mm256_and_si256(tmp, avx_nn_k_mask);
			}
			const __m256i tmp_pop = popcount_avx2_32(tmp);
			const int wt = compare_nn_on32(tmp_pop);

			ASSERT(wt < 1u << 8u);
			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap_ctz_rearrange<bucket_size>(wt, B + ctr, org_ptr);
			}
		}

		return ctr;
	}
	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint64_t
	/// \param: e1 end index
	/// \param: random value z
	template<const uint32_t limb>
	size_t avx2_sort_nn_on64_simple(const size_t e1,
							 	   	const uint64_t z,
							 	   	Element *__restrict__ L) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 64);
		ASSERT(k > 32);


		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi64x(z);
		alignas(32) const __m256i offset = _mm256_setr_epi64x(0*enl, 1*enl, 2*enl, 3*enl);

		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*8);
		Element *org_ptr = L;

		// #pragma unroll 4
		for (; i < (e1+3)/4; i++, ptr += 4, org_ptr += 4) {
			const __m256i ptr_tmp = _mm256_i64gather_epi64((const long long int *)ptr, offset, 8);
			__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			if constexpr (k < 64) { tmp = _mm256_and_si256(tmp, avx_nn_k_mask); }
			const __m256i tmp_pop = popcount_avx2_64(tmp);
			const uint32_t wt = compare_nn_on64(tmp_pop);
			ASSERT(wt < 1u << 4u);

			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap<4>(wt, L + ctr, org_ptr);
			}
		}


		return ctr;
	}


	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint64_t
	/// NOTE: hardcoded unroll parameter u=8
	/// NOTE: this version rearranges the elements in the list L
	///       in the following manner:
	///				TODO
	/// \param: e1 end index
	/// \param: random value z
	/// \param: L: input list
	/// \param: B: output bucket (rearranged)
	template<const uint32_t limb, const uint32_t bucket_size>
	size_t avx2_sort_nn_on64_rearrange(const size_t e1,
							 		   const uint64_t z,
	                                   Element *__restrict__ L,
	                                   T *B) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 64);
		ASSERT(k > 32);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		__m256i z256 = _mm256_set1_epi64x(z);
		__m256i offset = _mm256_setr_epi64x(0*enl, 1*enl, 2*enl, 3*enl);
		const __m256i avx_nn_weight64 = _mm256_set1_epi64x(dk+NN_LOWER+epsilon);

		size_t ctr = 0;

		/// NOTE: I need 2 ptrs tracking the current position, because of the limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*8);
		Element *org_ptr = L;

		constexpr uint32_t u = 8;
		constexpr uint32_t off = u*4;
		for (; i+off <= e1; i += off, ptr += off, org_ptr += off) {
			__m256i ptr_tmp0 = _mm256_i64gather_epi64(ptr +  0, offset, 8);
			ptr_tmp0 = _mm256_xor_si256(ptr_tmp0, z256);
			if constexpr (k < 64) { ptr_tmp0 = _mm256_and_si256(ptr_tmp0, avx_nn_k_mask); }
			__m256i ptr_tmp1 = _mm256_i64gather_epi64(ptr +  4, offset, 8);
			ptr_tmp1 = _mm256_xor_si256(ptr_tmp1, z256);
			if constexpr (k < 64) { ptr_tmp1 = _mm256_and_si256(ptr_tmp1, avx_nn_k_mask); }
			__m256i ptr_tmp2 = _mm256_i64gather_epi64(ptr +  8, offset, 8);
			ptr_tmp2 = _mm256_xor_si256(ptr_tmp2, z256);
			if constexpr (k < 64) { ptr_tmp2 = _mm256_and_si256(ptr_tmp2, avx_nn_k_mask); }
			__m256i ptr_tmp3 = _mm256_i64gather_epi64(ptr + 12, offset, 8);
			ptr_tmp3 = _mm256_xor_si256(ptr_tmp3, z256);
			if constexpr (k < 64) { ptr_tmp3 = _mm256_and_si256(ptr_tmp3, avx_nn_k_mask); }
			__m256i ptr_tmp4 = _mm256_i64gather_epi64(ptr + 16, offset, 8);
			ptr_tmp4 = _mm256_xor_si256(ptr_tmp4, z256);
			if constexpr (k < 64) { ptr_tmp4 = _mm256_and_si256(ptr_tmp4, avx_nn_k_mask); }
			__m256i ptr_tmp5 = _mm256_i64gather_epi64(ptr + 20, offset, 8);
			ptr_tmp5 = _mm256_xor_si256(ptr_tmp5, z256);
			if constexpr (k < 64) { ptr_tmp5 = _mm256_and_si256(ptr_tmp5, avx_nn_k_mask); }
			__m256i ptr_tmp6 = _mm256_i64gather_epi64(ptr + 24, offset, 8);
			ptr_tmp6 = _mm256_xor_si256(ptr_tmp6, z256);
			if constexpr (k < 64) { ptr_tmp6 = _mm256_and_si256(ptr_tmp6, avx_nn_k_mask); }
			__m256i ptr_tmp7 = _mm256_i64gather_epi64(ptr + 28, offset, 8);
			ptr_tmp7 = _mm256_xor_si256(ptr_tmp7, z256);
			if constexpr (k < 64) { ptr_tmp7 = _mm256_and_si256(ptr_tmp7, avx_nn_k_mask); }

			__m256i tmp_pop = popcount_avx2_64(ptr_tmp0);
			uint32_t wt = compare_nn_on64(tmp_pop);

			tmp_pop = popcount_avx2_64(ptr_tmp1);
			wt ^= compare_nn_on64(tmp_pop) << 4u;

			tmp_pop = popcount_avx2_64(ptr_tmp2);
			wt ^= compare_nn_on64(tmp_pop) << 8u;

			tmp_pop = popcount_avx2_64(ptr_tmp3);
			wt ^= compare_nn_on64(tmp_pop) << 12u;

			tmp_pop = popcount_avx2_64(ptr_tmp4);
			wt ^= compare_nn_on64(tmp_pop) << 16u;

			tmp_pop = popcount_avx2_64(ptr_tmp5);
			wt ^= compare_nn_on64(tmp_pop) << 20u;

			tmp_pop = popcount_avx2_64(ptr_tmp6);
			wt ^= compare_nn_on64(tmp_pop) << 24u;

			tmp_pop = popcount_avx2_64(ptr_tmp7);
			wt ^= compare_nn_on64(tmp_pop) << 28u;
			//ASSERT(uint64_t(wt) < (1ull << 32ull));

			ASSERT(ctr <= LIST_SIZE);
			ASSERT(ctr <= e1);

			if (wt) {
				ctr += swap_ctz_rearrange<bucket_size>(wt, B + ctr, org_ptr);
			}
		}

		// #pragma unroll 4
		for (; i < (e1+3)/4; i++, ptr += 4, org_ptr += 4) {
			const __m256i ptr_tmp = _mm256_i64gather_epi64(ptr, offset, 8);
			__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			if constexpr (k < 64) { tmp = _mm256_and_si256(tmp, avx_nn_k_mask); }
			const __m256i tmp_pop = popcount_avx2_64(tmp);

			//const __m256i gt_mask = _mm256_cmpgt_epi64(mask, tmp_pop);
			const __m256i gt_mask = _mm256_cmpeq_epi64(avx_nn_weight64, tmp_pop);
			const int wt = _mm256_movemask_pd((__m256d) gt_mask);
			ASSERT(wt < 1u << 4u);
			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap_ctz_rearrange<bucket_size>(wt, B + ctr, org_ptr);
			}
		}

		return ctr;
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint64_t
	/// NOTE: hardcoded unroll parameter u=8
	/// \param: e1 end index
	/// \param: random value z
	template<const uint32_t limb>
	size_t avx2_sort_nn_on64(const size_t e1,
					    	 const uint64_t z,
							 Element *__restrict__ L) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 64);
		ASSERT(k > 32);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi64x(z);
		alignas(32) const __m256i offset = _mm256_setr_epi64x(0*enl, 1*enl, 2*enl, 3*enl);
		const __m256i avx_nn_weight64 = _mm256_set1_epi64x(dk+NN_LOWER+epsilon);

		size_t ctr = 0;

		/// NOTE: I need 2 ptrs tracking the current position, because of the limb shift
		Element *ptr = (Element *)(((uint8_t *)L) + limb*8);
		Element *org_ptr = L;

		constexpr uint32_t u = 8;
		constexpr uint32_t off = u*4;
		for (; i+off <= e1; i += off, ptr += off, org_ptr += off) {
			__m256i ptr_tmp0 = _mm256_i64gather_epi64(ptr +  0, offset, 8);
			ptr_tmp0 = _mm256_xor_si256(ptr_tmp0, z256);
			if constexpr (k < 64) { ptr_tmp0 = _mm256_and_si256(ptr_tmp0, avx_nn_k_mask); }
			__m256i ptr_tmp1 = _mm256_i64gather_epi64(ptr +  4, offset, 8);
			ptr_tmp1 = _mm256_xor_si256(ptr_tmp1, z256);
			if constexpr (k < 64) { ptr_tmp1 = _mm256_and_si256(ptr_tmp1, avx_nn_k_mask); }
			__m256i ptr_tmp2 = _mm256_i64gather_epi64(ptr +  8, offset, 8);
			ptr_tmp2 = _mm256_xor_si256(ptr_tmp2, z256);
			if constexpr (k < 64) { ptr_tmp2 = _mm256_and_si256(ptr_tmp2, avx_nn_k_mask); }
			__m256i ptr_tmp3 = _mm256_i64gather_epi64(ptr + 12, offset, 8);
			ptr_tmp3 = _mm256_xor_si256(ptr_tmp3, z256);
			if constexpr (k < 64) { ptr_tmp3 = _mm256_and_si256(ptr_tmp3, avx_nn_k_mask); }
			__m256i ptr_tmp4 = _mm256_i64gather_epi64(ptr + 16, offset, 8);
			ptr_tmp4 = _mm256_xor_si256(ptr_tmp4, z256);
			if constexpr (k < 64) { ptr_tmp4 = _mm256_and_si256(ptr_tmp4, avx_nn_k_mask); }
			__m256i ptr_tmp5 = _mm256_i64gather_epi64(ptr + 20, offset, 8);
			ptr_tmp5 = _mm256_xor_si256(ptr_tmp5, z256);
			if constexpr (k < 64) { ptr_tmp5 = _mm256_and_si256(ptr_tmp5, avx_nn_k_mask); }
			__m256i ptr_tmp6 = _mm256_i64gather_epi64(ptr + 24, offset, 8);
			ptr_tmp6 = _mm256_xor_si256(ptr_tmp6, z256);
			if constexpr (k < 64) { ptr_tmp6 = _mm256_and_si256(ptr_tmp6, avx_nn_k_mask); }
			__m256i ptr_tmp7 = _mm256_i64gather_epi64(ptr + 28, offset, 8);
			ptr_tmp7 = _mm256_xor_si256(ptr_tmp7, z256);
			if constexpr (k < 64) { ptr_tmp7 = _mm256_and_si256(ptr_tmp7, avx_nn_k_mask); }

			__m256i tmp_pop = popcount_avx2_64(ptr_tmp0);
			uint32_t wt = compare_nn_on64(tmp_pop);

			tmp_pop = popcount_avx2_64(ptr_tmp1);
			wt ^= compare_nn_on64(tmp_pop) << 4u;

			tmp_pop = popcount_avx2_64(ptr_tmp2);
			wt ^= compare_nn_on64(tmp_pop) << 8u;

			tmp_pop = popcount_avx2_64(ptr_tmp3);
			wt ^= compare_nn_on64(tmp_pop) << 12u;

			tmp_pop = popcount_avx2_64(ptr_tmp4);
			wt ^= compare_nn_on64(tmp_pop) << 16u;

			tmp_pop = popcount_avx2_64(ptr_tmp5);
			wt ^= compare_nn_on64(tmp_pop) << 20u;

			tmp_pop = popcount_avx2_64(ptr_tmp6);
			wt ^= compare_nn_on64(tmp_pop) << 24u;

			tmp_pop = popcount_avx2_64(ptr_tmp7);
			wt ^= compare_nn_on64(tmp_pop) << 28u;
			//ASSERT(uint64_t(wt) < (1ull << 32ull));

			ASSERT(ctr <= LIST_SIZE);
			ASSERT(ctr <= e1);

			if (wt) {
				ctr += swap_ctz(wt, L + ctr, org_ptr);
			}
		}

		// #pragma unroll 4
		for (; i < (e1+3)/4; i++, ptr += 4, org_ptr += 4) {
			const __m256i ptr_tmp = _mm256_i64gather_epi64(ptr, offset, 8);
			__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
			if constexpr (k < 64) { tmp = _mm256_and_si256(tmp, avx_nn_k_mask); }
			const __m256i tmp_pop = popcount_avx2_64(tmp);
	
			//const __m256i gt_mask = _mm256_cmpgt_epi64(mask, tmp_pop);
			const __m256i gt_mask = _mm256_cmpeq_epi64(avx_nn_weight64, tmp_pop);
			const int wt = _mm256_movemask_pd((__m256d) gt_mask);
			ASSERT(wt < 1u << 4u);
			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap<4>(wt, L + ctr, org_ptr);
			}
		}
	
	
		return ctr;
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint32_t
	/// NOTE: dont call this function at first.
	/// NOTE: the current implementation will overflow the given e1, e2 in multiples of u*4
	/// NOTE: make sure that `new_e1` and `new_e2` are zero
	/// \tparam limb current limb
	/// \tparam u number of checks to unroll
	/// \param e1 end of List L1
	/// \param e2 end of List L2
	/// \param new_e1 new end of List L1
	/// \param new_e2 new end of List L2
	/// \param z random element to match on
	template<const uint32_t limb, const uint32_t u>
	void avx2_sort_nn_on_double32(const size_t e1,
								  const size_t e2,
								  size_t &new_e1,
								  size_t &new_e2,
								  const uint32_t z) noexcept {
		static_assert(u <= 8);
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(e2 <= LIST_SIZE);
		ASSERT(k <= 32);
		ASSERT(new_e1 == 0);
		ASSERT(new_e2 == 0);
		ASSERT(dk <= 16);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi32(z);
		alignas(32) const __m256i offset = _mm256_setr_epi32(0*enl, 1*enl, 2*enl, 3*enl,
		                                                     4*enl, 5*enl, 6*enl, 7*enl);

		/// NOTE: I need 2 ptrs tracking the current position, because of the limb shift
		Element *ptr_L1 = (Element *)(((uint8_t *)L1) + limb*4);
		Element *org_ptr_L1 = L1;
		Element *ptr_L2 = (Element *)(((uint8_t *)L2) + limb*4);
		Element *org_ptr_L2 = L2;

		constexpr uint32_t off = 8*u;
		const size_t min_e = (std::min(e1, e2) + off - 1);
		for (; i+off <= min_e; i += off, ptr_L1 += off, org_ptr_L1 += off,
		                                 ptr_L2 += off, org_ptr_L2 += off) {
			uint64_t wt_L1 = 0, wt_L2 = 0;

			#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				/// load the left list
				__m256i ptr_tmp_L1 = _mm256_i32gather_epi32(ptr_L1 +  8*j, offset, 8);
				ptr_tmp_L1 = _mm256_xor_si256(ptr_tmp_L1, z256);
				if constexpr (k < 32) { ptr_tmp_L1 = _mm256_and_si256(ptr_tmp_L1, avx_nn_k_mask); }
				ptr_tmp_L1 = popcount_avx2_32(ptr_tmp_L1);
				wt_L1 ^= compare_nn_on32(ptr_tmp_L1) << (8u * j);

				/// load the right list
				__m256i ptr_tmp_L2 = _mm256_i32gather_epi32(ptr_L2 +  8*j, offset, 8);
				ptr_tmp_L2 = _mm256_xor_si256(ptr_tmp_L2, z256);
				if constexpr (k < 32) { ptr_tmp_L2 = _mm256_and_si256(ptr_tmp_L2, avx_nn_k_mask); }
				ptr_tmp_L2 = popcount_avx2_32(ptr_tmp_L2);
				wt_L2 ^= compare_nn_on32(ptr_tmp_L2) << (8u * j);
			}

			if (wt_L1) {
				new_e1 += swap_ctz(wt_L1, L1 + new_e1, org_ptr_L1);
			}

			if (wt_L2) {
				new_e2 += swap_ctz(wt_L2, L2 + new_e2, org_ptr_L2);
			}

			ASSERT(new_e1 <= LIST_SIZE);
			ASSERT(new_e2 <= LIST_SIZE);
		}

		// tail work
		// #pragma unroll 4
		//size_t i2 = i;
		//for (; i < e1; i += 8, ptr_L1 += 8, org_ptr_L1 += 8) {
		//	const __m256i ptr_tmp = _mm256_i32gather_epi32(ptr_L1, offset, 8);
		//	__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
		//	if constexpr (k < 32) { tmp = _mm256_and_si256(tmp, avx_nn_k_mask); }
		//	const __m256i tmp_pop = popcount_avx2_32(tmp);
		//	const int wt = compare_nn_on32(tmp_pop);
		//	if (wt) {
		//		new_e1 += swap<8>(wt, L1 + new_e1, org_ptr_L1);
		//	}
		//}
		//for (; i2 < e2; i2 += 8, ptr_L2 += 8, org_ptr_L2 += 8) {
		//	const __m256i ptr_tmp = _mm256_i32gather_epi32(ptr_L2, offset, 8);
		//	__m256i tmp = _mm256_xor_si256(ptr_tmp, z256);
		//	if constexpr (k < 32) { tmp = _mm256_and_si256(tmp, avx_nn_k_mask); }
		//	const __m256i tmp_pop = popcount_avx2_32(tmp);
		//	const int wt = compare_nn_on32(tmp_pop);
		//	if (wt) {
		//		new_e2 += swap<8>(wt, L2 + new_e2, org_ptr_L1);
		//	}
		//}
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint64_t
	/// NOTE: dont call this function at first.
	/// NOTE: the current implementation will overflow the given e1, e2 in multiples of u*4
	/// NOTE: make sure that `new_e1` and `new_e2` are zero
	/// \tparam limb current limb
	/// \tparam u number of checks to unroll
	/// \param e1 end of List L1
	/// \param e2 end of List L2
	/// \param new_e1 new end of List L1
	/// \param new_e2 new end of List L2
	/// \param z random element to match on
	template<const uint32_t limb, const uint32_t u>
	void avx2_sort_nn_on_double64(const size_t e1,
							      const size_t e2,
								  size_t &new_e1,
	                              size_t &new_e2,
								  const uint64_t z) noexcept {
		static_assert(u <= 16);
		static_assert(u > 0);
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 64);
		ASSERT(k > 32);
		ASSERT(new_e1 == 0);
		ASSERT(new_e2 == 0);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const __m256i z256 = _mm256_set1_epi64x(z);
		alignas(32) const __m256i offset = _mm256_setr_epi64x(0*enl, 1*enl, 2*enl, 3*enl);

		/// NOTE: I need 2 ptrs tracking the current position, because of the limb shift
		Element *ptr_L1 = (Element *)(((uint8_t *)L1) + limb*8);
		Element *org_ptr_L1 = L1;
		Element *ptr_L2 = (Element *)(((uint8_t *)L2) + limb*8);
		Element *org_ptr_L2 = L2;

		const size_t min_e = (std::min(e1, e2) + 4*u - 1);
		for (; i+(4*u) <= min_e; i += (4*u), ptr_L1 += (4*u), org_ptr_L1 += (4*u),
											 ptr_L2 += (4*u), org_ptr_L2 += (4*u)) {
			uint32_t wt_L1 = 0, wt_L2 = 0;

			#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				__m256i ptr_tmp_L1 = _mm256_i64gather_epi64(ptr_L1 +  4*j, offset, 8);
				ptr_tmp_L1 = _mm256_xor_si256(ptr_tmp_L1, z256);
				if constexpr (k < 64) { ptr_tmp_L1 = _mm256_and_si256(ptr_tmp_L1, avx_nn_k_mask); }
				ptr_tmp_L1 = popcount_avx2_64(ptr_tmp_L1);
				wt_L1 ^= compare_nn_on64(ptr_tmp_L1) << (4u * j);

				__m256i ptr_tmp_L2 = _mm256_i64gather_epi64(ptr_L2 +  4*j, offset, 8);
				ptr_tmp_L2 = _mm256_xor_si256(ptr_tmp_L2, z256);
				if constexpr (k < 64) { ptr_tmp_L2 = _mm256_and_si256(ptr_tmp_L2, avx_nn_k_mask); }
				ptr_tmp_L2 = popcount_avx2_64(ptr_tmp_L2);
				wt_L2 ^= compare_nn_on64(ptr_tmp_L2) << (4u * j);
			}

			if (wt_L1) {
				new_e1 += swap_ctz(wt_L1, L1 + new_e1, org_ptr_L1);
			}

			if (wt_L2) {
				new_e2 += swap_ctz(wt_L2, L2 + new_e2, org_ptr_L2);
			}
		}

	}

	/// runs the Esser, Kübler, Zweydinger NN on a the two lists
	/// dont call ths function normally.
	/// \tparam level current level of the
	/// \param e1 end of list L1
	/// \param e2 end of list L2
	template<const uint32_t level>
	void avx2_nn_internal(const size_t e1,
	                      const size_t e2) noexcept {
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(e2 <= LIST_SIZE);

		/// NOTE: is this really the only wat to get around the restriction
		/// of partly specialized template functions?
		if constexpr (level < 1) {
			bruteforce(e1, e2);
		} else {
			size_t new_e1 = 0,
			       new_e2 = 0;

			if constexpr (k <= 32) {
				const uint32_t z =  fastrandombytes_uint64();
				// avx2_sort_nn_on_double32<r-level, 4>(e1, e2, new_e1, new_e2, z);
				new_e1 = avx2_sort_nn_on32<r - level>(e1, z, L1);
				new_e2 = avx2_sort_nn_on32<r - level>(e2, z, L2);
			} else if constexpr (k <= 64) {
				const uint64_t z = fastrandombytes_uint64();
				//avx2_sort_nn_on_double64<r-level, 4>(e1, e2, new_e1, new_e2, z);
				new_e1 = avx2_sort_nn_on64<r - level>(e1, z, L1);
				new_e2 = avx2_sort_nn_on64<r - level>(e2, z, L2);
			} else  {
				ASSERT(false);
			}

			ASSERT(new_e1 <= LIST_SIZE);
			ASSERT(new_e2 <= LIST_SIZE);
			ASSERT(new_e1 <= e1);
			ASSERT(new_e2 <= e2);



			if (unlikely(new_e1 == 0 or new_e2 == 0)) {
				return;
			}

			if ((new_e1 < BRUTEFORCE_THRESHHOLD) || (new_e2 < BRUTEFORCE_THRESHHOLD)) {
#ifdef DEBUG
				std::cout << level << " " << new_e1 << " " << e1 << " " << new_e2 << " " << e2 << "\n";
#endif
				bruteforce(new_e1, new_e2);
				return;
			}

			for (uint32_t i = 0; i < N; i++) {
#ifdef DEBUG
				if (i == 0) {
					std::cout << level << " " << i << " " << new_e1 << " " << e1 << " " << new_e2 << " " << e2 << "\n";
				}
#endif

				/// predict the future:
				if constexpr (USE_REARRANGE) {
					const uint32_t next_size = uint32_t(survive_prob * double(new_e1));
					//printf("%d\n", next_size);
					if (next_size < BUCKET_SIZE) {
						size_t new_new_e1, new_new_e2;
						if constexpr (k <= 32) {
							const uint32_t z =  fastrandombytes_uint64();
							new_new_e1 = avx2_sort_nn_on32_rearrange<r - level + 1, BUCKET_SIZE>(new_e1, z, L1, LB);
							new_new_e2 = avx2_sort_nn_on32_rearrange<r - level + 1, BUCKET_SIZE>(new_e2, z, L2, RB);
						} else if constexpr (k <= 64) {
							const uint64_t z = fastrandombytes_uint64();
							new_new_e1 = avx2_sort_nn_on64_rearrange<r - level + 1, BUCKET_SIZE>(new_e1, z, L1, LB);
							new_new_e2 = avx2_sort_nn_on64_rearrange<r - level + 1, BUCKET_SIZE>(new_e2, z, L2, RB);
						} else {
							ASSERT(false);
						}

						/// Now bruteforce the (rearranges) buckets
						bruteforce_avx2_256_64_4x4_rearrange<BUCKET_SIZE>(new_new_e1, new_new_e2);
					}
				} else {
					/// normal code path
					avx2_nn_internal<level - 1>(new_e1, new_e2);
				}

				if (unlikely(solutions_nr)){
#if DEBUG
					std::cout << "sol: " << level << " " << i << " " << new_e1 << " " << new_e2 << "\n";
#endif
					break;
				}
			}
		}
	}

	/// core entry function
	/// \param e1
	/// \param e2
	void avx2_nn(const size_t e1=LIST_SIZE,
	             const size_t e2=LIST_SIZE) noexcept {
		//config.print();
		//madvise(L1, LIST_SIZE*sizeof(T)*ELEMENT_NR_LIMBS, POSIX_MADV_WILLNEED | POSIX_MADV_SEQUENTIAL | MADV_HUGEPAGE);
		//madvise(L2, LIST_SIZE*sizeof(T)*ELEMENT_NR_LIMBS, POSIX_MADV_WILLNEED | POSIX_MADV_SEQUENTIAL | MADV_HUGEPAGE);
		constexpr size_t P = 1;//n;//256ull*256ull*256ull*256ull;

		for (size_t i = 0; i < P*N; ++i) {
			if constexpr(32 < n and n <= 256) {
				avx2_nn_internal<r>(e1, e2);
	 		} else {
				ASSERT(false);
			}
			if (solutions_nr > 0) {
				//std::cout << "outer: " << i << "\n";
				break;
			}
		}
	}

#ifdef __AVX512F__
	alignas(64) const __m512i avx512_weight32 = _mm512_set1_epi32(d+1);
	alignas(64) const __m512i avx512_weight64 = _mm512_set1_epi64(d+1);
	alignas(64) const __m512i avx512_exact_weight32 = _mm512_set1_epi32(d);
	alignas(64) const __m512i avx512_exact_weight64 = _mm512_set1_epi64(d);

	///
	/// \param in
	/// \return
	__m512i	popcount_avx512_32(const __m512i in) {
		return _mm512_popcnt_epi32(in);
	}

	///
	/// \param in
	/// \return
	__m512i	popcount_avx512_64(const __m512i in) {
		return _mm512_popcnt_epi64(in);
	}

	///
	/// NOTE: upper bound `d` is inclusive
	/// \param in
	/// \return
	template<bool exact=false>
	int compare_512_32(const __m512i in1, const __m512i in2) {
		if constexpr(EXACT) {
			return _mm512_cmpeq_epi32_mask(in1, in2);
		} else {
			const __m512i tmp1 = _mm512_xor_si512(in1, in2);
			const __m512i pop = popcount_avx512_32(tmp1);

			if constexpr (exact) {
				return _mm512_cmpeq_epi32_mask(avx512_exact_weight32, pop);
			} else  {
				return _mm512_cmpgt_epi32_mask(avx512_weight32, pop);
			}
		}
	}

	///
	/// NOTE: upper bound `d` is inclusive
	/// \param in
	/// \return
	template<bool exact=false>
	int compare_512_64(const __m512i in1, const __m512i in2) {
		if constexpr(EXACT) {
			return _mm512_cmpeq_epi64_mask(in1, in2);
		} else {
			const __m512i tmp1 = _mm512_xor_si512(in1, in2);
			const __m512i pop = popcount_avx512_32(tmp1);

			if constexpr (exact) {
				return _mm512_cmpeq_epi64_mask(avx512_exact_weight64, pop);
			} else  {
				return _mm512_cmpgt_epi64_mask(avx512_weight64, pop);
			}
		}
	}

	///
	/// \param e1
	/// \param e2
	void bruteforce_avx512_32_16x16(const size_t e1,
	                               const size_t e2) noexcept {
		ASSERT(n <= 32);
		ASSERT(ELEMENT_NR_LIMBS == 1);
		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		ASSERT(e1 >= 64);
		ASSERT(e2 >= 64);

		ASSERT(d < 16);

		__m512i *ptr_l = (__m512i *)L1;
		const __m512i perm = _mm512_setr_epi32(15, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14);

		for (size_t i = 0; i < (e1+15)/16; i++) {
			const __m512i li = _mm512_load_si512(ptr_l);
			__m512i *ptr_r = (__m512i *)L2;

			for (size_t j = 0; j < (e2+15)/16; j++) {
				__m512i ri = _mm512_load_si512(ptr_r);
				const int m = compare_512_32(li, ri);

				if (m) {
					const size_t jprime = j*16;
					const size_t iprime = i*16;

					if (compare_u64_ptr((T *) (L1 + iprime), (T *)(L2 + jprime))) {
						// TODO
						printf("k\n");
					}
				}

				// shuffle 
				ri = _mm512_permutexvar_epi32(perm, ri);
			}
 		}
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint32_t
	/// NOTE: dont call this function at first.
	/// NOTE: the current implementation will overflow the given e1, e2 in multiples of u*4
	/// NOTE: make sure that `new_e1` and `new_e2` are zero
	/// \tparam limb current limb
	/// \tparam u number of checks to unroll
	/// \param e1 end of List L1
	/// \param e2 end of List L2
	/// \param new_e1 new end of List L1
	/// \param new_e2 new end of List L2
	/// \param z random element to match on
	template<const uint32_t limb, const uint32_t u>
	void avx512_sort_nn_on_double32(const size_t e1,
								    const size_t e2,
								    size_t &new_e1,
								    size_t &new_e2,
								    const uint64_t z) noexcept {
		static_assert(u <= 8);
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(e2 <= LIST_SIZE);
		ASSERT(k <= 32);
		ASSERT(new_e1 == 0);
		ASSERT(new_e2 == 0);
		ASSERT(dk <= 16);


	}
#endif
};

#endif//NN_CODE_WINDOWED_AVX2_H
