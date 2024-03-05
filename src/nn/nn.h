#ifndef CRYPTANALYSISLIB_NN_H
#define CRYPTANALYSISLIB_NN_H

#include <array>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "alloc/alloc.h"
#include "helper.h"
#include "popcount/popcount.h"
#include "random.h"
#include "simd/simd.h"


using namespace cryptanalysislib;

/// configuration for the following algorithm: https://arxiv.org/abs/2102.02597
class NN_Config {
private:
	// disable the normal constructor
	NN_Config() : n(0), r(0), N(0), k(0), d(0), dk(0), dk_bruteforce_weight(0),
	              dk_bruteforce_size(0),
	              LIST_SIZE(0), epsilon(0), BRUTEFORCE_THRESHOLD(0) {}

public:
	const uint32_t n,            // length of the input vectors
	        r,                   // number of limbs to separate n on (number of levels)
	        N,                   // number of leaves per leve
	        k,                   // size of each limb
	        d,                   // weight difference to match on the golden solution
	        dk,                  // weight difference to match on each limb
	        dk_bruteforce_weight,// max. weight different to allow on each limb during the bruteforce step
	        dk_bruteforce_size,  // number of bits to check `dk_bruteforce_weight` on, should be 32/64
	        LIST_SIZE,           // list size on scale
	        epsilon,
	        BRUTEFORCE_THRESHOLD;// max. number of elements to be in both lists, until we switch to the bruteforce

	// special bucket config. If set to true the algorithm will switch from a
	// column approach to a row approach if the expected number of elements in
	// the next iteration will be below `BUCKET_SIZE`.
	// The normal approach: column approach: is fetch the current limb of the
	// row to analyse via a gather instruction and search for close elements
	// on these columns. In the `row` approach the remaining < `BUCKET_SIZE`
	// elements are saved in a transposed fashion. Thus we can now to a aligned
	// load to fetch the first 4-64-limbs (= 64 columns) of 4 distinct rows with
	// one simd instruction, and hence to a fast check if the golden element
	// is in the buckets.
	const bool USE_REARRANGE = false;
	const double survive_prob = 0.025;
	const uint32_t BUCKET_SIZE = 1024;

	constexpr NN_Config(const uint32_t n,
	                    const uint32_t r,
	                    const uint32_t N,
	                    const uint32_t k,
	                    const uint32_t ls,
	                    const uint32_t dk,
	                    const uint32_t d,
	                    const uint32_t epsilon,
	                    const uint32_t bf,
	                    const uint32_t dk_bruteforce_weight = 0,
	                    const uint32_t dk_bruteforce_size = 0) noexcept : n(n), r(r), N(N), k(k), d(d), dk(dk), dk_bruteforce_weight(dk_bruteforce_weight),
	                                                                      dk_bruteforce_size(dk_bruteforce_size),
	                                                                      LIST_SIZE(ls), epsilon(epsilon), BRUTEFORCE_THRESHOLD(bf){};

	///
	/// helper function, only printing the internal parameters
	void print() const noexcept {
		std::cout
		        << "{ \"n\": " << n
		        << ", \"r\": " << r
		        << ", \"N\":" << N
		        << ", \"k\":" << k
		        << ", \"|L|\": " << LIST_SIZE
		        << ", \"dk\n:" << dk
		        << ", \"dk_bruteforce_size\": " << dk_bruteforce_size
		        << ", \"dk_bruteforce_weight\": " << dk_bruteforce_weight
		        << ", \"d\": " << d
		        << ", \"e\": " << epsilon
		        << ", \"bf\": " << BRUTEFORCE_THRESHOLD
		        << ", \"k\": " << n / r
		        << std::endl;
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

	/// special mask needed if k is now a multiple of 32/64
	/// to remove all uneeded bits from a limb.
	alignas(32) constexpr static uint32x8_t SIMD_NN_K_MASK32 = uint32x8_t::set1((uint32_t) (1ul << (k % 32u)) - 1ul);
	alignas(32) constexpr static uint64x4_t SIMD_NN_K_MASK64 = uint64x4_t::set1((1ul << (k % 64u)) - 1ul);

	/// Base types
	using T = uint64_t;// NOTE do not change.
	constexpr static size_t T_BITSIZE = sizeof(T) * 8;
	constexpr static size_t ELEMENT_NR_LIMBS = (n + T_BITSIZE - 1) / T_BITSIZE;
	using Element = T[ELEMENT_NR_LIMBS];

	/// The Probability that a element will end up in the subsequent list.
	constexpr static bool USE_REARRANGE = config.USE_REARRANGE;
	constexpr static double survive_prob = config.survive_prob;
	constexpr static uint32_t BUCKET_SIZE = config.BUCKET_SIZE;
	alignas(64) uint64_t LB[USE_REARRANGE ? BUCKET_SIZE * ELEMENT_NR_LIMBS : 1u];
	alignas(64) uint64_t RB[USE_REARRANGE ? BUCKET_SIZE * ELEMENT_NR_LIMBS : 1u];

	// if set to true the final solution  is accepted if
	// its weight is <=w.
	// if false the final solution is only correct if == w
	constexpr static bool FINAL_SOL_WEIGHT = true;

	// if set to true, speciallizes functions which compute both lists at
	// the sametime
	constexpr static bool USE_DOUBLE_SEARCH = false;

	// instance
	alignas(64) Element *L1 = nullptr,
	                    *L2 = nullptr;

	// solution
	size_t solution_l = 0, solution_r = 0, solutions_nr = 0;
	std::vector<std::pair<size_t, size_t>> solutions;

	~NN() noexcept {
		/// probably its ok to assert some stuff here
		static_assert(k <= n);
		static_assert(dk_bruteforce_size >= dk_bruteforce_weight);

		if (L1) { free(L1); }
		if (L2) { free(L2); }
	}

	/// transposes the two lists into the two buckets
	void transpose(const size_t list_size) {
		if constexpr (!USE_REARRANGE) {
			ASSERT(false);
			return;
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
				wt += popcount::popcount(e1[i]);
			}

			e1[ELEMENT_NR_LIMBS - 1] = fastrandombytes_uint64() & mask;
			e2[ELEMENT_NR_LIMBS - 1] = 0;
			wt += popcount::popcount(e1[ELEMENT_NR_LIMBS - 1]);

			if (wt < d) {
				return;
			}
		}
	}

	/// chooses e1 completely random and e2 a weight d vector.
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
			uint32_t pos = i;
			while (pos == i) {
				pos = fastrandombytes_uint64() % (n - i - 1);
			}

			const uint32_t from_limb = 0;
			const uint32_t from_pos = i;
			const T from_mask = 1ull << from_pos;

			const uint32_t to_limb = pos / T_BITSIZE;
			const uint32_t to_pos = pos % T_BITSIZE;
			const T to_mask = 1ull << to_pos;

			const T from_read = (e2[from_limb] & from_mask) >> from_pos;
			const T to_read = (e2[to_limb] & to_mask) >> to_pos;
			//e2[to_limb]   ^= (-from_read ^ e2[to_limb]) & (1ul << to_pos);
			//e2[from_limb] ^= (-to_read ^ e2[from_limb]) & (1ul << from_pos);
			e2[to_limb] = ((e2[to_limb] & ~to_mask) | (from_read << to_pos));
			e2[from_limb] = ((e2[from_limb] & ~from_mask) | (to_read << from_pos));
		}

		uint32_t wt = 0;
		for (uint32_t i = 0; i < ELEMENT_NR_LIMBS - 1; i++) {
			wt += popcount::popcount(e2[i]);
			e1[i] = fastrandombytes_uint64();
			e2[i] ^= e1[i];
		}

		wt += popcount::popcount(e2[ELEMENT_NR_LIMBS - 1]);
		ASSERT(wt == d);

		e1[ELEMENT_NR_LIMBS - 1] = fastrandombytes_uint64() & mask;
		e2[ELEMENT_NR_LIMBS - 1] ^= e1[ELEMENT_NR_LIMBS - 1];
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

	/// generates a special instance in which either the solution is
	/// zero (`create_zero==1`) or normal
	/// \param insert_sol
	void generate_special_instance(bool insert_sol = true, bool create_zero = true) noexcept {
		constexpr size_t list_size = (ELEMENT_NR_LIMBS * LIST_SIZE * sizeof(T));
		L1 = (Element *) cryptanalysislib::aligned_alloc(64, list_size);
		L2 = (Element *) cryptanalysislib::aligned_alloc(64, list_size);
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
		L1 = (Element *) cryptanalysislib::aligned_alloc(PAGE_SIZE, list_size);
		L2 = (Element *) cryptanalysislib::aligned_alloc(PAGE_SIZE, list_size);
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
		if (!insert_sol) {
			return;
		}

		// generate solution:
		solution_l = fastrandombytes_uint64() % LIST_SIZE;
		solution_r = fastrandombytes_uint64() % LIST_SIZE;
		DEBUG_MACRO(std::cout << "sols at: " << solution_l << " " << solution_r << "\n";)

		if constexpr (EXACT) {
			Element sol;
			generate_random_element(sol);

			// inject the solution
			for (uint32_t i = 0; i < ELEMENT_NR_LIMBS; ++i) {
				L1[solution_l][i] = sol[i];
				L2[solution_r][i] = sol[i];
			}
		} else {
			Element sol1 = {0}, sol2 = {0};
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
			wt += popcount::popcount(L1[li][i] ^ L2[lr][i]);
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
		if (solutions_nr == 0) {
			return false;
		}

		for (uint32_t i = 0; i < solutions_nr; i++) {
			bool equal = true;
			if constexpr (EXACT) {
				for (uint32_t j = 0; j < ELEMENT_NR_LIMBS; j++) {
					equal &= L1[solutions[i].first][j] == L2[solutions[i].second][j];
				}
			} else {
				uint32_t wt = 0;
				for (uint32_t j = 0; j < ELEMENT_NR_LIMBS; j++) {
					wt += popcount::popcount(L1[solutions[i].first][j] ^ L2[solutions[i].second][j]);
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
			return uint32_t(popcount::popcount(a ^ b)) <= d;
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
			return uint32_t(popcount::popcount(a ^ b)) <= d;
		}
	}

	/// copies `from` to `to` if the bit in `gt_mask` is set.
	/// \param wt int mask selecting the upper elements to shift down
	/// \param to
	/// \param from
	/// \return
	template<const uint32_t limit>
	inline size_t swap(uint32_t wt,
	                   Element *__restrict__ to,
	                   Element *__restrict__ from) const noexcept {
		ASSERT(wt < (1u << limit));

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
		const uint32_t bit_limit = popcount::popcount(wt);
		for (uint32_t i = 0; i < bit_limit; ++i) {
			const uint32_t pos = __builtin_ctz(wt);
			std::swap(to[i], from[pos]);

			// clear the set bit.
			wt ^= 1u << pos;
		}

		return bit_limit;
	}

	/// Helper function: used be `simd_sort_nn_on32_rearrange` or similar.
	/// Basically it transpose a set of possible solutions into a Bucket
	/// \tparam bucket_size max number of elements in a bucket
	/// \param wt solution indicies in `from`. E.g. a 1 in `wt` says that this position
	///			in `from` is a valid solution
	/// \param to: output bucket
	/// \param from: input
	/// \return number of solutions found
	template<const uint32_t bucket_size>
	constexpr inline size_t swap_ctz_rearrange(uint32_t wt,
	                                           T *__restrict__ to,
	                                           Element *__restrict__ from) const noexcept {
		if constexpr (!USE_REARRANGE) {
			ASSERT(false);
			return 0;
		}

		const uint32_t bit_limit = popcount::popcount(wt);
		for (uint32_t i = 0; i < bit_limit; ++i) {
			const uint32_t pos = __builtin_ctz(wt);

#pragma unroll
			for (uint32_t j = 0; j < ELEMENT_NR_LIMBS; j++) {
				ASSERT(i + j * bucket_size < (ELEMENT_NR_LIMBS * bucket_size));
				to[i + j * bucket_size] = from[pos][j];
			}

			// clear the set bit.
			wt ^= 1u << pos;
		}

		return bit_limit;
	}


	/// executes the comparison operator in the NN subroutine on 32 bit limbs
	/// \param tmp input: a register containing the popcount of 8 limbs
	/// \return a integer mask with only at most the first 8 bits are set:
	/// 		bit x == 1 <=> (limb x == dk || limb x < dk || dk-eps < limb x < dk+eps)
	///			depending on the exact config
	constexpr inline uint32_t compare_nn_on32(const uint32x8_t tmp) const noexcept {
		// sanity check: we only can compute one of those settings.
		static_assert(NN_EQUAL + NN_LOWER + NN_BOUNDS == 1);

		if constexpr (NN_EQUAL) {
			static_assert(epsilon == 0);
			constexpr uint32x8_t avx_nn_weight32 = uint32x8_t::set1(dk);
			return tmp == avx_nn_weight32;
		}

		if constexpr (NN_LOWER) {
			static_assert(epsilon == 0);
			constexpr uint32x8_t avx_nn_weight32 = uint32x8_t::set1(dk + NN_LOWER);
			return avx_nn_weight32 > tmp;
		}

		if constexpr (NN_BOUNDS) {
			static_assert(epsilon > 0);
			static_assert(epsilon < dk);

			constexpr uint32x8_t avx_nn_weight32 = uint32x8_t::set1(dk + NN_LOWER + epsilon);
			constexpr uint32x8_t avx_nn_weight_lower32 = uint32x8_t::set1(dk - epsilon);

			const uint32x8_t lt_mask = uint32x8_t::gt_(tmp, avx_nn_weight_lower32);
			const uint32x8_t gt_mask = uint32x8_t::gt_(avx_nn_weight32, tmp);
			return uint32x8_t::move(lt_mask & gt_mask);
		}
	}

	/// executes the comparison operator in the NN subroutine on 64 bit limbs
	/// \param tmp input: a register containing the popcount of 4 limbs
	/// \return a integer mask with only at most the first 4 bits are set:
	/// 		bit x == 1 <=> (limb x == dk || limb x < dk || dk-eps < limb x < dk+eps)
	///			depending on the exact config
	constexpr inline uint32_t compare_nn_on64(const uint64x4_t tmp) const noexcept {
		static_assert(NN_EQUAL + NN_LOWER + NN_BOUNDS == 1);
		if constexpr (NN_EQUAL) {
			constexpr uint64x4_t avx_nn_weight64 = uint64x4_t::set1(dk);
			return avx_nn_weight64 == tmp;
		}

		if constexpr (NN_LOWER) {
			constexpr uint64x4_t avx_nn_weight64 = uint64x4_t::set1(dk + NN_LOWER);
			return avx_nn_weight64 > tmp;
		}

		if constexpr (NN_BOUNDS) {
			static_assert(epsilon > 0);
			static_assert(epsilon < dk);
			constexpr uint64x4_t avx_nn_weight64 = uint64x4_t::set1(dk + NN_LOWER + epsilon);
			constexpr uint64x4_t avx_nn_weight_lower64 = uint64x4_t::set1(dk - epsilon);

			const uint64x4_t lt_mask = uint64x4_t::gt_(avx_nn_weight_lower64, tmp);
			const uint64x4_t gt_mask = uint64x4_t::gt_(avx_nn_weight64, tmp);
			return uint64x4_t::move(lt_mask & gt_mask);
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
				wt += popcount::popcount(a[i] ^ b[i]);
			}

			ASSERT(!(a[ELEMENT_NR_LIMBS - 1] & mask));
			ASSERT(!(b[ELEMENT_NR_LIMBS - 1] & mask));

			if constexpr (FINAL_SOL_WEIGHT) {
				return wt <= d;
			} else {
				return wt == d;
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
			bruteforce_simd_64_uxv<4, 4>(e1, e2);
		} else if constexpr (64 < n and n <= 128) {
			bruteforce_simd_128_32_2_uxv<4, 4>(e1, e2);
		} else if constexpr (128 < n and n <= 256) {
			// TODO optimal value
			if (e1 < 10 && e2 < 10) {
				bruteforce_256(e1, e2);
				return;
			}

			// in the low weight case with have better implementations
			if constexpr (d < 16) {
				bruteforce_simd_256_64_4x4(e1, e2);
				return;
			}

			// generic best implementations for every weight
			bruteforce_simd_256_64_4x4(e1, e2);
		} else {
			ASSERT(false);
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
					if (popcount::popcount(L1[i][limb_pos] ^ L2[j][limb_pos]) <= d) {
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
		using Element2 = uint32_t[3];
		Element2 *LL1 = (Element2 *) L1;
		Element2 *LL2 = (Element2 *) L2;
		for (size_t i = s1; i < e1; i++) {
			for (size_t j = s2; j < e2; j++) {
				if constexpr (EXACT) {
					const uint32_t t = (LL1[i][0] == LL2[j][0]) & (LL1[i][1] == LL2[j][1]) & (LL1[i][1] == LL2[j][1]);
					if (t) {
						found_solution(i, j);
					}
				} else {
					const uint32_t t = (popcount::popcount(LL1[i][0] ^ LL2[j][0]) <= d) +
					                   (popcount::popcount(LL1[i][1] ^ LL2[j][1]) <= d) +
					                   (popcount::popcount(LL1[i][2] ^ LL2[j][2]) <= d);
					if (t == 3) {
						solutions.resize(solutions_nr + 1);
						solutions[solutions_nr++] = std::pair<size_t, size_t>{i, j};
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


	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint32
	/// \param: e1 end index
	/// \param: random value z
	template<const uint32_t limb>
	size_t simd_sort_nn_on32_simple(const size_t e1,
	                                const uint32_t z,
	                                Element *__restrict__ L) const noexcept {
		static_assert(sizeof(T) == 8);
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 32);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		const size_t s1 = 0;
		alignas(32) const uint32x8_t z256 = uint32x8_t::set1(z);
		alignas(32) constexpr uint32x8_t offset = uint32x8_t::setr(0 * enl, 1 * enl, 2 * enl, 3 * enl,
		                                                           4 * enl, 5 * enl, 6 * enl, 7 * enl);

		// NR of partial solutions found
		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *) (((uint8_t *) L) + limb * 4);
		Element *org_ptr = L;

		for (size_t i = s1; i < (e1 + 7) / 8; i++, ptr += 8, org_ptr += 8) {
			const uint32x8_t ptr_tmp = uint32x8_t::template gather<8>(ptr, offset);
			uint32x8_t tmp = ptr_tmp ^ z256;
			if constexpr (k < 32) { tmp &= SIMD_NN_K_MASK32; }
			const uint32x8_t tmp_pop = uint32x8_t::popcnt(tmp);
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

	///
	/// \tparam limb
	/// \param e1
	/// \param z
	/// \param L
	/// \return
	template<const uint32_t limb>
	size_t simd_sort_nn_on32(const size_t e1,
	                         const uint32_t z,
	                         Element *__restrict__ L) noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 32);


		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const uint32x8_t z256 = uint32x8_t::set1(z);
		alignas(32) constexpr uint32x8_t offset = uint32x8_t::setr(0 * enl, 1 * enl, 2 * enl, 3 * enl,
		                                                           4 * enl, 5 * enl, 6 * enl, 7 * enl);
		// size of the output list
		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *) (((uint8_t *) L) + limb * 4);
		Element *org_ptr = L;

		constexpr uint32_t u = 4;
		constexpr uint32_t off = u * 8;
		for (; i + off <= e1; i += off, ptr += off, org_ptr += off) {
			uint32x8_t ptr_tmp0 = uint32x8_t::gather<8>(ptr + 0, offset);
			ptr_tmp0 ^= z256;
			if constexpr (k < 32) { ptr_tmp0 &= SIMD_NN_K_MASK32; }
			uint32x8_t ptr_tmp1 = uint32x8_t::gather<8>(ptr + 8, offset);
			ptr_tmp1 ^= z256;
			if constexpr (k < 32) { ptr_tmp1 &= SIMD_NN_K_MASK32; }
			uint32x8_t ptr_tmp2 = uint32x8_t::gather<8>(ptr + 16, offset);
			ptr_tmp2 ^= z256;
			if constexpr (k < 32) { ptr_tmp2 &= SIMD_NN_K_MASK32; }
			uint32x8_t ptr_tmp3 = uint32x8_t::gather<8>(ptr + 24, offset);
			ptr_tmp3 ^= z256;
			if constexpr (k < 32) { ptr_tmp3 &= SIMD_NN_K_MASK32; }

			uint32_t wt = 0;
			uint32x8_t tmp_pop = uint32x8_t::popcnt(ptr_tmp0);
			wt = compare_nn_on32(tmp_pop);

			tmp_pop = uint32x8_t::popcnt(ptr_tmp1);
			wt ^= compare_nn_on32(tmp_pop) << 8u;

			tmp_pop = uint32x8_t::popcnt(ptr_tmp2);
			wt ^= compare_nn_on32(tmp_pop) << 16u;

			tmp_pop = uint32x8_t::popcnt(ptr_tmp3);
			wt ^= compare_nn_on32(tmp_pop) << 24u;

			if (wt) {
				ctr += swap_ctz(wt, L + ctr, org_ptr);
			}
		}

		// tail work
		// #pragma unroll 4
		for (; i + 8 < e1 + 7; i += 8, ptr += 8, org_ptr += 8) {
			const uint32x8_t ptr_tmp = uint32x8_t::gather<8>(ptr + 0, offset);
			uint32x8_t tmp = ptr_tmp ^ z256;
			if constexpr (k < 32) { tmp &= SIMD_NN_K_MASK32; }
			const uint32x8_t tmp_pop = uint32x8_t::popcnt(tmp);
			const uint32_t wt = compare_nn_on32(tmp_pop);

			ASSERT(wt < (1u << 8u));
			// now `wt` contains the incises of matches. Meaning if bit 1 in
			// `wt` is set (and bit 0 not), we need to swap the second (0 indexed)
			// uint64_t from L + ctr with the first element from L + i.
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
	size_t simd_sort_nn_on32_rearrange(const size_t e1,
	                                   const uint32_t z,
	                                   Element *__restrict__ L,
	                                   T *__restrict__ B) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 32);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const uint32x8_t z256 = uint32x8_t::set1(z);
		alignas(32) constexpr uint32x8_t offset = uint32x8_t::setr(0 * enl, 1 * enl, 2 * enl, 3 * enl,
		                                                           4 * enl, 5 * enl, 6 * enl, 7 * enl);

		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *) (((uint8_t *) L) + limb * 4);
		Element *org_ptr = L;

		constexpr uint32_t u = 4;
		constexpr uint32_t off = u * 8;
		for (; i + off <= e1; i += off, ptr += off, org_ptr += off) {
			uint32x8_t ptr_tmp0 = uint32x8_t::gather<8>(ptr + 0, offset);
			ptr_tmp0 ^= z256;
			if constexpr (k < 32) { ptr_tmp0 &= SIMD_NN_K_MASK32; }
			uint32x8_t ptr_tmp1 = uint32x8_t::gather<8>(ptr + 8, offset);
			ptr_tmp1 ^= z256;
			if constexpr (k < 32) { ptr_tmp1 &= SIMD_NN_K_MASK32; }
			uint32x8_t ptr_tmp2 = uint32x8_t::gather<8>(ptr + 16, offset);
			ptr_tmp2 ^= z256;
			if constexpr (k < 32) { ptr_tmp2 &= SIMD_NN_K_MASK32; }
			uint32x8_t ptr_tmp3 = uint32x8_t::gather<8>(ptr + 24, offset);
			ptr_tmp3 ^= z256;
			if constexpr (k < 32) { ptr_tmp3 &= SIMD_NN_K_MASK32; }

			uint32_t wt = 0;
			uint32x8_t tmp_pop = uint32x8_t::popcnt(ptr_tmp0);
			wt = compare_nn_on32(tmp_pop);

			tmp_pop = uint32x8_t::popcnt(ptr_tmp1);
			wt ^= compare_nn_on32(tmp_pop) << 8u;

			tmp_pop = uint32x8_t::popcnt(ptr_tmp2);
			wt ^= compare_nn_on32(tmp_pop) << 16u;

			tmp_pop = uint32x8_t::popcnt(ptr_tmp3);
			wt ^= compare_nn_on32(tmp_pop) << 24u;

			if (wt) {
				ctr += swap_ctz_rearrange<bucket_size>(wt, B + ctr, org_ptr);
			}
		}

		// tail work
		// #pragma unroll 4
		for (; i + 8 < e1 + 7; i += 8, ptr += 8, org_ptr += 8) {
			const uint32x8_t ptr_tmp = uint32x8_t::gather<8>(ptr + 0, offset);
			uint32x8_t tmp = ptr_tmp ^ z256;
			if constexpr (k < 32) { tmp &= SIMD_NN_K_MASK32; }
			const uint32x8_t tmp_pop = uint32x8_t::popcnt(tmp);
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
	size_t simd_sort_nn_on64_simple(const size_t e1,
	                                const uint64_t z,
	                                Element *__restrict__ L) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 64);
		ASSERT(k > 32);


		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		alignas(32) const uint64x4_t z256 = uint64x4_t::set1(z);
		alignas(32) constexpr uint64x4_t offset = uint64x4_t::setr(0 * enl, 1 * enl, 2 * enl, 3 * enl);

		size_t ctr = 0;

		/// NOTE: i need 2 ptr tracking the current position, because of the
		/// limb shift
		Element *ptr = (Element *) (((uint8_t *) L) + limb * 8);
		Element *org_ptr = L;

		// #pragma unroll 4
		for (; i < (e1 + 3) / 4; i++, ptr += 4, org_ptr += 4) {
			const uint64x4_t ptr_tmp = uint64x4_t::template gather<8>(ptr, offset);
			uint64x4_t tmp = ptr_tmp ^ z256;
			if constexpr (k < 64) { tmp &= SIMD_NN_K_MASK64; }
			const uint64x4_t tmp_pop = uint64x4_t::popcnt(tmp);
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


	///
	/// \tparam limb
	/// \param e1
	/// \param z
	/// \param L
	/// \return
	template<const uint32_t limb>
	size_t simd_sort_nn_on64(const size_t e1,
	                         const uint64_t z,
	                         Element *__restrict__ L) const noexcept {
		ASSERT(limb <= ELEMENT_NR_LIMBS);
		ASSERT(e1 <= LIST_SIZE);
		ASSERT(k <= 64);
		ASSERT(k > 32);

		/// just a shorter name, im lazy.
		constexpr uint32_t enl = ELEMENT_NR_LIMBS;

		size_t i = 0;
		const uint64x4_t z256 = uint64x4_t::set1(z);
		constexpr uint64x4_t offset = uint64x4_t::setr(0 * enl, 1 * enl, 2 * enl, 3 * enl);
		uint64x4_t tmp_pop;

		size_t ctr = 0;

		/// NOTE: I need 2 ptrs tracking the current position, because of the limb shift
		Element *ptr = (Element *) (((uint8_t *) L) + limb * 8);
		Element *org_ptr = L;

		constexpr uint32_t u = 8;
		constexpr uint32_t off = u * 4;
		for (; i + off <= e1; i += off, ptr += off, org_ptr += off) {
			uint64x4_t ptr_tmp0 = uint64x4_t::template gather<8>(ptr + 0, offset);
			ptr_tmp0 ^= z256;
			if constexpr (k < 64) { ptr_tmp0 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp1 = uint64x4_t::template gather<8>(ptr + 4, offset);
			ptr_tmp1 ^= z256;
			if constexpr (k < 64) { ptr_tmp1 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp2 = uint64x4_t::template gather<8>(ptr + 8, offset);
			ptr_tmp2 ^= z256;
			if constexpr (k < 64) { ptr_tmp2 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp3 = uint64x4_t::template gather<8>(ptr + 12, offset);
			ptr_tmp3 ^= z256;
			if constexpr (k < 64) { ptr_tmp3 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp4 = uint64x4_t::template gather<8>(ptr + 16, offset);
			ptr_tmp4 ^= z256;
			if constexpr (k < 64) { ptr_tmp4 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp5 = uint64x4_t::template gather<8>(ptr + 20, offset);
			ptr_tmp5 ^= z256;
			if constexpr (k < 64) { ptr_tmp5 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp6 = uint64x4_t::template gather<8>(ptr + 24, offset);
			ptr_tmp6 ^= z256;
			if constexpr (k < 64) { ptr_tmp6 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp7 = uint64x4_t::template gather<8>(ptr + 28, offset);
			ptr_tmp7 ^= z256;
			if constexpr (k < 64) { ptr_tmp7 &= SIMD_NN_K_MASK64; }


			tmp_pop = uint64x4_t::popcnt(ptr_tmp0);
			uint32_t wt = compare_nn_on64(tmp_pop);

			tmp_pop = uint64x4_t::popcnt(ptr_tmp1);
			wt ^= compare_nn_on64(tmp_pop) << 4u;

			tmp_pop = uint64x4_t::popcnt(ptr_tmp2);
			wt ^= compare_nn_on64(tmp_pop) << 8u;

			tmp_pop = uint64x4_t::popcnt(ptr_tmp3);
			wt ^= compare_nn_on64(tmp_pop) << 12u;

			tmp_pop = uint64x4_t::popcnt(ptr_tmp4);
			wt ^= compare_nn_on64(tmp_pop) << 16u;

			tmp_pop = uint64x4_t::popcnt(ptr_tmp5);
			wt ^= compare_nn_on64(tmp_pop) << 20u;

			tmp_pop = uint64x4_t::popcnt(ptr_tmp6);
			wt ^= compare_nn_on64(tmp_pop) << 24u;

			tmp_pop = uint64x4_t::popcnt(ptr_tmp7);
			wt ^= compare_nn_on64(tmp_pop) << 28u;
			ASSERT(uint64_t(wt) < (1ull << 32ull));
			ASSERT(ctr <= LIST_SIZE);
			ASSERT(ctr <= e1);

			if (wt) {
				ctr += swap_ctz(wt, L + ctr, org_ptr);
			}
		}

		// #pragma unroll 4
		for (; i < (e1 + 3) / 4; i++, ptr += 4, org_ptr += 4) {
			uint64x4_t ptr_tmp = uint64x4_t::template gather<8>(ptr, offset);
			ptr_tmp ^= z256;
			if constexpr (k < 64) { ptr_tmp &= SIMD_NN_K_MASK64; }
			const uint64x4_t tmp_pop = uint64x4_t::popcnt(ptr_tmp);
			const uint32_t wt = compare_nn_on64(tmp_pop) << 28u;
			ASSERT(wt < (1u << 4u));
			// now `wt` contains the incises of matches. Meaning if bit 1 in `wt` is set (and bit 0 not),
			// we need to swap the second (0 indexed) uint64_t from L + ctr with the first element from L + i.
			// The core problem is, that we need 64bit indices and not just 32bit
			if (wt) {
				ctr += swap_ctz(wt, L + ctr, org_ptr);
			}
		}

		return ctr;
	}

	/// NOTE: assumes T=uint64
	/// NOTE: only matches weight dk on uint64_t
	/// NOTE: hardcoded unroll parameter u=8
	/// NOTE: this version rearranges the elements in the list L into buckets
	/// 		thus transposing them into buckets
	/// \param: e1 end index
	/// \param: random value z
	/// \param: L: input list
	/// \param: B: output bucket (rearranged)
	template<const uint32_t limb, const uint32_t bucket_size>
	size_t simd_sort_nn_on64_rearrange(const size_t e1,
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
		const uint64x4_t z256 = uint64x4_t::set1(z);
		constexpr uint64x4_t offset = uint64x4_t::setr(0 * enl, 1 * enl, 2 * enl, 3 * enl);
		uint64x4_t tmp_pop;

		size_t ctr = 0;

		/// NOTE: I need 2 ptrs tracking the current position, because of the limb shift
		Element *ptr = (Element *) (((uint8_t *) L) + limb * 8);
		Element *org_ptr = L;

		constexpr uint32_t u = 8;
		constexpr uint32_t off = u * 4;
		for (; i + off <= e1; i += off, ptr += off, org_ptr += off) {
			uint64x4_t ptr_tmp0 = uint64x4_t::template gather<8>(ptr + 0, offset);
			ptr_tmp0 ^= z256;
			if constexpr (k < 64) { ptr_tmp0 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp1 = uint64x4_t::template gather<8>(ptr + 4, offset);
			ptr_tmp1 ^= z256;
			if constexpr (k < 64) { ptr_tmp1 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp2 = uint64x4_t::template gather<8>(ptr + 8, offset);
			ptr_tmp2 ^= z256;
			if constexpr (k < 64) { ptr_tmp2 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp3 = uint64x4_t::template gather<8>(ptr + 12, offset);
			ptr_tmp3 ^= z256;
			if constexpr (k < 64) { ptr_tmp3 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp4 = uint64x4_t::template gather<8>(ptr + 16, offset);
			ptr_tmp4 ^= z256;
			if constexpr (k < 64) { ptr_tmp4 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp5 = uint64x4_t::template gather<8>(ptr + 20, offset);
			ptr_tmp5 ^= z256;
			if constexpr (k < 64) { ptr_tmp5 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp6 = uint64x4_t::template gather<8>(ptr + 24, offset);
			ptr_tmp6 ^= z256;
			if constexpr (k < 64) { ptr_tmp6 &= SIMD_NN_K_MASK64; }
			uint64x4_t ptr_tmp7 = uint64x4_t::template gather<8>(ptr + 28, offset);
			ptr_tmp7 ^= z256;
			if constexpr (k < 64) { ptr_tmp7 &= SIMD_NN_K_MASK64; }


			tmp_pop = uint64x4_t::popcnt(ptr_tmp0);
			uint32_t wt = compare_nn_on64(tmp_pop);

			tmp_pop = uint64x4_t::popcnt(ptr_tmp1);
			wt ^= compare_nn_on64(tmp_pop) << 4u;

			tmp_pop = uint64x4_t::popcnt(ptr_tmp2);
			wt ^= compare_nn_on64(tmp_pop) << 8u;

			tmp_pop = uint64x4_t::popcnt(ptr_tmp3);
			wt ^= compare_nn_on64(tmp_pop) << 12u;

			tmp_pop = uint64x4_t::popcnt(ptr_tmp4);
			wt ^= compare_nn_on64(tmp_pop) << 16u;

			tmp_pop = uint64x4_t::popcnt(ptr_tmp5);
			wt ^= compare_nn_on64(tmp_pop) << 20u;

			tmp_pop = uint64x4_t::popcnt(ptr_tmp6);
			wt ^= compare_nn_on64(tmp_pop) << 24u;

			tmp_pop = uint64x4_t::popcnt(ptr_tmp7);
			wt ^= compare_nn_on64(tmp_pop) << 28u;
			//ASSERT(uint64_t(wt) < (1ull << 32ull));

			ASSERT(ctr <= LIST_SIZE);
			ASSERT(ctr <= e1);

			if (wt) {
				ctr += swap_ctz_rearrange<bucket_size>(wt, B + ctr, org_ptr);
			}
		}

		// #pragma unroll 4
		for (; i < (e1 + 3) / 4; i++, ptr += 4, org_ptr += 4) {
			uint64x4_t ptr_tmp = uint64x4_t::template gather<8>(ptr, offset);
			ptr_tmp ^= z256;
			if constexpr (k < 64) { ptr_tmp &= SIMD_NN_K_MASK64; }
			const uint64x4_t tmp_pop = uint64x4_t::popcnt(ptr_tmp);
			const int wt = compare_nn_on64(tmp_pop) << 28u;
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
	/// NOTE: only matches weight dk on uint32_t
	/// NOTE: dont call this function at first.
	/// NOTE: the current implementation will overflow the given e1, e2 in multiples of u*4
	/// NOTE: make sure that `new_e1` and `new_e2` are zero
	/// \tparam limb current limb
	/// \tparam u number of checks to unroll, <= 4
	/// \param e1 end of List L1
	/// \param e2 end of List L2
	/// \param new_e1 new end of List L1
	/// \param new_e2 new end of List L2
	/// \param z random element to match on
	template<const uint32_t limb, const uint32_t u>
	void simd_sort_nn_on_double32(const size_t e1,
	                              const size_t e2,
	                              size_t &new_e1,
	                              size_t &new_e2,
	                              const uint32_t z) noexcept {
		static_assert(u <= 4);
		static_assert(u > 0);
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
		alignas(32) const uint32x8_t z256 = uint32x8_t::set1(z);
		alignas(32) constexpr uint32x8_t offset = uint32x8_t::setr(0 * enl, 1 * enl, 2 * enl, 3 * enl,
		                                                           4 * enl, 5 * enl, 6 * enl, 7 * enl);

		/// NOTE: I need 2 ptrs tracking the current position, because of the limb shift
		Element *ptr_L1 = (Element *) (((uint8_t *) L1) + limb * 4);
		Element *org_ptr_L1 = L1;
		Element *ptr_L2 = (Element *) (((uint8_t *) L2) + limb * 4);
		Element *org_ptr_L2 = L2;

		constexpr uint32_t off = 8 * u;
		const size_t min_e = (std::min(e1, e2) + off - 1);
		for (; i + off <= min_e; i += off, ptr_L1 += off, org_ptr_L1 += off,
		                         ptr_L2 += off, org_ptr_L2 += off) {
			uint64_t wt_L1 = 0, wt_L2 = 0;

#pragma unroll u
			for (uint32_t j = 0; j < u; ++j) {
				/// load the left list
				uint32x8_t ptr_tmp_L1 = uint32x8_t::template gather<8>(ptr_L1 + 8 * j, offset);
				ptr_tmp_L1 ^= z256;
				if constexpr (k < 32) { ptr_tmp_L1 &= SIMD_NN_K_MASK32; }
				ptr_tmp_L1 = uint32x8_t::popcnt(ptr_tmp_L1);
				wt_L1 ^= compare_nn_on32(ptr_tmp_L1) << (8u * j);


				/// load the right list
				uint32x8_t ptr_tmp_L2 = uint32x8_t::template gather<8>(ptr_L2 + 8 * j, offset);
				ptr_tmp_L2 ^= z256;
				if constexpr (k < 32) { ptr_tmp_L2 &= SIMD_NN_K_MASK32; }
				ptr_tmp_L2 = uint32x8_t::popcnt(ptr_tmp_L2);
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
	void simd_sort_nn_on_double64(const size_t e1,
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
		alignas(32) const uint64x4_t z256 = uint64x4_t::set1(z);
		alignas(32) constexpr uint64x4_t offset = uint64x4_t::setr(0 * enl, 1 * enl, 2 * enl, 3 * enl);

		/// NOTE: I need 2 ptrs tracking the current position, because of the limb shift
		Element *ptr_L1 = (Element *) (((uint8_t *) L1) + limb * 8);
		Element *org_ptr_L1 = L1;
		Element *ptr_L2 = (Element *) (((uint8_t *) L2) + limb * 8);
		Element *org_ptr_L2 = L2;

		const size_t min_e = (std::min(e1, e2) + 4 * u - 1);
		for (; i + (4 * u) <= min_e; i += (4 * u), ptr_L1 += (4 * u), org_ptr_L1 += (4 * u),
		                             ptr_L2 += (4 * u), org_ptr_L2 += (4 * u)) {
			uint32_t wt_L1 = 0, wt_L2 = 0;

#pragma unroll
			for (uint32_t j = 0; j < u; ++j) {
				/// left list
				uint64x4_t ptr_tmp_L1 = uint64x4_t::template gather<8>(ptr_L1 + 4 * j, offset);
				ptr_tmp_L1 ^= z256;
				if constexpr (k < 64) { ptr_tmp_L1 &= SIMD_NN_K_MASK64; }
				ptr_tmp_L1 = uint64x4_t::popcnt(ptr_tmp_L1);
				wt_L1 ^= compare_nn_on64(ptr_tmp_L1) << (4u * j);

				/// right list
				uint64x4_t ptr_tmp_L2 = uint64x4_t::template gather<8>(ptr_L2 + 4 * j, offset);
				ptr_tmp_L2 ^= z256;
				if constexpr (k < 64) { ptr_tmp_L2 &= SIMD_NN_K_MASK64; }
				ptr_tmp_L2 = uint64x4_t::popcnt(ptr_tmp_L2);
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
	void simd_nn_internal(const size_t e1,
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
				const uint32_t z = fastrandombytes_uint64();
				if constexpr (USE_DOUBLE_SEARCH) {
					simd_sort_nn_on_double32<r - level, 4>(e1, e2, new_e1, new_e2, z);
				} else {
					new_e1 = simd_sort_nn_on32<r - level>(e1, z, L1);
					new_e2 = simd_sort_nn_on32<r - level>(e2, z, L2);
				}
			} else if constexpr (k <= 64) {
				const uint64_t z = fastrandombytes_uint64();
				if constexpr (USE_DOUBLE_SEARCH) {
					simd_sort_nn_on_double64<r - level, 4>(e1, e2, new_e1, new_e2, z);
				} else {
					new_e1 = simd_sort_nn_on64<r - level>(e1, z, L1);
					new_e2 = simd_sort_nn_on64<r - level>(e2, z, L2);
				}
			} else {
				ASSERT(false);
			}

			ASSERT(new_e1 <= LIST_SIZE);
			ASSERT(new_e2 <= LIST_SIZE);
			ASSERT(new_e1 <= e1);
			ASSERT(new_e2 <= e2);

			/// early exit if we filtered everything out
			if (unlikely(new_e1 == 0 or new_e2 == 0)) { return; }

			if ((new_e1 < BRUTEFORCE_THRESHHOLD) || (new_e2 < BRUTEFORCE_THRESHHOLD)) {
				DEBUG_MACRO(std::cout << level << " " << new_e1 << " " << e1 << " " << new_e2 << " " << e2 << "\n";)
				bruteforce(new_e1, new_e2);
				return;
			}

			for (uint32_t i = 0; i < N; i++) {
				if (i == 0) {
					DEBUG_MACRO(std::cout << level << " " << i << " " << new_e1 << " " << e1 << " " << new_e2 << " " << e2 << "\n";)
				}

				/// predict the future:
				if constexpr (USE_REARRANGE) {
					const uint32_t next_size = uint32_t(survive_prob * double(new_e1));
					if (next_size < BUCKET_SIZE) {
						size_t new_new_e1, new_new_e2;
						if constexpr (k <= 32) {
							const uint32_t z = fastrandombytes_uint64();
							new_new_e1 = simd_sort_nn_on32_rearrange<r - level + 1, BUCKET_SIZE>(new_e1, z, L1, LB);
							new_new_e2 = simd_sort_nn_on32_rearrange<r - level + 1, BUCKET_SIZE>(new_e2, z, L2, RB);
						} else if constexpr (k <= 64) {
							const uint64_t z = fastrandombytes_uint64();
							new_new_e1 = simd_sort_nn_on64_rearrange<r - level + 1, BUCKET_SIZE>(new_e1, z, L1, LB);
							new_new_e2 = simd_sort_nn_on64_rearrange<r - level + 1, BUCKET_SIZE>(new_e2, z, L2, RB);
						} else {
							ASSERT(false);
						}

						/// Now bruteforce the (rearranges) buckets
						bruteforce_simd_256_64_4x4_rearrange<BUCKET_SIZE>(new_new_e1, new_new_e2);
					}
				} else {
					/// normal code path
					simd_nn_internal<level - 1>(new_e1, new_e2);
				}

				if (unlikely(solutions_nr)) {
					DEBUG_MACRO(std::cout << "sol: " << level << " " << i << " " << new_e1 << " " << new_e2 << "\n";)
					break;
				}
			}
		}
	}

	/// runs the Esser, Kübler, Zweydinger NN on a the two lists
	/// dont call ths function normally.
	/// \tparam level current level of the
	/// \param e1 end of list L1
	/// \param e2 end of list L2
	template<const uint32_t level>
	void nn_internal(const size_t e1,
	                 const size_t e2) noexcept {
	}

	/// core entry function for the implementation of the Esser, Kuebler, Zweydinger NN algorithm
	/// \param e1 size of the left list
	/// \param e2 size of the right list
	constexpr void run(const size_t e1 = LIST_SIZE,
	                   const size_t e2 = LIST_SIZE) noexcept {
		constexpr size_t P = 1;//n;//256ull*256ull*256ull*256ull;

		for (size_t i = 0; i < P * N; ++i) {
			if constexpr (32 < n and n <= 256) {
				simd_nn_internal<r>(e1, e2);
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
	constexpr int compare_256_32(const uint32x8_t in1,
	                             const uint32x8_t in2) const noexcept {
		if constexpr (exact) {
			return uint32x8_t::cmp(in1, in2);
		}

		const uint32x8_t pop = uint32x8_t::popcnt(in1 ^ in2);

		if constexpr (dk_bruteforce_weight > 0) {
			if constexpr (EXACT) {
				constexpr uint32x8_t weight = uint32x8_t::set1(dk_bruteforce_weight);
				return weight == pop;
			} else {
				constexpr uint32x8_t weight = uint32x8_t::set1(dk_bruteforce_weight + 1u);
				return weight > pop;
			}

			// just to make sure that the compiler will not compiler the
			// following code
			return 0;
		}

		if constexpr (EXACT) {
			constexpr uint32x8_t weight = uint32x8_t::set1(d);
			return weight == pop;
		} else {
			constexpr uint32x8_t weight = uint32x8_t::set1(d + 1);
			return weight > pop;
		}
	}

	/// \tparam exact if set to true: a simple equality check is done
	/// \param in1 first input
	/// \param in2 second input
	/// \return compresses equality check:
	///			[bit0 = in1.v64[0] == in2.v64[0],
	/// 						....,
	/// 		 bit3 = in1.v64[3] == in2.v64[3]]
	template<const bool exact = false>
	[[nodiscard]] constexpr int compare_256_64(const uint64x4_t in1,
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
	/// NOTE: only compares a single 32 bit column of the list.
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
			const uint32_t *ptr_r = (uint32_t *) L2;

			for (size_t j = s2; j < s2 + (e2 + 7) / 8; ++j, ptr_r += 16) {
				/// NOTE the 8: this is needed, als internally all limbs are T=uint64_t
				const uint32x8_t ri = uint32x8_t::template gather<8>((const int *) ptr_r, loadr);
				const int m = compare_256_32(li, ri);

				if (m) {
					const size_t jprime = j * 8 + __builtin_ctz(m);
					if (compare_u64_ptr((T *) (L1 + i), (T *) (L2 + jprime))) {
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

		for (size_t i = s1; i < e1; ++i) {
			const uint64x4_t li = uint64x4_t::set1(L1[i][0]);

			/// NOTE: only possible because L2 is a continuous memory block
			T *ptr_r = (T *) L2;

			for (size_t j = s2; j < s2 + (e2 + 3) / 4; ++j, ptr_r += 4) {
				const uint64x4_t ri = uint64x4_t::load(ptr_r);
				const int m = compare_256_64(li, ri);

				if (m) {
					const size_t jprime = j * 4 + __builtin_ctz(m);

					if (compare_u64_ptr((T *) (L1 + i), (T *) (L2 + jprime))) {
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
			uint64x4_t *ptr_r = (uint64x4_t *) L2;

			for (size_t j = s2; j < s2 + (e2 + 3) / 4; ++j, ptr_r += 1) {
				const uint64x4_t ri = uint64x4_t::load(ptr_r);
				const int m = compare_256_64(li, ri);

				if (m) {
					const size_t jprime = j * 4 + __builtin_ctz(m);

					if (compare_u64_ptr((T *) (L1 + i), (T *) (L2 + jprime))) {
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
			uint64x4_t *ptr_r = (uint64x4_t *) L2;

			for (size_t j = s2; j < s2 + (e2 + 3) / 4; j += v, ptr_r += v) {

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
							const size_t jprime = j * 4 + a2 * 4 + __builtin_ctz(m);
							const size_t iprime = i + a1;

							if (compare_u64_ptr((T *) (L1 + iprime), (T *) (L2 + jprime))) {
								found_solution(iprime, jprime);
							}
						}// if
					}    // for v
				}        // for u
			}            // for right list
		}                // for left list
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
		static_assert(sizeof(T) == 8);
		static_assert((v > 0) && (u > 0));
		ASSERT(ELEMENT_NR_LIMBS == 1);
		ASSERT(n <= 64);
		ASSERT(n >= 33);

		constexpr size_t s1 = 0, s2 = 0;
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		uint64x4_t lii[u], rii[v];
		auto *ptr_l = (uint64x4_t *) L1;

		for (size_t i = s1; i < s1 + (e1 + 3) / 4; i += u, ptr_l += u) {

#pragma unroll
			for (uint32_t s = 0; s < u; ++s) {
				lii[s] = uint64x4_t::load(ptr_l + s);
			}

			/// NOTE: only possible because L2 is a continuous memory block
			auto *ptr_r = (uint64x4_t *) L2;

			for (size_t j = s2; j < s2 + (e2 + 3) / 4; j += v, ptr_r += v) {

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
							const size_t jprime = j * 4 + a2 * 4 + __builtin_ctz(m);
							const size_t iprime = i * 4 + a1 * 4 + __builtin_ctz(m);
							if (compare_u64_ptr((T *) (L1 + iprime), (T *) (L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}

						/// this is left rotate: [3, 2, 1, 0] -> [2, 1, 0, 3]
						tmp2 = uint64x4_t::template permute<0b10010011>(tmp2);
						m = compare_256_64(tmp1, tmp2);
						if (m) {
							const size_t jprime = j * 4 + a2 * 4 + __builtin_ctz(m) - 1;
							const size_t iprime = i * 4 + a1 * 4 + __builtin_ctz(m);
							if (compare_u64_ptr((T *) (L1 + iprime), (T *) (L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}

						tmp2 = uint64x4_t::template permute<0b10010011>(tmp2);
						m = compare_256_64(tmp1, tmp2);
						if (m) {
							const size_t jprime = j * 4 + a2 * 4 + __builtin_ctz(m) - 2;
							const size_t iprime = i * 4 + a1 * 4 + __builtin_ctz(m);
							if (compare_u64_ptr((T *) (L1 + iprime), (T *) (L2 + jprime))) {
								//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
								found_solution(iprime, jprime);
							}
						}

						tmp2 = uint64x4_t::template permute<0b10010011>(tmp2);
						m = compare_256_64(tmp1, tmp2);
						if (m) {
							const size_t jprime = j * 4 + a2 * 4 + __builtin_ctz(m) - 3;
							const size_t iprime = i * 4 + a1 * 4 + __builtin_ctz(m);
							if (compare_u64_ptr((T *) (L1 + iprime), (T *) (L2 + jprime))) {
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
		constexpr cryptanalysislib::_uint32x4_t loadr1 = cryptanalysislib::_uint32x4_t::setr(0u, 2u, 4u, 6u);
		constexpr cryptanalysislib::_uint32x4_t loadr2 = cryptanalysislib::_uint32x4_t::setr(1u, 3u, 5u, 7u);

		for (size_t i = s1; i < e1; ++i) {
			const uint64x4_t li1 = uint64x4_t::set1(L1[i][0]);
			const uint64x4_t li2 = uint64x4_t::set1(L1[i][1]);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *) L2;

			for (size_t j = s2; j < (s2 + e2 + 3); j += 4, ptr_r += 8) {
				const auto ri1 = uint64x4_t::template gather<8>((const long long int *) ptr_r, loadr1);
				uint32_t m1 = compare_256_64(li1, ri1);

				if (m1) {
					const auto ri2 = uint64x4_t::template gather<8>((const long long int *) ptr_r, loadr2);
					m1 &= compare_256_64(li2, ri2);

					while (m1) {
						const uint32_t ctz = __builtin_ctz(m1);
						const size_t jprime = j + ctz;

						if (compare_u64_ptr((T *) (L1 + i), (T *) (L2 + jprime))) {
							//std::cout << L1[i][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
							found_solution(i, jprime);
						}// if solution

						m1 ^= 1u << ctz;
					}// if m2
				}    // if m1
			}        // right for loop
		}            // left for loop
	}

	/// helper functions which is capable to identify full solutions
	///	after a series of multiple nn steps
	/// \tparam u max unrolling parameter of the left list
	/// \tparam v max unrolling parameter of the right list
	/// \param mask of possible solutions. A `1` in the mask correspond
	/// 		to an index of a possible solution in the two lists.
	/// \param m1
	/// \param round
	/// \param i found indext
	/// \param j found index
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

			const int32_t off_l = test_i * 8 + inner_i2;
			const int32_t off_r = test_j * 8 + ((8 + inner_j2 - round) % 8);


			const T *test_tl = ((T *) L1) + i * 2 + off_l * 2;
			const T *test_tr = ((T *) L2) + j * 2 + off_r * 2;
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
	                                  const size_t s1 = 0,
	                                  const size_t s2 = 0) noexcept {
		static_assert((u <= 8) && (u > 0));
		static_assert((v <= 8) && (v > 0));
		ASSERT(n <= 128);
		ASSERT(n > 64);
		ASSERT(2 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// some constants
		constexpr uint8x32_t zero = uint8x32_t::set1(0);
		constexpr uint32x8_t shuffl = uint32x8_t::setr(7, 0, 1, 2, 3, 4, 5, 6);
		constexpr uint32x8_t loadr = uint32x8_t::setr(0, 4, 8, 12, 16, 20, 24, 28);
		constexpr size_t ptr_ctr_l = u * 8,
		                 ptr_ctr_r = v * 8;
		constexpr size_t ptr_inner_ctr_l = 8 * 4,
		                 ptr_inner_ctr_r = 8 * 4;

		/// container for the unrolling
		uint32x8_t lii_1[u]{}, rii_1[v]{}, lii_2[u]{}, rii_2[v]{};

		/// container for the solutions masks
		/// the init with zero is important, as otherwise the
		/// the aligned load would read uninitialized memory
		alignas(32) uint8_t m1[roundToAligned<32>(u * v)] = {0};

		auto *ptr_l = (uint32_t *) L1;
		for (size_t i = s1; i < s1 + e1; i += ptr_ctr_l, ptr_l += ptr_ctr_l * 4) {

#pragma unroll
			for (uint32_t s = 0; s < u; ++s) {
				lii_1[s] = uint32x8_t::template gather<4>(ptr_l + s * ptr_inner_ctr_l + 0, loadr);
				lii_2[s] = uint32x8_t::template gather<4>(ptr_l + s * ptr_inner_ctr_l + 1, loadr);
			}

			auto *ptr_r = (uint32_t *) L2;
			for (size_t j = s2; j < (s2 + e2); j += ptr_ctr_r, ptr_r += ptr_ctr_r * 4) {

// load the fi
#pragma unroll
				for (uint32_t s = 0; s < v; ++s) {
					rii_1[s] = uint32x8_t::template gather<4>(ptr_r + s * ptr_inner_ctr_r + 0, loadr);
					rii_2[s] = uint32x8_t::template gather<4>(ptr_r + s * ptr_inner_ctr_r + 1, loadr);
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
#pragma unroll
						for (uint32_t f2 = 0; f2 < v; ++f2) {
							m1[f1 * u + f2] = compare_256_32(lii_1[f1], rii_1[f2]);
						}
					}

					// early exit
					uint32_t mask = uint8x32_t::load(m1) > zero;
					if (unlikely(mask == 0)) {
						continue;
					}

// second limb
#pragma unroll
					for (uint32_t f1 = 0; f1 < u; ++f1) {
#pragma unroll
						for (uint32_t f2 = 0; f2 < v; ++f2) {
							m1[f1 * u + f2] &= compare_256_32(lii_2[f1], rii_2[f2]);
						}
					}


					// early exit from the second limb computations
					mask = uint8x32_t::load(m1) > zero;
					if (likely(mask == 0)) {
						continue;
					}

					// maybe write back a solution
					bruteforce_simd_32_2_uxv_helper<u, v>(mask, m1, l, i, j);
				}// 8x8 shuffle
			}    // j: enumerate right side
		}        // i: enumerate left side
	}            // end func

	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 4.
	/// NOTE: this check every element on the left against 4 on the right
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_simd_256(const size_t e1,
	                         const size_t e2,
	                         const size_t s1 = 0,
	                         const size_t s2 = 0) noexcept {

		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);

		/// difference of the memory location in the right list
		constexpr cryptanalysislib::_uint32x4_t loadr1 = cryptanalysislib::_uint32x4_t::setr((4ull << 32u), (8ul) | (12ull << 32u));
		constexpr cryptanalysislib::_uint32x4_t loadr2 = cryptanalysislib::_uint32x4_t::setr(1ull | (5ull << 32u), (9ul) | (13ull << 32u));
		constexpr cryptanalysislib::_uint32x4_t loadr3 = cryptanalysislib::_uint32x4_t::setr(2ull | (6ull << 32u), (10ul) | (14ull << 32u));
		constexpr cryptanalysislib::_uint32x4_t loadr4 = cryptanalysislib::_uint32x4_t::setr(3ull | (7ull << 32u), (11ul) | (15ull << 32u));

		for (size_t i = s1; i < e1; ++i) {
			const uint64x4_t li1 = uint64x4_t::set1(L1[i][0]);
			const uint64x4_t li2 = uint64x4_t::set1(L1[i][1]);
			const uint64x4_t li3 = uint64x4_t::set1(L1[i][2]);
			const uint64x4_t li4 = uint64x4_t::set1(L1[i][3]);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *) L2;

			for (size_t j = s2; j < s2 + (e2 + 3) / 4; ++j, ptr_r += 16) {
				const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *) ptr_r, loadr1);
				const int m1 = compare_256_64(li1, ri);

				if (m1) {
					const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *) ptr_r, loadr2);
					const int m1 = compare_256_64(li2, ri);

					if (m1) {
						const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *) ptr_r, loadr3);
						const int m1 = compare_256_64(li3, ri);

						if (m1) {
							const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *) ptr_r, loadr4);
							const int m1 = compare_256_64(li4, ri);
							if (m1) {
								const size_t jprime = j * 4 + __builtin_ctz(m1);
								if (compare_u64_ptr((T *) (L1 + i), (T *) (L2 + jprime))) {
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
		constexpr cryptanalysislib::_uint32x4_t loadr1 = cryptanalysislib::_uint32x4_t::setr((4ull << 32u), (8ul) | (12ull << 32u));
		constexpr cryptanalysislib::_uint32x4_t loadr2 = cryptanalysislib::_uint32x4_t::setr(1ull | (5ull << 32u), (9ul) | (13ull << 32u));
		constexpr cryptanalysislib::_uint32x4_t loadr3 = cryptanalysislib::_uint32x4_t::setr(2ull | (6ull << 32u), (10ul) | (14ull << 32u));
		constexpr cryptanalysislib::_uint32x4_t loadr4 = cryptanalysislib::_uint32x4_t::setr(3ull | (7ull << 32u), (11ul) | (15ull << 32u));

		alignas(128) uint64x4_t li[u * 4u];
		alignas(32) uint32_t m1s[8] = {0};// this clearing is important

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
				li[ui + 0 * u] = uint64x4_t::set1(L1[i + ui][0]);
				li[ui + 1 * u] = uint64x4_t::set1(L1[i + ui][1]);
				li[ui + 2 * u] = uint64x4_t::set1(L1[i + ui][2]);
				li[ui + 3 * u] = uint64x4_t::set1(L1[i + ui][3]);
			}


			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			T *ptr_r = (T *) L2;

			for (size_t j = s2; j < s2 + (e2 + 3) / 4; ++j, ptr_r += 16) {
				//#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *) ptr_r, loadr1);
					const uint32_t tmp = compare_256_64(li[0 * u + mi], ri);
					m1s[mi] = tmp ? tmp ^ m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *) ptr_r, loadr2);
					const uint32_t tmp = compare_256_64(li[1 * u + mi], ri);
					m1s[mi] = tmp ? tmp ^ m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}

#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *) ptr_r, loadr3);
					const uint32_t tmp = compare_256_64(li[2 * u + mi], ri);
					m1s[mi] = tmp ? tmp ^ m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint64x4_t ri = uint64x4_t::template gather<8>((const long long int *) ptr_r, loadr4);
					const uint32_t tmp = compare_256_64(li[3 * u + mi], ri);
					m1s[mi] = tmp ? tmp ^ m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp) {
					ASSERT(popcount::template popcount<uint32_t>(m1s_tmp) == 1);
					const uint32_t m1s_ctz = __builtin_ctz(m1s_tmp);
					const uint32_t bla = __builtin_ctz(m1s[m1s_ctz]);
					const size_t iprime = i + m1s_ctz;
					const size_t jprime = j * 4 + bla;

					if (compare_u64_ptr((T *) (L1 + iprime), (T *) (L2 + jprime))) {
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
	                                const size_t s1 = 0,
	                                const size_t s2 = 0) noexcept {
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
		constexpr uint32x8_t loadr1 = uint32x8_t::setr(0u, 8u, 16u, 24u, 32u, 40u, 48u, 56u);
		constexpr uint32x8_t loadr_add = uint32x8_t::set1(1u);
		uint32x8_t loadr{};

		alignas(32) uint32x8_t li[u * 8u]{};
		alignas(32) uint32_t m1s[8] = {0};// this clearing is important

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
					const uint32_t tmp = ((uint32_t *) L1[i + ui])[uii];
					li[ui + uii * u] = uint32x8_t::set1(tmp);
				}
			}


			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			uint32_t *ptr_r = (uint32_t *) L2;

			for (size_t j = s2; j < s2 + (e2 + 7) / 8; ++j, ptr_r += 64) {
				loadr = loadr1;
#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *) ptr_r, loadr);
					const uint32_t tmp = compare_256_32(li[0 * u + mi], ri);
					m1s[mi] = tmp ? tmp ^ m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *) ptr_r, loadr);
					const uint32_t tmp = compare_256_32(li[1 * u + mi], ri);
					m1s[mi] = tmp ? tmp ^ m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *) ptr_r, loadr);
					const uint32_t tmp = compare_256_32(li[2 * u + mi], ri);
					m1s[mi] = tmp ? tmp ^ m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *) ptr_r, loadr);
					const uint32_t tmp = compare_256_32(li[3 * u + mi], ri);
					m1s[mi] = tmp ? tmp ^ m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *) ptr_r, loadr);
					const uint32_t tmp = compare_256_32(li[4 * u + mi], ri);
					m1s[mi] = tmp ? tmp ^ m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *) ptr_r, loadr);
					const uint32_t tmp = compare_256_32(li[5 * u + mi], ri);
					m1s[mi] = tmp ? tmp ^ m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *) ptr_r, loadr);
					const uint32_t tmp = compare_256_32(li[6 * u + mi], ri);
					m1s[mi] = tmp ? tmp ^ m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp == 0) {
					continue;
				}


				loadr = loadr + loadr_add;
#pragma unroll
				for (uint32_t mi = 0; mi < u; mi++) {
					const uint32x8_t ri = uint32x8_t::template gather<4>((const int *) ptr_r, loadr);
					const uint32_t tmp = compare_256_32(li[7 * u + mi], ri);
					m1s[mi] = tmp ? tmp ^ m1s_mask : 0;
				}

				m1s_tmp = uint32x8_t::move(uint32x8_t::load(m1s));
				if (m1s_tmp) {
					ASSERT(popcount::template popcount<uint32_t>(m1s_tmp) == 1);
					const uint32_t m1s_ctz = __builtin_ctz(m1s_tmp);
					const uint32_t bla = __builtin_ctz(m1s[m1s_ctz]);
					const size_t iprime = i + m1s_ctz;
					const size_t jprime = j * 8 + bla;
					//std::cout << L1[iprime][0] << " " << L2[jprime][0] << " " << L2[jprime+1][0] << " " << L2[jprime-1][0] << "\n";
					if (compare_u64_ptr((T *) (L1 + iprime), (T *) (L2 + jprime))) {
						found_solution(iprime, jprime);
					}
				}
			}
		}
	}

	/// specialized helper function which is capable of recovering solutions
	/// after multiple internal runs of `brutefroce_simd_256_32_8x8`
	/// \tparam off
	/// \tparam rotation
	/// \param m1sx a set bit in this corresponds to a partial solution in
	/// 		in the two lists
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
		static_assert(off % 32 == 0);

		while (m1sx > 0) {
			const uint32_t ctz1 = __builtin_ctz(m1sx);
			const uint32_t ctz = off + ctz1;
			const uint32_t m1sc = m1s[ctz];
			const uint32_t m1sc_ctz = __builtin_ctz(m1sc);

			const uint32_t test_j = ctz % 8;
			const uint32_t test_i = ctz / 8;

			// NOTE: the signed is important
			const int32_t off_l = test_i * 8 + m1sc_ctz;
			const int32_t off_r = test_j * 8 + (8 - rotation + m1sc_ctz) % 8;

			const uint64_t *test_tl = (uint64_t *) (ptr_l + off_l * 8);
			const uint64_t *test_tr = (uint64_t *) (ptr_r + off_r * 8);
			if (compare_u64_ptr(test_tl, test_tr)) {
				found_solution(i + off_l, j + off_r);
			}

			m1sx ^= 1u << ctz1;
		}
	}

	/// unrolled bruteforce step.
	/// stack: uint64_t[64]
	/// a1-a8, b1-b7: __m256i
	inline void BRUTEFORCE256_32_8x8_STEP(
	        uint8_t *stack,
	        const uint32x8_t a1, const uint32x8_t a2, const uint32x8_t a3, const uint32x8_t a4,
	        const uint32x8_t a5, const uint32x8_t a6, const uint32x8_t a7, const uint32x8_t a8,
	        const uint32x8_t b1, const uint32x8_t b2, const uint32x8_t b3, const uint32x8_t b4,
	        const uint32x8_t b5, const uint32x8_t b6, const uint32x8_t b7, const uint32x8_t b8) {
		stack[0] = (uint8_t) compare_256_32(a1, b1);
		stack[1] = (uint8_t) compare_256_32(a1, b2);
		stack[2] = (uint8_t) compare_256_32(a1, b3);
		stack[3] = (uint8_t) compare_256_32(a1, b4);
		stack[4] = (uint8_t) compare_256_32(a1, b5);
		stack[5] = (uint8_t) compare_256_32(a1, b6);
		stack[6] = (uint8_t) compare_256_32(a1, b7);
		stack[7] = (uint8_t) compare_256_32(a1, b8);
		stack[8] = (uint8_t) compare_256_32(a2, b1);
		stack[9] = (uint8_t) compare_256_32(a2, b2);
		stack[10] = (uint8_t) compare_256_32(a2, b3);
		stack[11] = (uint8_t) compare_256_32(a2, b4);
		stack[12] = (uint8_t) compare_256_32(a2, b5);
		stack[13] = (uint8_t) compare_256_32(a2, b6);
		stack[14] = (uint8_t) compare_256_32(a2, b7);
		stack[15] = (uint8_t) compare_256_32(a2, b8);
		stack[16] = (uint8_t) compare_256_32(a3, b1);
		stack[17] = (uint8_t) compare_256_32(a3, b2);
		stack[18] = (uint8_t) compare_256_32(a3, b3);
		stack[19] = (uint8_t) compare_256_32(a3, b4);
		stack[20] = (uint8_t) compare_256_32(a3, b5);
		stack[21] = (uint8_t) compare_256_32(a3, b6);
		stack[22] = (uint8_t) compare_256_32(a3, b7);
		stack[23] = (uint8_t) compare_256_32(a3, b8);
		stack[24] = (uint8_t) compare_256_32(a4, b1);
		stack[25] = (uint8_t) compare_256_32(a4, b2);
		stack[26] = (uint8_t) compare_256_32(a4, b3);
		stack[27] = (uint8_t) compare_256_32(a4, b4);
		stack[28] = (uint8_t) compare_256_32(a4, b5);
		stack[29] = (uint8_t) compare_256_32(a4, b6);
		stack[30] = (uint8_t) compare_256_32(a4, b7);
		stack[31] = (uint8_t) compare_256_32(a4, b8);
		stack[32] = (uint8_t) compare_256_32(a5, b1);
		stack[33] = (uint8_t) compare_256_32(a5, b2);
		stack[34] = (uint8_t) compare_256_32(a5, b3);
		stack[35] = (uint8_t) compare_256_32(a5, b4);
		stack[36] = (uint8_t) compare_256_32(a5, b5);
		stack[37] = (uint8_t) compare_256_32(a5, b6);
		stack[38] = (uint8_t) compare_256_32(a5, b7);
		stack[39] = (uint8_t) compare_256_32(a5, b8);
		stack[40] = (uint8_t) compare_256_32(a6, b1);
		stack[41] = (uint8_t) compare_256_32(a6, b2);
		stack[42] = (uint8_t) compare_256_32(a6, b3);
		stack[43] = (uint8_t) compare_256_32(a6, b4);
		stack[44] = (uint8_t) compare_256_32(a6, b5);
		stack[45] = (uint8_t) compare_256_32(a6, b6);
		stack[46] = (uint8_t) compare_256_32(a6, b7);
		stack[47] = (uint8_t) compare_256_32(a6, b8);
		stack[48] = (uint8_t) compare_256_32(a7, b1);
		stack[49] = (uint8_t) compare_256_32(a7, b2);
		stack[50] = (uint8_t) compare_256_32(a7, b3);
		stack[51] = (uint8_t) compare_256_32(a7, b4);
		stack[52] = (uint8_t) compare_256_32(a7, b5);
		stack[53] = (uint8_t) compare_256_32(a7, b6);
		stack[54] = (uint8_t) compare_256_32(a7, b7);
		stack[55] = (uint8_t) compare_256_32(a7, b8);
		stack[56] = (uint8_t) compare_256_32(a8, b1);
		stack[57] = (uint8_t) compare_256_32(a8, b2);
		stack[58] = (uint8_t) compare_256_32(a8, b3);
		stack[59] = (uint8_t) compare_256_32(a8, b4);
		stack[60] = (uint8_t) compare_256_32(a8, b5);
		stack[61] = (uint8_t) compare_256_32(a8, b6);
		stack[62] = (uint8_t) compare_256_32(a8, b7);
		stack[63] = (uint8_t) compare_256_32(a8, b8);
	}


	/// NOTE: this is hyper optimized for the case if there is only one solution with extremely low weight.
	/// \param e1 size of the list L1
	/// \param e2 size of the list L2
	void bruteforce_simd_256_32_8x8(const size_t e1,
	                                const size_t e2,
	                                const size_t s1 = 0,
	                                const size_t s2 = 0) noexcept {
		ASSERT(n <= 256);
		ASSERT(n > 128);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);
		ASSERT(d < 16);

		uint32_t *ptr_l = (uint32_t *) L1;

		/// difference of the memory location in the right list
		constexpr uint32x8_t loadr1 = uint32x8_t::setr(0, 8, 16, 24, 32, 40, 48, 56);
		constexpr uint32x8_t shuffl = uint32x8_t::setr(7, 0, 1, 2, 3, 4, 5, 6);

		alignas(32) uint8_t m1s[64];

		/// helper to detect zeros
		const uint8x32_t zero = uint8x32_t::set1(0);

		for (size_t i = s1; i < s1 + e1; i += 64, ptr_l += 512) {
			const uint32x8_t l1 = uint32x8_t::template gather<4>((const int *) (ptr_l + 0), loadr1);
			const uint32x8_t l2 = uint32x8_t::template gather<4>((const int *) (ptr_l + 64), loadr1);
			const uint32x8_t l3 = uint32x8_t::template gather<4>((const int *) (ptr_l + 128), loadr1);
			const uint32x8_t l4 = uint32x8_t::template gather<4>((const int *) (ptr_l + 192), loadr1);
			const uint32x8_t l5 = uint32x8_t::template gather<4>((const int *) (ptr_l + 256), loadr1);
			const uint32x8_t l6 = uint32x8_t::template gather<4>((const int *) (ptr_l + 320), loadr1);
			const uint32x8_t l7 = uint32x8_t::template gather<4>((const int *) (ptr_l + 384), loadr1);
			const uint32x8_t l8 = uint32x8_t::template gather<4>((const int *) (ptr_l + 448), loadr1);

			uint32_t *ptr_r = (uint32_t *) L2;
			for (size_t j = s1; j < s2 + e2; j += 64, ptr_r += 512) {
				uint32x8_t r1 = uint32x8_t::template gather<4>((const int *) (ptr_r + 0), loadr1);
				uint32x8_t r2 = uint32x8_t::template gather<4>((const int *) (ptr_r + 64), loadr1);
				uint32x8_t r3 = uint32x8_t::template gather<4>((const int *) (ptr_r + 128), loadr1);
				uint32x8_t r4 = uint32x8_t::template gather<4>((const int *) (ptr_r + 192), loadr1);
				uint32x8_t r5 = uint32x8_t::template gather<4>((const int *) (ptr_r + 256), loadr1);
				uint32x8_t r6 = uint32x8_t::template gather<4>((const int *) (ptr_r + 320), loadr1);
				uint32x8_t r7 = uint32x8_t::template gather<4>((const int *) (ptr_r + 384), loadr1);
				uint32x8_t r8 = uint32x8_t::template gather<4>((const int *) (ptr_r + 448), loadr1);

				BRUTEFORCE256_32_8x8_STEP(m1s, l1, l2, l3, l4, l5, l6, l7, l8, r1, r2, r3, r4, r5, r6, r7, r8);
				uint32_t m1s1 = zero != uint8x32_t::load(m1s + 0);
				uint32_t m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper<0, 0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
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
				m1s1 = zero != uint8x32_t::load(m1s + 0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper<0, 1>(m1s1, m1s, ptr_l, ptr_r, i, j); }
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
				m1s1 = zero != uint8x32_t::load(m1s + 0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper<0, 2>(m1s1, m1s, ptr_l, ptr_r, i, j); }
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
				m1s1 = zero != uint8x32_t::load(m1s + 0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper<0, 3>(m1s1, m1s, ptr_l, ptr_r, i, j); }
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
				m1s1 = zero != uint8x32_t::load(m1s + 0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper<0, 4>(m1s1, m1s, ptr_l, ptr_r, i, j); }
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
				m1s1 = zero != uint8x32_t::load(m1s + 0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper<0, 5>(m1s1, m1s, ptr_l, ptr_r, i, j); }
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
				m1s1 = zero != uint8x32_t::load(m1s + 0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper<0, 6>(m1s1, m1s, ptr_l, ptr_r, i, j); }
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
				m1s1 = zero != uint8x32_t::load(m1s + 0);
				m1s2 = zero != uint8x32_t::load(m1s + 32);
				if (m1s1 != 0) { bruteforce_avx2_256_32_8x8_helper<0, 7>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_32_8x8_helper<32, 7>(m1s2, m1s, ptr_l, ptr_r, i, j); }
			}
		}
	}

	/// \tparam off: either 0 or 32, depening of the lower, or upper 32 elements
	/// from the ouput of `BRUTEFORCE256_64_4x4_STEP2` is analysed for a collision
	/// \param m1sx compressed `uint32_t[32]`, with a the k-th bit in `m1sx` is
	/// 	set, <=> uint32_t[k] != 0; This allows for fast checking for non
	/// 	solutions. As the calling function assumes that in expectation no
	/// 	close elements will be found.
	/// \param m1s: output of `BRUTEFORCE256_64_4x4_STEP2`
	/// \param ptr_l pointer to the first element of the check in the left list
	/// \param ptr_r pointer to the first element of the check in the right list
	/// \param i left list counter
	/// \param j right list counter
	template<const uint32_t off>
	inline void bruteforce_avx2_256_64_4x4_helper(uint32_t m1sx,
	                                              const uint8_t *__restrict__ m1s,
	                                              const uint64_t *__restrict__ ptr_l,
	                                              const uint64_t *__restrict__ ptr_r,
	                                              const size_t i, const size_t j) noexcept {
		static_assert(off % 32 == 0);

		while (m1sx > 0) {
			const uint32_t ctz1 = __builtin_ctz(m1sx);
			const uint32_t ctz = off + ctz1;
			const uint32_t m1sc = m1s[ctz];
			const uint32_t inner_ctz = __builtin_ctz(m1sc);

			const uint32_t test_j = ctz % 4;
			const uint32_t test_inner = (inner_ctz + (ctz / 16)) % 4;
			const uint32_t test_i = (ctz % 16) / 4;

			const uint32_t off_l = test_i * 4 + inner_ctz;
			const uint32_t off_r = test_j * 4 + test_inner;

			const T *test_tl = ptr_l + off_l * 4;
			const T *test_tr = ptr_r + off_r * 4;
			if (compare_u64_ptr(test_tl, test_tr)) {
				found_solution(i + off_l, j + off_r);
			}

			m1sx ^= 1u << ctz1;
		}
	}

	/// this function takes each of the 64-bit limbs of all the a_i and compares
	/// them against each of the 64-bit of the b_i
	/// \param stack output: the bit k \in [0,..,3] at index [i*4 + j*16]
	/// 	(with i = 0, ..., 3 and j = 0, ..., 3)
	/// 	is set if a_i_k == b_j_{j_k % 4} (maybe not equal but close, depending on the exact config of the algorithm.)
	///	where a_i_k is the k-64-bit-limb of a_i is.
	///	NOTE: this is hyper optimized for the case, where no solution is expected.
	/// \param a0 const input
	/// \param a1 const input
	/// \param a2 const input
	/// \param a3 const input
	/// \param b0 input, is permuted!
	/// \param b1 input, is permuted!
	/// \param b2 input, is permuted!
	/// \param b3 input, is permuted!
	inline void BRUTEFORCE256_64_4x4_STEP2(uint8_t *stack,
	                                       const uint64x4_t a0, const uint64x4_t a1, const uint64x4_t a2, const uint64x4_t a3,
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
	/// NOTE: only checks the first limb. If this passes the weight check all
	/// 	others are checked within `check_solution`.
	/// NOTE:
	/// \param e1 end index of list 1
	/// \param e2 end index of list 2
	void bruteforce_simd_256_64_4x4(const size_t e1,
	                                const size_t e2,
	                                const size_t s1 = 0,
	                                const size_t s2 = 0) noexcept {
		ASSERT(n <= 256);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);
		ASSERT(dk < 32);

		/// NOTE is already aligned
		T *ptr_l = (T *) L1;

		/// difference of the memory location in the right list
		const cryptanalysislib::_uint32x4_t loadr1 = cryptanalysislib::_uint32x4_t::setr((4ull << 32u), (8ul) | (12ull << 32u));
		alignas(32) uint8_t m1s[64];

		/// allowed weight to match on
		const uint8x32_t zero = uint8x32_t::set1(0);

		for (size_t i = s1; i < s1 + e1; i += 16, ptr_l += 64) {
			const uint64x4_t l1 = uint64x4_t::template gather<8>((const long long int *) (ptr_l + 0), loadr1);
			const uint64x4_t l2 = uint64x4_t::template gather<8>((const long long int *) (ptr_l + 16), loadr1);
			const uint64x4_t l3 = uint64x4_t::template gather<8>((const long long int *) (ptr_l + 32), loadr1);
			const uint64x4_t l4 = uint64x4_t::template gather<8>((const long long int *) (ptr_l + 48), loadr1);

			/// reset right list pointer
			T *ptr_r = (T *) L2;

#pragma unroll 4
			for (size_t j = s1; j < s2 + e2; j += 16, ptr_r += 64) {
				uint64x4_t r1 = uint64x4_t::template gather<8>((const long long int *) (ptr_r + 0), loadr1);
				uint64x4_t r2 = uint64x4_t::template gather<8>((const long long int *) (ptr_r + 16), loadr1);
				uint64x4_t r3 = uint64x4_t::template gather<8>((const long long int *) (ptr_r + 32), loadr1);
				uint64x4_t r4 = uint64x4_t::template gather<8>((const long long int *) (ptr_r + 48), loadr1);

				BRUTEFORCE256_64_4x4_STEP2(m1s, l1, l2, l3, l4, r1, r2, r3, r4);
				/// NOTE: we can load the data aligned.
				uint32_t m1s1 = uint8x32_t::load<true>(m1s + 0) > zero;
				uint32_t m1s2 = uint8x32_t::load<true>(m1s + 32) > zero;

				if (m1s1 != 0) { bruteforce_avx2_256_64_4x4_helper<0>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_avx2_256_64_4x4_helper<32>(m1s2, m1s, ptr_l, ptr_r, i, j); }
			}
		}
	}


	template<const uint32_t off, const uint32_t bucket_size>
	void bruteforce_simd_256_64_4x4_rearrange_helper(uint32_t m1sx,
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
			const uint32_t test_i = (ctz % 16) / 4;

			const uint32_t off_l = test_i * 4 + inner_ctz;
			const uint32_t off_r = test_j * 4 + test_inner;
			ASSERT(off_l < 16);
			ASSERT(off_r < 16);

			uint32_t wt = 0;
			for (uint32_t s = 0; s < ELEMENT_NR_LIMBS; s++) {
				const T t1 = ptr_l[off_l + s * bucket_size];
				const T t2 = ptr_r[off_r + s * bucket_size];
				wt += popcount::popcount(t1 ^ t2);
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
	/// NOTE: this assumes that the `last` list (whatever the last is: normally
	/// 	its the list with < BUCKET_SIZE elements) is in the REARRANGE/TRANSPOSED
	/// 	buckets/
	/// \param e1 end index of list 1
	/// \param e2 end index of list 2
	template<const uint32_t bucket_size>
	void bruteforce_simd_256_64_4x4_rearrange(const size_t e1,
	                                          const size_t e2,
	                                          const size_t s1 = 0,
	                                          const size_t s2 = 0) noexcept {
		ASSERT(e1 <= bucket_size);
		ASSERT(e2 <= bucket_size);
		ASSERT(n <= 256);
		ASSERT(4 == ELEMENT_NR_LIMBS);
		ASSERT(e1 >= s1);
		ASSERT(e2 >= s2);
		ASSERT(dk < 32);
		ASSERT(USE_REARRANGE);

		/// NOTE is already aligned
		T *ptr_l = (T *) LB;

		/// difference of the memory location in the right list
		alignas(32) uint8_t m1s[64];

		/// allowed weight to match on
		constexpr uint8x32_t zero = uint8x32_t::set1(0);

		size_t i = s1;
#pragma unroll 4
		for (; i < s1 + e1; i += 16, ptr_l += 16) {
			const uint64x4_t l1 = uint64x4_t::template load<true>(ptr_l + 0);
			const uint64x4_t l2 = uint64x4_t::template load<true>(ptr_l + 4);
			const uint64x4_t l3 = uint64x4_t::template load<true>(ptr_l + 8);
			const uint64x4_t l4 = uint64x4_t::template load<true>(ptr_l + 12);

			/// reset right list pointer
			T *ptr_r = (T *) RB;

#pragma unroll 4
			for (size_t j = s1; j < s2 + e2; j += 16, ptr_r += 16) {
				uint64x4_t r1 = uint64x4_t::template load<true>(ptr_r + 0);
				uint64x4_t r2 = uint64x4_t::template load<true>(ptr_r + 4);
				uint64x4_t r3 = uint64x4_t::template load<true>(ptr_r + 8);
				uint64x4_t r4 = uint64x4_t::template load<true>(ptr_r + 12);

				BRUTEFORCE256_64_4x4_STEP2(m1s, l1, l2, l3, l4, r1, r2, r3, r4);
				uint32_t m1s1 = zero < uint8x32_t::load(m1s + 0u);
				uint32_t m1s2 = zero < uint8x32_t::load(m1s + 32u);

				if (m1s1 != 0) { bruteforce_simd_256_64_4x4_rearrange_helper<0, bucket_size>(m1s1, m1s, ptr_l, ptr_r, i, j); }
				if (m1s2 != 0) { bruteforce_simd_256_64_4x4_rearrange_helper<32, bucket_size>(m1s2, m1s, ptr_l, ptr_r, i, j); }
			}
		}
	}

	/// TODO tests?:
	/// bruteforce the two lists between the given start and end indices.
	/// NOTE: uses avx2
	/// NOTE: only in limb comparison possible. inter limb (e.g. bit 43...83) is impossible.
	/// NOTE: assumes that list size is multiple of 4.
	/// NOTE: in comparison to `bruteforce_avx_256` this implementation used no `gather`
	///		instruction, but rather direct (aligned) loads.
	/// NOTE: only works if exact matching is active
	/// \param e1 end index of list 1
	/// \param e2 end index list 2
	void bruteforce_asimd_256_v2(const size_t e1,
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
		uint64x4_t *ptr_l = (uint64x4_t *) L1;

		for (size_t i = s1; i < e1; ++i, ptr_l += 1) {
			const uint64x4_t li1 = uint64x4_t::load(ptr_l);

			/// NOTE: only possible because L2 is a continuous memory block
			/// NOTE: reset every loop
			uint64x4_t *ptr_r = (uint64x4_t *) L2;

			for (size_t j = s2; j < s2 + e2; ++j, ptr_r += 1) {
				const uint64x4_t ri = uint64x4_t::load(ptr_r);
				const uint64x4_t tmp1 = li1 ^ ri;
				if (zero == tmp1) {
					found_solution(i, j);
				}// if solution found
			}    // right list
		}        // left list
	}
};


#endif//SMALLSECRETLWE_NN_H
