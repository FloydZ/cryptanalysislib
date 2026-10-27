#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <random>
#include <type_traits>
#include <utility>
#include <vector>

#include "helper.cuh"
#include "nn/nn.cuh"
#include "nn/nn_tensor.cuh"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib::nn::tensor;

constexpr size_t LS = 1u << 14u;

static std::mt19937_64 gen;

/// the implementations in `nn/nn_tensor.cuh`
enum class Variant {
	XOR,	// C: `tensor_bruteforce`
	AND,	// D: `tensor_bruteforce_and`
	HYBRID,	// F: `tensor_bruteforce_hybrid`
	MXF4	// G: `tensor_bruteforce_mxf4`
};

/// minimal version of `NN<config>` from `nn/nn.h`, which cannot be compiled
/// by nvcc. The lists are allocated as managed memory.
/// \tparam n number of bits of each element
/// \tparam d weight difference to match on the golden solution
/// \tparam LIST_SIZE number of elements in each list
template<const uint32_t n, const uint32_t d, const size_t LIST_SIZE>
class TensorNN {
public:
	using T = uint32_t;
	constexpr static uint32_t ELEMENT_NR_LIMBS = n / 32u;
	static_assert(n % 32u == 0);

	T *L1 = nullptr, *L2 = nullptr;
	size_t solution_l = 0, solution_r = 0, solutions_nr = 0;
	std::vector<std::pair<size_t, size_t>> solutions;

	TensorNN() {
		checkCudaErrors(cudaMallocManaged(&L1, sizeof(T) * LIST_SIZE * ELEMENT_NR_LIMBS));
		checkCudaErrors(cudaMallocManaged(&L2, sizeof(T) * LIST_SIZE * ELEMENT_NR_LIMBS));
	}

	~TensorNN() {
		cudaFree(L1);
		cudaFree(L2);
	}

	/// \param insert_sol if true, a solution with weight exactly `d` is inserted
	/// 		at a random position
	void generate_random_instance(const bool insert_sol = true) noexcept {
		for (size_t i = 0; i < LIST_SIZE * ELEMENT_NR_LIMBS; i++) {
			L1[i] = gen();
			L2[i] = gen();
		}

		if (insert_sol) {
			insert_solution(gen() % LIST_SIZE, gen() % LIST_SIZE);
		}
	}

	/// sets L2[r] = L1[l] ^ e with wt(e) = w
	void insert_solution(const size_t l, const size_t r, const uint32_t w = d) noexcept {
		solution_l = l;
		solution_r = r;
		for (uint32_t i = 0; i < ELEMENT_NR_LIMBS; i++) {
			L2[r * ELEMENT_NR_LIMBS + i] = L1[l * ELEMENT_NR_LIMBS + i];
		}

		std::vector<uint32_t> pos(n);
		for (uint32_t i = 0; i < n; i++) {
			pos[i] = i;
		}
		std::shuffle(pos.begin(), pos.end(), gen);
		for (uint32_t i = 0; i < w; i++) {
			L2[r * ELEMENT_NR_LIMBS + pos[i] / 32u] ^= T(1u) << (pos[i] % 32u);
		}
	}

	[[nodiscard]] uint32_t weight(const size_t l, const size_t r) const noexcept {
		uint32_t wt = 0;
		for (uint32_t j = 0; j < ELEMENT_NR_LIMBS; j++) {
			wt += __builtin_popcount(L1[l * ELEMENT_NR_LIMBS + j] ^ L2[r * ELEMENT_NR_LIMBS + j]);
		}
		return wt;
	}

	void found_solution(const size_t li, const size_t lr) noexcept {
		solutions.resize(solutions_nr + 1);
		solutions[solutions_nr++] = std::pair<size_t, size_t>{li, lr};
	}

	/// calls `launch(L1, L2, f1, f2)` and adds the solution, if one was found
	template<typename F>
	void run(F launch) noexcept {
		size_t *f1, *f2;
		checkCudaErrors(cudaMallocManaged(&f1, sizeof(size_t)));
		checkCudaErrors(cudaMallocManaged(&f2, sizeof(size_t)));
		*f1 = size_t(-1);
		*f2 = size_t(-1);

		launch(L1, L2, f1, f2);
		checkCudaErrors(cudaDeviceSynchronize());

		if (*f1 != size_t(-1)) {
			found_solution(*f1, *f2);
		}

		cudaFree(f1);
		cudaFree(f2);
	}

	/// searches with the tensor cores in L1[0..e1) and L2[0..e2), with the
	/// default tuning parameters of the variant
	template<const Variant V = Variant::XOR>
	void bruteforce_tensor(const size_t e1, const size_t e2) noexcept {
		run([&](const T *L, const T *R, size_t *f1, size_t *f2) {
			if constexpr (V == Variant::XOR) {
				launch_tensor_bruteforce<T, ELEMENT_NR_LIMBS, d>(L, R, e1, e2, f1, f2);
			} else if constexpr (V == Variant::AND) {
				launch_tensor_bruteforce_and<T, ELEMENT_NR_LIMBS, d>(L, R, e1, e2, f1, f2);
			} else if constexpr (V == Variant::HYBRID) {
				launch_tensor_bruteforce_hybrid<T, ELEMENT_NR_LIMBS, d>(L, R, e1, e2, f1, f2);
			} else {
				launch_tensor_bruteforce_mxf4<T, ELEMENT_NR_LIMBS, d>(L, R, e1, e2, f1, f2);
			}
		});
	}

	// checks whether all submitted solutions are correct
	[[nodiscard]] bool all_solutions_correct() const noexcept {
		if (solutions_nr == 0) {
			return false;
		}

		for (uint32_t i = 0; i < solutions_nr; i++) {
			if ((solutions[i].first >= LIST_SIZE) ||
				(solutions[i].second >= LIST_SIZE) ||
				(weight(solutions[i].first, solutions[i].second) > d)) {
				return false;
			}
		}

		return true;
	}
};

/// all tests below run for every variant
template<typename V>
class TensorBruteforce : public Test {};
using Variants = ::testing::Types<std::integral_constant<Variant, Variant::XOR>,
								  std::integral_constant<Variant, Variant::AND>,
								  std::integral_constant<Variant, Variant::HYBRID>,
								  std::integral_constant<Variant, Variant::MXF4>>;
TYPED_TEST_SUITE(TensorBruteforce, Variants);

TYPED_TEST(TensorBruteforce, n128) {
	TensorNN<128, 20, LS> algo{};
	algo.generate_random_instance();
	algo.template bruteforce_tensor<TypeParam::value>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	EXPECT_EQ(algo.solutions[0].first, algo.solution_l);
	EXPECT_EQ(algo.solutions[0].second, algo.solution_r);
}

TYPED_TEST(TensorBruteforce, n256) {
	TensorNN<256, 50, LS> algo{};
	algo.generate_random_instance();
	algo.template bruteforce_tensor<TypeParam::value>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	EXPECT_EQ(algo.solutions[0].first, algo.solution_l);
	EXPECT_EQ(algo.solutions[0].second, algo.solution_r);
}

/// small `d`: the early abort after the first 128 bits is nearly always taken
TYPED_TEST(TensorBruteforce, n384) {
	TensorNN<384, 10, LS> algo{};
	algo.generate_random_instance();
	algo.template bruteforce_tensor<TypeParam::value>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	EXPECT_EQ(algo.solutions[0].first, algo.solution_l);
	EXPECT_EQ(algo.solutions[0].second, algo.solution_r);
}

TYPED_TEST(TensorBruteforce, NoSolution) {
	TensorNN<128, 20, LS> algo{};
	algo.generate_random_instance(false);
	algo.template bruteforce_tensor<TypeParam::value>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 0);
}

/// a pair with weight d+1 is not a solution
TYPED_TEST(TensorBruteforce, WeightBound) {
	TensorNN<128, 20, LS> algo{};
	algo.generate_random_instance(false);
	algo.insert_solution(gen() % LS, gen() % LS, 21);
	algo.template bruteforce_tensor<TypeParam::value>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 0);

	algo.insert_solution(algo.solution_l, algo.solution_r, 0);
	algo.template bruteforce_tensor<TypeParam::value>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

/// list sizes which are not a multiple of any tile size, solution at the
/// very last elements
TYPED_TEST(TensorBruteforce, Tail) {
	constexpr size_t ls = 1001;
	TensorNN<256, 20, ls> algo{};
	algo.generate_random_instance(false);
	algo.insert_solution(ls - 1, ls - 1);
	algo.template bruteforce_tensor<TypeParam::value>(ls, ls);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	EXPECT_EQ(algo.solutions[0].first, ls - 1);
	EXPECT_EQ(algo.solutions[0].second, ls - 1);
}

/// only the first `e1`/`e2` elements are searched
TYPED_TEST(TensorBruteforce, Bounds) {
	TensorNN<128, 20, LS> algo{};
	algo.generate_random_instance(false);
	algo.insert_solution(99, LS - 1);
	algo.template bruteforce_tensor<TypeParam::value>(99, LS);
	EXPECT_EQ(algo.solutions_nr, 0);
	algo.template bruteforce_tensor<TypeParam::value>(100, LS - 1);
	EXPECT_EQ(algo.solutions_nr, 0);
	algo.template bruteforce_tensor<TypeParam::value>(100, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TYPED_TEST(TensorBruteforce, SingleElement) {
	TensorNN<128, 20, 1> algo{};
	algo.generate_random_instance();
	algo.template bruteforce_tensor<TypeParam::value>(1, 1);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TYPED_TEST(TensorBruteforce, MultipleSolutions) {
	TensorNN<128, 20, LS> algo{};
	algo.generate_random_instance(false);
	algo.insert_solution(17, 4711);
	algo.insert_solution(LS - 9, 3);
	algo.insert_solution(1234, LS - 2);
	algo.template bruteforce_tensor<TypeParam::value>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

/// the solution at every position within a R chunk. For the hybrid variant
/// this covers both the tensor core and the CUDA core part of each chunk.
TYPED_TEST(TensorBruteforce, ChunkPositions) {
	constexpr size_t ls = 1024;
	TensorNN<256, 20, ls> algo{};
	algo.generate_random_instance(false);
	for (size_t r = 0; r < 512; r += 7) {
		const size_t l = gen() % ls;
		algo.insert_solution(l, r);
		algo.template bruteforce_tensor<TypeParam::value>(ls, ls);
		ASSERT_EQ(algo.solutions_nr, 1);
		EXPECT_EQ(algo.solutions[0].first, l);
		EXPECT_EQ(algo.solutions[0].second, r);

		// remove the solution again
		algo.solutions_nr = 0;
		for (uint32_t i = 0; i < algo.ELEMENT_NR_LIMBS; i++) {
			algo.L2[r * algo.ELEMENT_NR_LIMBS + i] = gen();
		}
	}
}

/// instance of the tests with other tuning parameters
/// NOTE: one global alias, as nvcc 12.9 (cudafe++) emits the invalid
/// 		`unsigned long(...)` for the `size_t` template argument, if the same
/// 		`TensorNN` type is spelled out in more than one test.
constexpr size_t PARAMS_LS = 3333;
constexpr uint32_t PARAMS_D = 30;
using ParamsNN = TensorNN<256, PARAMS_D, PARAMS_LS>;

/// other tuning parameters
TEST(TensorBruteforceParameters, XOR) {
	using T = uint32_t;
	constexpr size_t ls = PARAMS_LS;
	constexpr uint32_t d = PARAMS_D;
	ParamsNN algo{};
	algo.generate_random_instance(false);
	algo.insert_solution(ls - 1, 0);

	algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
		launch_tensor_bruteforce<T, 8, d, 1, 1, 8>(L, R, ls, ls, f1, f2); });
	EXPECT_EQ(algo.solutions_nr, 1);
	algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
		launch_tensor_bruteforce<T, 8, d, 8, 4, 64>(L, R, ls, ls, f1, f2); });
	EXPECT_EQ(algo.solutions_nr, 2);
	algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
		launch_tensor_bruteforce<T, 8, d, 2, 16, 256>(L, R, ls, ls, f1, f2); });
	EXPECT_EQ(algo.solutions_nr, 3);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(TensorBruteforceParameters, AND) {
	using T = uint32_t;
	constexpr size_t ls = PARAMS_LS;
	constexpr uint32_t d = PARAMS_D;
	ParamsNN algo{};
	algo.generate_random_instance(false);
	algo.insert_solution(ls - 1, 0);

	algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
		launch_tensor_bruteforce_and<T, 8, d, 1, 1, 8>(L, R, ls, ls, f1, f2); });
	EXPECT_EQ(algo.solutions_nr, 1);
	algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
		launch_tensor_bruteforce_and<T, 8, d, 8, 4, 64>(L, R, ls, ls, f1, f2); });
	EXPECT_EQ(algo.solutions_nr, 2);
	algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
		launch_tensor_bruteforce_and<T, 8, d, 4, 4, 128>(L, R, ls, ls, f1, f2); });
	EXPECT_EQ(algo.solutions_nr, 3);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(TensorBruteforceParameters, HYBRID) {
	using T = uint32_t;
	constexpr size_t ls = PARAMS_LS;
	constexpr uint32_t d = PARAMS_D;
	ParamsNN algo{};
	algo.generate_random_instance(false);

	// once in the tensor core and once in the CUDA core part of a chunk
	for (const size_t r : {size_t(0), ls - 1}) {
		algo.insert_solution(gen() % ls, r);
		algo.solutions_nr = 0;

		// 1 popc warp, 1 row per popc thread
		algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
			launch_tensor_bruteforce_hybrid<T, 8, d, 1, 4, 64, 1, 8>(L, R, ls, ls, f1, f2); });
		EXPECT_EQ(algo.solutions_nr, 1);
		// only a single tensor core tile per chunk
		algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
			launch_tensor_bruteforce_hybrid<T, 8, d, 4, 8, 128, 8, 120>(L, R, ls, ls, f1, f2); });
		EXPECT_EQ(algo.solutions_nr, 2);
		// 2 rows per popc thread
		algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
			launch_tensor_bruteforce_hybrid<T, 8, d, 2, 8, 256, 2, 128>(L, R, ls, ls, f1, f2); });
		EXPECT_EQ(algo.solutions_nr, 3);
		// only CUDA cores
		algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
			launch_tensor_bruteforce_hybrid<T, 8, d, 2, 8, 128, 4, 128>(L, R, ls, ls, f1, f2); });
		EXPECT_EQ(algo.solutions_nr, 4);
		EXPECT_EQ(algo.all_solutions_correct(), true);
		for (uint32_t i = 0; i < algo.solutions_nr; i++) {
			EXPECT_EQ(algo.solutions[i].first, algo.solution_l);
			EXPECT_EQ(algo.solutions[i].second, algo.solution_r);
		}

		// remove the solution again
		for (uint32_t i = 0; i < algo.ELEMENT_NR_LIMBS; i++) {
			algo.L2[r * algo.ELEMENT_NR_LIMBS + i] = gen();
		}
	}
}

/// mxf4: other tuning parameters, all steps before the check (S0 = S), and
/// a single step
TEST(TensorBruteforceParameters, MXF4) {
	using T = uint32_t;
	constexpr size_t ls = PARAMS_LS;
	constexpr uint32_t d = PARAMS_D;
	ParamsNN algo{};
	algo.generate_random_instance(false);
	algo.insert_solution(ls - 1, 0);

	// one warp, one tile per iteration and chunk
	algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
		launch_tensor_bruteforce_mxf4<T, 8, d, 1, 1, 8, 0, 1>(L, R, ls, ls, f1, f2); });
	EXPECT_EQ(algo.solutions_nr, 1);
	// TM = 4, large chunks, 4 tiles per iteration
	algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
		launch_tensor_bruteforce_mxf4<T, 8, d, 4, 4, 256, 0, 4>(L, R, ls, ls, f1, f2); });
	EXPECT_EQ(algo.solutions_nr, 2);
	// no early abort: all 4 steps
	algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
		launch_tensor_bruteforce_mxf4<T, 8, d, 2, 8, 128, 4>(L, R, ls, ls, f1, f2); });
	EXPECT_EQ(algo.solutions_nr, 3);
	// early abort after a single step, which for d = 30 is often not taken
	algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
		launch_tensor_bruteforce_mxf4<T, 8, d, 2, 8, 64, 1>(L, R, ls, ls, f1, f2); });
	EXPECT_EQ(algo.solutions_nr, 4);
	// 3 steps, 8 tiles per iteration, 1 CTA per SM
	algo.run([](const T *L, const T *R, size_t *f1, size_t *f2) {
		launch_tensor_bruteforce_mxf4<T, 8, d, 2, 16, 128, 3, 8, 1>(L, R, ls, ls, f1, f2); });
	EXPECT_EQ(algo.solutions_nr, 5);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	for (uint32_t i = 0; i < algo.solutions_nr; i++) {
		EXPECT_EQ(algo.solutions[i].first, algo.solution_l);
		EXPECT_EQ(algo.solutions[i].second, algo.solution_r);
	}
}

/// mxf4 on element sizes which are not a multiple of 128 bits: inserts a
/// solution of weight d and one of weight d+1, searches, and checks.
/// NOTE: one alias per size, see `ParamsNN`
template<typename NN>
void check_mxf4_size(const size_t ls, const size_t l, const size_t r) {
	using T = uint32_t;
	constexpr uint32_t d = NN::d_;
	NN algo{};
	algo.generate_random_instance(false);
	algo.insert_solution(gen() % ls, gen() % ls, d + 1);
	algo.run([ls](const T *L, const T *R, size_t *f1, size_t *f2) {
		launch_tensor_bruteforce_mxf4<T, NN::ELEMENT_NR_LIMBS, d>(L, R, ls, ls, f1, f2); });
	EXPECT_EQ(algo.solutions_nr, 0);

	algo.insert_solution(l, r);
	algo.run([ls](const T *L, const T *R, size_t *f1, size_t *f2) {
		launch_tensor_bruteforce_mxf4<T, NN::ELEMENT_NR_LIMBS, d>(L, R, ls, ls, f1, f2); });
	ASSERT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	EXPECT_EQ(algo.solutions[0].first, l);
	EXPECT_EQ(algo.solutions[0].second, r);
}

/// `TensorNN` with the weight as a member, for `check_mxf4_size`
template<const uint32_t n, const uint32_t d, const size_t LIST_SIZE>
struct SizeNN : public TensorNN<n, d, LIST_SIZE> {
	constexpr static uint32_t d_ = d;
};

constexpr size_t SIZE_LS = 2000;
/// NOTE: n = 32 needs a tiny d: 4e6 random pairs contain ~5 of weight <= 3
using NN32 = SizeNN<32, 1, SIZE_LS>;
using NN64 = SizeNN<64, 8, SIZE_LS>;
using NN96 = SizeNN<96, 12, SIZE_LS>;
using NN160 = SizeNN<160, 20, SIZE_LS>;
using NN192 = SizeNN<192, 30, SIZE_LS>;
using NN512 = SizeNN<512, 100, SIZE_LS>;
using NN512s = SizeNN<512, 10, SIZE_LS>;

/// n = 32: a single word per element, one step
TEST(TensorBruteforceMXF4, n32) {
	check_mxf4_size<NN32>(SIZE_LS, 1234, SIZE_LS - 1);
}

/// n = 64: one step, uint2 copies
TEST(TensorBruteforceMXF4, n64) {
	check_mxf4_size<NN64>(SIZE_LS, 0, 7);
}

/// n = 96: odd number of words, the last step has a single word
TEST(TensorBruteforceMXF4, n96) {
	check_mxf4_size<NN96>(SIZE_LS, SIZE_LS - 1, 0);
}

/// n = 160: odd number of words, 3 steps
TEST(TensorBruteforceMXF4, n160) {
	check_mxf4_size<NN160>(SIZE_LS, 300, 301);
}

/// n = 192: 3 steps, row stride 6
TEST(TensorBruteforceMXF4, n192) {
	check_mxf4_size<NN192>(SIZE_LS, 999, 1000);
}

/// n = 512: 8 steps, row stride 18, large d: 2 steps before the check
TEST(TensorBruteforceMXF4, n512) {
	check_mxf4_size<NN512>(SIZE_LS, 17, 1999);
}

/// n = 512, small d: one step before the check
TEST(TensorBruteforceMXF4, n512_small_d) {
	check_mxf4_size<NN512s>(SIZE_LS, 18, 18);
}

/// the automatic choice of the steps before the early abort check
TEST(TensorBruteforceMXF4, FirstSteps) {
	// Pr[Bin(64, 1/2) <= 20] ~ 0.0017 (~20% of 128 pairs),
	// Pr[Bin(128, 1/2) <= 50] ~ 0.008 (~66% of 128 pairs)
	EXPECT_EQ(first_steps(20, 2), 2u);
	EXPECT_EQ(first_steps(20, 2, 128, 1.), 1u);
	EXPECT_EQ(first_steps(20, 4), 2u);
	EXPECT_EQ(first_steps(10, 4), 1u);
	EXPECT_EQ(first_steps(50, 4), 3u);
	EXPECT_EQ(first_steps(50, 4, 1, 1.), 2u);
	EXPECT_EQ(first_steps(100, 8), 4u);
	EXPECT_EQ(first_steps(10, 1), 1u);
	EXPECT_EQ(first_steps(60, 2), 2u);
	for (const uint32_t n : {8u, 64u, 128u, 512u}) {
		for (const uint32_t d : {0u, 1u, 20u, 50u, 100u, 512u}) {
			// reference via lgamma, as `filter_probability` in nn/full_length.cuh
			double p = 0.;
			for (uint32_t k = 0; (k <= d) && (k <= n); k++) {
				p += std::exp(std::lgamma(n + 1.) - std::lgamma(k + 1.) - std::lgamma(n - k + 1.) - n * std::log(2.));
			}
			p = p > 1. ? 1. : p;
			EXPECT_NEAR(binomial_tail(n, d), p, 1e-9 * p + 1e-300) << n << " " << d;
		}
	}
	EXPECT_EQ(binomial_tail(8, 8), 1.);
	EXPECT_EQ(mxf4_row_stride(1), 2u);
	EXPECT_EQ(mxf4_row_stride(4), 4u);
	EXPECT_EQ(mxf4_row_stride(8), 10u);
	EXPECT_EQ(mxf4_row_stride(16), 18u);
}

/// measures the throughput of `launch(L, R, f1, f2)` on two random lists of
/// `ls` elements of `n` bits, with at least 2^34 pairs in total.
/// \return Gpairs/s
template<const uint32_t n, typename F>
double measure(const size_t ls, F launch) {
	using T = uint32_t;
	constexpr uint32_t nr_limbs = n / 32u;
	T *L, *R;
	size_t *f1, *f2;
	checkCudaErrors(cudaMalloc(&L, sizeof(T) * ls * nr_limbs));
	checkCudaErrors(cudaMalloc(&R, sizeof(T) * ls * nr_limbs));
	checkCudaErrors(cudaMalloc(&f1, sizeof(size_t)));
	checkCudaErrors(cudaMalloc(&f2, sizeof(size_t)));
	checkCudaErrors(cudaMemset(f1, 0xff, sizeof(size_t)));
	checkCudaErrors(cudaMemset(f2, 0xff, sizeof(size_t)));

	std::vector<T> h(ls * nr_limbs);
	for (auto &x : h) { x = gen(); }
	checkCudaErrors(cudaMemcpy(L, h.data(), sizeof(T) * ls * nr_limbs, cudaMemcpyHostToDevice));
	for (auto &x : h) { x = gen(); }
	checkCudaErrors(cudaMemcpy(R, h.data(), sizeof(T) * ls * nr_limbs, cudaMemcpyHostToDevice));

	const double pairs = double(ls) * double(ls);
	const uint32_t iters = std::max(3.0, double(1ull << 34u) / pairs);

	cudaEvent_t start, end;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&end));
	// warmup
	launch(L, R, f1, f2);
	checkCudaErrors(cudaEventRecord(start));
	for (uint32_t i = 0; i < iters; i++) {
		launch(L, R, f1, f2);
	}
	checkCudaErrors(cudaEventRecord(end));
	checkCudaErrors(cudaEventSynchronize(end));
	float ms;
	checkCudaErrors(cudaEventElapsedTime(&ms, start, end));

	cudaEventDestroy(start);
	cudaEventDestroy(end);
	cudaFree(L);
	cudaFree(R);
	cudaFree(f1);
	cudaFree(f2);
	return pairs * iters / ms / 1e6;
}

/// compares all variants (with their default tuning parameters) with
/// `bruteforce` from `nn/nn.cuh` for list sizes 2^10, ..., 2^18.
template<const uint32_t n, const uint32_t d>
void bench() {
	using T = uint32_t;
	constexpr uint32_t nr_limbs = n / 32u;
	printf("n=%u d=%u [Gpairs/s]\n", n, d);
	printf("  log|L|     popc      xor      and   hybrid     mxf4\n");
	for (uint32_t l = 10; l <= 18; l++) {
		const size_t ls = size_t(1u) << l;
		const double popc = measure<n>(ls, [ls](T *L, T *R, size_t *f1, size_t *f2) {
			constexpr uint32_t threads = 1024;
			const uint32_t blocks = (ls + threads - 1) / threads;
			bruteforce<T, nr_limbs, d><<<blocks, threads>>>(L, R, ls, ls, f1, f2);
		});
		double x = 0., a = 0., h = 0.;
		if constexpr ((n % 128u) == 0) {
			x = measure<n>(ls, [ls](T *L, T *R, size_t *f1, size_t *f2) {
				launch_tensor_bruteforce<T, nr_limbs, d>(L, R, ls, ls, f1, f2);
			});
			a = measure<n>(ls, [ls](T *L, T *R, size_t *f1, size_t *f2) {
				launch_tensor_bruteforce_and<T, nr_limbs, d>(L, R, ls, ls, f1, f2);
			});
			h = measure<n>(ls, [ls](T *L, T *R, size_t *f1, size_t *f2) {
				launch_tensor_bruteforce_hybrid<T, nr_limbs, d>(L, R, ls, ls, f1, f2);
			});
		}
		const double m = measure<n>(ls, [ls](T *L, T *R, size_t *f1, size_t *f2) {
			launch_tensor_bruteforce_mxf4<T, nr_limbs, d>(L, R, ls, ls, f1, f2);
		});
		printf("  %6u %8.1f %8.1f %8.1f %8.1f %8.1f\n", l, popc, x, a, h, m);
	}
}

/// run with: --gtest_also_run_disabled_tests --gtest_filter=TensorBench.DISABLED_bench
TEST(TensorBench, DISABLED_bench) {
	bench<128, 20>();
	bench<256, 20>();
	bench<256, 50>();
}

/// the element sizes only the mxf4 variant supports (popc for comparison)
/// run with: --gtest_also_run_disabled_tests --gtest_filter=TensorBench.DISABLED_bench_sizes
TEST(TensorBench, DISABLED_bench_sizes) {
	bench<32, 3>();
	bench<64, 8>();
	bench<96, 12>();
	bench<192, 30>();
	bench<512, 100>();
}

/// prints the throughput of one set of tuning parameters, see `DISABLED_tune`
/// \tparam P1, P2 hybrid: POPC_WARPS, RC_POPC; mxf4: TN, MINB
/// \tparam S0 mxf4: steps before the early abort check, 0 = automatic
template<const Variant V, const uint32_t TM, const uint32_t WARPS, const uint32_t RC,
		 const uint32_t P1=0, const uint32_t P2=0, const uint32_t n=128, const uint32_t d=20,
		 const uint32_t S0=0>
void tune(const size_t ls) {
	using T = uint32_t;
	constexpr uint32_t nr_limbs = n / 32u;
	const double r = measure<n>(ls, [ls](T *L, T *R, size_t *f1, size_t *f2) {
		if constexpr (V == Variant::XOR) {
			launch_tensor_bruteforce<T, nr_limbs, d, TM, WARPS, RC>(L, R, ls, ls, f1, f2);
		} else if constexpr (V == Variant::AND) {
			launch_tensor_bruteforce_and<T, nr_limbs, d, TM, WARPS, RC>(L, R, ls, ls, f1, f2);
		} else if constexpr (V == Variant::HYBRID) {
			launch_tensor_bruteforce_hybrid<T, nr_limbs, d, TM, WARPS, RC, P1, P2>(L, R, ls, ls, f1, f2);
		} else {
			launch_tensor_bruteforce_mxf4<T, nr_limbs, d, TM, WARPS, RC, S0, P1, P2>(L, R, ls, ls, f1, f2);
		}
	});
	const char *name = V == Variant::XOR ? "xor" : V == Variant::AND ? "and" : V == Variant::HYBRID ? "hybrid" : "mxf4";
	printf("%-6s n=%3u d=%3u TM=%2u WARPS=%2u RC=%3u P1=%3u P2=%3u S0=%u: %7.1f Gpairs/s\n",
		   name, n, d, TM, WARPS, RC, P1, P2, S0, r);
}

/// sweep over the tuning parameters (n=128, d=20, |L|=|R|=2^17), which was
/// used to choose the defaults of the launchers in `nn/nn_tensor.cuh`.
/// run with: --gtest_also_run_disabled_tests --gtest_filter=TensorBench.DISABLED_tune
TEST(TensorBench, DISABLED_tune) {
	constexpr size_t ls = 1u << 17u;
	tune<Variant::XOR, 4, 8, 128>(ls);
	tune<Variant::XOR, 8, 8, 128>(ls);
	tune<Variant::XOR, 8, 4, 128>(ls);
	tune<Variant::XOR, 8, 16, 256>(ls);
	tune<Variant::XOR, 2, 8, 128>(ls);
	tune<Variant::XOR, 2, 8, 256>(ls);

	tune<Variant::AND, 4, 8, 128>(ls);
	tune<Variant::AND, 8, 8, 128>(ls);
	tune<Variant::AND, 4, 8, 256>(ls);
	tune<Variant::AND, 2, 8, 128>(ls);
	tune<Variant::AND, 2, 8, 256>(ls);
	tune<Variant::AND, 1, 8, 256>(ls);
	tune<Variant::AND, 2, 4, 256>(ls);
	tune<Variant::AND, 2, 16, 256>(ls);
	tune<Variant::AND, 4, 4, 256>(ls);

	tune<Variant::HYBRID, 4, 8, 128, 4, 32>(ls);
	tune<Variant::HYBRID, 4, 8, 128, 4, 64>(ls);
	tune<Variant::HYBRID, 4, 8, 128, 8, 80>(ls);
	tune<Variant::HYBRID, 4, 8, 256, 8, 136>(ls);
	tune<Variant::HYBRID, 4, 8, 256, 8, 144>(ls);
	tune<Variant::HYBRID, 4, 8, 256, 8, 160>(ls);
	tune<Variant::HYBRID, 4, 8, 256, 4, 160>(ls);
	tune<Variant::HYBRID, 2, 8, 256, 4, 96>(ls);
	tune<Variant::HYBRID, 2, 8, 256, 4, 112>(ls);
	tune<Variant::HYBRID, 2, 8, 256, 4, 128>(ls);
	tune<Variant::HYBRID, 2, 8, 256, 4, 144>(ls);
	tune<Variant::HYBRID, 2, 8, 256, 2, 128>(ls);
	tune<Variant::HYBRID, 2, 4, 256, 2, 128>(ls);
	tune<Variant::HYBRID, 2, 16, 256, 8, 160>(ls);
	tune<Variant::HYBRID, 4, 4, 256, 4, 160>(ls);
}

/// sweep over the tuning parameters of the mxf4 variant (n=128, d=20,
/// |L|=|R|=2^17)
/// run with: --gtest_also_run_disabled_tests --gtest_filter=TensorBench.DISABLED_tune_mxf4
TEST(TensorBench, DISABLED_tune_mxf4) {
	constexpr size_t ls = 1u << 17u;
	// TM, WARPS, RC, TN, MINB, n, d, S0 (0: automatic)
	tune<Variant::MXF4, 2, 8, 256, 2, 2>(ls);
	tune<Variant::MXF4, 2, 8, 256, 4, 2>(ls);
	tune<Variant::MXF4, 3, 8, 256, 2, 2>(ls);
	tune<Variant::MXF4, 4, 8, 256, 1, 2>(ls);
	tune<Variant::MXF4, 4, 8, 256, 2, 2>(ls);
	tune<Variant::MXF4, 4, 8, 128, 2, 2>(ls);
	tune<Variant::MXF4, 4, 8, 256, 2, 2, 128, 20, 1>(ls);
	tune<Variant::MXF4, 6, 8, 256, 1, 2>(ls);
	tune<Variant::MXF4, 6, 8, 256, 2, 2>(ls);
	tune<Variant::MXF4, 2, 12, 256, 2, 2>(ls);
	tune<Variant::MXF4, 2, 16, 256, 1, 2>(ls);
	tune<Variant::MXF4, 2, 16, 256, 2, 2>(ls);
	tune<Variant::MXF4, 2, 16, 256, 4, 2>(ls);
	tune<Variant::MXF4, 3, 16, 256, 1, 2>(ls);
	tune<Variant::MXF4, 3, 16, 256, 2, 2>(ls);
	tune<Variant::MXF4, 4, 16, 256, 1, 2>(ls);
	tune<Variant::MXF4, 4, 16, 256, 1, 1>(ls);
	tune<Variant::MXF4, 4, 12, 256, 1, 2>(ls);
	tune<Variant::MXF4, 4, 12, 256, 2, 2>(ls);
	tune<Variant::MXF4, 4, 4, 256, 2, 4>(ls);
	tune<Variant::MXF4, 2, 8, 256, 2, 3>(ls);
	tune<Variant::MXF4, 2, 8, 256, 4, 3>(ls);
	tune<Variant::MXF4, 3, 8, 256, 2, 3>(ls);
	// n = 256
	tune<Variant::MXF4, 2, 8, 256, 2, 2, 256, 20>(ls);
	tune<Variant::MXF4, 2, 8, 256, 4, 2, 256, 20>(ls);
	tune<Variant::MXF4, 3, 8, 256, 2, 2, 256, 20>(ls);
	tune<Variant::MXF4, 4, 8, 256, 1, 2, 256, 20>(ls);
	tune<Variant::MXF4, 4, 8, 256, 2, 2, 256, 20>(ls);
	tune<Variant::MXF4, 4, 8, 256, 1, 1, 256, 20>(ls);
	tune<Variant::MXF4, 2, 16, 256, 2, 2, 256, 20>(ls);
	tune<Variant::MXF4, 2, 12, 256, 2, 2, 256, 20>(ls);
	tune<Variant::MXF4, 2, 8, 256, 2, 3, 256, 20>(ls);
	tune<Variant::MXF4, 2, 8, 256, 2, 2, 256, 20, 1>(ls);
	tune<Variant::MXF4, 2, 8, 256, 2, 2, 256, 50>(ls);
	tune<Variant::MXF4, 3, 8, 256, 2, 2, 256, 50>(ls);
	tune<Variant::MXF4, 4, 8, 256, 1, 2, 256, 50>(ls);
	tune<Variant::MXF4, 4, 8, 256, 2, 2, 256, 50>(ls);
	tune<Variant::MXF4, 2, 16, 256, 2, 2, 256, 50>(ls);
	// n = 512
	tune<Variant::MXF4, 1, 8, 256, 2, 2, 512, 100>(ls);
	tune<Variant::MXF4, 1, 8, 256, 4, 2, 512, 100>(ls);
	tune<Variant::MXF4, 1, 16, 256, 2, 2, 512, 100>(ls);
	tune<Variant::MXF4, 2, 8, 256, 2, 2, 512, 100>(ls);
	tune<Variant::MXF4, 2, 8, 256, 1, 2, 512, 100>(ls);
	tune<Variant::MXF4, 2, 8, 256, 2, 1, 512, 100>(ls);
	tune<Variant::MXF4, 2, 8, 256, 2, 2, 512, 100, 3>(ls);
	tune<Variant::MXF4, 2, 8, 256, 2, 2, 512, 100, 5>(ls);
	// n = 64
	tune<Variant::MXF4, 4, 8, 256, 2, 2, 64, 8>(ls);
	tune<Variant::MXF4, 8, 8, 256, 1, 2, 64, 8>(ls);
	tune<Variant::MXF4, 8, 8, 256, 2, 2, 64, 8>(ls);
	tune<Variant::MXF4, 6, 8, 256, 2, 2, 64, 8>(ls);
	tune<Variant::MXF4, 4, 16, 256, 2, 2, 64, 8>(ls);
	tune<Variant::MXF4, 6, 16, 256, 1, 2, 64, 8>(ls);
	tune<Variant::MXF4, 4, 12, 256, 2, 2, 64, 8>(ls);
}

int main(int argc, char **argv) {
	const uint64_t seed = time(NULL);
	printf("seed: %lu\n", seed);
	gen.seed(seed);
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
