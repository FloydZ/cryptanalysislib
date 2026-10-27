#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <random>
#include <vector>

// the roots below the bruteforce threshold on the tensor cores
#define NN_CUDA_TENSOR
#include "helper.cuh"
#include "nn/nn.cuh"

using ::testing::InitGoogleTest;
using ::testing::Test;

static std::mt19937_64 gen;

/// the tree implementation of `nn/nn.cuh` (`nn_tree_root`, `nn_tree_child`
/// and the tensor core bruteforce of the roots), launched as in
/// `NN::cuda_launch` of nn/nn.h. The parameters are those of `NN_Config`.
/// \tparam n number of bits of each element
/// \tparam r number of limbs the tree filters on (depth)
/// \tparam N branching factor of the tree
/// \tparam k number of bits of each limb the filter uses
/// \tparam LS number of elements in each list
/// \tparam dk weight bound of the filter
/// \tparam d weight bound of the solution
/// \tparam bf bruteforce threshold
template<const uint32_t n, const uint32_t r, const uint32_t N, const uint32_t k,
         const size_t LS, const uint32_t dk, const uint32_t d, const uint32_t bf>
class TreeNN {
public:
	using T = uint32_t;
	using LT = std::conditional_t<(k <= 32), uint32_t, uint64_t>;
	constexpr static uint32_t nr_limbs = n / 32u;
	constexpr static uint32_t THREADS = 256;
	/// limbs per element of the dense lists of the tensor core bruteforce
	constexpr static uint32_t TL = ((n + 127u) / 128u) * 4u;
	static_assert(n % 32u == 0);
	static_assert(r * sizeof(LT) <= nr_limbs * sizeof(T));

	T *L1 = nullptr, *L2 = nullptr;
	uint32_t *root = nullptr, *B = nullptr, *cnt = nullptr, *tcnt = nullptr;
	size_t *f = nullptr, *F = nullptr;
	T *DL = nullptr, *DR = nullptr;
	uint32_t blocks = 0;
	size_t solution_l = 0, solution_r = 0;

	TreeNN() {
		checkCudaErrors(cudaMallocManaged(&L1, sizeof(T) * LS * nr_limbs));
		checkCudaErrors(cudaMallocManaged(&L2, sizeof(T) * LS * nr_limbs));
		checkCudaErrors(cudaMallocManaged(&f, 2 * sizeof(size_t)));
		checkCudaErrors(cudaMalloc(&root, size_t(N) * 2 * LS * sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&cnt, (2 * N + 1) * sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&DL, size_t(N) * LS * TL * sizeof(T)));
		checkCudaErrors(cudaMalloc(&DR, size_t(N) * LS * TL * sizeof(T)));
		checkCudaErrors(cudaMalloc(&tcnt, 2 * N * sizeof(uint32_t)));
		checkCudaErrors(cudaMalloc(&F, 2 * N * sizeof(size_t)));

		int sms = 0, occupancy = 0;
		checkCudaErrors(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));
		checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
				&occupancy, nn_tree_child<T, LT, nr_limbs, d, dk, k, N, r, false>, THREADS, 0));
		blocks = sms * occupancy;
		checkCudaErrors(cudaMalloc(&B, size_t(blocks) * 2 * (r > 1 ? r - 1 : 1) * LS * sizeof(uint32_t)));
	}

	~TreeNN() {
		cudaFree(L1); cudaFree(L2); cudaFree(f); cudaFree(root); cudaFree(cnt);
		cudaFree(DL); cudaFree(DR); cudaFree(tcnt); cudaFree(F); cudaFree(B);
	}

	/// random lists, and a pair of weight exactly `d` at random positions if
	/// `insert_sol`
	void generate_random_instance(const bool insert_sol = true) noexcept {
		for (size_t i = 0; i < LS * nr_limbs; i++) {
			L1[i] = gen();
			L2[i] = gen();
		}
		if (insert_sol) {
			insert_solution(gen() % LS, gen() % LS);
		}
	}

	/// sets L2[rr] = L1[l] ^ e with wt(e) = w
	void insert_solution(const size_t l, const size_t rr, const uint32_t w = d) noexcept {
		solution_l = l;
		solution_r = rr;
		for (uint32_t i = 0; i < nr_limbs; i++) {
			L2[rr * nr_limbs + i] = L1[l * nr_limbs + i];
		}
		std::vector<uint32_t> pos(n);
		for (uint32_t i = 0; i < n; i++) { pos[i] = i; }
		std::shuffle(pos.begin(), pos.end(), gen);
		for (uint32_t i = 0; i < w; i++) {
			L2[rr * nr_limbs + pos[i] / 32u] ^= T(1u) << (pos[i] % 32u);
		}
	}

	[[nodiscard]] uint32_t weight(const size_t l, const size_t rr) const noexcept {
		uint32_t wt = 0;
		for (uint32_t j = 0; j < nr_limbs; j++) {
			wt += __builtin_popcount(L1[l * nr_limbs + j] ^ L2[rr * nr_limbs + j]);
		}
		return wt;
	}

	/// one run of the tree with a fresh seed, as `NN::cuda_launch`
	/// \tparam tensor if true the roots below `bf` are bruteforced on the
	/// 		tensor cores
	/// \return true if a pair was found; its positions are in f[0], f[1]
	template<const bool tensor>
	bool run(const size_t e1 = LS, const size_t e2 = LS) noexcept {
		checkCudaErrors(cudaMemset(cnt, 0, (2 * N + 1) * sizeof(uint32_t)));
		checkCudaErrors(cudaMemset(f, 0xff, 2 * sizeof(size_t)));
		const uint64_t seed = gen();
		const size_t e = e1 > e2 ? e1 : e2;
		const dim3 grid((e + 4 * THREADS - 1) / (4 * THREADS), N);
		nn_tree_root<T, LT, nr_limbs, dk, k><<<grid, THREADS>>>
			(L1, L2, e1, e2, root, root + N * LS, cnt, LS, seed);
		checkCudaErrors(cudaGetLastError());
		if constexpr (tensor) {
			checkCudaErrors(cudaMemset(F, 0xff, 2 * N * sizeof(size_t)));
			nn_tree_gather<T, nr_limbs, TL, r><<<grid, THREADS>>>
				(L1, L2, root, root + N * LS, cnt, DL, DR, tcnt, LS, bf);
			checkCudaErrors(cudaGetLastError());
			nn_tree_tensor<T, TL, d><<<dim3(32, 1, N), (8 + 4) * 32>>>(DL, DR, tcnt, LS, F);
			checkCudaErrors(cudaGetLastError());
			nn_tree_tensor_result<<<1, 1>>>(root, root + N * LS, F, N, LS, f, f + 1);
			checkCudaErrors(cudaGetLastError());
		}
		nn_tree_child<T, LT, nr_limbs, d, dk, k, N, r, tensor><<<blocks, THREADS>>>
			(L1, L2, root, root + N * LS, cnt, B, LS, bf, f, f + 1, cnt + 2 * N, seed);
		checkCudaErrors(cudaGetLastError());
		checkCudaErrors(cudaDeviceSynchronize());
		return f[0] != size_t(-1);
	}

	/// checks the survivors of all roots: unique positions < e, and the
	/// counts match
	void check_roots(const size_t e1, const size_t e2) const {
		std::vector<uint32_t> h(size_t(N) * 2 * LS), c(2 * N);
		checkCudaErrors(cudaMemcpy(h.data(), root, h.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(c.data(), cnt, c.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost));
		for (uint32_t y = 0; y < N; y++) {
			for (uint32_t list = 0; list < 2; list++) {
				const size_t e = list ? e2 : e1;
				ASSERT_LE(c[2 * y + list], e);
				std::vector<uint32_t> s(h.begin() + (list * N + y) * LS, h.begin() + (list * N + y) * LS + c[2 * y + list]);
				std::sort(s.begin(), s.end());
				ASSERT_EQ(std::unique(s.begin(), s.end()), s.end()) << "root " << y;
				if (!s.empty()) { ASSERT_LT(s.back(), e); }
				// Pr[wt(32 or 64 random bits) <= dk] * e survivors in expectation
				// ASSERT_GT(c[2 * y + list], 0u);
			}
		}
	}
};

/// one tree run must find a pair of weight <= d; a few runs, as each one
/// succeeds only with high probability
template<typename NN, const bool tensor>
void check_found(NN &algo, const uint32_t tries = 5) {
	bool found = false;
	for (uint32_t i = 0; (i < tries) && !found; i++) {
		found = algo.template run<tensor>();
		if (found) {
			ASSERT_LT(algo.f[0], NN::nr_limbs ? (size_t)-1 : 0);
			EXPECT_LE(algo.weight(algo.f[0], algo.f[1]), NN::nr_limbs ? 1u << 30u : 0u);
		}
	}
	EXPECT_TRUE(found);
}

/// n = 128, 2 limbs of 64 bits: the parameters of
/// tests/nn/parameters.cpp (MO640Params_n128_r2_64, "100%")
constexpr size_t LS14 = 1u << 14u;
using NN128 = TreeNN<128, 2, 250, 64, LS14, 24, 11, 600>;
/// n = 256, 8 limbs of 32 bits (tests/nn/parameters.cpp, `config`)
using NN256 = TreeNN<256, 8, 150, 32, LS14, 10, 14, 1000>;
/// n = 256, 4 limbs of 64 bits (tests/nn/simd.cpp)
using NN256_64 = TreeNN<256, 4, 320, 64, LS14, 22, 16, 512>;
/// r = 1: the roots are always bruteforced
using NN64_r1 = TreeNN<64, 1, 64, 32, LS14, 10, 5, 512>;
/// k = 20 < 32: masked limbs
using NN96 = TreeNN<96, 3, 100, 20, LS14, 6, 6, 700>;

TEST(NNTree, n128_r2) {
	NN128 algo{};
	algo.generate_random_instance();
	bool found = false;
	for (uint32_t i = 0; (i < 5) && !found; i++) {
		found = algo.run<false>();
	}
	ASSERT_TRUE(found);
	EXPECT_LE(algo.weight(algo.f[0], algo.f[1]), 11u);
	// any pair of weight <= d is a valid solution, but random pairs of
	// weight <= 11 on 128 bits do not exist in 2^28 pairs
	EXPECT_EQ(algo.f[0], algo.solution_l);
	EXPECT_EQ(algo.f[1], algo.solution_r);
	algo.check_roots(LS14, LS14);
}

TEST(NNTree, n128_r2_tensor) {
	NN128 algo{};
	algo.generate_random_instance();
	bool found = false;
	for (uint32_t i = 0; (i < 5) && !found; i++) {
		found = algo.run<true>();
	}
	ASSERT_TRUE(found);
	EXPECT_EQ(algo.f[0], algo.solution_l);
	EXPECT_EQ(algo.f[1], algo.solution_r);
}

TEST(NNTree, n256_r8) {
	NN256 algo{};
	algo.generate_random_instance();
	bool found = false;
	for (uint32_t i = 0; (i < 5) && !found; i++) {
		found = algo.run<false>();
	}
	ASSERT_TRUE(found);
	EXPECT_EQ(algo.f[0], algo.solution_l);
	EXPECT_EQ(algo.f[1], algo.solution_r);
	algo.check_roots(LS14, LS14);
}

TEST(NNTree, n256_r8_tensor) {
	NN256 algo{};
	algo.generate_random_instance();
	bool found = false;
	for (uint32_t i = 0; (i < 5) && !found; i++) {
		found = algo.run<true>();
	}
	ASSERT_TRUE(found);
	EXPECT_EQ(algo.f[0], algo.solution_l);
	EXPECT_EQ(algo.f[1], algo.solution_r);
}

TEST(NNTree, n256_r4_64) {
	NN256_64 algo{};
	algo.generate_random_instance();
	bool found = false;
	for (uint32_t i = 0; (i < 5) && !found; i++) {
		found = algo.run<false>();
	}
	ASSERT_TRUE(found);
	EXPECT_EQ(algo.f[0], algo.solution_l);
	EXPECT_EQ(algo.f[1], algo.solution_r);
}

/// r = 1: every root is bruteforced by the children (slices) resp. by the
/// tensor cores. NOTE: a single run succeeds only with ~60%: the pair
/// survives a root with ~1.5%, and there are N = 64 roots.
TEST(NNTree, n64_r1) {
	NN64_r1 algo{};
	algo.generate_random_instance();
	bool found = false;
	for (uint32_t i = 0; (i < 10) && !found; i++) {
		found = algo.run<false>();
	}
	ASSERT_TRUE(found);
	EXPECT_LE(algo.weight(algo.f[0], algo.f[1]), 5u);
	found = false;
	for (uint32_t i = 0; (i < 10) && !found; i++) {
		found = algo.run<true>();
	}
	ASSERT_TRUE(found);
	EXPECT_LE(algo.weight(algo.f[0], algo.f[1]), 5u);
}

/// k = 20 bits of each 32 bit limb, 3 limbs of 96
TEST(NNTree, n96_k20) {
	NN96 algo{};
	algo.generate_random_instance();
	bool found = false;
	for (uint32_t i = 0; (i < 5) && !found; i++) {
		found = algo.run<false>();
	}
	ASSERT_TRUE(found);
	EXPECT_EQ(algo.f[0], algo.solution_l);
	EXPECT_EQ(algo.f[1], algo.solution_r);
}

TEST(NNTree, NoSolution) {
	NN128 algo{};
	algo.generate_random_instance(false);
	EXPECT_FALSE(algo.run<false>());
	EXPECT_FALSE(algo.run<true>());
}

/// the solution outside of e1, e2 is not found
TEST(NNTree, Bounds) {
	NN128 algo{};
	algo.generate_random_instance(false);
	algo.insert_solution(LS14 - 1, 0);
	EXPECT_FALSE(algo.run<false>(LS14 - 1, LS14));
	EXPECT_FALSE(algo.run<true>(LS14 - 1, LS14));
	bool found = false;
	for (uint32_t i = 0; (i < 5) && !found; i++) {
		found = algo.run<false>(LS14, 1);
	}
	EXPECT_TRUE(found);
}

/// \return ms of one run (the tree searches until the first pair is found)
template<typename NN, const bool tensor>
double measure(NN &algo, const uint32_t runs = 20) {
	cudaEvent_t start, end;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&end));
	checkCudaErrors(cudaEventRecord(start));
	uint32_t found = 0, wrong = 0;
	for (uint32_t i = 0; i < runs; i++) {
		if (algo.template run<tensor>()) {
			found += 1;
			wrong += algo.weight(algo.f[0], algo.f[1]) > NN::nr_limbs * 0 + algo.weight(algo.solution_l, algo.solution_r);
		}
	}
	checkCudaErrors(cudaEventRecord(end));
	checkCudaErrors(cudaEventSynchronize(end));
	float ms;
	checkCudaErrors(cudaEventElapsedTime(&ms, start, end));
	printf("  %u/%u found, %u wrong\n", found, runs, wrong);
	return ms / runs;
}

/// NOTE: the index buffers `B` of `nn_tree_child` are blocks * 2 * (r-1) *
/// 		LS entries: for |L| = 2^20, r = 8 and ~1400 persistent blocks that
/// 		is 80 GB, hence 2^16 here (`NN::cuda_alloc` has the same problem).
/// run with: --gtest_also_run_disabled_tests --gtest_filter=NNTreeBench.DISABLED_bench
TEST(NNTreeBench, DISABLED_bench) {
	constexpr size_t LS = 1u << 16u;
	{
		// n=256 r=8 k=32: tests/nn/parameters.cpp
		TreeNN<256, 8, 150, 32, LS, 10, 14, 1000> algo{};
		algo.generate_random_instance();
		printf("n=256 r=8 N=150 k=32 dk=10 d=14 bf=1000 |L|=2^16, %u persistent blocks\n", algo.blocks);
		printf("  tensor roots:  %.1f ms\n", measure<decltype(algo), true>(algo));
		printf("  children only: %.1f ms\n", measure<decltype(algo), false>(algo));
		printf("  tensor roots:  %.1f ms\n", measure<decltype(algo), true>(algo));
		printf("  children only: %.1f ms\n", measure<decltype(algo), false>(algo));
	}
	{
		TreeNN<128, 2, 250, 64, LS, 24, 11, 600> algo{};
		algo.generate_random_instance();
		printf("n=128 r=2 N=250 k=64 dk=24 d=11 bf=600 |L|=2^16, %u persistent blocks\n", algo.blocks);
		printf("  children only: %.1f ms\n", measure<decltype(algo), false>(algo));
		printf("  tensor roots:  %.1f ms\n", measure<decltype(algo), true>(algo));
	}
}

int main(int argc, char **argv) {
	const uint64_t seed = time(NULL);
	printf("seed: %lu\n", seed);
	gen.seed(seed);
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
