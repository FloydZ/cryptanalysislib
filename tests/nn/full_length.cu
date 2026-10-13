#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <random>
#include <utility>
#include <vector>

#include "helper.cuh"
#include "nn/full_length.cuh"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib::nn::full_length;

static std::mt19937_64 gen;

/// \return the smallest t with filter_probability(n, t) >= p
uint32_t threshold(const uint32_t n, const double p) {
	uint32_t t = 0;
	while (filter_probability(n, t) < p) {
		t++;
	}
	return t;
}

/// random list of N elements of W words in managed memory, and S random
/// centers, plus the reference masks computed on the CPU.
template<const uint32_t W, const uint32_t S>
struct Instance {
	size_t N;
	uint32_t t;
	uint32_t *L = nullptr;
	std::vector<uint32_t> centers;
	std::vector<uint64_t> masks;
	Filters<W, S> P;

	Instance(const size_t N, const uint32_t t) : N(N), t(t), centers(S * W) {
		checkCudaErrors(cudaMallocManaged(&L, sizeof(uint32_t) * (N ? N : 1) * W));
		for (size_t i = 0; i < N * W; i++) { L[i] = gen(); }
		for (auto &c : centers) { c = gen(); }
		update();
	}

	~Instance() { cudaFree(L); }

	/// recomputes `P` and the reference masks, after `L`, `centers` or `t`
	/// were changed
	void update() {
		P = make_filters<uint32_t, W, S>(centers.data(), t);
		masks.assign(N, 0);
		for (size_t i = 0; i < N; i++) {
			for (uint32_t c = 0; c < S; c++) {
				uint32_t d = 0;
				for (uint32_t w = 0; w < W; w++) {
					d += __builtin_popcount(L[i * W + w] ^ centers[c * W + w]);
				}
				if (d <= t) {
					masks[i] |= uint64_t(1u) << c;
				}
			}
		}
	}

	/// sets L[i] = centers[c] ^ e with wt(e) = w
	void set_distance(const size_t i, const uint32_t c, const uint32_t w) {
		std::vector<uint32_t> pos(32 * W);
		for (uint32_t k = 0; k < 32 * W; k++) { pos[k] = k; }
		std::shuffle(pos.begin(), pos.end(), gen);
		for (uint32_t k = 0; k < W; k++) { L[i * W + k] = centers[c * W + k]; }
		for (uint32_t k = 0; k < w; k++) { L[i * W + pos[k] / 32] ^= 1u << (pos[k] % 32); }
	}
};

/// kernel configuration of `launch_filter`
template<const uint32_t TW_, const uint32_t PW_, const uint32_t E_, const TensorOp OP_ = TensorOp::AUTO>
struct Config {
	constexpr static uint32_t TW = TW_, PW = PW_, E = E_;
	constexpr static TensorOp OP = OP_;
};

/// runs `launch_filter` with configuration `C`
/// \return the buckets, the caller must free them
template<typename C, const uint32_t W, const uint32_t S>
Buckets run_filter(const Instance<W, S> &I, uint64_t *M, uint32_t cap = 0, const double p = -1.) {
	if (cap == 0) {
		cap = bucket_capacity(I.N, 32 * W, I.t);
	}
	Buckets B = alloc_buckets(S, cap);
	launch_filter<uint32_t, W, S, C::TW, C::PW, C::E, C::OP>(I.L, I.N, I.P, M, B, p);
	checkCudaErrors(cudaDeviceSynchronize());
	return B;
}

/// copies the buckets to the host
struct HostBuckets {
	std::vector<uint32_t> index, count;
	uint32_t capacity;

	explicit HostBuckets(const Buckets &B) : index(size_t(B.nr_buckets) * B.capacity),
	                                         count(B.nr_buckets),
	                                         capacity(B.capacity) {
		checkCudaErrors(cudaMemcpy(index.data(), B.index, sizeof(uint32_t) * index.size(), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(count.data(), B.count, sizeof(uint32_t) * count.size(), cudaMemcpyDeviceToHost));
	}
};

/// checks that bucket c contains exactly the elements with bit c set in
/// `masks`
void check_buckets(const Buckets &B, const std::vector<uint64_t> &masks) {
	const HostBuckets H(B);
	for (uint32_t c = 0; c < B.nr_buckets; c++) {
		std::vector<uint32_t> expected;
		for (size_t i = 0; i < masks.size(); i++) {
			if ((masks[i] >> c) & 1u) { expected.push_back(i); }
		}

		ASSERT_EQ(H.count[c], expected.size()) << "bucket " << c;
		ASSERT_LE(H.count[c], B.capacity);
		std::vector<uint32_t> got(H.index.begin() + size_t(c) * B.capacity,
		                          H.index.begin() + size_t(c) * B.capacity + H.count[c]);
		std::sort(got.begin(), got.end());
		ASSERT_EQ(got, expected) << "bucket " << c;
	}
}

/// runs `launch_filter` with configuration `C` and checks the masks and the
/// buckets against the reference
template<typename C, const uint32_t W, const uint32_t S>
void check_filter(const Instance<W, S> &I, const double p = -1., const uint32_t cap = 0) {
	uint64_t *M;
	checkCudaErrors(cudaMallocManaged(&M, sizeof(uint64_t) * (I.N ? I.N : 1)));
	Buckets B = run_filter<C>(I, M, cap, p);
	for (size_t i = 0; i < I.N; i++) {
		ASSERT_EQ(M[i], I.masks[i]) << "element " << i;
	}
	check_buckets(B, I.masks);
	free_buckets(B);
	cudaFree(M);
}

/// all tests below run for every configuration
template<typename C>
class FullLengthFilter : public Test {};
/// NOTE: MXF4 falls back to S8 if not compiled for sm_120a
using Configs = ::testing::Types<Config<0, 8, 2048>,                    // CUDA cores only
                                 Config<8, 0, 2048, TensorOp::B1>,      // tensor cores only
                                 Config<8, 0, 2048, TensorOp::S8>,
                                 Config<8, 0, 4096, TensorOp::MXF4>,
                                 Config<4, 4, 1024>>;                   // both
TYPED_TEST_SUITE(FullLengthFilter, Configs);

TYPED_TEST(FullLengthFilter, n128_s64) {
	Instance<4, 64> I(1u << 15u, threshold(128, 1. / 32.));
	check_filter<TypeParam>(I);
}

TYPED_TEST(FullLengthFilter, n256_s64) {
	Instance<8, 64> I(1u << 14u, threshold(256, 1. / 32.));
	check_filter<TypeParam>(I);
}

/// 3 words: zero padded 128 bit chunk on the tensor cores
TYPED_TEST(FullLengthFilter, n96_s40) {
	Instance<3, 40> I(10000, threshold(96, 1. / 16.));
	check_filter<TypeParam>(I);
}

/// m16n8k256 + m16n8k128, 3 n-tiles with 7 padding columns
TYPED_TEST(FullLengthFilter, n384_s17) {
	Instance<12, 17> I(10000, threshold(384, 1. / 8.));
	check_filter<TypeParam>(I);
}

TYPED_TEST(FullLengthFilter, n512_s64) {
	Instance<16, 64> I(10000, threshold(512, 1. / 32.));
	check_filter<TypeParam>(I);
}

/// 2 words: uint2 loads on the CUDA cores
TYPED_TEST(FullLengthFilter, n64_s33) {
	Instance<2, 33> I(10000, threshold(64, 1. / 32.));
	check_filter<TypeParam>(I);
}

TYPED_TEST(FullLengthFilter, n128_s1) {
	Instance<4, 1> I(10000, threshold(128, 1. / 4.));
	check_filter<TypeParam>(I);
}

/// N is not a multiple of 32 or E
TYPED_TEST(FullLengthFilter, Tail) {
	for (const size_t N : {size_t(1), size_t(31), size_t(33), size_t(1001), size_t(4097)}) {
		Instance<4, 64> I(N, threshold(128, 1. / 8.));
		check_filter<TypeParam>(I);
	}
}

TYPED_TEST(FullLengthFilter, Empty) {
	Instance<4, 64> I(0, 50);
	check_filter<TypeParam>(I);
}

/// elements at distance exactly t and t+1 of a center
TYPED_TEST(FullLengthFilter, Boundary) {
	constexpr uint32_t t = 40;
	Instance<4, 64> I(4096, t);
	for (size_t i = 0; i < I.N; i++) {
		I.set_distance(i, i % 64, t + (i / 64) % 2);
	}
	I.update();
	for (size_t i = 0; i < I.N; i++) {
		ASSERT_EQ((I.masks[i] >> (i % 64)) & 1u, (i / 64) % 2 == 0);
	}
	// 32 elements per bucket, far more than for random elements
	check_filter<TypeParam>(I, -1., 64);
}

/// every element passes every filter, and none
TYPED_TEST(FullLengthFilter, AllNone) {
	Instance<4, 64> I(3000, 128);
	check_filter<TypeParam>(I);
	for (size_t i = 0; i < I.N; i++) {
		ASSERT_EQ(I.masks[i], ~uint64_t(0));
	}

	I.t = 0;
	I.update();
	check_filter<TypeParam>(I);
}

/// a far too small `p`: the staging area overflows, the hits are inserted via
/// global atomics instead
TYPED_TEST(FullLengthFilter, StagingOverflow) {
	Instance<4, 64> I(1u << 14u, threshold(128, 1. / 8.));
	check_filter<TypeParam>(I, 1e-9);
}

/// too small buckets: the count is correct, the stored entries are a subset
TYPED_TEST(FullLengthFilter, BucketOverflow) {
	Instance<4, 64> I(1u << 14u, threshold(128, 1. / 32.));
	constexpr uint32_t cap = 100;
	uint64_t *M;
	checkCudaErrors(cudaMalloc(&M, sizeof(uint64_t) * I.N));
	Buckets B = run_filter<TypeParam>(I, M, cap);
	const HostBuckets H(B);
	for (uint32_t c = 0; c < 64; c++) {
		uint32_t expected = 0;
		for (size_t i = 0; i < I.N; i++) { expected += (I.masks[i] >> c) & 1u; }
		ASSERT_EQ(H.count[c], expected);
		ASSERT_GT(H.count[c], cap);

		std::vector<uint32_t> got(H.index.begin() + c * cap, H.index.begin() + (c + 1) * cap);
		std::sort(got.begin(), got.end());
		ASSERT_EQ(std::unique(got.begin(), got.end()), got.end());
		for (uint32_t e = 0; e < cap; e++) {
			ASSERT_LT(H.index[c * cap + e], I.N);
			ASSERT_EQ((I.masks[H.index[c * cap + e]] >> c) & 1u, 1u);
		}
	}
	free_buckets(B);
	cudaFree(M);
}

/// \return all pairs (i, j) with m1[i] & m2[j] != 0, sorted
std::vector<std::pair<uint32_t, uint32_t>> reference_pairs(const std::vector<uint64_t> &m1,
                                                           const std::vector<uint64_t> &m2) {
	std::vector<std::pair<uint32_t, uint32_t>> ret;
	for (uint32_t i = 0; i < m1.size(); i++) {
		if (!m1[i]) { continue; }
		for (uint32_t j = 0; j < m2.size(); j++) {
			if (m1[i] & m2[j]) { ret.emplace_back(i, j); }
		}
	}
	return ret;
}

/// runs the join and compares the pairs with the reference
/// \param out_capacity if != size_t(-1): capacity of the output
void check_join(const Buckets &B1, const uint64_t *M1, const Buckets &B2, const uint64_t *M2,
                const std::vector<uint64_t> &m1, const std::vector<uint64_t> &m2,
                size_t out_capacity = size_t(-1)) {
	const auto expected = reference_pairs(m1, m2);
	unsigned long long *cnt;
	checkCudaErrors(cudaMallocManaged(&cnt, sizeof(unsigned long long)));

	// count only
	launch_join(B1, M1, B2, M2, nullptr, 0, cnt);
	checkCudaErrors(cudaDeviceSynchronize());
	ASSERT_EQ(*cnt, expected.size());

	if (out_capacity == size_t(-1)) {
		out_capacity = expected.size();
	}
	uint2 *out;
	checkCudaErrors(cudaMallocManaged(&out, sizeof(uint2) * (out_capacity ? out_capacity : 1)));
	launch_join(B1, M1, B2, M2, out, out_capacity, cnt);
	checkCudaErrors(cudaDeviceSynchronize());
	ASSERT_EQ(*cnt, expected.size());

	const size_t n = std::min(out_capacity, expected.size());
	std::vector<std::pair<uint32_t, uint32_t>> got(n);
	for (size_t i = 0; i < n; i++) {
		got[i] = {out[i].x, out[i].y};
	}
	std::sort(got.begin(), got.end());
	if (n == expected.size()) {
		ASSERT_EQ(got, expected);
	} else {
		// a subset without duplicates
		ASSERT_EQ(std::unique(got.begin(), got.end()), got.end());
		for (const auto &e : got) {
			ASSERT_TRUE(std::binary_search(expected.begin(), expected.end(), e));
		}
	}

	cudaFree(out);
	cudaFree(cnt);
}

/// the full pipeline: filter both lists, join
template<typename C, const uint32_t W, const uint32_t S>
void check_pipeline(const size_t N1, const size_t N2, const uint32_t t,
                    const size_t out_capacity = size_t(-1)) {
	Instance<W, S> I1(N1, t), I2(N2, t);
	I2.centers = I1.centers;
	I2.update();
	uint64_t *M1, *M2;
	checkCudaErrors(cudaMalloc(&M1, sizeof(uint64_t) * N1));
	checkCudaErrors(cudaMalloc(&M2, sizeof(uint64_t) * N2));
	Buckets B1 = run_filter<C>(I1, M1);
	Buckets B2 = run_filter<C>(I2, M2);
	check_join(B1, M1, B2, M2, I1.masks, I2.masks, out_capacity);
	free_buckets(B1);
	free_buckets(B2);
	cudaFree(M1);
	cudaFree(M2);
}

TYPED_TEST(FullLengthFilter, Join_n128_s64) {
	check_pipeline<TypeParam, 4, 64>(5000, 7000, threshold(128, 1. / 32.));
}

/// large buckets, i.e. more than one shared memory tile per bucket
TYPED_TEST(FullLengthFilter, Join_n128_s4) {
	check_pipeline<TypeParam, 4, 4>(20000, 3000, threshold(128, 1. / 4.));
}

TYPED_TEST(FullLengthFilter, Join_n256_s33) {
	check_pipeline<TypeParam, 8, 33>(3000, 2000, threshold(256, 1. / 16.));
}

/// every pair matches every filter: all pairs are emitted in bucket 0 only
TYPED_TEST(FullLengthFilter, Join_All) {
	check_pipeline<TypeParam, 4, 64>(300, 500, 128);
}

TYPED_TEST(FullLengthFilter, Join_None) {
	check_pipeline<TypeParam, 4, 64>(300, 500, 0);
}

/// too small output: the count is correct, the output is a subset
TYPED_TEST(FullLengthFilter, Join_OutputOverflow) {
	check_pipeline<TypeParam, 4, 64>(3000, 3000, threshold(128, 1. / 32.), 1000);
}

/// measures `launch()` on the default stream
/// \return ms per call
template<typename F>
double measure(F launch, const uint32_t iters = 10) {
	cudaEvent_t start, end;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&end));
	launch();
	checkCudaErrors(cudaEventRecord(start));
	for (uint32_t i = 0; i < iters; i++) {
		launch();
	}
	checkCudaErrors(cudaEventRecord(end));
	checkCudaErrors(cudaEventSynchronize(end));
	float ms;
	checkCudaErrors(cudaEventElapsedTime(&ms, start, end));
	cudaEventDestroy(start);
	cudaEventDestroy(end);
	return ms / iters;
}

/// filter step on N random elements with the configuration C
template<const uint32_t W, const uint32_t S, typename C>
void bench_filter(const char *name, const uint32_t *L, const size_t N,
                  const Filters<W, S> &P, uint64_t *M, const Buckets &B) {
	const double ms = measure([&]() {
		launch_filter<uint32_t, W, S, C::TW, C::PW, C::E, C::OP>(L, N, P, M, B);
	});
	printf("  %-24s %8.3f ms %8.1f Gpairs/s %8.1f Gelem/s %7.1f GB/s (L+M)\n", name, ms,
	       double(N) * S / ms / 1e6, double(N) / ms / 1e6,
	       double(N) * (W * 4 + 8) / ms / 1e6);
}

/// all variants of the filter step, with and without buckets
template<const uint32_t W, const uint32_t S>
void bench(const size_t N, const double p) {
	const uint32_t t = threshold(32 * W, p);
	uint32_t *L, *D;
	uint64_t *M;
	checkCudaErrors(cudaMalloc(&L, sizeof(uint32_t) * N * W));
	checkCudaErrors(cudaMalloc(&D, sizeof(uint32_t) * N * W));
	checkCudaErrors(cudaMalloc(&M, sizeof(uint64_t) * N));
	std::vector<uint32_t> h(N * W);
	for (auto &x : h) { x = gen(); }
	checkCudaErrors(cudaMemcpy(L, h.data(), sizeof(uint32_t) * N * W, cudaMemcpyHostToDevice));
	std::vector<uint32_t> centers(S * W);
	for (auto &x : centers) { x = gen(); }
	const auto P = make_filters<uint32_t, W, S>(centers.data(), t);
	Buckets B = alloc_buckets(S, bucket_capacity(N, 32 * W, t));

	printf("n=%u S=%u t=%u N=2^%.0f p=%.4f\n", 32 * W, S, t, std::log2(double(N)), filter_probability(32 * W, t));
	// the memory bound
	const double ms = measure([&]() {
		checkCudaErrors(cudaMemcpyAsync(D, L, sizeof(uint32_t) * N * W, cudaMemcpyDeviceToDevice));
	});
	printf("  %-24s %8.3f ms %43.1f GB/s (read + write)\n", "memcpy of L", ms, 8. * N * W / ms / 1e6);

	const Buckets none{};
	bench_filter<W, S, Config<0, 8, 2048>>("popc         only M", L, N, P, M, none);
	bench_filter<W, S, Config<8, 0, 2048, TensorOp::B1>>("b1           only M", L, N, P, M, none);
	bench_filter<W, S, Config<8, 0, 2048, TensorOp::S8>>("s8           only M", L, N, P, M, none);
	bench_filter<W, S, Config<4, 4, 2048, TensorOp::S8>>("s8+popc 4+4  only M", L, N, P, M, none);
	bench_filter<W, S, Config<8, 0, 2048, TensorOp::MXF4>>("mxf4         only M", L, N, P, M, none);
	bench_filter<W, S, Config<4, 4, 2048, TensorOp::MXF4>>("mxf4+popc 4+4 only M", L, N, P, M, none);
	bench_filter<W, S, Config<0, 8, 2048>>("popc", L, N, P, M, B);
	bench_filter<W, S, Config<8, 0, 2048, TensorOp::B1>>("b1", L, N, P, M, B);
	bench_filter<W, S, Config<8, 0, 2048, TensorOp::S8>>("s8", L, N, P, M, B);
	bench_filter<W, S, Config<4, 4, 2048, TensorOp::S8>>("s8+popc 4+4", L, N, P, M, B);
	bench_filter<W, S, Config<8, 0, 2048, TensorOp::MXF4>>("mxf4", L, N, P, M, B);
	bench_filter<W, S, Config<4, 4, 2048, TensorOp::MXF4>>("mxf4+popc 4+4", L, N, P, M, B);

	free_buckets(B);
	cudaFree(L);
	cudaFree(D);
	cudaFree(M);
}

/// NOTE: mxf4 is only measured if compiled for sm_120a, otherwise it is s8
/// run with: --gtest_also_run_disabled_tests --gtest_filter=FullLengthBench.DISABLED_filter
TEST(FullLengthBench, DISABLED_filter) {
	bench<4, 64>(1u << 24u, 1. / 32.);
	bench<8, 64>(1u << 23u, 1. / 32.);
	bench<16, 64>(1u << 22u, 1. / 32.);
	bench<4, 16>(1u << 24u, 1. / 8.);
}

/// the join of two lists of 2^20 elements, n=128, S=64, ~1800 elements per
/// bucket, ~2^27.6 pairs
/// run with: --gtest_also_run_disabled_tests --gtest_filter=FullLengthBench.DISABLED_join
TEST(FullLengthBench, DISABLED_join) {
	constexpr uint32_t W = 4, S = 64;
	constexpr size_t N = 1u << 20u;
	const uint32_t t = threshold(32 * W, 1. / 1024.);
	const double p = filter_probability(32 * W, t);
	uint32_t *L1, *L2;
	uint64_t *M1, *M2;
	checkCudaErrors(cudaMalloc(&L1, sizeof(uint32_t) * N * W));
	checkCudaErrors(cudaMalloc(&L2, sizeof(uint32_t) * N * W));
	checkCudaErrors(cudaMalloc(&M1, sizeof(uint64_t) * N));
	checkCudaErrors(cudaMalloc(&M2, sizeof(uint64_t) * N));
	std::vector<uint32_t> h(N * W);
	for (auto &x : h) { x = gen(); }
	checkCudaErrors(cudaMemcpy(L1, h.data(), sizeof(uint32_t) * N * W, cudaMemcpyHostToDevice));
	for (auto &x : h) { x = gen(); }
	checkCudaErrors(cudaMemcpy(L2, h.data(), sizeof(uint32_t) * N * W, cudaMemcpyHostToDevice));
	std::vector<uint32_t> centers(S * W);
	for (auto &x : centers) { x = gen(); }
	const auto P = make_filters<uint32_t, W, S>(centers.data(), t);
	const uint32_t cap = bucket_capacity(N, 32 * W, t);
	Buckets B1 = alloc_buckets(S, cap), B2 = alloc_buckets(S, cap);
	launch_filter<uint32_t, W, S, 8, 0>(L1, N, P, M1, B1);
	launch_filter<uint32_t, W, S, 8, 0>(L2, N, P, M2, B2);

	const size_t out_capacity = size_t(1.2 * S * (N * p) * (N * p));
	uint2 *out;
	unsigned long long *cnt, pairs;
	checkCudaErrors(cudaMalloc(&out, sizeof(uint2) * out_capacity));
	checkCudaErrors(cudaMalloc(&cnt, sizeof(unsigned long long)));
	const double count = measure([&]() { launch_join(B1, M1, B2, M2, nullptr, 0, cnt); });
	checkCudaErrors(cudaMemcpy(&pairs, cnt, sizeof(pairs), cudaMemcpyDeviceToHost));
	const double write = measure([&]() { launch_join(B1, M1, B2, M2, out, out_capacity, cnt); });
	printf("N=2^20 n=128 S=64 p=%.5f pairs=%llu\n", p, pairs);
	printf("  count only %8.3f ms %8.1f Gpairs/s\n", count, pairs / count / 1e6);
	printf("  write      %8.3f ms %8.1f Gpairs/s %7.1f GB/s\n", write, pairs / write / 1e6, 8. * pairs / write / 1e6);

	free_buckets(B1);
	free_buckets(B2);
	cudaFree(out);
	cudaFree(cnt);
	cudaFree(L1);
	cudaFree(L2);
	cudaFree(M1);
	cudaFree(M2);
}

int main(int argc, char **argv) {
	const uint64_t seed = time(NULL);
	printf("seed: %lu\n", seed);
	gen.seed(seed);
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
