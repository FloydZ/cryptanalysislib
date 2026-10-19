#include <gtest/gtest.h>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <random>
#include <vector>

#include "helper.cuh"
#include "algorithm/histogram.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib::internal::cuda;

static std::mt19937_64 gen;

/// distributions of the samples
enum class Dist {
	/// uniform over all values
	Uniform,
	/// a single value: worst case for the atomics
	Constant,
	/// 4 values
	Small,
};

/// \return `n` samples from `dist`
template<typename T>
std::vector<T> samples(const size_t n, const Dist dist) {
	std::vector<T> ret(n);
	const T c = gen();
	for (auto &x : ret) {
		switch (dist) {
			case Dist::Uniform: x = gen(); break;
			case Dist::Constant: x = c; break;
			case Dist::Small: x = c + gen() % 4u; break;
		}
	}
	return ret;
}

/// \return the reference histogram
template<typename T, typename C>
std::vector<C> reference(const std::vector<T> &s) {
	std::vector<C> ret(histogram_bins<T>, 0);
	for (const T x : s) { ret[x] += 1; }
	return ret;
}

/// runs `DeviceHistogram::Histogram` on `n` samples starting `offset`
/// elements after a 256-byte aligned address
template<typename T, typename C, const int BT, const BlockHistogramAlgorithm A>
void check_device(const size_t n, const size_t offset, const Dist dist) {
	const auto s = samples<T>(n, dist);
	const auto expected = reference<T, C>(s);
	T *d_in;
	C *d_cnt;
	checkCudaErrors(cudaMalloc(&d_in, sizeof(T) * (n + offset + 1)));
	checkCudaErrors(cudaMalloc(&d_cnt, sizeof(C) * histogram_bins<T>));
	checkCudaErrors(cudaMemcpy(d_in + offset, s.data(), sizeof(T) * n, cudaMemcpyHostToDevice));
	// the histogram is overwritten
	checkCudaErrors(cudaMemset(d_cnt, 0xff, sizeof(C) * histogram_bins<T>));

	checkCudaErrors((DeviceHistogram::Histogram<BT, A>(d_in + offset, d_cnt, n)));
	std::vector<C> got(histogram_bins<T>);
	checkCudaErrors(cudaMemcpy(got.data(), d_cnt, sizeof(C) * got.size(), cudaMemcpyDeviceToHost));
	ASSERT_EQ(got, expected) << "n=" << n << " offset=" << offset << " dist=" << int(dist);

	cudaFree(d_in);
	cudaFree(d_cnt);
}

/// configuration of `DeviceHistogram`
template<typename T_, typename C_, const int BT_, const BlockHistogramAlgorithm A_>
struct Config {
	using T = T_;
	using C = C_;
	constexpr static int BT = BT_;
	constexpr static BlockHistogramAlgorithm A = A_;
};

template<typename C>
class CudaHistogram : public Test {};
using Configs = ::testing::Types<Config<uint8_t, uint32_t, 512, BLOCK_HISTO_ATOMIC>,
                                 Config<uint8_t, uint32_t, 256, BLOCK_HISTO_WARP_AGGREGATE>,
                                 Config<uint8_t, unsigned long long, 1024, BLOCK_HISTO_ATOMIC>,
                                 Config<uint16_t, uint32_t, 512, BLOCK_HISTO_ATOMIC>,
                                 Config<uint16_t, uint64_t, 256, BLOCK_HISTO_WARP_AGGREGATE>>;
TYPED_TEST_SUITE(CudaHistogram, Configs);

/// sizes around the tile and vector size
TYPED_TEST(CudaHistogram, Sizes) {
	using C = TypeParam;
	for (const size_t n : {size_t(0), size_t(1), size_t(15), size_t(16), size_t(17), size_t(4097),
	                       size_t(C::BT * 16 + 3), size_t(1u << 20u) + 13u}) {
		check_device<typename C::T, typename C::C, C::BT, C::A>(n, 0, Dist::Uniform);
	}
}

/// every misalignment of the start
TYPED_TEST(CudaHistogram, Offsets) {
	using C = TypeParam;
	for (size_t offset = 0; offset < 16; offset++) {
		for (const size_t n : {size_t(3), size_t(100003)}) {
			check_device<typename C::T, typename C::C, C::BT, C::A>(n, offset, Dist::Uniform);
		}
	}
}

TYPED_TEST(CudaHistogram, Skewed) {
	using C = TypeParam;
	for (const Dist dist : {Dist::Constant, Dist::Small}) {
		for (const size_t n : {size_t(77), size_t(1u << 22u) + 5u}) {
			check_device<typename C::T, typename C::C, C::BT, C::A>(n, 1, dist);
		}
	}
}

/// `BlockHistogram::Histogram` of a single tile per block
template<typename T, const int BT, const int ITEMS, const BlockHistogramAlgorithm A>
__global__ void block_histogram_kernel(const T *in, uint32_t *out) {
	constexpr int BINS = histogram_bins<T>;
	using BlockHistogramT = BlockHistogram<T, BT, ITEMS, BINS, A>;
	__shared__ typename BlockHistogramT::TempStorage temp_storage;
	__shared__ uint32_t histogram[BINS];

	T items[ITEMS];
	for (int i = 0; i < ITEMS; i++) {
		items[i] = in[(blockIdx.x * BT + threadIdx.x) * ITEMS + i];
	}
	BlockHistogramT(temp_storage).Histogram(items, histogram);
	__syncthreads();
	for (int i = threadIdx.x; i < BINS; i += BT) {
		out[blockIdx.x * BINS + i] = histogram[i];
	}
}

template<const BlockHistogramAlgorithm A>
void check_block(const Dist dist) {
	constexpr int BT = 128, ITEMS = 5, BLOCKS = 3;
	using T = uint8_t;
	const auto s = samples<T>(BT * ITEMS * BLOCKS, dist);
	T *d_in;
	uint32_t *d_out;
	checkCudaErrors(cudaMalloc(&d_in, s.size()));
	checkCudaErrors(cudaMalloc(&d_out, sizeof(uint32_t) * 256 * BLOCKS));
	checkCudaErrors(cudaMemcpy(d_in, s.data(), s.size(), cudaMemcpyHostToDevice));
	block_histogram_kernel<T, BT, ITEMS, A><<<BLOCKS, BT>>>(d_in, d_out);
	checkCudaErrors(cudaGetLastError());
	std::vector<uint32_t> got(256 * BLOCKS);
	checkCudaErrors(cudaMemcpy(got.data(), d_out, sizeof(uint32_t) * got.size(), cudaMemcpyDeviceToHost));
	for (int b = 0; b < BLOCKS; b++) {
		const std::vector<T> part(s.begin() + b * BT * ITEMS, s.begin() + (b + 1) * BT * ITEMS);
		const std::vector<uint32_t> block(got.begin() + b * 256, got.begin() + (b + 1) * 256);
		ASSERT_EQ(block, (reference<T, uint32_t>(part))) << "block " << b;
	}
	cudaFree(d_in);
	cudaFree(d_out);
}

TEST(CudaBlockHistogram, Atomic) {
	check_block<BLOCK_HISTO_ATOMIC>(Dist::Uniform);
	check_block<BLOCK_HISTO_ATOMIC>(Dist::Small);
}

TEST(CudaBlockHistogram, WarpAggregate) {
	check_block<BLOCK_HISTO_WARP_AGGREGATE>(Dist::Uniform);
	check_block<BLOCK_HISTO_WARP_AGGREGATE>(Dist::Small);
}

/// the host wrapper of `parallel_cuda_policy` with host, device and managed
/// memory
TEST(CudaHistogramHost, Memory) {
	using T = uint16_t;
	using C = uint64_t;
	constexpr size_t n = (1u << 20u) + 7u;
	const auto s = samples<T>(n, Dist::Uniform);
	const auto expected = reference<T, C>(s);
	std::vector<C> got(histogram_bins<T>, 7);
	cudaStream_t stream;
	checkCudaErrors(cudaStreamCreate(&stream));

	// host memory
	ASSERT_TRUE((histogram<T, C>(got.data(), s.data(), n, stream, 0)));
	ASSERT_EQ(got, expected);

	// device memory in, host memory out
	T *d_in;
	checkCudaErrors(cudaMalloc(&d_in, sizeof(T) * n));
	checkCudaErrors(cudaMemcpy(d_in, s.data(), sizeof(T) * n, cudaMemcpyHostToDevice));
	std::fill(got.begin(), got.end(), 7);
	ASSERT_TRUE((histogram<T, C>(got.data(), d_in, n, stream, 0)));
	ASSERT_EQ(got, expected);

	// managed memory
	C *m_cnt;
	checkCudaErrors(cudaMallocManaged(&m_cnt, sizeof(C) * histogram_bins<T>));
	ASSERT_TRUE((histogram<T, C>(m_cnt, d_in, n, nullptr, 0)));
	ASSERT_EQ(std::vector<C>(m_cnt, m_cnt + histogram_bins<T>), expected);

	// no samples
	std::fill(got.begin(), got.end(), 7);
	ASSERT_TRUE((histogram<T, C>(got.data(), s.data(), 0, stream, 0)));
	ASSERT_EQ(got, std::vector<C>(histogram_bins<T>, 0));

	// invalid device
	ASSERT_FALSE((histogram<T, C>(got.data(), s.data(), n, stream, 1000)));

	cudaFree(m_cnt);
	cudaFree(d_in);
	cudaStreamDestroy(stream);
}

/// `parallel_cuda_policy` in `cryptanalysislib::algorithm::histogram`
TEST(CudaHistogramPolicy, Histogram) {
	const size_t n = (1u << 22) + 13u;
	const auto s16 = samples<uint16_t>(n, Dist::Uniform);
	std::vector<int8_t> s(n);
	for (auto &x : s) { x = int8_t(gen()); }
	std::vector<uint32_t> e(256, 0);
	for (const int8_t x : s) { e[uint8_t(x)] += 1; }
	const auto e16 = reference<uint16_t, uint64_t>(s16);

	std::vector<uint32_t> c(256, 7);
	std::vector<uint64_t> c16(1u << 16, 7);
	cryptanalysislib::algorithm::histogram(cryptanalysislib::par_cuda, c.data(), s.data(), n);
	EXPECT_EQ(c, e);
	cryptanalysislib::algorithm::histogram(cryptanalysislib::par_cuda, c16.data(), s16.data(), n);
	EXPECT_EQ(c16, e16);

	// the sequential CPU version
	std::fill(c.begin(), c.end(), 7u);
	cryptanalysislib::algorithm::histogram(cryptanalysislib::par_cuda.par_if(false), c.data(), s.data(), n);
	EXPECT_EQ(c, e);

	// an invalid device: falls back to the sequential CPU version
	std::fill(c.begin(), c.end(), 7u);
	cryptanalysislib::algorithm::histogram(cryptanalysislib::par_cuda.on_dev(1000), c.data(), s.data(), n);
	EXPECT_EQ(c, e);

	std::fill(c.begin(), c.end(), 7u);
	cryptanalysislib::algorithm::histogram(cryptanalysislib::par_cuda, c.data(), s.data(), 0);
	EXPECT_EQ(c, std::vector<uint32_t>(256, 0));
}

/// measures `launch()` on the default stream
/// \return ms per call
template<typename F>
double measure(F launch, const uint32_t iters = 20) {
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

/// `DeviceHistogram` with different configurations
template<typename T>
void bench(const size_t n, const Dist dist) {
	const auto s = samples<T>(n, dist);
	T *d_in, *d_copy;
	uint32_t *d_cnt;
	checkCudaErrors(cudaMalloc(&d_in, sizeof(T) * n));
	checkCudaErrors(cudaMalloc(&d_copy, sizeof(T) * n / 2));
	checkCudaErrors(cudaMalloc(&d_cnt, sizeof(uint32_t) * histogram_bins<T>));
	checkCudaErrors(cudaMemcpy(d_in, s.data(), sizeof(T) * n, cudaMemcpyHostToDevice));

	const double gb = double(n) * sizeof(T) / 1e6;
	printf("u%zu n=2^%.0f dist=%d\n", 8 * sizeof(T), std::log2(double(n)), int(dist));
	double ms = measure([&]() {
		checkCudaErrors(cudaMemcpyAsync(d_copy, d_in, sizeof(T) * n / 2, cudaMemcpyDeviceToDevice));
	});
	printf("  %-28s %8.3f ms %8.1f GB/s (read + write)\n", "memcpy of n/2 samples", ms, gb / ms);
	ms = measure([&]() { checkCudaErrors((DeviceHistogram::Histogram<256, BLOCK_HISTO_ATOMIC>(d_in, d_cnt, n))); });
	printf("  %-28s %8.3f ms %8.1f GB/s\n", "atomic          256", ms, gb / ms);
	ms = measure([&]() { checkCudaErrors((DeviceHistogram::Histogram<512, BLOCK_HISTO_ATOMIC>(d_in, d_cnt, n))); });
	printf("  %-28s %8.3f ms %8.1f GB/s\n", "atomic          512", ms, gb / ms);
	ms = measure([&]() { checkCudaErrors((DeviceHistogram::Histogram<1024, BLOCK_HISTO_ATOMIC>(d_in, d_cnt, n))); });
	printf("  %-28s %8.3f ms %8.1f GB/s\n", "atomic          1024", ms, gb / ms);
	ms = measure([&]() { checkCudaErrors((DeviceHistogram::Histogram<512, BLOCK_HISTO_WARP_AGGREGATE>(d_in, d_cnt, n))); });
	printf("  %-28s %8.3f ms %8.1f GB/s\n", "warp aggregate  512", ms, gb / ms);

	cudaFree(d_in);
	cudaFree(d_copy);
	cudaFree(d_cnt);
}

/// run with: --gtest_also_run_disabled_tests --gtest_filter=CudaHistogramBench.DISABLED_bench
TEST(CudaHistogramBench, DISABLED_bench) {
	for (const Dist dist : {Dist::Uniform, Dist::Small, Dist::Constant}) {
		bench<uint8_t>(1u << 28u, dist);
		bench<uint16_t>(1u << 27u, dist);
	}
}

/// `parallel_cuda_policy` on host memory, i.e. including the copies
/// run with: --gtest_also_run_disabled_tests --gtest_filter=CudaHistogramBench.DISABLED_policy
TEST(CudaHistogramBench, DISABLED_policy) {
	std::vector<uint32_t> c(256);
	for (const size_t n : {size_t(1u << 14u), size_t(1u << 20u), size_t(1u << 24u), size_t(1u << 28u)}) {
		const auto s = samples<uint8_t>(n, Dist::Uniform);
		const double ms = measure([&]() {
			cryptanalysislib::algorithm::histogram(cryptanalysislib::par_cuda, c.data(), s.data(), n);
		}, 5);
		printf("u8 n=2^%.0f host memory %8.3f ms %8.2f GB/s\n", std::log2(double(n)), ms, double(n) / ms / 1e6);
	}
}

int main(int argc, char **argv) {
	const uint64_t seed = time(NULL);
	printf("seed: %lu\n", seed);
	gen.seed(seed);
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
