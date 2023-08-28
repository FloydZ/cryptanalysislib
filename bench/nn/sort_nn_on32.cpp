#include "benchmark/benchmark.h"
#include <cstdint>
#include <cstdlib>

#define ENABLE_BENCHMARK
#include "nn/avx2.h"

constexpr size_t LS = 1u << 24u;
constexpr size_t d = 16;
constexpr size_t dk = 8;
constexpr static WindowedAVX2_Config config{256, 8, 200, 32, LS, dk, d, 0, 512};
WindowedAVX2<config> algo{};

// Define another benchmark
static void BM_avx2_sort_nn_on32_simple_0(benchmark::State& state) {
	for (auto _ : state) {
		const uint64_t z = fastrandombytes_uint64();
		algo.avx2_sort_nn_on32_simple<0>(state.range(0), z, algo.L1);
		algo.avx2_sort_nn_on32_simple<0>(state.range(0), z, algo.L2);
	}
	state.SetComplexityN(state.range(0));
}
static void BM_avx2_sort_nn_on32_simple_1(benchmark::State& state) {
	for (auto _ : state) {
		const uint64_t z = fastrandombytes_uint64();
		algo.avx2_sort_nn_on32_simple<1>(state.range(0), z, algo.L1);
		algo.avx2_sort_nn_on32_simple<1>(state.range(0), z, algo.L2);
	}
	state.SetComplexityN(state.range(0));
}

static void BM_avx2_sort_nn_on32_0(benchmark::State& state) {
	for (auto _ : state) {
		const uint64_t z = fastrandombytes_uint64();
		algo.avx2_sort_nn_on32<0>(state.range(0), z, algo.L1);
		algo.avx2_sort_nn_on32<0>(state.range(0), z, algo.L2);
	}
	state.SetComplexityN(state.range(0));
}
static void BM_avx2_sort_nn_on32_1(benchmark::State& state) {
	for (auto _ : state) {
		const uint64_t z = fastrandombytes_uint64();
		algo.avx2_sort_nn_on32<1>(state.range(0), z, algo.L1);
		algo.avx2_sort_nn_on32<1>(state.range(0), z, algo.L2);
	}
	state.SetComplexityN(state.range(0));
}

static void BM_avx2_sort_nn_on_double32_0_8(benchmark::State& state) {
	for (auto _ : state) {
		const uint64_t z = fastrandombytes_uint64();
		size_t ne1=0, ne2=0;
		algo.avx2_sort_nn_on_double32<0, 8>(state.range(0), state.range(0), ne1, ne2, z);
	}
	state.SetComplexityN(state.range(0));
}
static void BM_avx2_sort_nn_on_double32_1_8(benchmark::State& state) {
	for (auto _ : state) {
		const uint64_t z = fastrandombytes_uint64();
		size_t ne1=0, ne2=0;
		algo.avx2_sort_nn_on_double32<1, 8>(state.range(0), state.range(0), ne1, ne2, z);
	}
	state.SetComplexityN(state.range(0));
}
static void BM_avx2_sort_nn_on_double32_0_4(benchmark::State& state) {
	for (auto _ : state) {
		const uint64_t z = fastrandombytes_uint64();
		size_t ne1=0, ne2=0;
		algo.avx2_sort_nn_on_double32<0, 4>(state.range(0), state.range(0), ne1, ne2, z);
	}
	state.SetComplexityN(state.range(0));
}
static void BM_avx2_sort_nn_on_double32_1_4(benchmark::State& state) {
	for (auto _ : state) {
		const uint64_t z = fastrandombytes_uint64();
		size_t ne1=0, ne2=0;
		algo.avx2_sort_nn_on_double32<1, 4>(state.range(0), state.range(0), ne1, ne2, z);
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_avx2_sort_nn_on32_simple_0)->RangeMultiplier(2)->Range(1024, 1u<<18)->Complexity();
BENCHMARK(BM_avx2_sort_nn_on32_simple_1)->RangeMultiplier(2)->Range(1024, 1u<<18)->Complexity();
BENCHMARK(BM_avx2_sort_nn_on32_0)->RangeMultiplier(2)->Range(1024, 1u<<18)->Complexity();
BENCHMARK(BM_avx2_sort_nn_on32_1)->RangeMultiplier(2)->Range(1024, 1u<<18)->Complexity();
BENCHMARK(BM_avx2_sort_nn_on_double32_0_8)->RangeMultiplier(2)->Range(1024, 1u<<18)->Complexity();
BENCHMARK(BM_avx2_sort_nn_on_double32_1_8)->RangeMultiplier(2)->Range(1024, 1u<<18)->Complexity();
BENCHMARK(BM_avx2_sort_nn_on_double32_0_4)->RangeMultiplier(2)->Range(1024, 1u<<18)->Complexity();
BENCHMARK(BM_avx2_sort_nn_on_double32_1_4)->RangeMultiplier(2)->Range(1024, 1u<<18)->Complexity();
int main(int argc, char** argv) {
	random_seed(time(NULL));
	algo.generate_random_instance();

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}