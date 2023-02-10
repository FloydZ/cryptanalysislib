#include "benchmark/benchmark.h"
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#define ENABLE_BENCHMARK
#include "windowed_avx2.h"

constexpr size_t LS = 1u << 18u;
constexpr size_t d = 16;
constexpr size_t dk = 22;
constexpr static WindowedAVX2_Config config{256, 4, 200, LS, dk, d, 0, 512};
WindowedAVX2<config> algo{};

// Define another benchmark
static void BM_avx2_sort_nn_on64_simple_0(benchmark::State& state) {
	for (auto _ : state) {
		const uint64_t z = fastrandombytes_uint64();
		algo.avx2_sort_nn_on64_simple<0>(state.range(0), z, algo.L1);
	}
	state.SetComplexityN(state.range(0));
}
static void BM_avx2_sort_nn_on64_simple_1(benchmark::State& state) {
	for (auto _ : state) {
		const uint64_t z = fastrandombytes_uint64();
		algo.avx2_sort_nn_on64_simple<1>(state.range(0), z, algo.L1);
	}
	state.SetComplexityN(state.range(0));
}

static void BM_avx2_sort_nn_on64_0(benchmark::State& state) {
	for (auto _ : state) {
		const uint64_t z = fastrandombytes_uint64();
		algo.avx2_sort_nn_on64<0>(state.range(0), z, algo.L2);
	}
	state.SetComplexityN(state.range(0));
}
static void BM_avx2_sort_nn_on64_1(benchmark::State& state) {
	for (auto _ : state) {
		const uint64_t z = fastrandombytes_uint64();
		algo.avx2_sort_nn_on64<1>(state.range(0), z, algo.L2);
	}
	state.SetComplexityN(state.range(0));
}


BENCHMARK(BM_avx2_sort_nn_on64_simple_0)->RangeMultiplier(2)->Range(128, 1u<<18)->Complexity();
BENCHMARK(BM_avx2_sort_nn_on64_simple_1)->RangeMultiplier(2)->Range(128, 1u<<18)->Complexity();
BENCHMARK(BM_avx2_sort_nn_on64_0)->RangeMultiplier(2)->Range(128, 1u<<18)->Complexity();
BENCHMARK(BM_avx2_sort_nn_on64_1)->RangeMultiplier(2)->Range(128, 1u<<18)->Complexity();

int main(int argc, char** argv) {
	random_seed(time(NULL));
	algo.generate_random_instance();

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}