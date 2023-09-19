#include "benchmark/benchmark.h"
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#define ENABLE_BENCHMARK
#include "nn/nn.h"

constexpr size_t LS = 1u << 20u;
constexpr size_t d = 1;
constexpr size_t dk = 22;
constexpr static NN_Config config{256, 4, 300, 64, LS, dk, d, 0, 512};

#ifdef USE_AVX2

NN<config> algo{};

// Define another benchmark
static void BM_avx2_256_32_8x8(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_avx2_256_32_8x8(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_avx2_256_64_4x4(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_avx2_256_64_4x4(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

#
static void BM_NearestNeighborAVX(benchmark::State& state) {
	for (auto _ : state) {
		algo.nn(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

//BENCHMARK(BM_avx2_256_32_8x8)->RangeMultiplier(2)->Range(128, 1u<<16)->Complexity();
BENCHMARK(BM_avx2_256_64_4x4)->RangeMultiplier(2)->Range(1024, 1u<<16)->Complexity();
BENCHMARK(BM_NearestNeighborAVX)->RangeMultiplier(2)->Range(1024, 1u<<16)->Complexity();

#endif

int main(int argc, char** argv) {
	random_seed(time(NULL));
#ifdef USE_AVX2
	algo.generate_random_instance(false);
#endif

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
