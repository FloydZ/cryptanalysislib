#include "benchmark/benchmark.h"
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#define ENABLE_BENCHMARK
#include "nn/nn.h"

constexpr size_t LS = 1u << 20u;
constexpr size_t d = 16;
constexpr size_t dk = 22;
constexpr static WindowedAVX2_Config config{256, 4, 300, LS, dk, d, 0, 512};
WindowedAVX2<config> algo{};

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


static void BM_NearestNeighborAVX(benchmark::State& state) {
	for (auto _ : state) {
		algo.avx2_nn(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

//BENCHMARK(BM_avx2_256_32_8x8)->RangeMultiplier(2)->Range(128, 1u<<16)->Complexity();
BENCHMARK(BM_avx2_256_64_4x4)->RangeMultiplier(2)->Range(1024, 1u<<18)->Complexity();
BENCHMARK(BM_NearestNeighborAVX)->RangeMultiplier(2)->Range(1024, 1u<<18)->Complexity();

int main(int argc, char** argv) {
	random_seed(time(NULL));
	algo.generate_random_instance(false);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}