#include "benchmark/benchmark.h"
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#define ENABLE_BENCHMARK
#include "nn/nn.h"

constexpr size_t LS = 1u << 14u;
constexpr size_t d = 6;
constexpr size_t dk = 12;
constexpr static WindowedAVX2_Config config{128, 4, 300, 32, LS, dk, d, 0, 512};
WindowedAVX2<config> algo{};

static void BM_bruteforce_128(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_128(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_bruteforce_avx2_128(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_avx2_128(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}


static void BM_bruteforce_avx2_128_32_2_uxv_4x4(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_avx2_128_32_2_uxv<4, 4>(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_bruteforce_avx2_128_32_2_uxv_8x8(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_avx2_128_32_2_uxv<8, 8>(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_bruteforce_128)->RangeMultiplier(2)->Range(128, LS)->Complexity();
BENCHMARK(BM_bruteforce_avx2_128)->RangeMultiplier(2)->Range(128, LS)->Complexity();
BENCHMARK(BM_bruteforce_avx2_128_32_2_uxv_4x4)->RangeMultiplier(2)->Range(128, LS)->Complexity();
BENCHMARK(BM_bruteforce_avx2_128_32_2_uxv_8x8)->RangeMultiplier(2)->Range(128, LS)->Complexity();

int main(int argc, char** argv) {
	random_seed(time(NULL));
	algo.generate_random_instance(false);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
