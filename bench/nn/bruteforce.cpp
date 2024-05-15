#include "benchmark/benchmark.h"

#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#define ENABLE_BENCHMARK
#include "nn/nn.h"

constexpr size_t LS = 1u << 16u;
constexpr size_t d = 14;
constexpr size_t dk = 10;
constexpr static NN_Config config{256, 4, 1, 64, LS, dk, d, 0, 512};
NN<config> algo{};

static void BM_bruteforce_256(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_256(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_bruteforce_simd_256(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_simd_256(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_bruteforce_simd_256_ux4_1(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_simd_256_ux4<1>(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_bruteforce_simd_256_ux4_2(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_simd_256_ux4<2>(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_bruteforce_simd_256_ux4_4(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_simd_256_ux4<4>(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_bruteforce_simd_256_ux4_8(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_simd_256_ux4<8>(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}



static void BM_bruteforce_simd_256_32_ux8_1(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_simd_256_32_ux8<1>(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_bruteforce_simd_256_32_ux8_2(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_simd_256_32_ux8<2>(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_bruteforce_simd_256_32_ux8_4(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_simd_256_32_ux8<4>(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

static void BM_bruteforce_simd_256_32_ux8_8(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_simd_256_32_ux8<8>(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}


static void BM_bruteforce_simd_256_64_4x4(benchmark::State& state) {
	for (auto _ : state) {
		algo.bruteforce_simd_256_64_4x4(state.range(0), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}

BENCHMARK(BM_bruteforce_simd_256)->RangeMultiplier(2)->Range(1024, LS)->Complexity();

BENCHMARK(BM_bruteforce_simd_256_ux4_1)->RangeMultiplier(2)->Range(1024, LS)->Complexity();
BENCHMARK(BM_bruteforce_simd_256_ux4_2)->RangeMultiplier(2)->Range(1024, LS)->Complexity();
BENCHMARK(BM_bruteforce_simd_256_ux4_4)->RangeMultiplier(2)->Range(1024, LS)->Complexity();
BENCHMARK(BM_bruteforce_simd_256_ux4_8)->RangeMultiplier(2)->Range(1024, LS)->Complexity();

BENCHMARK(BM_bruteforce_simd_256_32_ux8_1)->RangeMultiplier(2)->Range(1024, LS)->Complexity();
BENCHMARK(BM_bruteforce_simd_256_32_ux8_2)->RangeMultiplier(2)->Range(1024, LS)->Complexity();
BENCHMARK(BM_bruteforce_simd_256_32_ux8_4)->RangeMultiplier(2)->Range(1024, LS)->Complexity();
BENCHMARK(BM_bruteforce_simd_256_32_ux8_8)->RangeMultiplier(2)->Range(1024, LS)->Complexity();

BENCHMARK(BM_bruteforce_simd_256_64_4x4)->RangeMultiplier(2)->Range(1024, LS)->Complexity();

BENCHMARK(BM_bruteforce_256)->RangeMultiplier(2)->Range(1024, LS)->Complexity();


int main(int argc, char** argv) {
	random_seed(time(NULL));
	algo.generate_random_instance(false);

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
