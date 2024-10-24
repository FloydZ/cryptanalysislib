#include <benchmark/benchmark.h>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include "random.h"
using namespace cryptanalysislib;
using namespace cryptanalysislib::random::internal;

static void BM_xorshf96(benchmark::State& state) {
	uint64_t data = rand();
	for (auto _ : state) {
		for (int64_t i = 0; i < state.range(0); ++i) {
			data += xorshf96_random_data();
		}
        benchmark::DoNotOptimize(data += 1);
	}
}

static void BM_xorshf128(benchmark::State& state) {
	uint64_t data = rand();
	for (auto _ : state) {
		for (int64_t i = 0; i < state.range(0); ++i) {
			data += xorshf128_random_data();
		}
        benchmark::DoNotOptimize(data += 1);
	}
}

static void BM_pcg64(benchmark::State& state) {
	uint64_t data = rand();
	for (auto _ : state) {
		for (int64_t i = 0; i < state.range(0); ++i) {
			data += pcg64_random_data();
		}
        benchmark::DoNotOptimize(data += 1);
	}
}

static void BM_lehmer64(benchmark::State& state) {
	uint64_t data = rand();
	for (auto _ : state) {
		for (int64_t i = 0; i < state.range(0); ++i) {
			data += lehmer64_random_data();
		}
        benchmark::DoNotOptimize(data += 1);
	}
}

constexpr uint64_t limit = 1<<10;
BENCHMARK(BM_xorshf96)->RangeMultiplier(2)->Range(1, limit);
BENCHMARK(BM_xorshf128)->RangeMultiplier(2)->Range(1, limit);
BENCHMARK(BM_pcg64)->RangeMultiplier(2)->Range(1, limit);
BENCHMARK(BM_lehmer64)->RangeMultiplier(2)->Range(1, limit);


int main(int argc, char** argv) {
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
