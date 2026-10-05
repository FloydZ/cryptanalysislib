#include <benchmark/benchmark.h>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include "random.h"
using namespace cryptanalysislib;

static void BM_random_bounded(benchmark::State& state) {
	uint64_t data = 123;
	for (auto _ : state) {
		for (int64_t i = 0; i < state.range(0); ++i) {
			data += rng(data);
		}
	}
}

static void BM_random_bounded_v2(benchmark::State& state) {
	uint64_t data = 123;
	for (auto _ : state) {
		for (int64_t i = 0; i < state.range(0); ++i) {
			data += rng_v2(data);
		}
	}
}

constexpr uint64_t limit = 1<<14;
BENCHMARK(BM_random_bounded)->RangeMultiplier(2)->Range(1, limit);
BENCHMARK(BM_random_bounded_v2)->RangeMultiplier(2)->Range(1, limit);


int main(int argc, char** argv) {
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
