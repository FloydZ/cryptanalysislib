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

__m256i *ptr;

static void BM_popcount_avx2_64(benchmark::State& state) {
	__m256i tmp;
	for (auto _ : state) {
		for (size_t ctr = 0; ctr < state.range(0)/4; ctr++) {
			tmp = algo.popcount_avx2_64(ptr[ctr]);
			ptr[ctr] = _mm256_add_epi8(ptr[ctr], tmp);
		}
	}

	state.SetComplexityN(state.range(0));
}

static void BM_popcount_avx2_64_simple(benchmark::State& state) {
	__m256i tmp ;
	size_t ctr = 0;
	for (auto _ : state) {
		for (size_t ctr = 0; ctr < state.range(0)/4; ctr++) {
			tmp = algo.popcount_avx2_64_simple(ptr[ctr]);
			ptr[ctr] = _mm256_add_epi8(ptr[ctr], tmp);
		}
	}

	state.SetComplexityN(state.range(0));
}



BENCHMARK(BM_popcount_avx2_64)->RangeMultiplier(2)->Range(128, LS)->Complexity();
BENCHMARK(BM_popcount_avx2_64_simple)->RangeMultiplier(2)->Range(128, LS)->Complexity();

int main(int argc, char** argv) {
	random_seed(time(NULL));
	algo.generate_random_instance();
	ptr = (__m256i *)algo.L1;

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}