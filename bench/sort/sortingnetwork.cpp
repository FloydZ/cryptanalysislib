#include <benchmark/benchmark.h>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include "random.h"
#include "sort/sort.h"

uint32_t data[128] __attribute__((aligned(64)));

int c(const void *a, const void *b){
	if ( *(uint32_t *)a <  *(uint32_t *)b ) return -1;
	if ( *(uint32_t *)a == *(uint32_t *)b ) return 0;
	if ( *(uint32_t *)a >  *(uint32_t *)b ) return 1;
	return 0;

}
static void bench_stdsort(benchmark::State& state) {
	for (auto _ : state) {
		state.PauseTiming();
		for (uint32_t i = 0; i < 128; ++i) {
			data[i] = fastrandombytes_uint64();
		}
		state.ResumeTiming();

		qsort(data, 128, 4, c);
	}
}

BENCHMARK(bench_stdsort)->Range(128, 128);

#ifdef USE_AVX2
static void bench_sortingnetwort_sort_u32x128(benchmark::State& state) {
	for (auto _ : state) {
		state.PauseTiming();
		for (uint32_t i = 0; i < 128; ++i) {
			data[i] = fastrandombytes_uint64();
		}
		state.ResumeTiming();
		sortingnetwork_sort_u32x128((__m256i *)data);
	}
}

BENCHMARK(bench_sortingnetwort_sort_u32x128)->Range(128, 128);
#endif

int main(int argc, char** argv) {
	random_seed(time(NULL));

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
