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

static void bench_qsort(benchmark::State& state) {
	for (auto _ : state) {
		state.PauseTiming();
		for (uint32_t i = 0; i < state.range(0); ++i) {
			data[i] = fastrandombytes_uint64();
		}
		state.ResumeTiming();

		qsort(data, 128, 4, c);
	}
}

static void bench_stdsort(benchmark::State& state) {
	for (auto _ : state) {
		state.PauseTiming();
		for (uint32_t i = 0; i < state.range(0); ++i) {
			data[i] = fastrandombytes_uint64();
		}
		state.ResumeTiming();

		std::sort(std::begin(data), std::end(data));
	}
}

static void bench_skasort(benchmark::State& state) {
	for (auto _ : state) {
		state.PauseTiming();
		for (uint32_t i = 0; i < state.range(0); ++i) {
			data[i] = fastrandombytes_uint64();
		}
		state.ResumeTiming();

		ska_sort(std::begin(data), std::end(data));
	}
}

BENCHMARK(bench_stdsort)->Range(64, 64)->Range(128, 128);
BENCHMARK(bench_qsort)->Range(64, 64)->Range(128, 128);
BENCHMARK(bench_skasort)->Range(64, 64)->Range(128, 128);

#ifdef USE_AVX2
static void bench_sortingnetwort_sort_u32x128(benchmark::State& state) {
	__m256i *d = (__m256i *)data;
	for (auto _ : state) {
		state.PauseTiming();
		for (uint32_t i = 0; i < 128; ++i) {
			data[i] = fastrandombytes_uint64();
		}
		state.ResumeTiming();
		sortingnetwork_sort_u32x128(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15]);
	}
}

static void bench_sortingnetwort_sort_u32x64(benchmark::State& state) {
	__m256i *d = (__m256i *)data;
	for (auto _ : state) {
		state.PauseTiming();
		for (uint32_t i = 0; i < 128; ++i) {
			data[i] = fastrandombytes_uint64();
		}
		state.ResumeTiming();
		sortingnetwork_sort_u32x64(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);
	}
}

static void bench_sortingnetwort_sort_u32x128_v2(benchmark::State& state) {
	for (auto _ : state) {
		state.PauseTiming();
		for (uint32_t i = 0; i < 128; ++i) {
			data[i] = fastrandombytes_uint64();
		}
		state.ResumeTiming();
		sortingnetwork_sort_u32x128_2((__m256i *)data);
	}
}

BENCHMARK(bench_sortingnetwort_sort_u32x64)->Range(128, 128);
BENCHMARK(bench_sortingnetwort_sort_u32x128)->Range(128, 128);
BENCHMARK(bench_sortingnetwort_sort_u32x128_v2)->Range(128, 128);
#endif

int main(int argc, char** argv) {
	random_seed(time(NULL));

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
