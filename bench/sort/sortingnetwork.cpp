#include <benchmark/benchmark.h>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include "random.h"
#include "sort/sort.h"

constexpr size_t LS = 1u<<18u;
alignas(64) std::vector<uint32_t> data(LS);

uint32_t data2[128] __attribute__((aligned(64)));

int c(const void *a, const void *b){
	if ( *(uint32_t *)a <  *(uint32_t *)b ) return -1;
	if ( *(uint32_t *)a == *(uint32_t *)b ) return 0;
	if ( *(uint32_t *)a >  *(uint32_t *)b ) return 1;
	return 0;

}

static void bench_qsort(benchmark::State& state) {
	for (uint32_t i = 0; i < state.range(0); ++i) {
		data[i] = fastrandombytes_uint64();
	}
	for (auto _ : state) {
		qsort(data.data(), state.range(0), 4, c);
	}
}

static void bench_stdsort(benchmark::State& state) {
	for (uint32_t i = 0; i < state.range(0); ++i) {
		data[i] = fastrandombytes_uint64();
	}
	for (auto _ : state) {
		std::sort(data.begin(), data.end());
	}
}

static void bench_skasort(benchmark::State& state) {
	for (uint32_t i = 0; i < state.range(0); ++i) {
		data[i] = fastrandombytes_uint64();
	}
	for (auto _ : state) {
		ska_sort(data.begin(), data.end());
	}
}

//BENCHMARK(bench_stdsort)->Range(64, 64)->Range(128, 128);
BENCHMARK(bench_qsort)->Range(64, 64)->Range(128, 128);

BENCHMARK(bench_stdsort)->RangeMultiplier(2)->Range(16, LS);
BENCHMARK(bench_skasort)->RangeMultiplier(2)->Range(16, LS);

#ifdef USE_AVX2
static void bench_sortingnetwort_sort_u32x128(benchmark::State& state) {
	__m256i *d = (__m256i *)data2;
	for (uint32_t i = 0; i < 128; ++i) {
		data2[i] = fastrandombytes_uint64();
	}
	for (auto _ : state) {
		sortingnetwork_sort_u32x128(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7],d[8],d[9],d[10],d[11],d[12],d[13],d[14],d[15]);
	}
}

static void bench_sortingnetwort_sort_u32x64(benchmark::State& state) {
	__m256i *d = (__m256i *)data2;
	for (uint32_t i = 0; i < 128; ++i) {
		data2[i] = fastrandombytes_uint64();
	}
	for (auto _ : state) {
		sortingnetwork_sort_u32x64(d[0],d[1],d[2],d[3],d[4],d[5],d[6],d[7]);
	}
}

static void bench_sortingnetwort_sort_u32x128_v2(benchmark::State& state) {
	for (uint32_t i = 0; i < 128; ++i) {
		data2[i] = fastrandombytes_uint64();
	}
	for (auto _ : state) {
		sortingnetwork_sort_u32x128_2((__m256i *)data2);
	}
}

static void bench_djb_sort(benchmark::State& state) {
	for (uint32_t i = 0; i < LS; ++i) {
		data[i] = fastrandombytes_uint64();
	}
	for (auto _ : state) {
		int32_sort_2power((int32_t *)data.data(), state.range(0), 0);
	}
}

BENCHMARK(bench_sortingnetwort_sort_u32x64)->Range(128, 128);
BENCHMARK(bench_sortingnetwort_sort_u32x128)->Range(128, 128);
BENCHMARK(bench_sortingnetwort_sort_u32x128_v2)->Range(128, 128);
BENCHMARK(bench_djb_sort)->DenseRange(16, 128, 16);
BENCHMARK(bench_djb_sort)->RangeMultiplier(2)->Range(16, LS);
#endif


#ifdef USE_AVX512F
static void bench_sortingnetwort_sort_u32x16_avx512(benchmark::State& state) {
	for (uint32_t i = 0; i < 128; ++i) {
		data[i] = fastrandombytes_uint64();
	}
	for (auto _ : state) {
		sortingnetwork_sort_u32x16(*((__m512i *)data.data()));
	}
}
BENCHMARK(bench_sortingnetwort_sort_u32x16_avx512)->Range(16, 16);
#endif


int main(int argc, char** argv) {
	random_seed(time(NULL));

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
