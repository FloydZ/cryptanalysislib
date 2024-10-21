#include <benchmark/benchmark.h>
#include <numeric>

#include "algorithm/gcd.h"
#include "cpucycles.h"
#include "random.h"

using namespace cryptanalysislib;

template<typename T>
static void BM_gcd_recursive_v0(benchmark::State &state) {
    uint64_t c = 0;

	for (auto _: state) {
		const T a = rng<T>();
		const T b = rng<T>();

        c -= cpucycles();
        auto r = cryptanalysislib::internal::gcd_recursive_v0<T>(a, b);
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_gcd_recursive_v1(benchmark::State &state) {
    uint64_t c = 0;

	for (auto _: state) {
		const T a = rng<T>();
		const T b = rng<T>();

        c -= cpucycles();
        auto r = cryptanalysislib::internal::gcd_recursive_v1<T>(a, b);
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_gcd_recursive_v2(benchmark::State &state) {
    uint64_t c = 0;

	for (auto _: state) {
		const T a = rng<T>();
		const T b = rng<T>();

        c -= cpucycles();
        auto r = cryptanalysislib::internal::gcd_recursive_v2<T>(a, b);
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_gcd_nonrecursive_v1(benchmark::State &state) {
    uint64_t c = 0;

	for (auto _: state) {
		const T a = rng<T>();
		const T b = rng<T>();

        c -= cpucycles();
        auto r = cryptanalysislib::internal::gcd_non_recursive_v1<T>(a, b);
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_gcd_nonrecursive_v2(benchmark::State &state) {
    uint64_t c = 0;

	for (auto _: state) {
		const T a = rng<T>();
		const T b = rng<T>();

        c -= cpucycles();
        auto r = cryptanalysislib::internal::gcd_non_recursive_v2<T>(a, b);
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_gcd_binary(benchmark::State &state) {
    uint64_t c = 0;

	for (auto _: state) {
		const T a = rng<T>();
		const T b = rng<T>();

        c -= cpucycles();
        auto r = cryptanalysislib::internal::gcd_binary<T>(a, b);
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}


BENCHMARK(BM_gcd_recursive_v0<uint8_t>)->RangeMultiplier(2);
BENCHMARK(BM_gcd_recursive_v0<uint32_t>)->RangeMultiplier(2);
BENCHMARK(BM_gcd_recursive_v0<uint64_t>)->RangeMultiplier(2);

BENCHMARK(BM_gcd_recursive_v1<uint8_t>)->RangeMultiplier(2);
BENCHMARK(BM_gcd_recursive_v1<uint32_t>)->RangeMultiplier(2);
BENCHMARK(BM_gcd_recursive_v1<uint64_t>)->RangeMultiplier(2);

BENCHMARK(BM_gcd_recursive_v2<uint8_t>)->RangeMultiplier(2);
BENCHMARK(BM_gcd_recursive_v2<uint32_t>)->RangeMultiplier(2);
BENCHMARK(BM_gcd_recursive_v2<uint64_t>)->RangeMultiplier(2);


BENCHMARK(BM_gcd_nonrecursive_v1<uint8_t>)->RangeMultiplier(2);
BENCHMARK(BM_gcd_nonrecursive_v1<uint32_t>)->RangeMultiplier(2);
BENCHMARK(BM_gcd_nonrecursive_v1<uint64_t>)->RangeMultiplier(2);

BENCHMARK(BM_gcd_nonrecursive_v2<uint8_t>)->RangeMultiplier(2);
BENCHMARK(BM_gcd_nonrecursive_v2<uint32_t>)->RangeMultiplier(2);
BENCHMARK(BM_gcd_nonrecursive_v2<uint64_t>)->RangeMultiplier(2);


BENCHMARK(BM_gcd_binary<uint8_t>)->RangeMultiplier(2);
BENCHMARK(BM_gcd_binary<uint32_t>)->RangeMultiplier(2);
BENCHMARK(BM_gcd_binary<uint64_t>)->RangeMultiplier(2);


BENCHMARK_MAIN();
