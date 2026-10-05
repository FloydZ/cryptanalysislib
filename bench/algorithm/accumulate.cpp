#include <benchmark/benchmark.h>
#include <numeric>

#include "algorithm/accumulate.h"
#include "cpucycles.h"
constexpr size_t LS = 1u << 20u;

using namespace cryptanalysislib;
 
template<typename T>
static void BM_stdaccumulate(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = std::accumulate(fr.begin(), fr.end(), 1);
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_accumulate(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = cryptanalysislib::accumulate(fr.begin(), fr.end(), 1);
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_accumulate_uXX_simd(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = cryptanalysislib::internal::accumulate_simd_int_plus<T>(fr.data(), state.range(0), 1);
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_accumulate_multithreaded(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = cryptanalysislib::accumulate(par_if(true), fr.begin(), fr.end(), 1);
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}


BENCHMARK(BM_stdaccumulate<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdaccumulate<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdaccumulate<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_accumulate<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_accumulate<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_accumulate<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_accumulate_uXX_simd<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_accumulate_uXX_simd<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_accumulate_uXX_simd<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_accumulate_multithreaded<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_accumulate_multithreaded<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_accumulate_multithreaded<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK_MAIN();
