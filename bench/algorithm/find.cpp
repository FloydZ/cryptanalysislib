#include <benchmark/benchmark.h>

#include "algorithm/find.h"
constexpr size_t LS = 1u << 20u;

using namespace cryptanalysislib;
 
template<typename T>
static void BM_stdfind(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        const auto r1 = std::find(fr.begin(), fr.end(), 0);
        c += cpucycles();
        auto t1 = std::distance(fr.begin(), r1);
		benchmark::DoNotOptimize(t1+=1);
	}

	state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_find(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        const auto r1 = cryptanalysislib::find(fr.begin(), fr.end(), 0);
        c += cpucycles();
        auto t1 = (size_t)std::distance(fr.begin(), r1);
		benchmark::DoNotOptimize(t1+=1);
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_simd_find(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r1 = cryptanalysislib::internal::find_uXX_simd<T>(fr.data(), state.range(0), 0);
        c += cpucycles();
		benchmark::DoNotOptimize(r1+=1);
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
}


template<typename T>
static void BM_find_multithreaded(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        const auto r1 = cryptanalysislib::find(par_if(true), fr.begin(), fr.end(), 0);
        c += cpucycles();
        auto t1 = std::distance(fr.begin(), r1);
		benchmark::DoNotOptimize(t1+=1);
	}

	state.counters["cycles"] = (double)c/(double)state.iterations();
}

BENCHMARK(BM_stdfind<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdfind<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdfind<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_find<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_find<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_find<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_simd_find<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_simd_find<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_simd_find<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_find_multithreaded<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_find_multithreaded<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_find_multithreaded<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK_MAIN();
