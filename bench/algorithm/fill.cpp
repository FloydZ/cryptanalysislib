#include <benchmark/benchmark.h>

#include "algorithm/fill.h"
constexpr size_t LS = 1u << 20u;

using namespace cryptanalysislib;
 
template<typename T>
static void BM_stdfill(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

	uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        std::fill(fr.begin(), fr.end(), 0);
        c += cpucycles();
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_fill(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        cryptanalysislib::fill(fr.begin(), fr.end(), 0);
        c += cpucycles();
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_fill_multithreaded(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        cryptanalysislib::fill(par_if(true), fr.begin(), fr.end(), 0);
        c += cpucycles();
		benchmark::ClobberMemory();
	}

	state.counters["cycles"] = (double)c/(double)state.iterations();
}

BENCHMARK(BM_stdfill<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdfill<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdfill<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_fill<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_fill<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_fill<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_fill_multithreaded<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_fill_multithreaded<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_fill_multithreaded<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK_MAIN();
