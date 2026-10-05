#include <benchmark/benchmark.h>

#include "algorithm/exclusive_scan.h"
constexpr size_t LS = 1u << 20u;

using namespace cryptanalysislib;
 
template<typename T>
static void BM_stdexclusive_scan(benchmark::State &state) {
	static std::vector<T> fr;
	static std::vector<T> to;

    to.resize(state.range(0));
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);
    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        std::exclusive_scan(fr.begin(), fr.end(), to.begin(), 0);
        c += cpucycles();
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_exclusive_scan(benchmark::State &state) {
	static std::vector<T> fr;
	static std::vector<T> to;
    to.resize(state.range(0));
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        cryptanalysislib::exclusive_scan(fr.begin(), fr.end(), to.begin());
        c += cpucycles();
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}


template<typename T>
static void BM_exclusive_scan_multithreaded(benchmark::State &state) {
	static std::vector<T> fr;
	static std::vector<T> to;
    to.resize(state.range(0));
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        cryptanalysislib::exclusive_scan(par_if(true), fr.begin(), fr.end(), to.begin());
        c += cpucycles();
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}


BENCHMARK(BM_stdexclusive_scan<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdexclusive_scan<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdexclusive_scan<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_exclusive_scan<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_exclusive_scan<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_exclusive_scan<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_exclusive_scan_multithreaded<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_exclusive_scan_multithreaded<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_exclusive_scan_multithreaded<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK_MAIN();
