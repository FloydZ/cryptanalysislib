#include <benchmark/benchmark.h>

#include "algorithm/copy.h"
constexpr size_t LS = 1u << 20u;

using namespace cryptanalysislib;
 
template<typename T>
static void BM_stdcopy(benchmark::State &state) {
	static std::vector<T> fr;
	static std::vector<T> to;

    to.resize(state.range(0));
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);
    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        std::copy(fr.begin(), fr.end(), to.begin());
        c += cpucycles();
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_copy(benchmark::State &state) {
	static std::vector<T> fr;
	static std::vector<T> to;
    to.resize(state.range(0));
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        cryptanalysislib::copy(fr.begin(), fr.end(), to.begin());
        c += cpucycles();
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}


template<typename T>
static void BM_copy_multithreaded(benchmark::State &state) {
	static std::vector<T> fr;
	static std::vector<T> to;
    to.resize(state.range(0));
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        cryptanalysislib::copy(par_if(true), fr.begin(), fr.end(), to.begin());
        c += cpucycles();
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}


BENCHMARK(BM_stdcopy<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
//BENCHMARK(BM_stdcopy<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
//BENCHMARK(BM_stdcopy<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_copy<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
//BENCHMARK(BM_copy<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
//BENCHMARK(BM_copy<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_copy_multithreaded<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
//BENCHMARK(BM_copy_multithreaded<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
//BENCHMARK(BM_copy_multithreaded<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK_MAIN();
