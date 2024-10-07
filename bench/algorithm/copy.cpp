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
	for (auto _: state) {
        std::copy(fr.begin(), fr.end(), to.begin());
		benchmark::ClobberMemory();
	}
}

template<typename T>
static void BM_copy(benchmark::State &state) {
	static std::vector<T> fr;
	static std::vector<T> to;
    to.resize(state.range(0));
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

	for (auto _: state) {
        cryptanalysislib::copy(fr.begin(), fr.end(), to.begin());
		benchmark::ClobberMemory();
	}
}


template<typename T>
static void BM_copy_multithreaded(benchmark::State &state) {
	static std::vector<T> fr;
	static std::vector<T> to;
    to.resize(state.range(0));
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

	for (auto _: state) {
        cryptanalysislib::copy(par_if(true), fr.begin(), fr.end(), to.begin());
		benchmark::ClobberMemory();
	}
}


BENCHMARK(BM_stdcopy<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_copy<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_copy_multithreaded<uint8_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK_MAIN();
