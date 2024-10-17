#include <benchmark/benchmark.h>

#include "algorithm/equal.h"
constexpr size_t LS = 1u << 20u;

using namespace cryptanalysislib;
 
template<typename T>
static void BM_stdequal(benchmark::State &state) {
	static std::vector<T> fr;
	static std::vector<T> to;

    to.resize(state.range(0));
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);
    std::fill(to.begin(), to.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        bool b = std::equal(fr.begin(), fr.end(), to.begin());
        c += cpucycles();
		benchmark::DoNotOptimize(b+1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_equal(benchmark::State &state) {
	static std::vector<T> fr;
	static std::vector<T> to;
    to.resize(state.range(0));
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);
    std::fill(to.begin(), to.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        bool b = cryptanalysislib::equal(fr.begin(), fr.end(), to.begin());
        c += cpucycles();
		benchmark::DoNotOptimize(b+1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}


template<typename T>
static void BM_equal_multithreaded(benchmark::State &state) {
	static std::vector<T> fr;
	static std::vector<T> to;
    to.resize(state.range(0));
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);
    std::fill(to.begin(), to.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        bool b = cryptanalysislib::equal(par_if(true), fr.begin(), fr.end(), to.begin());
        c += cpucycles();
		benchmark::DoNotOptimize(b+1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}


BENCHMARK(BM_stdequal<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
//BENCHMARK(BM_stdequal<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
//BENCHMARK(BM_stdequal<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_equal<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
//BENCHMARK(BM_equal<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
//BENCHMARK(BM_equal<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_equal_multithreaded<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
//BENCHMARK(BM_equal_multithreaded<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
//BENCHMARK(BM_equal_multithreaded<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK_MAIN();
