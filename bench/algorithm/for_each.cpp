#include <benchmark/benchmark.h>

#include "algorithm/for_each.h"
constexpr size_t LS = 1u << 20u;

using namespace cryptanalysislib;


template<typename T>
static void BM_stdfor_each(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);
	auto p = [](T &a) { a = a*a; };

	for (auto _: state) {
        std::for_each(fr.begin(), fr.end(), p);
		benchmark::ClobberMemory();
	}
}

template<typename T>
static void BM_for_each(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);
	auto p = [](T &a) {a = a*a;};

	for (auto _: state) {
        cryptanalysislib::for_each(fr.begin(), fr.end(), p);
		benchmark::ClobberMemory();
	}
}


template<typename T>
static void BM_for_each_multithreaded(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);
	auto p = [](T &a) {a = a*a;};

	for (auto _: state) {
        cryptanalysislib::for_each(par_if(true), fr.begin(), fr.end(), p);
		benchmark::ClobberMemory();
	}
}


BENCHMARK(BM_stdfor_each<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdfor_each<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdfor_each<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_for_each<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_for_each<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_for_each<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_for_each_multithreaded<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_for_each_multithreaded<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_for_each_multithreaded<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK_MAIN();
