#include <benchmark/benchmark.h>

#include "algorithm/count.h"
constexpr size_t LS = 1u << 20u;

using namespace cryptanalysislib;
 
template<typename T>
static void BM_stdcount(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

	for (auto _: state) {
        auto r = std::count(fr.begin(), fr.end(), 1);
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
}

template<typename T>
static void BM_count(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

	for (auto _: state) {
        auto r = cryptanalysislib::count(fr.begin(), fr.end(), 1);
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
}

template<typename T>
static void BM_count_uXX_simd(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

	for (auto _: state) {
        auto r = cryptanalysislib::internal::count_uXX_simd<T>(fr.data(), state.range(0), 1);
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
}

// template<typename T>
// static void BM_count_multithreaded(benchmark::State &state) {
// 	static std::vector<T> fr;
// 	static std::vector<T> to;
//     to.resize(state.range(0));
//     fr.resize(state.range(0));
//     std::fill(fr.begin(), fr.end(), 1);
//
// 	for (auto _: state) {
//         cryptanalysislib::count(par_if(true), fr.begin(), fr.end(), to.begin());
// 		benchmark::ClobberMemory();
// 	}
// }


BENCHMARK(BM_stdcount<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdcount<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdcount<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_count<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_count<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_count<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_count_uXX_simd<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_count_uXX_simd<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_count_uXX_simd<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

//BENCHMARK(BM_count_multithreaded<uint8_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK_MAIN();
