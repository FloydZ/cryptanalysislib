#include <benchmark/benchmark.h>
#include <numeric>

#include "algorithm/all_of.h"
#include "cpucycles.h"
constexpr size_t LS = 1u << 20u;

using namespace cryptanalysislib;
 
template<typename T>
static void BM_stdall_of(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = std::all_of(fr.begin(), fr.end(), [](const T &a) {
	        return a == 1;
        });
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_all_of(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = cryptanalysislib::all_of(fr.begin(), fr.end(), [](const T &a) {
	        return a == 1;
        });
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_all_of_multithreaded(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = cryptanalysislib::all_of(par_if(true), fr.begin(), fr.end(), [](const T &a) {
	        return a == 1;
        });
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}



template<typename T>
static void BM_stdnone_of(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = std::none_of(fr.begin(), fr.end(), [](const T &a) {
	        return a == 1;
        });
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_none_of(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = cryptanalysislib::none_of(fr.begin(), fr.end(), [](const T &a) {
	        return a == 1;
        });
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_none_of_multithreaded(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = cryptanalysislib::none_of(par_if(true), fr.begin(), fr.end(), [](const T &a) {
	        return a == 1;
        });
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}


template<typename T>
static void BM_stdany_of(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = std::any_of(fr.begin(), fr.end(), [](const T &a) {
	        return a == 1;
        });
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_any_of(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = cryptanalysislib::any_of(fr.begin(), fr.end(), [](const T &a) {
	        return a == 1;
        });
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_any_of_multithreaded(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = cryptanalysislib::none_of(par_if(true), fr.begin(), fr.end(), [](const T &a) {
	        return a == 1;
        });
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}



BENCHMARK(BM_stdall_of<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdall_of<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdall_of<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_all_of<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_all_of<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_all_of<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_all_of_multithreaded<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_all_of_multithreaded<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_all_of_multithreaded<uint64_t>)->RangeMultiplier(2)->Range(32, LS);



BENCHMARK(BM_stdnone_of<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdnone_of<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdnone_of<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_none_of<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_none_of<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_none_of<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_none_of_multithreaded<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_none_of_multithreaded<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_none_of_multithreaded<uint64_t>)->RangeMultiplier(2)->Range(32, LS);



BENCHMARK(BM_stdany_of<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdany_of<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stdany_of<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_any_of<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_any_of<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_any_of<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_any_of_multithreaded<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_any_of_multithreaded<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_any_of_multithreaded<uint64_t>)->RangeMultiplier(2)->Range(32, LS);


BENCHMARK_MAIN();
