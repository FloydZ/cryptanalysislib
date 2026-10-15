#include <benchmark/benchmark.h>
#include <algorithm>
#include <vector>

#include "random.h"
#include "algorithm/remove.h"

using namespace cryptanalysislib;
constexpr size_t LS = 1u << 22u;

/// 25% of the elements are removed
static std::vector<uint32_t> data, work;
static void generate(const size_t n) {
	data.resize(n); work.resize(n);
	for (auto &x : data) { x = rng<uint32_t>() % 4u; }
}

static void BM_std_remove(benchmark::State &state) {
	generate(state.range(0));
	for (auto _: state) {
		work = data;
		auto r = std::remove(work.begin(), work.end(), 1u);
		benchmark::DoNotOptimize(r);
	}
}

static void BM_remove(benchmark::State &state) {
	generate(state.range(0));
	for (auto _: state) {
		work = data;
		auto r = cryptanalysislib::remove(work.begin(), work.end(), 1u);
		benchmark::DoNotOptimize(r);
	}
}

static void BM_remove_par(benchmark::State &state) {
	generate(state.range(0));
	for (auto _: state) {
		work = data;
		auto r = cryptanalysislib::remove(par_if(true), work.begin(), work.end(), 1u);
		benchmark::DoNotOptimize(r);
	}
}

BENCHMARK(BM_std_remove)->RangeMultiplier(16)->Range(1024, LS);
BENCHMARK(BM_remove)->RangeMultiplier(16)->Range(1024, LS);
BENCHMARK(BM_remove_par)->RangeMultiplier(16)->Range(1024, LS)->UseRealTime();
BENCHMARK_MAIN();
