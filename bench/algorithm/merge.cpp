#include <benchmark/benchmark.h>
#include <algorithm>
#include <vector>

#include "random.h"
#include "algorithm/merge.h"

using namespace cryptanalysislib;
constexpr size_t LS = 1u << 20u;

static std::vector<uint32_t> a, b, out;
static void generate(const size_t n) {
	a.resize(n); b.resize(n); out.resize(2 * n);
	for (auto &x : a) { x = rng<uint32_t>(); }
	for (auto &x : b) { x = rng<uint32_t>(); }
	std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end());
}

static void BM_std_merge(benchmark::State &state) {
	generate(state.range(0));
	for (auto _: state) {
		std::merge(a.begin(), a.end(), b.begin(), b.end(), out.begin());
		benchmark::ClobberMemory();
	}
}

static void BM_merge(benchmark::State &state) {
	generate(state.range(0));
	for (auto _: state) {
		cryptanalysislib::merge(a.begin(), a.end(), b.begin(), b.end(), out.begin());
		benchmark::ClobberMemory();
	}
}

static void BM_merge_par(benchmark::State &state) {
	generate(state.range(0));
	for (auto _: state) {
		cryptanalysislib::merge(par_if(true), a.begin(), a.end(), b.begin(), b.end(), out.begin());
		benchmark::ClobberMemory();
	}
}

BENCHMARK(BM_std_merge)->RangeMultiplier(8)->Range(64, LS);
BENCHMARK(BM_merge)->RangeMultiplier(8)->Range(64, LS);
BENCHMARK(BM_merge_par)->RangeMultiplier(8)->Range(64, LS)->UseRealTime();
BENCHMARK_MAIN();
