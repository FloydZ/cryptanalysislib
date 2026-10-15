#include <benchmark/benchmark.h>
#include <set>
#include <vector>

#include "container/cartesian_tree.h"
#include "random.h"

using namespace cryptanalysislib;

static std::vector<uint32_t> keys(const size_t n) {
	std::vector<uint32_t> v(n);
	for (auto &x : v) { x = rng<uint32_t>(); }
	return v;
}

static void BM_cartesian_tree_insert(benchmark::State &state) {
	const auto v = keys(state.range(0));
	for (auto _: state) {
		cartesian_tree<uint32_t> t;
		for (const auto x : v) { t.insert(x); }
		benchmark::DoNotOptimize(t.size());
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(v.size()));
}

static void BM_std_set_insert(benchmark::State &state) {
	const auto v = keys(state.range(0));
	for (auto _: state) {
		std::set<uint32_t> t;
		for (const auto x : v) { t.insert(x); }
		benchmark::DoNotOptimize(t.size());
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(v.size()));
}

static void BM_cartesian_tree_contains(benchmark::State &state) {
	const auto v = keys(state.range(0));
	cartesian_tree<uint32_t> t;
	for (const auto x : v) { t.insert(x); }
	for (auto _: state) {
		size_t c = 0;
		for (const auto x : v) { c += t.contains(x); }
		benchmark::DoNotOptimize(c);
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(v.size()));
}

BENCHMARK(BM_cartesian_tree_insert)->RangeMultiplier(16)->Range(1u << 10, 1u << 18);
BENCHMARK(BM_std_set_insert)->RangeMultiplier(16)->Range(1u << 10, 1u << 18);
BENCHMARK(BM_cartesian_tree_contains)->RangeMultiplier(16)->Range(1u << 10, 1u << 18);
BENCHMARK_MAIN();
