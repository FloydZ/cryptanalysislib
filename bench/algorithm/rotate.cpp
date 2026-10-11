#include <benchmark/benchmark.h>
#include <algorithm>
#include <numeric>
#include <vector>

#include "algorithm/rotate.h"

using namespace cryptanalysislib;

/// rotation of n uint32_t by left = n/3
template<typename F>
static void bench_rotation(benchmark::State &state, F f) {
	const size_t n = state.range(0), left = n / 3;
	std::vector<uint32_t> v(n);
	std::iota(v.begin(), v.end(), 0u);
	for (auto _: state) {
		f(v.data(), left, n - left);
		benchmark::ClobberMemory();
	}
}

static void BM_std_rotate(benchmark::State &s) { bench_rotation(s, [](uint32_t *a, size_t l, size_t r) { std::rotate(a, a + l, a + l + r); }); }
static void BM_contrev(benchmark::State &s)    { bench_rotation(s, [](uint32_t *a, size_t l, size_t r) { contrev_rotation(a, l, r); }); }
static void BM_trinity(benchmark::State &s)    { bench_rotation(s, [](uint32_t *a, size_t l, size_t r) { trinity_rotation<uint32_t, 32>(a, l, r); }); }
static void BM_griesmills(benchmark::State &s) { bench_rotation(s, [](uint32_t *a, size_t l, size_t r) { griesmills_rotation(a, l, r); }); }
static void BM_grail(benchmark::State &s)      { bench_rotation(s, [](uint32_t *a, size_t l, size_t r) { grail_rotation(a, l, r); }); }
static void BM_piston(benchmark::State &s)     { bench_rotation(s, [](uint32_t *a, size_t l, size_t r) { piston_rotation(a, l, r); }); }
static void BM_helix(benchmark::State &s)      { bench_rotation(s, [](uint32_t *a, size_t l, size_t r) { helix_rotation(a, l, r); }); }
static void BM_drill(benchmark::State &s)      { bench_rotation(s, [](uint32_t *a, size_t l, size_t r) { drill_rotation(a, l, r); }); }

#define RANGE ->RangeMultiplier(16)->Range(64, 1u << 20)
BENCHMARK(BM_std_rotate) RANGE;
BENCHMARK(BM_contrev) RANGE;
BENCHMARK(BM_trinity) RANGE;
BENCHMARK(BM_griesmills) RANGE;
BENCHMARK(BM_grail) RANGE;
BENCHMARK(BM_piston) RANGE;
BENCHMARK(BM_helix) RANGE;
BENCHMARK(BM_drill) RANGE;
BENCHMARK_MAIN();
