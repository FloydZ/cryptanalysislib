#include <benchmark/benchmark.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "random.h"
#include "sort/heap/is_heap.h"
#include "sort/heap/push_heap.h"

/// a max heap of `n` random elements
static std::vector<int32_t> random_heap(const size_t n) {
	std::vector<int32_t> v(n);
	for (auto &x : v) { x = int32_t(cryptanalysislib::rng()); }
	std::make_heap(v.begin(), v.end());
	return v;
}

template<bool (*F)(const int32_t *, const int32_t *)>
static void BM_is_heap(benchmark::State &state) {
	const auto v = random_heap(state.range(0));
	for (auto _ : state) {
		benchmark::DoNotOptimize(F(v.data(), v.data() + v.size()));
	}
	state.SetComplexityN(state.range(0));
}

static bool std_is_heap(const int32_t *b, const int32_t *e) { return std::is_heap(b, e); }
static bool fwd_is_heap(const int32_t *b, const int32_t *e) { return is_heap_fwd(b, e, std::less<>{}); }
static bool rnd_is_heap(const int32_t *b, const int32_t *e) { return is_heap_rnd(b, e, std::less<>{}); }

BENCHMARK(BM_is_heap<std_is_heap>)->RangeMultiplier(8)->Range(64, 1u << 18);
BENCHMARK(BM_is_heap<fwd_is_heap>)->RangeMultiplier(8)->Range(64, 1u << 18);
BENCHMARK(BM_is_heap<rnd_is_heap>)->RangeMultiplier(8)->Range(64, 1u << 18);
#ifdef USE_AVX2
BENCHMARK(BM_is_heap<is_heap_sse_epi32>)->RangeMultiplier(8)->Range(64, 1u << 18);
BENCHMARK(BM_is_heap<is_heap_avx2_epi32>)->RangeMultiplier(8)->Range(64, 1u << 18);
#endif
#ifdef USE_AVX512F
BENCHMARK(BM_is_heap<is_heap_avx512_epi32>)->RangeMultiplier(8)->Range(64, 1u << 18);
#endif

/// pushes `n` random elements into a heap of size `state.range(0)`
template<typename F>
static void push_bench(benchmark::State &state, F f) {
	const auto base = random_heap(state.range(0));
	std::vector<int32_t> v;
	v.reserve(base.size() + 1);
	for (auto _ : state) {
		state.PauseTiming();
		v = base;
		v.push_back(int32_t(cryptanalysislib::rng()));
		state.ResumeTiming();
		f(v);
		benchmark::DoNotOptimize(v.data());
	}
}

static void BM_push_heap_std(benchmark::State &state) {
	push_bench(state, [](std::vector<int32_t> &v) { std::push_heap(v.begin(), v.end()); });
}
static void BM_push_heap_scalar(benchmark::State &state) {
	push_bench(state, [](std::vector<int32_t> &v) { push_heap_scalar(v.data(), v.data() + v.size(), std::less<>{}); });
}
BENCHMARK(BM_push_heap_std)->Arg(1000)->Arg(40000);
BENCHMARK(BM_push_heap_scalar)->Arg(1000)->Arg(40000);
#ifdef USE_AVX2
static void BM_push_heap_avx2(benchmark::State &state) {
	push_bench(state, [](std::vector<int32_t> &v) { push_heap_avx2(v.data(), v.size()); });
}
BENCHMARK(BM_push_heap_avx2)->Arg(1000)->Arg(40000);
#endif
#ifdef USE_AVX512F
static void BM_push_heap_avx512(benchmark::State &state) {
	push_bench(state, [](std::vector<int32_t> &v) { push_heap_avx512(v.data(), v.size()); });
}
BENCHMARK(BM_push_heap_avx512)->Arg(1000)->Arg(40000);
#endif

int main(int argc, char** argv) {
	cryptanalysislib::rng_seed(time(NULL));

	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
