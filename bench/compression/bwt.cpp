#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "compression/compression.h"

/// random text over 4 symbols, ending with END_MARKER
static std::vector<uint8_t> text(const size_t n) {
	std::vector<uint8_t> t(n);
	for (size_t i = 0; i + 1 < n; ++i) { t[i] = uint8_t('a' + rand() % 4); }
	t[n - 1] = END_MARKER;
	return t;
}

static void BM_bwt_inplace(benchmark::State &state) {
	const auto t = text(state.range(0));
	std::vector<uint8_t> T(t.size());
	for (auto _: state) {
		T = t;
		benchmark::DoNotOptimize(bwt_inplace(T.data(), int(T.size())));
	}
	state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(t.size()));
}

static void BM_bwt_lcp_inplace(benchmark::State &state) {
	const auto t = text(state.range(0));
	std::vector<uint8_t> T(t.size());
	std::vector<int> LCP(t.size());
	for (auto _: state) {
		T = t;
		bwt_lcp_inplace(T.data(), int(T.size()), LCP.data());
		benchmark::DoNotOptimize(LCP.data());
	}
	state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(t.size()));
}

static void BM_bwt_reverse(benchmark::State &state) {
	auto T = text(state.range(0));
	bwt_inplace(T.data(), int(T.size()));
	for (auto _: state) {
		uint8_t *r = bwt_reverse(T.data(), int(T.size()));
		benchmark::DoNotOptimize(r);
		free(r);
	}
	state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(T.size()));
}

// NOTE: O(n^2) algorithms
BENCHMARK(BM_bwt_inplace)->RangeMultiplier(4)->Range(256, 1u << 14);
BENCHMARK(BM_bwt_lcp_inplace)->RangeMultiplier(4)->Range(256, 1u << 14);
BENCHMARK(BM_bwt_reverse)->RangeMultiplier(4)->Range(256, 1u << 14);
BENCHMARK_MAIN();
