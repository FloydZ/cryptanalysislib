#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "compression/deflate.h"
#include "compression/inflate.h"

/// text over 9 symbols, compressible
static std::vector<uint8_t> text(const size_t n) {
	std::vector<uint8_t> t(n);
	for (size_t i = 0; i < n; ++i) { t[i] = uint8_t("abcdefgh "[rand() % 9]); }
	return t;
}

/// argument 0: input size, argument 1: compression level
static void BM_sdeflate(benchmark::State &state) {
	const auto t = text(state.range(0));
	std::vector<uint8_t> out(sdefl_bound(int(t.size())));
	auto s = std::make_unique<struct sdefl>();
	int len = 0;
	for (auto _: state) {
		len = sdeflate(s.get(), out.data(), t.data(), int(t.size()), int(state.range(1)));
		benchmark::DoNotOptimize(len);
	}
	state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(t.size()));
	state.counters["ratio"] = double(len) / double(t.size());
}

static void BM_sinflate(benchmark::State &state) {
	const auto t = text(state.range(0));
	std::vector<uint8_t> comp(sdefl_bound(int(t.size()))), out(t.size());
	auto s = std::make_unique<struct sdefl>();
	const int len = sdeflate(s.get(), comp.data(), t.data(), int(t.size()), SDEFL_LVL_DEF);
	for (auto _: state) {
		benchmark::DoNotOptimize(sinflate(out.data(), int(out.size()), comp.data(), len));
	}
	state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(t.size()));
}

BENCHMARK(BM_sdeflate)->ArgsProduct({{1u << 12, 1u << 16, 1u << 20}, {SDEFL_LVL_MIN, SDEFL_LVL_DEF, SDEFL_LVL_MAX}});
BENCHMARK(BM_sinflate)->RangeMultiplier(16)->Range(1u << 12, 1u << 20);
BENCHMARK_MAIN();
