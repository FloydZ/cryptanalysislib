#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "compression/horizontalpacking32.h"

using namespace cryptanalysislib;

/// argument 0: number of values, argument 1: bits per value
static void BM_horizontalpack32(benchmark::State &state) {
	const size_t n = state.range(0);
	const uint32_t b = uint32_t(state.range(1));
	std::vector<uint32_t> v(n);
	for (auto &x : v) { x = uint32_t(rand()); }
	std::vector<uint8_t> p(horizontalpacked_size(n, b));
	for (auto _: state) {
		horizontalpack32(v.data(), n, b, p.data());
		benchmark::DoNotOptimize(p.data());
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(n));
}

static void BM_horizontalunpack32(benchmark::State &state) {
	const size_t n = state.range(0);
	const uint32_t b = uint32_t(state.range(1));
	std::vector<uint32_t> v(n), o(n);
	for (auto &x : v) { x = uint32_t(rand()); }
	std::vector<uint8_t> p(horizontalpacked_size(n, b));
	horizontalpack32(v.data(), n, b, p.data());
	for (auto _: state) {
		horizontalunpack32(p.data(), n, b, o.data());
		benchmark::DoNotOptimize(o.data());
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(n));
}

static void BM_horizontalunpack32_scalar(benchmark::State &state) {
	const size_t n = state.range(0);
	const uint32_t b = uint32_t(state.range(1));
	std::vector<uint32_t> v(n), o(n);
	for (auto &x : v) { x = uint32_t(rand()); }
	std::vector<uint8_t> p(horizontalpacked_size(n, b));
	horizontalpack32(v.data(), n, b, p.data());
	for (auto _: state) {
		horizontalunpack32_scalar(p.data(), n, b, o.data());
		benchmark::DoNotOptimize(o.data());
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * int64_t(n));
}

BENCHMARK(BM_horizontalpack32)->ArgsProduct({{1u << 16}, {1, 7, 13, 32}});
BENCHMARK(BM_horizontalunpack32)->ArgsProduct({{1u << 16}, {1, 7, 13, 32}});
BENCHMARK(BM_horizontalunpack32_scalar)->ArgsProduct({{1u << 16}, {1, 7, 13, 32}});
BENCHMARK_MAIN();
