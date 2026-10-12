#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "compression/compress_float.h"

static std::vector<double> data(const size_t n) {
	std::vector<double> v(n);
	for (auto &x : v) { x = double(rand()) / RAND_MAX; }
	return v;
}

static void BM_to_float16(benchmark::State &state) {
	const auto v = data(state.range(0));
	std::vector<uint16_t> h(v.size());
	for (auto _: state) {
		cryptanalysislib::to_float16(h.data(), v.data(), v.size());
		benchmark::ClobberMemory();
	}
	state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(v.size() * sizeof(double)));
}

static void BM_from_float16(benchmark::State &state) {
	const auto v = data(state.range(0));
	std::vector<uint16_t> h(v.size());
	std::vector<double> r(v.size());
	cryptanalysislib::to_float16(h.data(), v.data(), v.size());
	for (auto _: state) {
		cryptanalysislib::from_float16(r.data(), h.data(), h.size());
		benchmark::ClobberMemory();
	}
	state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(v.size() * sizeof(double)));
}

BENCHMARK(BM_to_float16)->RangeMultiplier(16)->Range(64, 1u << 20);
BENCHMARK(BM_from_float16)->RangeMultiplier(16)->Range(64, 1u << 20);
BENCHMARK_MAIN();
