#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "compression/compression.h"

/// partly repetitive data (as in the tests)
static std::vector<uint8_t> data(const size_t n) {
	std::vector<uint8_t> t(n);
	for (size_t i = 0; i < n; ++i) { t[i] = uint8_t((i % 37) < 20 ? 'a' + (i % 7) : rand() % 4); }
	return t;
}

template<const unsigned long window>
static void BM_lzss_compress(benchmark::State &state) {
	const auto t = data(state.range(0));
	std::vector<uint8_t> c(cryptanalysislib::lzss<window>::bound(t.size()));
	size_t n = 0;
	for (auto _: state) {
		n = cryptanalysislib::lzss<window>::compress(c.data(), t.data(), t.size());
		benchmark::DoNotOptimize(n);
	}
	state.counters["ratio"] = double(n) / double(t.size());
	state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(t.size()));
}

static void BM_lzss_decompress(benchmark::State &state) {
	const auto t = data(state.range(0));
	std::vector<uint8_t> c(cryptanalysislib::lzss<>::bound(t.size())), out(t.size());
	cryptanalysislib::lzss<>::compress(c.data(), t.data(), t.size());
	for (auto _: state) {
		benchmark::DoNotOptimize(cryptanalysislib::lzss<>::decompress(out.data(), c.data()));
	}
	state.SetBytesProcessed(int64_t(state.iterations()) * int64_t(t.size()));
}

BENCHMARK(BM_lzss_compress<256>)->RangeMultiplier(8)->Range(1u << 10, 1u << 16);
BENCHMARK(BM_lzss_compress<MAX_WINDOWSIZE>)->RangeMultiplier(8)->Range(1u << 10, 1u << 16);
BENCHMARK(BM_lzss_decompress)->RangeMultiplier(8)->Range(1u << 10, 1u << 16);
BENCHMARK_MAIN();
