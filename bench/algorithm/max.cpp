#include <benchmark/benchmark.h>
#include <algorithm>
#include <vector>

#include "random.h"
#include "algorithm/max.h"

using namespace cryptanalysislib;
constexpr size_t LS = 1u << 16u;

template<typename T>
static void generate_data(std::vector<T> &out, const size_t size) noexcept {
	out.resize(size);
	for (size_t i = 0; i < size; ++i) { out[i] = rng<T>(); }
}

template<typename T>
static void BM_std_max_element(benchmark::State &state) {
	static std::vector<T> data;
	generate_data(data, state.range(0));
	for (auto _: state) {
		auto t = std::max_element(data.begin(), data.end());
		benchmark::DoNotOptimize(t);
	}
}

template<typename T>
static void BM_max_element(benchmark::State &state) {
	static std::vector<T> data;
	generate_data(data, state.range(0));
	for (auto _: state) {
		auto t = cryptanalysislib::max_element(data.begin(), data.end());
		benchmark::DoNotOptimize(t);
	}
}

BENCHMARK(BM_std_max_element<uint8_t>)->RangeMultiplier(4)->Range(32, LS);
BENCHMARK(BM_std_max_element<uint32_t>)->RangeMultiplier(4)->Range(32, LS);
BENCHMARK(BM_std_max_element<uint64_t>)->RangeMultiplier(4)->Range(32, LS);
BENCHMARK(BM_max_element<uint8_t>)->RangeMultiplier(4)->Range(32, LS);
BENCHMARK(BM_max_element<uint32_t>)->RangeMultiplier(4)->Range(32, LS);
BENCHMARK(BM_max_element<uint64_t>)->RangeMultiplier(4)->Range(32, LS);
BENCHMARK_MAIN();
