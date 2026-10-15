#include <benchmark/benchmark.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "random.h"
#include "sort/ska_sort.h"

using namespace cryptanalysislib;

/// sorted by `key`, `payload` is carried along
struct record {
	uint32_t key;
	uint32_t payload;
};

template<typename T>
static std::vector<T> random_input(const size_t n) {
	std::vector<T> v(n);
	for (size_t i = 0; i < n; ++i) {
		if constexpr (std::is_same_v<T, record>) {
			v[i] = record{uint32_t(rng()), uint32_t(i)};
		} else {
			v[i] = T(rng());
		}
	}
	return v;
}

/// NOTE: every iteration copies the unsorted input, the copy is part of the
/// 	measured time of all benchmarks
template<typename T>
static void BM_StdSort(benchmark::State &state) {
	const auto in = random_input<T>(state.range(0));
	std::vector<T> v(in.size());
	for (auto _: state) {
		v = in;
		if constexpr (std::is_same_v<T, record>) {
			std::sort(v.begin(), v.end(), [](const record &a, const record &b) { return a.key < b.key; });
		} else {
			std::sort(v.begin(), v.end());
		}
		benchmark::DoNotOptimize(v.data());
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * state.range(0));
}

template<typename T>
static void BM_SkaSort(benchmark::State &state) {
	const auto in = random_input<T>(state.range(0));
	std::vector<T> v(in.size());
	for (auto _: state) {
		v = in;
		if constexpr (std::is_same_v<T, record>) {
			ska_sort(v.begin(), v.end(), [](const record &a) { return a.key; });
		} else {
			ska_sort(v.begin(), v.end());
		}
		benchmark::DoNotOptimize(v.data());
	}
	state.SetItemsProcessed(int64_t(state.iterations()) * state.range(0));
}

BENCHMARK(BM_StdSort<uint32_t>)->RangeMultiplier(8)->Range(1u << 8, 1u << 23);
BENCHMARK(BM_StdSort<uint64_t>)->RangeMultiplier(8)->Range(1u << 8, 1u << 23);
BENCHMARK(BM_StdSort<record>)->RangeMultiplier(8)->Range(1u << 8, 1u << 23);
BENCHMARK(BM_SkaSort<uint32_t>)->RangeMultiplier(8)->Range(1u << 8, 1u << 23);
BENCHMARK(BM_SkaSort<uint64_t>)->RangeMultiplier(8)->Range(1u << 8, 1u << 23);
BENCHMARK(BM_SkaSort<record>)->RangeMultiplier(8)->Range(1u << 8, 1u << 23);

int main(int argc, char **argv) {
	rng_seed(42);

	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
