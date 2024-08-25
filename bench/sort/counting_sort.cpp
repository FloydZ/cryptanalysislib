#include <benchmark/benchmark.h>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include "sort/sort.h"
#include "../tests/search/common.h"

constexpr size_t LS = 1u << 10u;
size_t search_index;

template <class T>
void BM_counting_sort(benchmark::State& state) {
	std::vector<T> l((size_t)state.range(0));
	random_data<std::vector<T>, T>(l, search_index, state.range(0), 1);

	for (auto _ : state) {
		counting_sort_u8(l.data(), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}


BENCHMARK(BM_counting_sort<uint8_t>)->RangeMultiplier(2)->Range(128, LS)->Complexity();

int main(int argc, char** argv) {
	random_seed(time(NULL));

	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
