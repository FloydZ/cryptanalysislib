#include <benchmark/benchmark.h>

#include "sort/sort.h"

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

template <class T>
void BM_counting_stable_sort(benchmark::State& state) {
	std::vector<T> l((size_t)state.range(0));
	random_data<std::vector<T>, T>(l, search_index, state.range(0), 1);
	std::vector<T> output(state.range(0));

	for (auto _ : state) {
		counting_sort_stable_u8(output.data(), l.data(), state.range(0));
	}
	state.SetComplexityN(state.range(0));
}


//BENCHMARK(BM_counting_sort<uint8_t>)->DenseRange(32, 1024, 32); // ->RangeMultiplier(2)->Range(16, LS)->Complexity();
BENCHMARK(BM_counting_stable_sort<uint8_t>)->DenseRange(32, 1024, 32); // ->RangeMultiplier(2)->Range(16, LS)->Complexity();

int main(int argc, char** argv) {
	rng_seed(time(NULL));

	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
