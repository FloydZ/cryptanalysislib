#include "benchmark/benchmark.h"
#include "../bench_config.h"

#include "container/binary_packed_vector.h"
#include <helper.h>
//#include <sort.h>
#include "sort/sort.h"


constexpr static uint64_t sssize = 63;
constexpr static uint64_t k_lower = 30;
constexpr static uint64_t k_higher = sssize;
using ContainerT = BinaryContainer<sssize>;
constexpr uint64_t SIZE_LIST = 1ull << 20;


std::vector<ContainerT> data;

void setup(const benchmark::State& state) {
	data.resize(SIZE_LIST);
	for (uint32_t i = 0; i < SIZE_LIST; ++i) {
		data[i].random();
	}
}

static void BM_std_sort(benchmark::State& state) {
	setup(state);
	for (auto _ : state) {
		std::sort(data.begin(), data.end(), [](const ContainerT &a, const ContainerT &b){
			return a.is_greater<k_lower, k_higher>(b);
		});
	}
}

static void BM_crum_sort(benchmark::State& state) {
	for (auto _ : state) {
		crumsort<ContainerT>(data.data(), SIZE_LIST, [](const ContainerT *a, const ContainerT *b){
			return a->is_greater<k_lower, k_higher>(*b);
		});
	}
}

// run the benchmarks
//BENCHMARK(BM_std_sort)->Setup(setup);
//BENCHMARK(BM_crum_sort)->Setup(setup);
BENCHMARK(BM_std_sort);
BENCHMARK(BM_crum_sort);
BENCHMARK_MAIN();
