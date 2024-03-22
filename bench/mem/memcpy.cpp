#include <benchmark/benchmark.h>

#include "memory/memory.h"


constexpr size_t size = 9123978;

static void BM_stdmemcpy(benchmark::State &state) {
	uint8_t *ptr1 = (uint8_t *)malloc(size);
	uint8_t *ptr2 = (uint8_t *)malloc(size);
	for (auto _: state) {
		memcpy(ptr2, ptr1, size);
		benchmark::ClobberMemory();
	}
}

static void BM_memcpy(benchmark::State &state) {
	uint8_t *ptr1 = (uint8_t *)malloc(size);
	uint8_t *ptr2 = (uint8_t *)malloc(size);
	for (auto _: state) {
		cryptanalysislib::memcpy(ptr2, ptr1, size);
		benchmark::ClobberMemory();
	}
}

BENCHMARK(BM_stdmemcpy);
BENCHMARK(BM_memcpy);
BENCHMARK_MAIN();
