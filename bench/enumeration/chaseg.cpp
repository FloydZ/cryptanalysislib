#include <benchmark/benchmark.h>

#include "combination/chase.h"
#include "helper.h"

using T = uint64_t;
constexpr uint32_t n = 256;
constexpr uint32_t p = 3;

constexpr size_t list_size = bc(n, p) - 1;
constexpr uint32_t element_limbs = (n + 63) / 64;

static void BM_cbc(benchmark::State &state) {
	uint64_t res = 0;
	uint16_t pos1, pos2;

	uint64_t w1[element_limbs] = {0};
	Combinations_Binary_Chase<T, n, p> ch{};
	for (auto _: state) {
		ch.reset();
		for (size_t i = 0; i < list_size - 1; ++i) {
			// memcpy(w1, w2, element_limbs * 8);
			ch.left_step(w1, &pos1, &pos2);
			benchmark::DoNotOptimize(res += pos1 + pos2);
		}
	}
}

static void BM_ne(benchmark::State &state) {
	uint64_t res = 0;
	chase<n, p> c{};
	for (auto _: state) {
		c.enumerate([&](uint16_t p1, uint16_t p2) __attribute__((force_inline)) {
			benchmark::DoNotOptimize(res += p1 + p2);
		});
	}
}

BENCHMARK(BM_cbc);
BENCHMARK(BM_ne);
BENCHMARK_MAIN();
