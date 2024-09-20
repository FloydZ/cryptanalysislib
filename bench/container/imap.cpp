#include <benchmark/benchmark.h>
#include "container/imap.h"

static void BM_imap(benchmark::State &state) {
	imap_tree_t tree(state.range(0));
	uint32_t *slot;
	for (auto _: state) {
		for (int64_t i = 0; i < state.range(0); ++i) {
			slot = tree.assign(i);
			(void)slot;
		}
		benchmark::ClobberMemory();

		for (int64_t i = 0; i < state.range(0); ++i) {
			slot = tree.lookup(i);
			(void)slot;
		}
		benchmark::ClobberMemory();

		for (int64_t i = 0; i < state.range(0); ++i) {
			tree.remove(i);
		}
		benchmark::ClobberMemory();
	}
}

BENCHMARK(BM_imap)->RangeMultiplier(2)->Range(32, 1u<<10);
BENCHMARK_MAIN();
