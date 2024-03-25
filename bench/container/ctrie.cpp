#include <benchmark/benchmark.h>

#include "container/ctrie.h"
#include <unordered_map>

using V = uint64_t;
using K = uint64_t;
using CTrie = CacheTrie<K, V>;

constexpr size_t list_size =  1u<<20;
constexpr V f(V i) {
	return i*i/2;
}

static void BM_std_unordered_map(benchmark::State &state) {
	std::unordered_map<K, V> map;
	for (auto _: state) {
		for (size_t i = 0; i < list_size; ++i) {
			map[i] = f(i);
		}
		benchmark::ClobberMemory();

		V t = 0;
		for (size_t i = 0; i < list_size; ++i) {
			t += map.at(i);
		}
		benchmark::ClobberMemory();
	}
}

static void BM_ctrie(benchmark::State &state) {
	CTrie c{};
	for (auto _: state) {
		for (size_t i = 0; i < list_size; ++i) {
			 c.insert(i, f(i));
		}
		benchmark::ClobberMemory();

		V t = 0;
		for (size_t i = 0; i < list_size; ++i) {
			t += c.lookup(i);
		}
		benchmark::ClobberMemory();
	}
}


// BENCHMARK(BM_std_unordered_map);
BENCHMARK(BM_ctrie);
BENCHMARK_MAIN();
