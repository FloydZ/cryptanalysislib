#include <benchmark/benchmark.h>
#include "container/hashmap.h"

using V = uint64_t;
using K = uint64_t;

constexpr size_t list_size =  1u<<16;
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
        benchmark::DoNotOptimize(t += 1);
	}
}

static void BM_hopscotch_map(benchmark::State &state) {
	hopscotch_map<K, V> map;
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
        benchmark::DoNotOptimize(t += 1);
	}
}


/// lookups (hits and misses) of random keys, i.e. occupied neighborhoods
static void BM_hopscotch_map_find_random(benchmark::State &state) {
	hopscotch_map<K, V> map;
	std::vector<K> keys(2 * list_size);
	for (auto &k : keys) { k = cryptanalysislib::rng<K>(); }
	for (size_t i = 0; i < list_size; ++i) { map[keys[i]] = f(keys[i]); }

	for (auto _: state) {
		size_t t = 0;
		for (const auto k : keys) {
			t += map.count(k);
		}

		benchmark::DoNotOptimize(t += 1);
	}
}

BENCHMARK(BM_std_unordered_map);
BENCHMARK(BM_hopscotch_map);
BENCHMARK(BM_hopscotch_map_find_random);
BENCHMARK_MAIN();
