#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include "container/binary_packed_vector.h"
#include "helper.h"
#include "sort/sort.h"
#include "sort.h"
#include <cstdint>


using namespace std;

constexpr static uint64_t SIZE = 1<<22;

constexpr static uint32_t k_lower = 0;
constexpr static uint32_t k_higher = 16;

using T = uint32_t;
constexpr static T mask = ((T(1) << k_higher) - 1) ^ ((T(1) << k_lower) -1);

void random_data(std::vector<T> &data,
		const uint64_t size) {
	data.resize(SIZE);
	for (uint64_t i = 0; i < size; ++i) {
		data[i] = fastrandombytes_uint64() & mask;
	}
}

// std_sort
B63_BASELINE(Std_Sort, n) {
	vector<T> data;
	B63_SUSPEND {
		random_data(data, n);
	}

	std::sort(data.begin(), data.end(),
          [](const auto &e1, const auto &e2) {
	          return (e1&mask) < (e2&mask);
          }
	);

	assert(std::is_sorted(data.begin(), data.end()));
	B63_KEEP(data[0]);
}

B63_BENCHMARK(std_stable_sort, n) {
	vector<T> data;
	B63_SUSPEND {
		random_data(data, n);
	}

	std::stable_sort(data.begin(), data.end(),
	          [](const auto &e1, const auto &e2) {
		          return (e1&mask) < (e2&mask);
	          }
	);

	assert(std::is_sorted(data.begin(), data.end()));
	B63_KEEP(data[0]);
}

B63_BENCHMARK(SKASort, n) {
	vector<T> data;
	B63_SUSPEND {
		random_data(data, n);
	}

	ska_sort(data.begin(), data.end(),
	         [](auto e) -> decltype(auto) {
		         return e;
	         }
	);

	assert(std::is_sorted(data.begin(), data.end()));
	B63_KEEP(data[0]);
}




int main(int argc, char **argv) {
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	return 0;
}
