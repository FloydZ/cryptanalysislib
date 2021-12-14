#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include <helper.h>
#include <container.h>
#include <search.h>

using ContainerT = uint32_t;

// Size of the list to search
constexpr static uint64_t SIZE = 1<<20;
constexpr static uint64_t ssize = sizeof(ContainerT)*8; // size of the type in bits
// Where to search
constexpr static uint32_t k_lower = 0;
constexpr static uint32_t k_higher = 31;
constexpr static ContainerT mask = ((ContainerT(1) << k_higher) - 1) ^ ((ContainerT(1) << k_lower) -1);

void random_data(std::vector<ContainerT> &data, uint64_t &pos) {
	data.resize(SIZE);
	for (uint64_t i = 0; i < SIZE; ++i) {
		data[i] = fastrandombytes_uint64() & mask;
	}

	std::sort(data.begin(), data.end(),
	          [](const auto &e1, const auto &e2) {
		          return (e1) < (e2);
	          }
	);

	assert(std::is_sorted(data.begin(), data.end()));
	pos = fastrandombytes_uint64() % SIZE;
}

// std_sort
B63_BASELINE(Std_lowerbound, nn) {
	std::vector<ContainerT> data;
	uint64_t search, errors = 0;

	B63_SUSPEND {
		random_data(data, search);
	}

	for (uint64_t i = 0; i < nn; i++) {
		search = rand() % SIZE;
		auto v = std::lower_bound(data.begin(), data.end(), data[search],
								  [](const ContainerT e1, const ContainerT e2) {
			return (e1&mask) < (e2&mask);
		});
		//errors += std::distance(data.begin(), v) == search;
	}

	B63_KEEP(search);
}

B63_BENCHMARK(lower_bound_interpolation_search2, nn) {
	std::vector<ContainerT> data;
	uint64_t search, errors = 0;

	B63_SUSPEND {
		random_data(data, search);
	}

	for (uint64_t i = 0; i < nn; i++) {
		search = rand() % SIZE;
		auto v = lower_bound_interpolation_search2(data.begin(), data.end(), data[search],
												   [](const ContainerT e1) -> ContainerT { return e1&mask; }
		);
		//errors += std::distance(data.begin(), v) == search;
	}

	B63_KEEP(search);
}


B63_BENCHMARK(lower_bound_monobound_binary_search, nn) {
	std::vector<ContainerT> data;
	uint64_t search, errors = 0;

	B63_SUSPEND {
		random_data(data, search);
	}

	for (uint64_t i = 0; i < nn; i++) {
		search = rand() % SIZE;
		auto v = lower_bound_monobound_binary_search(data.begin(), data.end(), data[search],
		                                           [](const ContainerT e1) -> ContainerT { return e1&mask; }
		);
		//errors += std::distance(data.begin(), v) == search;
	}

	B63_KEEP(search);
}


int main(int argc, char **argv) {
	srand(time(NULL));
	B63_RUN_WITH("lpe:cycles,lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cache-references,lpe:instructions", argc, argv);
	return 0;
}