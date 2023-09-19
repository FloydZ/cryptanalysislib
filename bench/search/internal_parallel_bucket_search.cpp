#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include "container/binary_packed_vector.h"
#include "search.h"
#include <helper.h>

using namespace std;
constexpr static uint64_t SIZE = 1<<20;

constexpr static uint32_t asize = 64;
constexpr static uint32_t k_lower = 0;
constexpr static uint32_t k_higher = asize-1;

using ContainerT = uint64_t;
constexpr static ContainerT mask = ((ContainerT(1) << k_higher) - 1) ^ ((ContainerT(1) << k_lower) -1);

ContainerT random_data(std::vector<ContainerT> &data, const uint64_t size) {
	data.resize(SIZE);
	for (int i = 0; i < SIZE; ++i) {
		data[i] = fastrandombytes_uint64() & mask;
	}

	std::sort(data.begin(), data.end(),
         [](const auto &e1, const auto &e2) {
             return e1 < e2;
         }
	);

	assert(std::is_sorted(data.begin(), data.end()));
	return fastrandombytes_uint64() % SIZE;
}

// std_sort
B63_BASELINE(lower_bound, nn) {
	ContainerT search;
	vector<ContainerT> data;
	uint64_t pos = 0;
	B63_SUSPEND {
		search = random_data(data, nn);
	}

	for (int i = 0; i < nn; ++i) {
		auto r = std::lower_bound(data.begin(), data.end(), search,
		                          [](const auto &e1, const auto &e2) {
			                          return (e1&mask) < (e2&mask);
		                          }
		);
		pos += std::distance(data.begin(), r);
	}


	B63_KEEP(pos);
}

B63_BENCHMARK(standard_binary_search, nn) {
	ContainerT search;
	vector<ContainerT> data;
	uint64_t pos = 0;
	B63_SUSPEND {
		search = random_data(data, nn);
	}

	for (uint32_t i = 0; i < nn; ++i) {
		pos += standard_binary_search(data.data(), SIZE, search);
	}
	B63_KEEP(pos);
}

B63_BENCHMARK(boundless_binary_search, nn) {
	ContainerT search;
	vector<ContainerT> data;
	uint64_t pos = 0;
	B63_SUSPEND {
		search = random_data(data, nn);
	}
	for (uint32_t i = 0; i < nn; ++i) {
		pos += boundless_binary_search(data.data(), SIZE, search);
	}
	B63_KEEP(pos);
}

B63_BENCHMARK(doubletapped_binary_search, nn) {
	ContainerT search;
	vector<ContainerT> data;
	uint64_t pos = 0;
	B63_SUSPEND {
		search = random_data(data, nn);
	}
	for (uint32_t i = 0; i < nn; ++i) {
		pos += doubletapped_binary_search(data.data(), SIZE, search);
	}
	B63_KEEP(pos);
}

B63_BENCHMARK(monobound_binary_search, nn) {
	ContainerT search;
	vector<ContainerT> data;
	uint64_t pos = 0;
	B63_SUSPEND {
		search = random_data(data, nn);
	}
	for (uint32_t i = 0; i < nn; ++i) {
		pos += monobound_binary_search(data.data(), SIZE, search);
	}
	B63_KEEP(pos);
}

B63_BENCHMARK(tripletapped_binary_search, n) {
	ContainerT search;
	vector<ContainerT> data;
	uint64_t pos = 0;
	B63_SUSPEND {
		search = random_data(data, n);
	}
	for (uint32_t i = 0; i < n; ++i) {
		pos += tripletapped_binary_search(data.data(), SIZE, search);
	}
	B63_KEEP(pos);
}

B63_BENCHMARK(monobound_quaternary_search, nn) {
	ContainerT search;
	vector<ContainerT> data;
	uint64_t pos = 0;
	B63_SUSPEND {
		search = random_data(data, n);
	}
	for (uint32_t i = 0; i < nn; ++i) {
		pos += monobound_quaternary_search(data.data(), SIZE, search);
	}
	B63_KEEP(pos);
}

B63_BENCHMARK(Khuong_bin_search, nn) {
	ContainerT search;
	vector<ContainerT> data;
	uint64_t pos = 0;
	B63_SUSPEND {
		search = random_data(data, nn);
	}
	for (uint32_t i = 0; i < nn; ++i) {
		pos += Khuong_bin_search(data, search);
	}
	B63_KEEP(pos);
}

//B63_BENCHMARK(monobound_interpolated_search, n) {
//	ContainerT search;
//	vector<ContainerT> data;
//	B63_SUSPEND {
//		search = random_data(data, n);
//	}
//
//	uint64_t pos = monobound_interpolated_search(reinterpret_cast<int *>(data.data()), SIZE, search);
//	B63_KEEP(pos);
//}
//
//B63_BENCHMARK(adaptive_binary_search, n) {
//	ContainerT search;
//	vector<ContainerT> data;
//	B63_SUSPEND {
//		search = random_data(data, n);
//	}
//
//	uint64_t pos = adaptive_binary_search(reinterpret_cast<int *>(data.data()), SIZE, search);
//	B63_KEEP(pos);
//}

int main(int argc, char **argv) {
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions", argc, argv);
	return 0;
}
