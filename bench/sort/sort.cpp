#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include "container/binary_packed_vector.h"
#include "helper.h"
#include "sort/sort.h"


constexpr static uint64_t sssize = 63;
constexpr static uint64_t k_lower = 30;
constexpr static uint64_t k_higher = sssize;
using T = BinaryContainer<sssize>;

constexpr uint64_t SIZE_LIST = 1ull << 20;

// std_sort
B63_BASELINE(Std_Sort, nn) {
	std::vector<T> data;

	B63_SUSPEND {
		data.resize(SIZE_LIST);
		for (size_t i = 0; i < nn; ++i) {
			data[i].random();
		}
	}

	std::sort(data.begin(), data.end(), [](const T &a, const T &b) {
		return a.is_greater<k_lower, k_higher>(b);
	});
	B63_KEEP(data[0].data()[0]);
}

B63_BENCHMARK(SKASort, nn) {
	std::vector<T> data;
	B63_SUSPEND {
		data.resize(SIZE_LIST);
		for (size_t i = 0; i < nn; ++i) {
			data[i].random();
		}
	}

	ska_sort(data.begin(), data.end(), [](const T &a) {
		constexpr uint64_t mask = (1ul << (k_higher - k_lower)) - 1ull;
		return (a.data()[0] >> k_lower) & mask;
	});
	B63_KEEP(data[0].data()[0]);
}

B63_BENCHMARK(VergeSort, nn) {
	std::vector<T> data;
	B63_SUSPEND {
		data.resize(SIZE_LIST);
		for (size_t i = 0; i < nn; ++i) {
			data[i].random();
		}
	}

	vergesort::vergesort(data.begin(), data.end(), [](const T &in1, const T &in2) {
	  return in1.is_greater<k_lower, k_higher>(in2);
	});
	B63_KEEP(data[0].data()[0]);
}

B63_BENCHMARK(TimSort, nn) {
	std::vector<T> data;
	B63_SUSPEND {
		data.resize(SIZE_LIST);
		for (size_t i = 0; i < nn; ++i) {
			data[i].random();
		}
	}

	gfx::timsort(data.begin(), data.end(), [](const T &in1, const T &in2) {
		return in1.is_greater<k_lower, k_higher>(in2);
	}, {});
	B63_KEEP(data[0].data()[0]);
}


int main(int argc, char **argv) {
	srand(time(NULL));
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions", argc, argv);
	return 0;
}
