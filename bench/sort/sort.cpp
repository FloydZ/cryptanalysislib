#include "b63.h"
#include "counters/perf_events.h"

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

// std_sort
B63_BASELINE(Std_Sort, nn) {
	std::vector<ContainerT> data;

	B63_SUSPEND {
		data.resize(SIZE_LIST);
		for (int i = 0; i < nn; ++i) {
			data[i].random();
		}
	}

	std::sort(data.begin(), data.end(), [](const ContainerT &a, const ContainerT &b){
		return a.is_greater<k_lower, k_higher>(b);
	});
	B63_KEEP(data[0].data()[0]);
}


B63_BENCHMARK(crumsort, nn) {
	std::vector<ContainerT> data;
	B63_SUSPEND {
		data.resize(SIZE_LIST);
		for (int i = 0; i < nn; ++i) {
			data[i].random();
		}
	}
	crumsort<ContainerT>(data.data(), SIZE_LIST, [](const ContainerT *a, const ContainerT *b){
		return a->is_greater<k_lower, k_higher>(*b);
	});

	B63_KEEP(data[0].data()[0]);
}


//B63_BENCHMARK(quadsort, nn) {
//	std::vector<ContainerT> data;
//	B63_SUSPEND {
//		data.resize(SIZE_LIST);
//		for (int i = 0; i < nn; ++i) {
//			data[i].random();
//		}
//	}
//
//	quadsort<ContainerT>(data.data(), SIZE_LIST, [](const ContainerT *a, const ContainerT *b){
//		return a->is_greater<k_lower, k_higher>(*b);
//	});
//	B63_KEEP(data[0].data()[0]);
//}


int main(int argc, char **argv) {
	srand(time(NULL));
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions", argc, argv);
	return 0;
}