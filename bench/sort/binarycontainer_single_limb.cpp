#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include <helper.h>
#include <container.h>
#include <sort.h>

constexpr static uint64_t sssize = 63;
constexpr static uint64_t k_lower = 62;
constexpr static uint64_t k_higher = sssize;
using ContainerT = BinaryContainer<sssize>;

using namespace  std;

// std_sort
B63_BASELINE(Std_Sort, nn) {
	vector<ContainerT> data;

	B63_SUSPEND {
		data.resize(n);
		for (int i = 0; i < nn; ++i) {
			data[i].random();
		}
	}
	Std_Sort_Binary_Container<ContainerT>::sort(data, k_lower, k_higher);
	B63_KEEP(data[0].data()[0]);
}

B63_BENCHMARK(Bucket_Sort, nn) {
	vector<ContainerT> data;
	B63_SUSPEND {
		data.resize(n);
		for (int i = 0; i < nn; ++i) {
			data[i].random();
		}
	}

	Bucket_Sort_Binary_Container_Single_Limb<ContainerT>::sort(data, k_lower, k_higher);
	B63_KEEP(data[0].data()[0]);
}


B63_BENCHMARK(Shell_Sort, nn) {
	vector<ContainerT> data;
	B63_SUSPEND {
		data.resize(n);
		for (int i = 0; i < nn; ++i) {
			data[i].random();
		}
	}

	Shell_Sort_Binary_Container_Single_Limb<ContainerT>::sort(data, k_lower, k_higher);
	B63_KEEP(data[0].data()[0]);
}


int main(int argc, char **argv) {
	srand(time(NULL));
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions", argc, argv);
	return 0;
}
