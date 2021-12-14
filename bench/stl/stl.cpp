#include <vector>
#include <array>

#include "../bench_config.h"
#include "b63.h"
#include "counters/perf_events.h"

#include <helper.h>
#include <container.h>
#include <list.h>
#include <sort.h>

constexpr uint64_t lsize = (1<<20);

constexpr uint32_t l = 20;
using ContainerA        = BinaryContainer<k + l>;
using ContainerB        = BinaryContainer<n - 20>;
using DecodingValue     = Value_T<BinaryContainer<k + l>>;
using DecodingLabel     = Label_T<BinaryContainer<n - 20>>;
using DecodingMatrix    = mzd_t *;
using DecodingElement   = Element_T<DecodingValue, DecodingLabel, DecodingMatrix>;
using DecodingList      = List_T<DecodingElement>;

// std_sort
B63_BASELINE(ListConstructor, nn) {
	uint32_t res = 0;
	for (int i = 0; i < nn; ++i) {
		DecodingList L(nn*lsize);
		L.set_load(nn*lsize);
		res += L[0].get_label_container_ptr()[0];
	}

	B63_KEEP(res);
}

B63_BENCHMARK(ListMalloc, nn) {
	uint32_t res = 0;
	for (int i = 0; i < nn; ++i) {
		DecodingElement *L = (DecodingElement *) malloc(nn*lsize * sizeof(DecodingElement));
		res += L[0].get_label_container_ptr()[0];
		free(L);
	}

	B63_KEEP(res);
}

B63_BENCHMARK(CopyClass, nn) {
	DecodingList L1{nn*lsize};
	DecodingList L2{nn*lsize};
	uint32_t res = 0;

	B63_SUSPEND {
		L1.set_load(nn*lsize);
		L2.set_load(nn*lsize);
	}

	for (int i = 0; i < nn; ++i) {
		L1 = L2;
		res += L1[0].get_label_container_ptr()[0];
		L1[0].get_label_container_ptr()[0] += rand();
	}

	B63_KEEP(res);
}

B63_BENCHMARK(CopyMalloc, nn) {
	DecodingElement *L1 = (DecodingElement *) malloc(nn*lsize * sizeof(DecodingElement));
	DecodingElement *L2 = (DecodingElement *) malloc(nn*lsize * sizeof(DecodingElement));

	uint32_t res = 0;
	for (int i = 0; i < nn; ++i) {
		memcpy(L1, L2, sizeof(DecodingElement)*nn*lsize);
		res += L1[0].get_label_container_ptr()[0];
		L1[0].get_label_container_ptr()[0] += rand();
	}

	free(L1);
	free(L2);

	B63_KEEP(res);
}

int main(int argc, char **argv) {
	std::cout << "Size Container Value: " << sizeof(ContainerA) << "\n";
	std::cout << "Size Container Label: " << sizeof(ContainerB) << "\n";
	std::cout << "Size Value:           " << sizeof(DecodingValue) << "\n";
	std::cout << "Size Label:           " << sizeof(DecodingLabel) << "\n";
	std::cout << "Size Matrix:          " << sizeof(DecodingMatrix) << "\n";
	std::cout << "Size Element:         " << sizeof(DecodingElement) << "\n";
	std::cout << "Size List:            " << sizeof(DecodingList) << "\n";

	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions", argc, argv);
	return 0;
}