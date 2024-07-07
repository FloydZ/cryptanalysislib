#include "b63.h"
#include "counters/perf_events.h"

#include "matrix/matrix.h"
#include <vector>

using T = uint64_t;
constexpr uint32_t nrows = 256;
constexpr uint32_t ncols = 1024;
using M = FqMatrix<T, nrows, ncols, 5, true>;
using MT = FqMatrix<T, ncols, nrows, 5, true>;

B63_BASELINE(permute_with_transpose, nn) {
	M m = M{};
	MT mt = MT{};
	Permutation P(ncols);
	uint32_t rank;
	B63_SUSPEND {
		m.random();
	}

	uint64_t keep = 0;
	for (uint64_t i = 0; i < nn; i++) {
		m.permute_cols(mt, P);
		keep += m.get(i % nrows, 10);
		keep += P.values[i % ncols];
	}
	B63_KEEP(keep);
}

B63_BENCHMARK(permute_without, nn) {
	M m = M{};
	Permutation P(ncols);
	B63_SUSPEND {
		m.random();
	}

	uint64_t keep = 0;
	for (uint64_t i = 0; i < nn; i++) {
		m.permute_cols(P);
		keep += m.get(i % nrows, 10);
		keep += P.values[i % ncols];
	}
	B63_KEEP(keep);
}

int main(int argc, char **argv) {
	//B63_RUN_WITH("time,lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	B63_RUN_WITH("time", argc, argv);
	return 0;
}
