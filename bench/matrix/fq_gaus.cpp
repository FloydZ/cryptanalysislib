#include "b63.h"
#include "counters/perf_events.h"

#include "matrix/matrix.h"
#include <vector>

using T = uint64_t;
constexpr uint32_t nrows = 256;
constexpr uint32_t ncols = 1024;
using M = FqMatrix<T, nrows, ncols, 5, true>;
using MT = FqMatrix<T, ncols, nrows, 5, true>;

constexpr uint32_t l = 25;
constexpr uint32_t c = 20;

B63_BASELINE(markov_gaus, nn) {
	M m = M{};
	Permutation P(ncols);
	uint32_t rank;
	B63_SUSPEND {
		while (true) {
			m.random();
			rank = m.gaus(nrows - l);
			rank = m.fix_gaus(P, rank, nrows - l);
			if (rank >= nrows - l) { break; }
		}
	}

	uint64_t keep = 0;
	for (uint64_t i = 0; i < nn; i++) {
		keep += m.markov_gaus<c, nrows - l>(P);
		keep += m.get(10, 10);
	}
	B63_KEEP(keep);
}

B63_BENCHMARK(gaus, nn) {
	M m = M{};
	MT mt = MT{};
	Permutation P(ncols);
	B63_SUSPEND {
		m.random();
	}

	uint64_t keep = 0;
	for (uint64_t i = 0; i < nn; i++) {
		m.permute_cols(mt, P);
		uint32_t rank2 = m.gaus();
		rank2 = m.fix_gaus(P, rank2, nrows - l);
		keep += rank2;
		keep += m.get(10, 10);
	}
	B63_KEEP(keep);
}


B63_BENCHMARK(m4ri, nn) {
	M m = M{};
	MT mt = MT{};
	Permutation P(ncols);
	B63_SUSPEND {
		m.random();
	}

	uint64_t keep = 0;
	for (uint64_t i = 0; i < nn; i++) {
		m.permute_cols(mt, P);
		uint32_t rank2 = m.template m4ri<2>(nrows - l);
		keep += rank2;
		keep += m.get(10, 10);
	}
	B63_KEEP(keep);
}
int main(int argc, char **argv) {
	//B63_RUN_WITH("time,lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	B63_RUN_WITH("time", argc, argv);
	return 0;
}
