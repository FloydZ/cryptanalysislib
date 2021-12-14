#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include <helper.h>
#include <value.h>

B63_BASELINE(copy, nn) {
	kAryValue v1{}, v2{};
	B63_SUSPEND {
		v1.data()[0] = 1;
	}

	int32_t res = 0;

	for (; res < nn; res++) {
		v1 = v2;
		v1.data()[0] += 1;
		v2 = v1;
		v2.data()[0] += 1;
	}

	B63_SUSPEND {
		res += 1;
	}

	/* this is to prevent compiler from optimizing res out */
	B63_KEEP(res);
}

/*
 * This is another benchmark, which will be compared to baseline
 */
B63_BENCHMARK(move, nn) {
	kAryValue v1{}, v2{};
	B63_SUSPEND {
		v1.data()[0] = 1;
	}

	int32_t res = 0;

	for (; res < nn; res++) {
		v1 = std::move(v2);
		v1.data()[0] += 1;
		v2 = std::move(v1);
		v2.data()[0] += 1;
	}

	B63_SUSPEND {
		res += 1;
	}

	/* this is to prevent compiler from optimizing res out */
	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	return 0;
}