#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"
#include "helper.h"
#include <cstdint>


/// custom abs function
#define cabs(y, x) tmp = x >> 15u; \
				x ^= tmp; \
				x += tmp & 1u; \
				y = x;


B63_BASELINE(abs, nn) {
	int16_t v1, v2;
	B63_SUSPEND {
		v1 = rand();
		v2 = rand();
	}

	uint64_t res = 0;

	for (; res < nn; res++) {
		v2 = abs(v1);
		v1 = abs(v2);
		v1 += v2;
	}

	B63_SUSPEND {
		res += v1;
	}

	B63_KEEP(res);
}

/*
 * This is another benchmark, which will be compared to baseline
 */
B63_BENCHMARK(cabs, nn) {
	int16_t v1, v2;
	uint16_t tmp;
	B63_SUSPEND {
		v1 = rand();
		v2 = rand();
	}

	uint64_t res = 0;

	for (; res < nn; res++) {
		cabs(v2, v1);
		cabs(v1, v2);
		v1 += v2;
	}

	B63_SUSPEND {
	res += tmp;
	}

	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	return 0;
}
