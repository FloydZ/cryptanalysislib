#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include "helper.h"
#include "combination/ternary.h"

#define COMBINATIONS_SIZE_LOG 20u
#define COMBINATIONS_SIZE (1u<<COMBINATIONS_SIZE_LOG)

B63_BASELINE(Combinations_Chase, nn) {
	uint8_t *data;
	Combinations_Chase_Ternary<uint8_t> c{COMBINATIONS_SIZE_LOG, COMBINATIONS_SIZE_LOG/3, COMBINATIONS_SIZE_LOG/3};
	B63_SUSPEND {
		data = (uint8_t *)malloc(COMBINATIONS_SIZE_LOG * sizeof(uint8_t));
		c.left_init(data);
	}

	int32_t res = 0, i;

	for (; (res < nn) && (i != 0); res++) {
		i = c.left_step(data);
		res += i;
	}

	B63_SUSPEND { free(data); res += i; }
	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	return 0;
}