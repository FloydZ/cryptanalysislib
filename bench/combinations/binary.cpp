#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include "helper.h"
#include "combination/binary.h"
#include "list/list.h"

#define COMBINATIONS_SIZE_LOG 100u
#define COMBINATIONS_SIZE_k   5u
#define COMBINATIONS_SIZE (1u<<COMBINATIONS_SIZE_LOG)

#define __WRITE_BIT(w, spot, value) ((w) = (((w) & ~(m4ri_one << (spot))) | (__M4RI_CONVERT_TO_WORD(value) << (spot))))


B63_BASELINE(Combinations_Chase_Binary, nn) {
	constexpr uint64_t size = (COMBINATIONS_SIZE_LOG / sizeof(word)) + 1;

	word *data, *data2;
	Combinations_Chase_Binary<word> c{COMBINATIONS_SIZE_LOG, COMBINATIONS_SIZE_k, 0};
	B63_SUSPEND {
		data = (word *)malloc(size);
		data2 = (word *)malloc(size);

		c.left_init(data);
		c.left_step(data, true);
	}

	uint64_t res = 0, j, i = 1;
	uint16_t pos1, pos2;
	const uint32_t limbs = size/8;
	for (j = 0; j < nn; j++) {
		while (i != 0) {
			i = c.left_step(data);

			Combinations_Chase_Binary<word>::diff(data, data2, limbs, &pos1, &pos2);
			res += 1 + pos1;
			memcpy(data2, data, size);
		}
	}

	B63_SUSPEND { free(data); free(data2); res += i; }
	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	return 0;
}