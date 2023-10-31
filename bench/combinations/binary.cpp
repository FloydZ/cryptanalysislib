#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include "helper.h"
#include "combination/chase.h"
#include "list/list.h"

#define __WRITE_BIT(w, spot, value) ((w) = (((w) & ~(m4ri_one << (spot))) | (__M4RI_CONVERT_TO_WORD(value) << (spot))))

constexpr uint64_t w = 3;

B63_BASELINE(Combinations_Chase_Binary, nn) {
	constexpr uint64_t size = bc(n, w);
	constexpr uint32_t element_limbs = (n + 63)/64;

	std::array<std::pair<uint16_t, uint16_t>, size> cL;
	word *list;
	Combinations_Binary_Chase<word, n, w> c;
	B63_SUSPEND {
		list = (word *)calloc(size*element_limbs, 8);
	}

	uint64_t res = 0;
	uint16_t pos1, pos2;
	for (uint64_t j = 0; j < nn; j++) {
		c.reset();
		for (uint32_t i = 0; i < element_limbs; i++) {
			list[i] = 0;
		}
		c.left_step(list, &pos1, &pos2);

		for (uint64_t i = 1; i < size; ++i) {
			memcpy(list + i*element_limbs, list + (i-1)*element_limbs, element_limbs*8);
			c.left_step(list + i*element_limbs, &pos1, &pos2);

			cL[i-1] = std::pair<uint16_t, uint16_t>(pos1, pos2);
			res += 1 + list[i] + cL[i/(j+1)].first;
		}
	}

	B63_SUSPEND { free(list); }
	B63_KEEP(res);
}


int main(int argc, char **argv) {
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	return 0;
}