#include "b63.h"
#include "counters/perf_events.h"

#include "helper.h"
#include "combination/chase.h"

using T = uint64_t;
constexpr uint32_t n = 256;
constexpr uint32_t p = 2;
Combinations_Binary_Chase<T, n, p> ch{};

constexpr size_t list_size = bc(n, p) - 1;
constexpr uint32_t element_limbs = (n + 63) / 64;

B63_BASELINE(Combinations_Binary_Chase, nn) {
	uint64_t w1[element_limbs+1] = {0}, w2[element_limbs];
	uint64_t res = 0;

	uint16_t pos1, pos2;
	for (; res < nn; res++) {
		for (size_t i = 0; i < list_size - 1; ++i) {
			// memcpy(w1, w2, element_limbs * 8);
			ch.left_step(w1, &pos1, &pos2);
			res += pos1 + pos2;
			B63_KEEP(res);
		}

		ch.reset();
	}

	B63_KEEP(res);
}

B63_BENCHMARK(add_level1, nn) {
	uint64_t res = 0;
	chase<n, p> c{};

	for (; res < nn; res++) {
		c.enumerate([&](uint16_t p1, uint16_t p2) __attribute__((force_inline)){
			res += p1 + p2;
			B63_KEEP(res);
		});
	}

	B63_KEEP(res);
}


int main(int argc, char **argv) {
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	return 0;
}
