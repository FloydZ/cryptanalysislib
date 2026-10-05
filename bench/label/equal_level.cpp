#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include "helper.h"

B63_BASELINE(equal_alllevel, nn) {
	kAryLabel v1{}, v2{};
	uint32_t k_lower=0, k_higher=0;
	B63_SUSPEND {
		for (uint64_t i = 0; i < n; ++i) {
			v1.data()[i] = i;
			v2.data()[i] = i;
		}

		translate_level(&k_lower, &k_higher, -1, __level_translation_array);
	}

	uint64_t res = 0;

	for (; res < nn; res++) {
		res += v1.is_equal(v2, k_lower, k_higher);
	}

	B63_SUSPEND {
		res += 1;
	}

	// this is to prevent compiler from optimizing res out
	B63_KEEP(res);
}

B63_BENCHMARK(equal_level1, nn) {
	kAryLabel v1{}, v2{};
	uint32_t k_lower=0, k_higher=0;

	B63_SUSPEND {
		for (uint64_t i = 0; i < n; ++i) {
			v1.data()[i] = i;
			v2.data()[i] = i;
		}
		translate_level(&k_lower, &k_higher, 0, __level_translation_array);
	}

	uint64_t res = 0;

	for (; res < nn; res++) {
		res += v1.is_equal(v2, k_lower, k_higher);
	}

	B63_SUSPEND {
		res += 1;
	}

	// this is to prevent compiler from optimizing res out
	B63_KEEP(res);
}

B63_BENCHMARK(equal_level2, nn) {
	kAryLabel v1{}, v2{};
	uint32_t k_lower=0, k_higher=0;

	B63_SUSPEND {
		for (uint64_t i = 0; i < n; ++i) {
			v1.data()[i] = i;
			v2.data()[i] = i;
		}

		translate_level(&k_lower, &k_higher, 1, __level_translation_array);
	}

	uint64_t res = 0;

	for (; res < nn; res++) {
		res += v1.is_equal(v2, k_lower, k_higher);
	}

	B63_SUSPEND {
		res += 1;
	}

	// this is to prevent compiler from optimizing res out
	B63_KEEP(res);
}

B63_BENCHMARK(equal_level3, nn) {
	kAryLabel v1{}, v2{};
	uint32_t k_lower=0, k_higher=0;

	B63_SUSPEND {
		for (uint64_t i = 0; i < n; ++i) {
			v1.data()[i] = i;
			v2.data()[i] = i;
		}

		translate_level(&k_lower, &k_higher, 2, __level_translation_array);
	}

	uint64_t res = 0;

	for (; res < nn; res++) {
		res += v1.is_equal(v2, k_lower, k_higher);
	}

	B63_SUSPEND {
		res += 1;
	}

	// this is to prevent compiler from optimizing res out
	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	return 0;
}
