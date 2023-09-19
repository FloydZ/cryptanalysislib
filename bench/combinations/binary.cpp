#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include "helper.h"
#include "combinations.h"
#include "list/list.h"

#define COMBINATIONS_SIZE_LOG 100u
#define COMBINATIONS_SIZE_k   5u
#define COMBINATIONS_SIZE (1u<<COMBINATIONS_SIZE_LOG)

#define __WRITE_BIT(w, spot, value) ((w) = (((w) & ~(m4ri_one << (spot))) | (__M4RI_CONVERT_TO_WORD(value) << (spot))))


B63_BASELINE(Combinations_Chase, nn) {
	constexpr uint64_t size = COMBINATIONS_SIZE_LOG * sizeof(uint8_t);

	uint8_t *data, *data2;
	Combinations_Chase<uint8_t> c{COMBINATIONS_SIZE_LOG, COMBINATIONS_SIZE_k, 0};
	B63_SUSPEND {
		data = (uint8_t *)malloc(size);
		data2 = (uint8_t *)malloc(size);

		c.left_init(data);
		c.left_step(data, true);
	}

	uint64_t res = 0, j, i = 1;
	uint32_t pos1, pos2;;
	const uint32_t limbs = COMBINATIONS_SIZE_LOG;

	for (j = 0; j < nn; j++) {
		while (i != 0) {
			i = c.left_step(data);

			Combinations_Chase<uint8_t>::diff(data, data2, limbs, &pos1, &pos2);
			res += 1 + pos1;
			memcpy(data2, data, size);
		}
	}

	B63_SUSPEND { free(data); free(data2); res += i; }
	B63_KEEP(res);
}

B63_BENCHMARK(Combinations_Chase_Binary, nn) {
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

B63_BENCHMARK(Combinations_Chase_M4RI, nn) {
	mzd_t *data, *data2;
	Combinations_Chase_M4RI c{COMBINATIONS_SIZE_LOG, COMBINATIONS_SIZE_k, 0};
	B63_SUSPEND {
		data = mzd_init(1, COMBINATIONS_SIZE_LOG);
		data2 = mzd_init(1, COMBINATIONS_SIZE_LOG);

		c.left_init(data);
		c.left_step(data, true);
	}

	uint64_t res = 0, j, i = 1;
	uint16_t pos1, pos2;;
	const uint32_t limbs = data->width;

	for (j = 0; j < nn; j++) {
		while (i != 0) {
			i = c.left_step(data);

			Combinations_Chase_M4RI::diff(data, data2, limbs, &pos1, &pos2);

			res += 1 + pos1;
			mzd_copy(data2, data);
		}
	}

	B63_SUSPEND { mzd_free(data); mzd_free(data2); res += i; }
	B63_KEEP(res);
}

B63_BENCHMARK(Combinations_Chase2, nn) {
	Combinations_Chase2<BinaryContainer<COMBINATIONS_SIZE_LOG>> cc{COMBINATIONS_SIZE_LOG, COMBINATIONS_SIZE_k, 0};
	BinaryContainer<COMBINATIONS_SIZE_LOG> container, container2;
	container.one();

	B63_SUSPEND {
		for (int j = COMBINATIONS_SIZE_LOG-COMBINATIONS_SIZE_k; j < COMBINATIONS_SIZE_LOG; ++j) {
			container[j] = true;
		}
	}

	uint64_t res = 0, j, i = 1, pos1, pos2;

	for (j = 0; j < nn; j++) {
		while (i != 0) {
			i = cc.next(container, &pos1, &pos2);
			container2 = container;
			res += 1 + pos1;
		}
	}

	B63_SUSPEND { res += i; }
	B63_KEEP(res);
}

B63_BENCHMARK(Combinations_ChaseBinary2, nn) {
	constexpr uint64_t size = COMBINATIONS_SIZE_LOG * sizeof(word);
	word *data, *data2;

	B63_SUSPEND {
		data = (word *)malloc(size);
		data2 = (word *)malloc(size);
	}

	Combinations_Chase_Binary2 c{COMBINATIONS_SIZE_LOG, COMBINATIONS_SIZE_k, 0};

	for (int j = COMBINATIONS_SIZE_LOG-COMBINATIONS_SIZE_k; j < COMBINATIONS_SIZE_LOG; ++j) {
		__WRITE_BIT(data[j/m4ri_radix], j%m4ri_radix, 1);
	}


	uint64_t res = 0, j, i = 1;
	uint64_t pos1;

	for (j = 0; j < nn; j++) {
		while (i != 0) {
			i = c.next(data, &pos1);

			memcpy(data2, data, size);
			res += 1 + pos1;
		}
	}

	B63_SUSPEND { free(data); free(data2); res += i; }
	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("lpe:branches,lpe:branch-misses,lpe:bus-cycles,lpe:cache-misses,lpe:cache-references,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs,lpe:major-faults,lpe:minor-faults,lpe:page-faults,lpe:L1-dcache-load-misses,lpe:L1-dcache-loads,lpe:L1-dcache-prefetches,lpe:L1-dcache-store-misses,lpe:L1-dcache-stores,lpe:L1-icache-load-misses,lpe:L1-icache-loads,lpe:LLC-load-misses,lpe:LLC-loads,lpe:LLC-store-misses,lpe:LLC-stores,lpe:branch-load-misses,lpe:branch-loads,lpe:iTLB-load-misses,lpe:iTLB-loads,lpe:dTLB-load-misses,lpe:dTLB-loads,lpe:dTLB-store-misses,lpe:dTLB-stores", argc, argv);
	return 0;
}