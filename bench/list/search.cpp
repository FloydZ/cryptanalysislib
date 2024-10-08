#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"


#include "helper.h"
#include "list/list.h"

B63_BASELINE(NotFind, nn) {
	kAryList l{n};
	kAryElement e{};
	uint64_t k_lower=0, k_higher=0;
	B63_SUSPEND {
		kAryMatrix m;
		m.identity();
		l.random(n, m);

		// make sure we will never find the value
		e.zero();

		translate_level(&k_lower, &k_higher, -1, __level_translation_array);
	}

	int32_t res = 0;

	for (uint64_t i = 0; i < nn; i++) {
		res += l.search_level(e, k_lower, k_higher);
	}

	B63_SUSPEND {
		res += 1;
	}

	B63_KEEP(res);
}

B63_BENCHMARK(FindEnd, nn) {
	kAryList l{n};
	kAryElement e{};
	uint64_t k_lower=0, k_higher=0;

	B63_SUSPEND {
		kAryMatrix m;
		m.identity(1);
		l.random(n, m);

		// make sure we will find the element at the end
		e.zero();
		l.append(e);
		translate_level(&k_lower, &k_higher, -1, __level_translation_array);
	}

	int32_t res = 0;

	for (uint64_t i = 0; i < nn; i++) {
		res += l.search_level(e, k_lower, k_higher);
	}

	B63_SUSPEND {
		res += 1;
	}

	B63_KEEP(res);
}

B63_BENCHMARK(FindBegin, nn) {
	kAryList l{n};
	kAryElement e{};
	uint64_t k_lower=0, k_higher=0;

	B63_SUSPEND {
		// make sure we will find the element at the beginning
		e.zero();
		l.append(e);

		kAryMatrix m;
		m.identity(1);
		l.random(n, m);

		translate_level(&k_lower, &k_higher, -1, __level_translation_array);

	}

	int32_t res = 0;

	for (uint64_t i = 0; i < nn; i++) {
		res += l.search_level(e, k_lower, k_higher);
	}

	B63_SUSPEND {
		res += 1;
	}

	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("time,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs", argc, argv);
	return 0;
}
