#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include <helper.h>
#include <list.h>

B63_BASELINE(Simple, nn) {
	kAryList l{n};
	B63_SUSPEND {
		kAryMatrix m;
		m.gen_identity(n);
		l.generate_base_random(n, m);
	}

	int32_t res = 0;

	l.sort_level(-1, __level_translation_array);

	B63_SUSPEND {
		res += 1;
	}

	B63_KEEP(res);
}

// sort not implemented TODO
//B63_BENCHMARK(Parallel, nn) {
//	kAryList l{n};
//	B63_SUSPEND {
//		kAryMatrix m;
//		m.gen_identity(n);
//		l.generate_base_random(n, m);
//	}
//
//	int32_t res = 0;
//
//	l.sort_parallel(-1, __level_translation_array);
//
//	B63_SUSPEND {
//		res += 1;
//	}
//
//	/* this is to prevent compiler from optimizing res out */
//	B63_KEEP(res);
//}

int main(int argc, char **argv) {
	B63_RUN_WITH("time,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs", argc, argv);
	return 0;
}