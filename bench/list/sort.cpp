#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include <helper.h>
#include "list/list.h"

B63_BASELINE(Simple, nn) {
	kAryList l{n};
	B63_SUSPEND {
		kAryMatrix m;
		m.identity(n);
		l.random(n, m);
	}

	int32_t res = 0;

	for (uint64_t i = 0; i < nn; i++) {
		l.sort_level(-1, __level_translation_array);
		B63_SUSPEND {
			res += 1;
		}
	}

	B63_KEEP(res);
}


int main(int argc, char **argv) {
	B63_RUN_WITH("time,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs", argc, argv);
	return 0;
}
