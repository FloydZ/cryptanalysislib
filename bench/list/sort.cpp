#include "b63.h"
#include "counters/perf_events.h"

// #include "../bench_config.h"

#include <helper.h>
#include "container/kAry_type.h"
#include "list/list.h"
#include "tree.h"

constexpr uint32_t n    = 16ul;
constexpr uint32_t q    = (1ul << n);

using T 			= uint64_t;
//using Value     	= kAryContainer_T<T, n, 2>;
using Value     	= FqPackedVector<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;


B63_BASELINE(Base, nn) {
	List l{n};
	B63_SUSPEND {
		Matrix m;
		m.identity(n);
		l.random(nn, m);
	}

	int32_t res = 0;

	for (uint64_t i = 0; i < nn; i++) {
		l.sort_level(0, n);
		B63_SUSPEND {
			res += l[0].label.value();
		}
	}

	B63_KEEP(res);
}

B63_BENCHMARK(Constexpr, nn) {
	List l{n};
	B63_SUSPEND {
		Matrix m;
		m.identity(n);
		l.random(nn, m);
	}

	int32_t res = 0;

	for (uint64_t i = 0; i < nn; i++) {
		l.template sort_level<0, n>();
		B63_SUSPEND {
			res += l[0].label.value();
		}
	}

	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("time,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs", argc, argv);
	return 0;
}
