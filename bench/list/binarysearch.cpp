#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"
#include "helper.h"
#include "list/list.h"
#include "matrix/matrix.h"


using BinaryValue     = BinaryContainer<n>;
using BinaryLabel     = BinaryContainer<n>;
using BinaryMatrix    = FqMatrix<uint64_t, n, n, 2>;
using BinaryElement   = Element_T<BinaryValue, BinaryLabel, BinaryMatrix>;
using BinaryList      = List_T<BinaryElement>;


B63_BASELINE(NotFindBinary, nn) {
	BinaryList l{n};
	BinaryElement e{};
	uint64_t k_lower, k_higher;
	B63_SUSPEND {
		BinaryMatrix B;
		B.random();
		l.random(n, B);

		// make sure we will never find the value
		e.zero();

		translate_level(&k_lower, &k_higher, -1, __level_translation_array);
		l.sort_level(k_lower, k_higher);
	}

	uint64_t res = 0;
	for (uint64_t i = 0; i < nn; i++) {
		res += l.search_level_binary(e, k_lower, k_higher);
	}
	B63_KEEP(res);
}

B63_BENCHMARK(NotFindBinaryCustom, nn) {
	BinaryList l{n};
	BinaryElement e{};
	uint64_t k_lower, k_higher;
	B63_SUSPEND {
		BinaryMatrix B;
		B.random();
		l.random(n, B);

		translate_level(&k_lower, &k_higher, -1, __level_translation_array);
		l.sort_level(k_lower, k_higher);

		e.zero();
	}

	uint64_t res = 0;
	for (uint64_t i = 0; i < nn; i++) {
		res += l.search_level_binary_custom(e, k_lower, k_higher);
	}
	B63_KEEP(res);
}

B63_BENCHMARK(FindEnd, nn) {
	BinaryList l{n};
	BinaryElement e{};
	uint64_t k_lower, k_higher;

	B63_SUSPEND {
		BinaryMatrix B;
		B.random();
		l.random(n, B);

		// make sure we will find the element at the end
		e.zero();
		l.append(e);
		translate_level(&k_lower, &k_higher, -1, __level_translation_array);
		l.sort_level(k_lower, k_higher);
	}

	uint64_t res = 0;
	for (uint64_t i = 0; i < nn; i++) {
		res += l.search_level_binary(e, k_lower, k_higher);
	}
	B63_KEEP(res);
}

B63_BENCHMARK(FindBegin, nn) {
	BinaryList l{n};
	BinaryElement e{};
	uint64_t k_lower, k_higher;

	B63_SUSPEND {
		BinaryMatrix B;
		B.random();
		l.random(n, B);

		// make sure we will find the element at the beginning
		e.zero();
		l.append(e);

		translate_level(&k_lower, &k_higher, -1, __level_translation_array);
		l.sort_level(k_lower, k_higher);
	}

	uint64_t res = 0;
	for (uint64_t i = 0; i < nn; i++) {
		res += l.search_level_binary(e, k_lower, k_higher);
	}
	B63_KEEP(res);
}

B63_BENCHMARK(FindEndCustom, nn) {
	BinaryList l{n};
	BinaryElement e{};
	uint64_t k_lower, k_higher;

	B63_SUSPEND {
		BinaryMatrix B;
		B.random();
		l.random(n, B);

		// make sure we will find the element at the end
		e.zero();
		l.append(e);
		translate_level(&k_lower, &k_higher, -1, __level_translation_array);
		l.sort_level(k_lower, k_higher);
	}

	uint64_t res = 0;
	for (uint64_t i = 0; i < nn; i++) {
		res += l.search_level_binary_custom(e, k_lower, k_higher);
	}
	B63_KEEP(res);
}

B63_BENCHMARK(FindBeginCustom, nn) {
	BinaryList l{n};
	BinaryElement e{};
	uint64_t k_lower, k_higher;

	B63_SUSPEND {
		BinaryMatrix B;
		B.random();
		l.random(n, B);

		// make sure we will find the element at the beginning
		e.zero();
		l.append(e);

		translate_level(&k_lower, &k_higher, -1, __level_translation_array);
		l.sort_level(k_lower, k_higher);
	}

	uint64_t res = 0;
	for (uint64_t i = 0; i < nn; i++) {
		res += l.search_level_binary_custom(e, k_lower, k_higher);
	}
	B63_KEEP(res);
}

int main(int argc, char **argv) {
	B63_RUN_WITH("time,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs", argc, argv);
	return 0;
}
