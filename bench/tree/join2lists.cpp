#include "b63.h"
#include "counters/perf_events.h"

// #include "../bench_config.h"

#include <helper.h>
#include "container/kAry_type.h"
#include "list/list.h"
#include "tree.h"
#include "algorithm/random_index.h"

constexpr uint32_t n    = 16ul;
constexpr uint32_t q    = (1ul << n);

using T 			= uint64_t;
//using Value     	= kAryContainer_T<T, n, 2>;
using Value     	= BinaryVector<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;

constexpr size_t baselist_size = 10*sum_bc(n/2, n/4);
List out{baselist_size}, l1{baselist_size}, l2{baselist_size};
Matrix A;

B63_BASELINE(Base, nn) {
	Label target; target.zero();
	B63_SUSPEND {
		std::vector<uint32_t> weights(n/2);
		generate_random_indices(weights, n);
		for (uint32_t i = 0; i < n/2; ++i) {
			Label::add(target, target, A[0][weights[i]]);
		}
	}

	int32_t res = 0;
	const uint32_t k_lower = 0, k_higher = 8;
    Tree t{1, A, 0};

	for (uint64_t i = 0; i < nn; i++) {
		out.set_load(0);
		t.join2lists(out, l1, l2, target, k_lower, k_higher, true);
		B63_SUSPEND {
			res += out[0].label.value();
		}
	}

	B63_KEEP(res);
}

B63_BENCHMARK(Constexpr, nn) {
	Label target; target.zero();
	B63_SUSPEND {
		std::vector<uint32_t> weights(n/2);
		generate_random_indices(weights, n);
		for (uint32_t i = 0; i < n/2; ++i) {
			Label::add(target, target, A[0][weights[i]]);
		}
	}

	int32_t res = 0;
	for (uint64_t i = 0; i < nn; i++) {
		out.set_load(0);
		Tree::template join2lists<0, 8>(out, l1, l2, target, true);
		B63_SUSPEND {
			res += out[0].label.value();
		}
	}

	B63_KEEP(res);
}

B63_BENCHMARK(Constexpr_on_iT_v2, nn) {
	Label target; target.zero();
	B63_SUSPEND {
		std::vector<uint32_t> weights(n/2);
		generate_random_indices(weights, n);
		for (uint32_t i = 0; i < n/2; ++i) {
			Label::add(target, target, A[0][weights[i]]);
		}

		l2.sort_level<0, 8>();
	}

	int32_t res = 0;
	for (uint64_t i = 0; i < nn; i++) {
		out.set_load(0);
		Tree::template join2lists_on_iT_v2<0, 8>(out, l1, l2, target);
		B63_SUSPEND {
			res += out[0].label.value();
		}
	}

	B63_KEEP(res);
}
int main(int argc, char **argv) {
A.random();
	using Enumerator = MaxBinaryRandomEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);
	B63_RUN_WITH("time,lpe:cycles,lpe:instructions,lpe:ref-cycles,lpe:context-switches,lpe:cs", argc, argv);
	return 0;
}
