#include <benchmark/benchmark.h>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include "container/binary_packed_vector.h"
#include "container/fq_packed_vector.h"
#include "container/fq_vector.h"
#include "sort/sort.h"
#include "list/list.h"
#include "tree.h"
#include "../tests/search/common.h"

constexpr uint64_t n    = 32ul;
constexpr uint64_t q    = (1ul << n);


using T 			= uint64_t;
//using Value     	= kAryContainer_T<T, n, 2>;
using Value     	= BinaryVector<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;
using Enumerator 	= BinaryLexicographicEnumerator<List, n/2, n/4>;
constexpr size_t LS = Enumerator::size();

constexpr uint32_t k_lower = 0, k_higher = 16;

constexpr size_t baselist_size = sum_bc(n/2, n/4);
List out{baselist_size}, l1{baselist_size}, l2{baselist_size};
Matrix A;

Label target;

template <class TR>
void BM_join2lists(benchmark::State& state) {
	target.random();
	uint64_t ctr = 0;
    TR t{1, A, 0};

	l1.set_load(state.range(0));
	l2.set_load(state.range(0));
	for (auto _ : state) {
		out.set_load(0);
		t.join2lists(out, l1, l2, target, k_lower, k_higher, true);
		benchmark::DoNotOptimize(ctr += out[0].label.data());
	}
}

template <class TR>
void BM_join2lists_on_iT_v2(benchmark::State& state) {
	target.random();
	uint64_t ctr = 0;
	l1.set_load(state.range(0));
	l2.set_load(state.range(0));

    TR t{1, A, 0};
	for (auto _ : state) {
		out.set_load(0);
		t.join2lists_on_iT_v2(out, l1, l2, target, k_lower, k_higher, true);
		benchmark::DoNotOptimize(ctr += out[0].label.data());
	}
}

template <class TR>
void BM_join2lists_on_iT_v2_constexpr(benchmark::State& state) {
	target.random();
	uint64_t ctr = 0;
	l1.set_load(state.range(0));
	l2.set_load(state.range(0));

    TR t{1, A, 0};
	for (auto _ : state) {
		out.set_load(0);
		t.template join2lists_on_iT_v2
		        <k_lower, k_higher>(out, l1, l2, target, true);
		benchmark::DoNotOptimize(ctr += out[0].label.data());
	}
}

template <class TR>
void BM_join2lists_on_iT_hashmap_v2_constexpr(benchmark::State& state) {
	target.random();
	uint64_t ctr = 0;
	l1.set_load(state.range(0));
	l2.set_load(state.range(0));

	using D = typename Label::DataType;
	constexpr static SimpleHashMapConfig simpleHashMapConfig{
			10, 1ul<<(k_higher-k_lower), 1
	};
	using HM = SimpleHashMap<D, size_t, simpleHashMapConfig, Hash<D, k_lower, k_higher, 2>>;
	HM hm{};

    // TODO api changed
	// for (auto _ : state) {
	// 	out.set_load(0);
	// 	TR::template join2lists_on_iT_hashmap_v2
	// 			<k_lower, k_higher>(out, l1, l2, hm, target, true);
	// 	benchmark::DoNotOptimize(ctr += out[0].label.data());
	// }
}


BENCHMARK(BM_join2lists<Tree>)->RangeMultiplier(2)->Range(128, LS);
BENCHMARK(BM_join2lists_on_iT_v2<Tree>)->RangeMultiplier(2)->Range(128, LS);
BENCHMARK(BM_join2lists_on_iT_v2_constexpr<Tree>)->RangeMultiplier(2)->Range(128, LS);
BENCHMARK(BM_join2lists_on_iT_hashmap_v2_constexpr<Tree>)->RangeMultiplier(2)->Range(128, LS);

int main(int argc, char** argv) {
	rng_seed(time(NULL));

	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
