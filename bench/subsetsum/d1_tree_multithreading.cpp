#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <benchmark/benchmark.h>

#include "algorithm/subsetsum.h"
#include "container/kAry_type.h"
#include "helper.h"
#include "matrix/matrix.h"
#include "tree.h"

#include "params.h"

constexpr uint32_t n = 32;
constexpr uint64_t q = (1ul << n);
constexpr uint64_t p = 4;
constexpr uint32_t k_lower=0, k_upper=16;



using T 			= uint64_t;
using Value     	= BinaryVector<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;

Matrix A;
Label target;

void BM_Single(benchmark::State& state) {
	constexpr size_t baselist_size = sum_bc(n/2, p);
	List out{1u<<8, 16}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, 4>;
	Enumerator e{A};
	e.run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	using D = typename Label::DataType;
	constexpr static SimpleHashMapConfig simpleHashMapConfigL0 {
			100, 1ull<<(k_upper-k_lower), 1
	};

	using HML0 = SimpleHashMap<D, size_t, simpleHashMapConfigL0, Hash<D, k_lower, k_upper, 2>>;
	HML0 *hm = new HML0{};

	Tree t{1, A, 0};

	for (auto _ : state) {
		t.join2lists_on_iT_v2
		    <k_lower, k_upper>
		    (out, l1, l2, *hm, target);
		benchmark::ClobberMemory();
		hm->clear();
		out.set_load(0);
	}

	delete hm;
}

template <const uint32_t nthreads,
		  const uint32_t chunks>
void BM_Multi(benchmark::State& state) {
	constexpr size_t baselist_size = sum_bc(n/2, p);
	List out{1u<<8, 16}, l1{baselist_size, chunks}, l2{baselist_size, chunks};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, 4>;
	Enumerator e{A};
	e.run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	using D = typename Label::DataType;
	constexpr static SimpleHashMapConfig simpleHashMapConfigL0 {
			100, 1ull<<(k_upper-k_lower), 1
	};

	using HML0 = SimpleHashMap<D, size_t, simpleHashMapConfigL0, Hash<D, k_lower, k_upper, 2>>;
	HML0 *hm = new HML0{};

	Tree t{1, A, 0};
	for (auto _ : state) {
		t.join2lists_on_iT_v2
		    <k_lower, k_upper, 100, nthreads, chunks>
		    (par_if(true), out, l1, l2, target);
		benchmark::ClobberMemory();

		hm->clear();
		for (uint32_t i = 0; i < chunks; i++) {
			out.set_load(0, i);
		}
	}

	delete hm;
}


BENCHMARK(BM_Single);
BENCHMARK(BM_Multi<1, 1>);
BENCHMARK(BM_Multi<2, 4>);
BENCHMARK(BM_Multi<2, 2>);


int main(int argc, char** argv) {
	rng_seed(time(NULL));
	A.random();
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);
	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
