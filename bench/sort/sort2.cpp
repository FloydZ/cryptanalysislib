#include <benchmark/benchmark.h>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include "container/binary_packed_vector.h"
#include "container/fq_packed_vector.h"
#include "container/fq_vector.h"
#include "random.h"
#include "list/list.h"
#include "sort/sort.h"

constexpr size_t n = 20;
constexpr size_t q = 3;
constexpr size_t LS = 1u << 16u;
constexpr uint32_t k_lower = 0,
		           k_upper = n;

using Binary 		= BinaryVector<n>;
using FqVector_ 	= FqNonPackedVector<n, q, uint8_t>;
using FqPVector     = FqPackedVector<n, q, uint64_t>;
using Fq 			= kAry_Type_T<q>;

using Matrix 		= FqMatrix<uint8_t, n, n, q>;
using Value 		= Binary;

using BinaryElement 		= Element_T<Value, Binary, Matrix>;
using FqVectorElement 		= Element_T<Value, FqVector_, Matrix>;
using FqPackedVectorElement	= Element_T<Value, FqPVector, Matrix>;

using BinaryList 			= MetaListT<BinaryElement>;
using FqVectorList 			= MetaListT<FqVectorElement>;
using FqPackedVectorList 	= MetaListT<FqPackedVectorElement>;

size_t search_index = 0;
size_t nr_sols = 1;

template <class List>
void BM_StdSort(benchmark::State& state) {
	using T = typename List::ElementType;
	T t;

	List l{(size_t)state.range(0)};
    cryptanalysislib::random_data<List, T>(l, search_index, state.range(0), nr_sols, t);

	for (auto _ : state) {
		std::sort(l.begin(), l.end(), [](const T &a, const T &b) {
		  return a.is_lower(b);
		});
	}
	state.SetComplexityN(state.range(0));
}

template <class List>
void BM_SkaSort(benchmark::State& state) {
	using T = typename List::ElementType;
	T t;

	List l{(size_t)state.range(0)};
    cryptanalysislib::random_data<List, T>(l, search_index, state.range(0), nr_sols, t);

	for (auto _ : state) {
		ska_sort(l.begin(), l.end(), [](const T &a) __attribute__((always_inline)) {
		  return a.template hash<k_lower, k_upper>();
		});
	}
	state.SetComplexityN(state.range(0));
}


BENCHMARK(BM_StdSort<BinaryList>)->RangeMultiplier(2)->Range(128, LS)->Complexity();
BENCHMARK(BM_StdSort<FqVectorList>)->RangeMultiplier(2)->Range(128, LS)->Complexity();
BENCHMARK(BM_StdSort<FqPackedVectorList>)->RangeMultiplier(2)->Range(128, LS)->Complexity();

BENCHMARK(BM_SkaSort<BinaryList>)->RangeMultiplier(2)->Range(128, LS)->Complexity();
BENCHMARK(BM_SkaSort<FqVectorList>)->RangeMultiplier(2)->Range(128, LS)->Complexity();
BENCHMARK(BM_SkaSort<FqPackedVectorList>)->RangeMultiplier(2)->Range(128, LS)->Complexity();

int main(int argc, char** argv) {
	rng_seed(time(NULL));

	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
