#include <benchmark/benchmark.h>
#include <cinttypes>
#include <cstdint>
#include <cstdlib>

#include "search/search.h"

using T = uint64_t;
std::vector<T> data;

// Size of the list to search
constexpr static uint64_t SIZE = 1 << 24u;
constexpr static uint64_t ssize = sizeof(T) * 8;// size of the type in bits
// Where to search
constexpr static uint32_t k_lower = 0;
constexpr static uint32_t k_higher = 54;
constexpr static T MASK = ((T(1) << k_higher) - 1) ^ ((T(1) << k_lower) - 1);
constexpr static size_t MULT = 10;
constexpr static size_t NR_SOLS = 2;

static void DoSetup(const benchmark::State& state) {
	size_t search;
	random_data(data, search, state.range(0), NR_SOLS, MASK);
}

static void stdlowerbound_bench(benchmark::State& state) {
	uint64_t errors = 0;
	for (auto _ : state) {
		const size_t search = rng() % state.range(0);
		auto v = std::lower_bound(data.begin(), data.begin() + state.range(0), data[search],
		[](const T e1, const T e2) {
			return (e1 & MASK) < (e2 & MASK);
		});
		benchmark::DoNotOptimize(errors += (uint64_t) std::distance(data.begin(), v) == search);
	}
	state.SetComplexityN(state.range(0));
}

static void upper_bound_standard_binary_search_bench(benchmark::State& state) {
	uint64_t errors = 0;
	for (auto _ : state) {
		const size_t search = rng() % state.range(0);
		auto v = upper_bound_breaking_linear_search(data.begin(), data.begin() + state.range(0), data[search],
			[](const T e1) {
				return e1 & MASK;
			});
		benchmark::DoNotOptimize(errors += (uint64_t) std::distance(data.begin(), v) == search);
	}
	state.SetComplexityN(state.range(0));
}


static void lower_bound_standard_binary_search_bench(benchmark::State& state) {
	uint64_t errors = 0;
	for (auto _ : state) {
		const size_t search = rng() % state.range(0);
		auto v = lower_bound_standard_binary_search(data.begin(), data.begin() + state.range(0), data[search],
			[](const T e1) {
			  return e1 & MASK;
			});
		benchmark::DoNotOptimize(errors += (uint64_t) std::distance(data.begin(), v) == search);
	}
	state.SetComplexityN(state.range(0));
}
static void upper_bound_monobound_binary_search_bench(benchmark::State& state) {
	uint64_t errors = 0;
	for (auto _ : state) {
		const size_t search = rng() % state.range(0);
		auto v = upper_bound_monobound_binary_search(data.begin(), data.begin() + state.range(0), data[search],
			[](const T e1) {
			  return e1 & MASK;
			});
		benchmark::DoNotOptimize(errors += (uint64_t) std::distance(data.begin(), v) == search);
	}
	state.SetComplexityN(state.range(0));
}
static void lower_bound_monobound_binary_search_bench(benchmark::State& state) {
	uint64_t errors = 0;
	for (auto _ : state) {
		const size_t search = rng() % state.range(0);
		auto v = lower_bound_monobound_binary_search(data.begin(), data.begin() + state.range(0), data[search],
			[](const T e1) {
			  return e1 & MASK;
			});
		benchmark::DoNotOptimize(errors += (uint64_t) std::distance(data.begin(), v) == search);
	}
	state.SetComplexityN(state.range(0));
}
static void tripletapped_binary_search_bench(benchmark::State& state) {
	uint64_t errors = 0;
	for (auto _ : state) {
		const size_t search = rng() % state.range(0);
		auto v = tripletapped_binary_search(data.begin(), data.begin() + state.range(0), data[search],
			[](const T e1) {
			  return e1 & MASK;
			});
		benchmark::DoNotOptimize(errors += (uint64_t) std::distance(data.begin(), v) == search);
	}
	state.SetComplexityN(state.range(0));
}
static void branchless_lower_bound_bench(benchmark::State& state) {
	uint64_t errors = 0;
	for (auto _ : state) {
		const size_t search = rng() % state.range(0);
		auto v = branchless_lower_bound(data.begin(), data.begin() + state.range(0), data[search],
			[](const T e1) {
			  return e1 & MASK;
			});
		benchmark::DoNotOptimize(errors += (uint64_t) std::distance(data.begin(), v) == search);
	}
	state.SetComplexityN(state.range(0));
}

static void LowerBoundInterpolationSearch_bench(benchmark::State& state) {
	uint64_t errors = 0;
	for (auto _ : state) {
		const size_t search = rng() % state.range(0);
		auto v = LowerBoundInterpolationSearch(data.begin(), data.begin() + state.range(0), data[search],
			[](const T e1) {
			  return e1 & MASK;
			});
		benchmark::DoNotOptimize(errors += (uint64_t) std::distance(data.begin(), v) == search);
	}
	state.SetComplexityN(state.range(0));
}
static void lower_bound_interpolation_search2_bench(benchmark::State& state) {
	uint64_t errors = 0;
	for (auto _ : state) {
		const size_t search = rng() % state.range(0);
		auto v = lower_bound_interpolation_search2(data.begin(), data.begin() + state.range(0), data[search],
												   [](const T e1) {
													 return e1 & MASK;
												   });
		benchmark::DoNotOptimize(errors += (uint64_t) std::distance(data.begin(), v) == search);
	}
	state.SetComplexityN(state.range(0));
}
static void lower_bound_interpolation_search_3p_bench(benchmark::State& state) {
	uint64_t errors = 0;
	for (auto _ : state) {
		const size_t search = rng() % state.range(0);
		auto v = lower_bound_interpolation_3p_search(data.begin(), data.begin() + state.range(0), data[search],
			[](const T e1) {
				 return e1 & MASK;
			});
		benchmark::DoNotOptimize(errors += (uint64_t) std::distance(data.begin(), v) == search);
	}
	state.SetComplexityN(state.range(0));
}


BENCHMARK(stdlowerbound_bench)->RangeMultiplier(2)->Range(128, SIZE)->Complexity()->Setup(DoSetup);
BENCHMARK(upper_bound_standard_binary_search_bench)->RangeMultiplier(2)->Range(128, SIZE)->Complexity()->Setup(DoSetup);
BENCHMARK(lower_bound_standard_binary_search_bench)->RangeMultiplier(2)->Range(128, SIZE)->Complexity()->Setup(DoSetup);
BENCHMARK(upper_bound_monobound_binary_search_bench)->RangeMultiplier(2)->Range(128, SIZE)->Complexity()->Setup(DoSetup);
BENCHMARK(lower_bound_monobound_binary_search_bench)->RangeMultiplier(2)->Range(128, SIZE)->Complexity()->Setup(DoSetup);
BENCHMARK(tripletapped_binary_search_bench)->RangeMultiplier(2)->Range(128, SIZE)->Complexity()->Setup(DoSetup);
BENCHMARK(branchless_lower_bound_bench)->RangeMultiplier(2)->Range(128, SIZE)->Complexity()->Setup(DoSetup);

BENCHMARK(LowerBoundInterpolationSearch_bench)->RangeMultiplier(2)->Range(128, SIZE)->Complexity()->Setup(DoSetup);
BENCHMARK(lower_bound_interpolation_search2_bench)->RangeMultiplier(2)->Range(128, SIZE)->Complexity()->Setup(DoSetup);
BENCHMARK(lower_bound_interpolation_search_3p_bench)->RangeMultiplier(2)->Range(128, SIZE)->Complexity()->Setup(DoSetup);


int main(int argc, char** argv) {
	rng_seed(time(NULL));

    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
