#include "b63.h"
#include "counters/perf_events.h"

#include "../bench_config.h"

#include "helper.h"
#include "random.h"
#include "search/search.h"

using T = uint64_t;

// Size of the list to search
constexpr static uint64_t SIZE = 1 << 18u;
constexpr static uint64_t ssize = sizeof(T) * 8;// size of the type in bits
// Where to search
constexpr static uint32_t k_lower = 0;
constexpr static uint32_t k_higher = 54;
constexpr static T MASK = ((T(1) << k_higher) - 1) ^ ((T(1) << k_lower) - 1);
constexpr static size_t MULT = 10;
constexpr static size_t NR_SOLS = 2;

std::vector<T> data;

// std_sort
B63_BASELINE(Std_lowerbound, nn) {
	uint64_t search=0, errors = 0;

	for (uint64_t i = 0; i < MULT * nn; i++) {
		search = rng() % SIZE;
		auto v = std::lower_bound(data.begin(), data.end(), data[search],
		    [](const T e1, const T e2) {
		        return (e1 & MASK) < (e2 & MASK);
		    });
		errors += (uint64_t) std::distance(data.begin(), v) == search;
	}
	const uint64_t keep = search + errors;
	B63_KEEP(keep);
}

B63_BENCHMARK(upper_bound_standard_binary_search, nn) {
	uint64_t search=0, errors = 0;

	for (uint64_t i = 0; i < MULT * nn; i++) {
		search = rng() % SIZE;
		auto v = upper_bound_standard_binary_search(data.begin(), data.end(), data[search],
		  [](const T &e1) -> T {
		      return e1 & MASK;
		  });

		errors += (uint64_t) std::distance(data.begin(), v) == search;
	}
	const uint64_t keep = search + errors;
	B63_KEEP(keep);
}

B63_BENCHMARK(lower_bound_standard_binary_search, nn) {
	uint64_t search=0, errors = 0;

	for (uint64_t i = 0; i < MULT * nn; i++) {
		search = rng() % SIZE;
		auto v = lower_bound_standard_binary_search(data.begin(), data.end(), data[search],
		   [](const T &e1) -> T {
		       return e1 & MASK;
		   });

		errors += (uint64_t) std::distance(data.begin(), v) == search;
	}
	const uint64_t keep = search + errors;
	B63_KEEP(keep);
}

B63_BENCHMARK(upper_bound_monobound_binary_search, nn) {
	uint64_t search=0, errors = 0;

	for (uint64_t i = 0; i < MULT * nn; i++) {
		search = rng() % SIZE;
		auto v = upper_bound_monobound_binary_search(data.begin(), data.end(), data[search],
		  [](const T &e1) -> T {
		      return e1 & MASK;
		  });

		errors += (uint64_t) std::distance(data.begin(), v) == search;
	}
	const uint64_t keep = search + errors;
	B63_KEEP(keep);
}

B63_BENCHMARK(lower_bound_monobound_binary_search, nn) {
	uint64_t search=0, errors = 0;

	for (uint64_t i = 0; i < MULT * nn; i++) {
		search = rng() % SIZE;
		auto v = lower_bound_monobound_binary_search(data.begin(), data.end(), data[search],
		    [](const T &e1) -> T {
		        return e1 & MASK;
		    });

		errors += (uint64_t) std::distance(data.begin(), v) == search;
	}
	const uint64_t keep = search + errors;
	B63_KEEP(keep);
}

B63_BENCHMARK(tripletapped_binary_search, nn) {
	uint64_t search=0, errors = 0;

	for (uint64_t i = 0; i < MULT * nn; i++) {
		search = rng() % SIZE;
		auto v = tripletapped_binary_search(data.begin(), data.end(), data[search],
		    [](const T &e1) -> T {
		        return e1 & MASK;
		    });

		errors += (uint64_t) std::distance(data.begin(), v) == search;
	}
	const uint64_t keep = search + errors;
	B63_KEEP(keep);
}

B63_BENCHMARK(branchless_lower_bound_cmp, nn) {
	uint64_t search=0, errors = 0;

	for (uint64_t i = 0; i < MULT * nn; i++) {
		search = rng() % SIZE;
		auto v = branchless_lower_bound(data.begin(), data.end(), search,
			[](const T &e1, const T &e2) -> T {
			  return (e1 & MASK) < (e2 & MASK);
			});

		errors += (uint64_t) std::distance(data.begin(), v) == search;
	}

	const uint64_t keep = search + errors;
	B63_KEEP(keep);
}

B63_BENCHMARK(branchless_lower_bound, nn) {
	uint64_t search=0, errors = 0;

	for (uint64_t i = 0; i < MULT * nn; i++) {
		search = rng() % SIZE;
		auto v = branchless_lower_bound(data.begin(), data.end(), search,
		  [](const T &e1) {
		      return (e1 & MASK);
		  });

		errors += (uint64_t) std::distance(data.begin(), v) == search;
	}

	const uint64_t keep = search + errors;
	B63_KEEP(keep);
}

B63_BENCHMARK(lower_bound_interpolation_search2, nn) {
	uint64_t search=0, errors = 0;

	for (uint64_t i = 0; i < MULT * nn; i++) {
		search = rng() % SIZE;
		auto v = lower_bound_interpolation_search2(data.begin(), data.end(), data[search],
		                                           [](const T e1) -> T { return e1 & MASK; });
		errors += (uint64_t) std::distance(data.begin(), v) == search;
	}

	const uint64_t keep = search + errors;
	B63_KEEP(keep);
}

B63_BENCHMARK(LowerBoundInterpolationSearch, nn) {
	uint64_t search=0, errors = 0;

	for (uint64_t i = 0; i < MULT * nn; i++) {
		search = rng() % SIZE;
		auto v = LowerBoundInterpolationSearch(data.begin(), data.end(), data[search],
		                                       [](const T e1) -> T { return e1 & MASK; });
		errors += (uint64_t) std::distance(data.begin(), v) == search;
	}

	const uint64_t keep = search + errors;
	B63_KEEP(keep);
}

int main(int argc, char **argv) {
	rng_seed(time(NULL));

	size_t search;
	random_data(data, search, SIZE, NR_SOLS, MASK);
	// preheat?
	uint64_t errors = 0;
	for (uint64_t i = 0; i < 1u<<20u; i++) {
		size_t search = rng() % SIZE;
		auto v = std::lower_bound(data.begin(), data.end(), data[search],
			  [](const T e1, const T e2) {
				return (e1 & MASK) < (e2 & MASK);
			  });
		errors += (uint64_t) std::distance(data.begin(), v) == search;
	}
	B63_RUN_WITH("lpe:cycles,lpe:branches,lpe:branch-misses,lpe:cache-misses,lpe:cache-references,lpe:instructions", argc, argv);
	return errors == 0;
}
