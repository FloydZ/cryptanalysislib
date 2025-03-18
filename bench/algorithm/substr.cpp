#include <benchmark/benchmark.h>
#include <numeric>

#include "algorithm/substr_search.h"
#include "random.h"
#include "cpucycles.h"

constexpr size_t LS = 1u << 20u;

using namespace cryptanalysislib::algorithm::internal;



constexpr void new_instance(span_t &haystack, 
                            span_t &needle, 
                            const size_t n,
                            const size_t k) {
    haystack.data = (uint8_t *)malloc(n);
    haystack.len = n;
    needle.data = (uint8_t *)malloc(k);
    needle.len = k;
    
    for (size_t i = 0; i < n; i++) {
        haystack.data[i] = 0; //cryptanalysislib::rng(); 
    }
    for (size_t i = 0; i < k; i++) {
        needle.data[i] = 0; //cryptanalysislib::rng(); 
    }
    for (uint32_t i = 0; i < k; i++) {
        haystack.data[n - k + i] = needle.data[i];
    }
}


static void BM_naive_substr(benchmark::State &state) {
    span_t haystack, needle;
    new_instance(haystack, needle, state.range(0), state.range(1));

    uint64_t c = 0, r = 0;
	for (auto _: state) {
        c -= cpucycles();
        r += naive_substr(haystack, needle);
        c += cpucycles();


        for (uint32_t i = 0; i < state.range(1); i++) {
            needle.data[i]++;
        }
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
    free(haystack.data); free(needle.data);
}

static void BM_prefix_substr(benchmark::State &state) {
    span_t haystack, needle;
    new_instance(haystack, needle, state.range(0), state.range(1));

    uint64_t c = 0, r = 0;
	for (auto _: state) {
        c -= cpucycles();
        r += prefix_substr(haystack, needle);
        c += cpucycles();


        for (uint32_t i = 0; i < state.range(1); i++) {
            needle.data[i]++;
        }
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
    free(haystack.data); free(needle.data);
}

#ifdef USE_AVX2
static void BM_avx2_prefix_substr(benchmark::State &state) {
    span_t haystack, needle;
    new_instance(haystack, needle, state.range(0), state.range(1));

    uint64_t c = 0, r = 0;
	for (auto _: state) {
        c -= cpucycles();
        r += avx2_prefix_substr(haystack, needle);
        c += cpucycles();


        for (uint32_t i = 0; i < state.range(1); i++) {
            needle.data[i]++;
        }
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
    free(haystack.data); free(needle.data);
}
static void BM_avx2_speculative_substr(benchmark::State &state) {
    span_t haystack, needle;
    new_instance(haystack, needle, state.range(0), state.range(1));

    uint64_t c = 0, r = 0;
	for (auto _: state) {
        c -= cpucycles();
        r += avx2_speculative_substr(haystack, needle);
        c += cpucycles();


        for (uint32_t i = 0; i < state.range(1); i++) {
            needle.data[i]++;
        }
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
    free(haystack.data); free(needle.data);
}
static void BM_avx2_hybrid_substr(benchmark::State &state) {
    span_t haystack, needle;
    new_instance(haystack, needle, state.range(0), state.range(1));

    uint64_t c = 0, r = 0;
	for (auto _: state) {
        c -= cpucycles();
        r += avx2_hybrid_substr(haystack, needle);
        c += cpucycles();


        for (uint32_t i = 0; i < state.range(1); i++) {
            needle.data[i]++;
        }
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
    free(haystack.data); free(needle.data);
}
static void BM_avx2_strstr_anysize(benchmark::State &state) {
    span_t haystack, needle;
    new_instance(haystack, needle, state.range(0), state.range(1));

    uint64_t c = 0, r = 0;
	for (auto _: state) {
        c -= cpucycles();
        r += avx2_strstr_anysize(haystack, needle);
        c += cpucycles();


        for (uint32_t i = 0; i < state.range(1); i++) {
            needle.data[i]++;
        }
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
    free(haystack.data); free(needle.data);
}

BENCHMARK(BM_avx2_prefix_substr) ->ArgsProduct({
      benchmark::CreateRange(128, LS, /*multi=*/2),
      benchmark::CreateDenseRange(2, 4, /*step=*/1)
    });
BENCHMARK(BM_avx2_speculative_substr) ->ArgsProduct({
      benchmark::CreateRange(128, LS, /*multi=*/2),
      benchmark::CreateDenseRange(2, 4, /*step=*/1)
    });
BENCHMARK(BM_avx2_hybrid_substr) ->ArgsProduct({
      benchmark::CreateRange(128, LS, /*multi=*/2),
      benchmark::CreateDenseRange(2, 4, /*step=*/1)
    });
BENCHMARK(BM_avx2_strstr_anysize) ->ArgsProduct({
      benchmark::CreateRange(128, LS, /*multi=*/2),
      benchmark::CreateDenseRange(2, 4, /*step=*/1)
    });

#endif

#ifdef USE_AVX512F
static void BM_avx512_speculative_substr(benchmark::State &state) {
    span_t haystack, needle;
    new_instance(haystack, needle, state.range(0), state.range(1));

    uint64_t c = 0, r = 0;
	for (auto _: state) {
        c -= cpucycles();
        r += avx512_speculative_substr(haystack, needle);
        c += cpucycles();


        for (uint32_t i = 0; i < state.range(1); i++) {
            needle.data[i]++;
        }
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
    free(haystack.data); free(needle.data);
}

BENCHMARK(BM_avx512_speculative_substr) ->ArgsProduct({
      benchmark::CreateRange(128, LS, /*multi=*/2),
      benchmark::CreateDenseRange(2, 4, /*step=*/1)
    });
#endif

#ifdef USE_NEON
static void BM_neon_speculative_substr(benchmark::State &state) {
    span_t haystack, needle;
    new_instance(haystack, needle, state.range(0), state.range(1));

    uint64_t c = 0, r = 0;
	for (auto _: state) {
        c -= cpucycles();
        r += neon_speculative_substr(haystack, needle);
        c += cpucycles();


        for (uint32_t i = 0; i < state.range(1); i++) {
            needle.data[i]++;
        }
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
    free(haystack.data); free(needle.data);
}

BENCHMARK(BM_neon_speculative_substr) ->ArgsProduct({
      benchmark::CreateRange(128, LS, /*multi=*/2),
      benchmark::CreateDenseRange(2, 4, /*step=*/1)
    });
#endif

BENCHMARK(BM_naive_substr) ->ArgsProduct({
      benchmark::CreateRange(128, LS, /*multi=*/2),
      benchmark::CreateDenseRange(2, 4, /*step=*/1)
    });

BENCHMARK(BM_prefix_substr) ->ArgsProduct({
      benchmark::CreateRange(128, LS, /*multi=*/2),
      benchmark::CreateDenseRange(2, 4, /*step=*/1)
    });

BENCHMARK_MAIN();
