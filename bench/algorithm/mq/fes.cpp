#include <benchmark/benchmark.h>
#include <numeric>

#include "algorithm/mq/fes.h"
#include "random.h"

constexpr static uint32_t n = 22;
constexpr static uint32_t m = 16;
constexpr static uint32_t k = n > 16 ? 16 : n/2;


uint32_t Fq[496];
uint32_t Fl[34];
const int count = 32;
uint32_t buffer[m * count];
int size[m];

using namespace cryptanalysislib;

#ifdef USE_AVX2
static void BM_feslite_avx2_enum_16x16(benchmark::State &state) {
    uint64_t c = 0, r = 0;
	for (auto _: state) {
        c -= cpucycles();
        r += feslite_avx2_enum_16x16(n, m, Fq, Fl, count, buffer, size);
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}

    state.counters["cycles"] = (double)c/(double)state.iterations();
}


BENCHMARK(BM_feslite_avx2_enum_16x16);
#endif

int main(int argc, char **argv) {
	rng_seed(time(NULL));

	uint32_t mask = ((1ull << k) - 1u) & 0xffffffff;
	for (uint32_t i = 0; i < 496; i++) { Fq[i] = rng() & mask; }
	for (uint32_t i = 0; i < n + 1; i++) { Fl[i] = rng() & mask; }

	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
