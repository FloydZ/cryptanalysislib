#include <benchmark/benchmark.h>

#include "random.h"
#include "algorithm/argmin.h"

using namespace cryptanalysislib;
constexpr size_t LS = 1u << 16u;

template<typename T>
void generate_data(std::vector<T> &out,
                   const size_t size) noexcept {
	out.resize(size);
	for (size_t i = 0; i < size; ++i) {
		out[i] = rng<T>();
	}
}

template<typename T>
static void BM_stupid_argmin(benchmark::State &state) {
	static std::vector<T> data;
	generate_data(data, state.range(0));

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
		size_t t = cryptanalysislib::argmin(data.begin(), data.end());
        c += cpucycles();
		benchmark::DoNotOptimize(t+1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

BENCHMARK(BM_stupid_argmin<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stupid_argmin<uint64_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK_MAIN();
