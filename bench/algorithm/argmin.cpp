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

	for (auto _: state) {
		const size_t t = argmin(data.data(), state.range(0));
		benchmark::DoNotOptimize(t);
		benchmark::ClobberMemory();
	}
}

#ifdef USE_AVX2
static void BM_argmin_i32_avx2(benchmark::State &state) {
	static std::vector<int32_t> data;
	generate_data(data, state.range(0));

	for (auto _: state) {
		const size_t t = argmin_avx2_i32(data.data(), state.range(0));
		benchmark::DoNotOptimize(t);
		benchmark::ClobberMemory();
	}
}

static void BM_argmin_i32_avx2_bl16(benchmark::State &state) {
	static std::vector<int32_t> data;
	generate_data(data, state.range(0));

	for (auto _: state) {
		const size_t t = argmin_avx2_i32_bl16(data.data(), state.range(0));
		benchmark::DoNotOptimize(t);
		benchmark::ClobberMemory();
	}
}
static void BM_argmin_i32_avx2_bl32(benchmark::State &state) {
	static std::vector<int32_t> data;
	generate_data(data, state.range(0));

	for (auto _: state) {
		const size_t t = argmin_avx2_i32_bl32(data.data(), state.range(0));
		benchmark::DoNotOptimize(t);
		benchmark::ClobberMemory();
	}
}
BENCHMARK(BM_argmin_i32_avx2)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_argmin_i32_avx2_bl16)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_argmin_i32_avx2_bl32)->RangeMultiplier(2)->Range(32, LS);
#endif

BENCHMARK(BM_stupid_argmin<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stupid_argmin<uint64_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK_MAIN();
