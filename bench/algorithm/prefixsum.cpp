#include <benchmark/benchmark.h>

#include "random.h"
#include "algorithm/prefixsum.h"

using namespace cryptanalysislib;
using namespace cryptanalysislib::algorithm;
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
static void BM_stupid_prefixsum(benchmark::State &state) {
	static std::vector<T> data;
	generate_data(data, state.range(0));

	for (auto _: state) {
		for (size_t i = 1; i < data.size(); ++i) {
			data[i] += data[i-1];
		}
		benchmark::ClobberMemory();
	}
}

template<typename T>
static void BM_prefixsum(benchmark::State &state) {
	static std::vector<T> data;
	generate_data(data, state.range(0));

	for (auto _: state) {
		prefixsum(data.begin(), data.end());
		benchmark::ClobberMemory();
	}
}

template<typename T>
static void BM_prefix_sum(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);
	for (auto _: state) {
        cryptanalysislib::algorithm::prefixsum(fr.begin(), fr.end(), fr.begin(), std::plus<>());
		benchmark::ClobberMemory();
	}
}

#ifdef USE_AVX2
static void BM_prefix_sum_u32_avx2(benchmark::State &state) {
	using T = uint32_t;
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);
	for (auto _: state) {
        cryptanalysislib::internal::prefixsum_u32_avx2(fr.data(), state.range(0));
		benchmark::ClobberMemory();
	}
}

static void BM_prefix_sum_i32_avx2(benchmark::State &state) {
	using T = int32_t;
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);
	for (auto _: state) {
        cryptanalysislib::internal::prefixsum_i32_avx2(fr.data(), state.range(0));
		benchmark::ClobberMemory();
	}
}

static void BM_prefix_sum_i32_avx2_v2(benchmark::State &state) {
	using T = int32_t;
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 1);
	for (auto _: state) {
        cryptanalysislib::internal::prefixsum_i32_avx2(fr.data(), state.range(0));
		benchmark::ClobberMemory();
	}
}

BENCHMARK(BM_prefix_sum_u32_avx2)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_prefix_sum_i32_avx2)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_prefix_sum_i32_avx2_v2)->RangeMultiplier(2)->Range(32, LS);

#endif


BENCHMARK(BM_prefix_sum<int8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_prefix_sum<int32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_prefix_sum<int64_t>)->RangeMultiplier(2)->Range(32, LS);

//BENCHMARK(BM_stupid_prefixsum<uint32_t>)->DenseRange(32, 350, 32);
//BENCHMARK(BM_prefixsum<uint32_t>)->DenseRange(32, 350, 32);

BENCHMARK(BM_stupid_prefixsum<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_stupid_prefixsum<uint64_t>)->RangeMultiplier(2)->Range(128, LS);

BENCHMARK(BM_prefixsum<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_prefixsum<uint64_t>)->RangeMultiplier(2)->Range(128, LS);
BENCHMARK_MAIN();
