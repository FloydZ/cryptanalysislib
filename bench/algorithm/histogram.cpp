#include <benchmark/benchmark.h>

#include "random.h"
#include "algorithm/histogram.h"
constexpr size_t LS = 1u << 12u;
uint32_t cnt[256] __attribute__((aligned(64))) = {0};

template<typename T>
void generate_data(std::vector<T> &out,
                   const size_t size) noexcept {
	out.resize(size);
	for (size_t i = 0; i < size; ++i) {
		out[i] = fastrandombytes_T<T>();
	}
}

template<typename T>
static void BM_stupid_histogram(benchmark::State &state) {
	static std::vector<T> data;
	generate_data(data, state.range(0));

	for (auto _: state) {
		for (size_t i = 0; i < data.size(); ++i) {
			cnt[data[i]] += 1;
		}
		benchmark::ClobberMemory();
	}
}

template<typename T>
static void BM_histogram(benchmark::State &state) {
	static std::vector<T> data;
	generate_data(data, state.range(0));

	for (auto _: state) {
		// histogram(data.begin(), data.end());
		histogram_u8_1x(cnt, data.data(), state.range(0));
		benchmark::ClobberMemory();
	}
}

template<typename T>
static void BM_histogram_4x(benchmark::State &state) {
	static std::vector<T> data;
	generate_data(data, state.range(0));

	for (auto _: state) {
		histogram_u8_4x(cnt, data.data(), state.range(0));
		benchmark::ClobberMemory();
	}
}

#ifdef USE_AVX512F
template<typename T>
static void BM_histogram_avx512(benchmark::State &state) {
	static std::vector<T> data;
	generate_data(data, state.range(0));

	for (auto _: state) {
		// histogram(data.begin(), data.end());
		avx512_histogram_u8_1x(cnt, data.data(), state.range(0));
		benchmark::ClobberMemory();
	}
}
BENCHMARK(BM_histogram_avx512<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
#endif

BENCHMARK(BM_stupid_histogram<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_histogram<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_histogram_4x<uint8_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK_MAIN();
