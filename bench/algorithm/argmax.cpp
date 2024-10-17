#include <benchmark/benchmark.h>

#include "random.h"
#include "algorithm/argmax.h"

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
static void BM_argmax(benchmark::State &state) {
	static std::vector<T> data;
	generate_data(data, state.range(0));

    uint64_t c = 0;
	for (auto _: state) {
		//const size_t t = cryptanalysislib::argmax<T>(data.data(), state.range(0));
        c -= cpucycles();
		size_t t = cryptanalysislib::argmax(data.begin(), data.end());
        c += cpucycles();
		benchmark::DoNotOptimize(t+1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_argmax_u32_simd(benchmark::State &state) {
	static std::vector<uint32_t> data;
	generate_data(data, state.range(0));

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
		size_t t = argmax_simd_u32(data.data(), state.range(0));
        c += cpucycles();
		benchmark::DoNotOptimize(t+1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_argmax_u32_simd_bl16(benchmark::State &state) {
	static std::vector<uint32_t> data;
	generate_data(data, state.range(0));

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
		size_t t = argmax_simd_u32_bl16(data.data(), state.range(0));
        c += cpucycles();
		benchmark::DoNotOptimize(t+1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

static void BM_argmax_u32_simd_bl32(benchmark::State &state) {
	static std::vector<uint32_t> data;
	generate_data(data, state.range(0));

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
		size_t t = argmax_simd_u32_bl32(data.data(), state.range(0));
        c += cpucycles();
		benchmark::DoNotOptimize(t+1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}

template<typename T>
static void BM_argmax_multithreaded(benchmark::State &state) {
	static std::vector<T> fr;
    fr.resize(state.range(0));
    std::fill(fr.begin(), fr.end(), 2);

    uint64_t c = 0;
	for (auto _: state) {
        c -= cpucycles();
        auto r = cryptanalysislib::argmax(par_if(true), fr.begin(), fr.end());
        c += cpucycles();
        benchmark::DoNotOptimize(r += 1);
		benchmark::ClobberMemory();
	}
    state.counters["cycles"] = (double)c/(double)state.iterations();
}


BENCHMARK(BM_argmax_u32_simd)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_argmax_u32_simd_bl16)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_argmax_u32_simd_bl32)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_argmax<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_argmax<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_argmax<uint64_t>)->RangeMultiplier(2)->Range(32, LS);

BENCHMARK(BM_argmax_multithreaded<uint8_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_argmax_multithreaded<uint32_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK(BM_argmax_multithreaded<uint64_t>)->RangeMultiplier(2)->Range(32, LS);
BENCHMARK_MAIN();
