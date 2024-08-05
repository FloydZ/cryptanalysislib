#include <benchmark/benchmark.h>
#include <cstdint>
#include <cstdlib>

#include "math/math.h"
#include "random.h"

constexpr uint32_t mod_u32 = 97987823;

static void BM_mod(benchmark::State &state) {
	uint32_t a = fastrandombytes_uint64(),
			b = fastrandombytes_uint64(),
			c = fastrandombytes_uint64();

	for (auto _: state) {
		for (size_t i = 0; i < state.range(0); ++i) {
			a += (b + c) % mod_u32;
			b += (a + c) % mod_u32;
			c += (a + b) % mod_u32;
			benchmark::DoNotOptimize(c += i);
		}
	}
}

static void BM_div(benchmark::State &state) {
	uint32_t a = fastrandombytes_uint64(),
			 b = fastrandombytes_uint64(),
			 c = fastrandombytes_uint64();

	for (auto _: state) {
		for (size_t i = 0; i < state.range(0); ++i) {
			a += (b + c) / mod_u32;
			b += (a + c) / mod_u32;
			c += (a + b) / mod_u32;
			benchmark::DoNotOptimize(c += i);
		}
	}
}

static void BM_fastmod(benchmark::State &state) {
	uint32_t a = fastrandombytes_uint64(),
			b = fastrandombytes_uint64(),
			c = fastrandombytes_uint64();

	for (auto _: state) {
		for (size_t i = 0; i < state.range(0); ++i) {
			a += fastmod<mod_u32>(b + c);
			b += fastmod<mod_u32>(a + c);
			c += fastmod<mod_u32>(a + b);
			benchmark::DoNotOptimize(c += i);
		}
	}
}

static void BM_fastdiv(benchmark::State &state) {
	uint32_t a = fastrandombytes_uint64(),
	         b = fastrandombytes_uint64(),
	         c = fastrandombytes_uint64();

	for (auto _: state) {
		for (size_t i = 0; i < state.range(0); ++i) {
			a += fastdiv<mod_u32>(b + c);
			b += fastdiv<mod_u32>(a + c);
			c += fastdiv<mod_u32>(a + b);
			benchmark::DoNotOptimize(c += i);
		}
	}
}

BENCHMARK(BM_fastmod)->RangeMultiplier(2)->Range(1u<<10, 1u<<14);
BENCHMARK(BM_fastdiv)->RangeMultiplier(2)->Range(1u<<10, 1u<<14);
BENCHMARK(BM_mod)->RangeMultiplier(2)->Range(1u<<10, 1u<<14);
BENCHMARK(BM_div)->RangeMultiplier(2)->Range(1u<<10, 1u<<14);

int main(int argc, char **argv) {
	random_seed(time(NULL));

	::benchmark::Initialize(&argc, argv);
	if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
	::benchmark::RunSpecifiedBenchmarks();
	::benchmark::Shutdown();
	return 0;
}
