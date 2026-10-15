#include <benchmark/benchmark.h>
#include <cstdint>
#include <vector>

#include "random.h"

using namespace cryptanalysislib;
constexpr size_t LS = 1u << 12u; // multiple of 16: the SIMD versions have no tail handling

#if defined(USE_AVX2) && defined(USE_BMI2)
#include "algorithm/bits/pext.h"

static std::vector<uint32_t> data, mask, out;

/// `bits` = popcount of every mask
static void generate(const uint32_t bits) {
	data.resize(LS); mask.resize(LS); out.resize(LS);
	for (size_t i = 0; i < LS; ++i) {
		data[i] = rng<uint32_t>();
		uint32_t m = 0;
		while ((uint32_t)__builtin_popcount(m) < bits) { m |= 1u << (rng<uint32_t>() % 32u); }
		mask[i] = m;
	}
}

static void BM_pext_bmi2(benchmark::State &state) {
	generate(state.range(0));
	for (auto _: state) {
		for (size_t i = 0; i < LS; ++i) { out[i] = _pext_u32(data[i], mask[i]); }
		benchmark::ClobberMemory();
	}
}

static void BM_pext_emu(benchmark::State &state) {
	generate(state.range(0));
	for (auto _: state) {
		for (size_t i = 0; i < LS; ++i) { out[i] = pext32_emu(data[i], mask[i]); }
		benchmark::ClobberMemory();
	}
}

template<const bool early_exit>
static void BM_pext_avx2(benchmark::State &state) {
	generate(state.range(0));
	for (auto _: state) {
		avx2_pext_u32_reference<32, early_exit>(data.data(), mask.data(), out.data(), LS);
		benchmark::ClobberMemory();
	}
}

BENCHMARK(BM_pext_bmi2)->DenseRange(0, 32, 8);
BENCHMARK(BM_pext_emu)->DenseRange(0, 32, 8);
BENCHMARK(BM_pext_avx2<false>)->DenseRange(0, 32, 8);
BENCHMARK(BM_pext_avx2<true>)->DenseRange(0, 32, 8);

#ifdef USE_AVX512F
template<const bool early_exit>
static void BM_pext_avx512(benchmark::State &state) {
	generate(state.range(0));
	for (auto _: state) {
		avx512_pext_u32_reference<32, early_exit>(data.data(), mask.data(), out.data(), LS);
		benchmark::ClobberMemory();
	}
}
BENCHMARK(BM_pext_avx512<false>)->DenseRange(0, 32, 8);
BENCHMARK(BM_pext_avx512<true>)->DenseRange(0, 32, 8);
#endif
#endif

BENCHMARK_MAIN();
