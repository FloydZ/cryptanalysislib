#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;

// NOTE: `pdep32_emu` is portable, the BMI2/AVX2/AVX-512 tests are x86 only
#include "algorithm/bits/pdep.h"

/// portable reference: deposits the low bits of `v` at the set bits of `m`
static uint32_t pdep_ref(uint32_t v, uint32_t m) {
	uint32_t ret = 0;
	for (uint32_t i = 0, k = 0; i < 32; ++i) {
		if ((m >> i) & 1u) { ret |= ((v >> k++) & 1u) << i; }
	}
	return ret;
}

/// masks with every popcount 0..32, plus random ones
static void gen(std::vector<uint32_t> &data, std::vector<uint32_t> &mask, const size_t n) {
	data.resize(n); mask.resize(n);
	for (size_t i = 0; i < n; ++i) {
		data[i] = rng<uint32_t>();
		uint32_t m = 0;
		const uint32_t w = i % 33;
		while ((uint32_t)__builtin_popcount(m) < w) { m |= 1u << (rng<uint32_t>() % 32u); }
		mask[i] = (i & 1) ? m : rng<uint32_t>();
	}
}


TEST(pdep, emu) {
	std::vector<uint32_t> d, m; gen(d, m, 4096);
	for (size_t i = 0; i < d.size(); ++i) {
		EXPECT_EQ(pdep32_emu(d[i], m[i]), pdep_ref(d[i], m[i])) << std::hex << d[i] << " " << m[i];
	}
}

#if defined(USE_AVX2) && defined(USE_BMI2)
TEST(pdep, reference_vs_bmi2) {
	std::vector<uint32_t> d, m; gen(d, m, 4096);
	for (size_t i = 0; i < d.size(); ++i) {
		EXPECT_EQ(pdep_ref(d[i], m[i]), _pdep_u32(d[i], m[i])) << i;
	}
}

template<const bool early_exit>
static void check_avx2() {
	constexpr size_t n = 4096; // multiple of 8, the functions have no tail handling
	std::vector<uint32_t> d, m, out(n); gen(d, m, n);
	avx2_pdep_u32_reference<32, early_exit>(d.data(), m.data(), out.data(), n);
	for (size_t i = 0; i < n; ++i) { EXPECT_EQ(out[i], pdep_ref(d[i], m[i])) << i; }
}
TEST(pdep, avx2) { check_avx2<false>(); }
TEST(pdep, avx2_early_exit) { check_avx2<true>(); }

#ifdef USE_AVX512F
template<const bool early_exit>
static void check_avx512() {
	constexpr size_t n = 4096; // multiple of 16
	std::vector<uint32_t> d, m, out(n); gen(d, m, n);
	avx512_pdep_u32_reference<32, early_exit>(d.data(), m.data(), out.data(), n);
	for (size_t i = 0; i < n; ++i) { EXPECT_EQ(out[i], pdep_ref(d[i], m[i])) << i; }
}
TEST(pdep, avx512) { check_avx512<false>(); }
TEST(pdep, avx512_early_exit) { check_avx512<true>(); }
#endif
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
