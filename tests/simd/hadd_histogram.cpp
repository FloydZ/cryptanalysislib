#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// sum of the `bits` wide unsigned parts of `x`
template<typename T>
static uint64_t hadd_ref(const T x, const uint32_t bits) {
	uint64_t s = 0;
	for (uint32_t j = 0; j < sizeof(T) * 8u; j += bits) {
		s += (uint64_t(x) >> j) & ((1ull << bits) - 1ull);
	}
	return s;
}

/// number of nibbles of `x` equal to `n`
template<typename T>
static uint64_t histogram_ref(const T x, const uint32_t n) {
	uint64_t s = 0;
	for (uint32_t j = 0; j < sizeof(T) * 8u; j += 4) {
		s += ((uint64_t(x) >> j) & 0xFu) == n;
	}
	return s;
}

/// `hadd_epu8`, `hadd_epu16`, `hadd_epu32` (64 bit lanes only) and
/// `histogram_epi4`
template<typename S>
static void check_hadd_histogram() {
	using T = typename S::limb_type;
	constexpr uint32_t N = S::LIMBS;
	alignas(64) T a[N], o[N];
	for (uint32_t r = 0; r < 1000; ++r) {
		for (uint32_t i = 0; i < N; ++i) { a[i] = T(rng()); }
		if (r == 0) { for (uint32_t i = 0; i < N; ++i) { a[i] = T(-1); } }
		const S va = S::load(a);

		S::store(o, S::hadd_epu8(va));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(uint64_t(o[i]), hadd_ref(a[i], 8)); }
		S::store(o, S::hadd_epu16(va));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(uint64_t(o[i]), hadd_ref(a[i], 16)); }
		if constexpr (sizeof(T) == 8) {
			S::store(o, S::hadd_epu32(va));
			for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(uint64_t(o[i]), hadd_ref(a[i], 32)); }
		}

		for (uint32_t n = 0; n < 16; ++n) {
			S::store(o, S::histogram_epi4(va, n));
			for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(uint64_t(o[i]), histogram_ref(a[i], n)); }
		}
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(HaddHistogram, simd256) {
	check_hadd_histogram<uint32x8_t>();
	check_hadd_histogram<uint64x4_t>();
}
#endif

#ifdef USE_AVX512F
TEST(HaddHistogram, simd512) {
	check_hadd_histogram<uint32x16_t>();
	check_hadd_histogram<uint64x8_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
