#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `pack(mask)`: the indices of the set bits of `mask` in front, then zeros
template<typename S>
static void check_pack() {
	constexpr uint32_t N = S::LIMBS;
	alignas(64) uint32_t o[N];
	for (uint32_t m = 0; m < (1u << N); ++m) {
		S::store(o, S::pack(m));
		uint32_t k = 0;
		for (uint32_t i = 0; i < N; ++i) {
			if ((m >> i) & 1u) { EXPECT_EQ(o[k++], i); }
		}
		for (; k < N; ++k) { EXPECT_EQ(o[k], 0u); }
	}
}

/// `cvtepu8`: zero extends the first `LIMBS` bytes
template<typename S>
static void check_cvtepu8() {
	constexpr uint32_t N = S::LIMBS;
	alignas(64) uint8_t a[16];
	alignas(64) uint32_t o[N];
	for (uint32_t r = 0; r < 1000; ++r) {
		for (uint32_t i = 0; i < 16; ++i) { a[i] = uint8_t(rng()); }
		S::store(o, S::cvtepu8(_uint8x16_t::load(a)));
		for (uint32_t i = 0; i < N; ++i) { EXPECT_EQ(o[i], uint32_t(a[i])); }
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(PackCvtepu8, simd256) {
	check_pack<uint32x8_t>();
	check_cvtepu8<uint32x8_t>();
}
#endif

#ifdef USE_AVX512F
TEST(PackCvtepu8, simd512) {
	check_pack<uint32x16_t>();
	check_cvtepu8<uint32x16_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
