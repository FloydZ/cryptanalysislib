#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `set_bit(pos)` sets every lane to `1 << pos`
template<typename S>
static void check_set_bit() {
	using T = typename S::limb_type;
	constexpr uint32_t bits = sizeof(T) * 8u;
	for (uint32_t pos = 0; pos < bits; ++pos) {
		S a = S::set1(T(-1));
		a.set_bit(pos);
		for (uint32_t i = 0; i < S::LIMBS; ++i) {
			EXPECT_EQ(uint64_t(a.d[i]) & ((~0ull) >> (64u - bits)), 1ull << pos);
		}
	}
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(SetBit, simd128) {
	check_set_bit<_uint8x16_t>();
	check_set_bit<_uint16x8_t>();
	check_set_bit<_uint32x4_t>();
	check_set_bit<_uint64x2_t>();
}

TEST(SetBit, simd256) {
	check_set_bit<uint8x32_t>();
	check_set_bit<uint16x16_t>();
	check_set_bit<uint32x8_t>();
	check_set_bit<uint64x4_t>();
}
#endif

#ifdef USE_AVX512F
TEST(SetBit, simd512) {
	check_set_bit<uint8x64_t>();
	check_set_bit<uint16x32_t>();
	check_set_bit<uint32x16_t>();
	check_set_bit<uint64x8_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
