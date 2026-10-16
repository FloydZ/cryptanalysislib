#include <gtest/gtest.h>
#include <cstdint>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `set`/`setr` of the four 32 bit words
template<typename S>
static void check_set_u32() {
	const S a = S::set(0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u);
	EXPECT_EQ(uint32_t(a.v32[0]), 0x44444444u);
	EXPECT_EQ(uint32_t(a.v32[1]), 0x33333333u);
	EXPECT_EQ(uint32_t(a.v32[2]), 0x22222222u);
	EXPECT_EQ(uint32_t(a.v32[3]), 0x11111111u);

	const S b = S::setr(0x11111111u, 0x22222222u, 0x33333333u, 0x44444444u);
	EXPECT_EQ(uint32_t(b.v32[0]), 0x11111111u);
	EXPECT_EQ(uint32_t(b.v32[1]), 0x22222222u);
	EXPECT_EQ(uint32_t(b.v32[2]), 0x33333333u);
	EXPECT_EQ(uint32_t(b.v32[3]), 0x44444444u);
}

/// `set`/`setr` of the two 64 bit words
template<typename S>
static void check_set_u64() {
	const S a = S::set(0x1111111122222222ull, 0x3333333344444444ull);
	EXPECT_EQ(uint64_t(a.v64[0]), 0x3333333344444444ull);
	EXPECT_EQ(uint64_t(a.v64[1]), 0x1111111122222222ull);

	const S b = S::setr(0x1111111122222222ull, 0x3333333344444444ull);
	EXPECT_EQ(uint64_t(b.v64[0]), 0x1111111122222222ull);
	EXPECT_EQ(uint64_t(b.v64[1]), 0x3333333344444444ull);
}

#if defined(USE_AVX2) || defined(USE_ARM)
TEST(WordSet, simd128) {
	check_set_u32<_uint8x16_t>();
	check_set_u32<_int8x16_t>();
	check_set_u32<_uint16x8_t>();
	check_set_u32<_int16x8_t>();
	check_set_u64<_uint32x4_t>();
	check_set_u64<_int32x4_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
