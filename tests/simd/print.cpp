#include <gtest/gtest.h>
#include <cstdint>
#include <string>

#include "simd/simd.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// prints `S::set1(26)` decimal, hex and binary and compares the output
template<typename S>
static void check_print() {
	constexpr uint32_t bits = sizeof(typename S::limb_type) * 8u;
	const S a = S::set1(26);

	std::string dec, hex, bin;
	for (uint32_t i = 0; i < S::LIMBS; ++i) {
		dec += "26 ";
		hex += "1a ";
		bin += std::string(bits - 5u, '0') + "11010\n";
	}
	dec += "\n";

	testing::internal::CaptureStdout();
	a.print();
	EXPECT_EQ(testing::internal::GetCapturedStdout(), dec);

	testing::internal::CaptureStdout();
	a.print(false, true);
	EXPECT_EQ(testing::internal::GetCapturedStdout(), hex);

	testing::internal::CaptureStdout();
	a.print(true, false);
	EXPECT_EQ(testing::internal::GetCapturedStdout(), bin);
}

TEST(Print, simd128) {
	check_print<_uint8x16_t>();
	check_print<_uint16x8_t>();
	check_print<_uint32x4_t>();
	check_print<_uint64x2_t>();
}

#ifdef USE_AVX512F
TEST(Print, simd512) {
	check_print<uint8x64_t>();
	check_print<uint16x32_t>();
	check_print<uint32x16_t>();
	check_print<uint64x8_t>();
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
