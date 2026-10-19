#include <gtest/gtest.h>
#include <cstdint>
#include <numeric>

#include "random.h"
#include "algorithm/numeric/lcm.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `lcm` against `std::lcm`: signs, zeros, mixed types
TEST(lcm, simple) {
	for (int32_t a = -60; a <= 60; ++a) {
		for (int32_t b = -60; b <= 60; ++b) {
			ASSERT_EQ(cryptanalysislib::lcm(a, b), std::lcm(a, b)) << a << " " << b;
		}
	}

	for (uint32_t i = 0; i < 10000; ++i) {
		const uint64_t a = rng() % (1u << 20u), b = rng() % (1u << 20u);
		ASSERT_EQ(cryptanalysislib::lcm(a, b), std::lcm(a, b)) << a << " " << b;
	}

	EXPECT_EQ(cryptanalysislib::lcm(uint8_t(12), int64_t(-18)), 36);
	static_assert(std::is_same_v<decltype(cryptanalysislib::lcm(uint8_t(1), int64_t(1))), int64_t>);
	static_assert(std::is_same_v<decltype(cryptanalysislib::lcm(short(1), short(1))), short>);

	// the largest values which still fit
	EXPECT_EQ(cryptanalysislib::lcm(uint64_t(1) << 63u, uint64_t(1) << 62u), uint64_t(1) << 63u);
	EXPECT_EQ(cryptanalysislib::lcm(INT32_MIN + 2, -2), INT32_MAX - 1);
	EXPECT_EQ(cryptanalysislib::lcm(uint32_t(65537), uint32_t(65521)), 65537u * 65521u);
}

/// usable in constant expressions
TEST(lcm, constexpr_lcm) {
	static_assert(cryptanalysislib::lcm(4, 6) == 12);
	static_assert(cryptanalysislib::lcm(0, 6) == 0);
	static_assert(cryptanalysislib::lcm(-21u, 1u) == -21u);
	EXPECT_EQ(cryptanalysislib::lcm(7, -3), 21);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
