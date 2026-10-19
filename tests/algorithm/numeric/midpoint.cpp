#include <gtest/gtest.h>
#include <cstdint>
#include <limits>
#include <numeric>

#include "random.h"
#include "algorithm/numeric/midpoint.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// all pairs of 8 bit integers against `std::midpoint`
TEST(midpoint, all_8bit) {
	for (int a = -128; a <= 127; ++a) {
		for (int b = -128; b <= 127; ++b) {
			ASSERT_EQ(cryptanalysislib::midpoint(int8_t(a), int8_t(b)), std::midpoint(int8_t(a), int8_t(b))) << a << " " << b;
			ASSERT_EQ(cryptanalysislib::midpoint(uint8_t(a), uint8_t(b)), std::midpoint(uint8_t(a), uint8_t(b))) << a << " " << b;
		}
	}
}

/// random and extreme wider integers, rounding towards `a`
TEST(midpoint, integers) {
	for (uint32_t i = 0; i < 100000; ++i) {
		const uint64_t a = rng(), b = rng();
		ASSERT_EQ(cryptanalysislib::midpoint(a, b), std::midpoint(a, b));
		ASSERT_EQ(cryptanalysislib::midpoint(int64_t(a), int64_t(b)), std::midpoint(int64_t(a), int64_t(b)));
		ASSERT_EQ(cryptanalysislib::midpoint(int32_t(a), int32_t(b)), std::midpoint(int32_t(a), int32_t(b)));
	}

	constexpr int64_t mn = std::numeric_limits<int64_t>::min(), mx = std::numeric_limits<int64_t>::max();
	EXPECT_EQ(cryptanalysislib::midpoint(mn, mx), -1);
	EXPECT_EQ(cryptanalysislib::midpoint(mx, mn), 0);
	EXPECT_EQ(cryptanalysislib::midpoint(uint64_t(-1), uint64_t(0)), uint64_t(1) << 63u);
	EXPECT_EQ(cryptanalysislib::midpoint(3, 6), 4);
	EXPECT_EQ(cryptanalysislib::midpoint(6, 3), 5);
}

/// floating point: large values do not overflow, small ones do not underflow
TEST(midpoint, floating_point) {
	for (uint32_t i = 0; i < 10000; ++i) {
		const double a = double(int64_t(rng())) * 1e-10, b = double(int64_t(rng())) * 1e288;
		ASSERT_EQ(cryptanalysislib::midpoint(a, b), std::midpoint(a, b));
		ASSERT_EQ(cryptanalysislib::midpoint(b, a), std::midpoint(b, a));
		ASSERT_EQ(cryptanalysislib::midpoint(float(a), float(a * 3)), std::midpoint(float(a), float(a * 3)));
	}

	constexpr double mx = std::numeric_limits<double>::max(), dmin = std::numeric_limits<double>::denorm_min();
	EXPECT_EQ(cryptanalysislib::midpoint(mx, mx), mx);
	EXPECT_EQ(cryptanalysislib::midpoint(-mx, mx), 0.0);
	EXPECT_EQ(cryptanalysislib::midpoint(dmin, mx), std::midpoint(dmin, mx));
	EXPECT_EQ(cryptanalysislib::midpoint(dmin, 3 * dmin), 2 * dmin);
	EXPECT_EQ(cryptanalysislib::midpoint(1.0, 2.0), 1.5);
}

/// pointers: rounded towards the first argument
TEST(midpoint, pointers) {
	int a[10] = {};
	EXPECT_EQ(cryptanalysislib::midpoint(a, a + 9), a + 4);
	EXPECT_EQ(cryptanalysislib::midpoint(a + 9, a), a + 5);
	const int *c = a;
	EXPECT_EQ(cryptanalysislib::midpoint(c, c + 2), c + 1);
}

/// usable in constant expressions
TEST(midpoint, constexpr_midpoint) {
	static_assert(cryptanalysislib::midpoint(uint32_t(-1), uint32_t(-3)) == uint32_t(-2));
	static_assert(cryptanalysislib::midpoint(-7, 2) == -3);
	static_assert(cryptanalysislib::midpoint(1.0f, 4.0f) == 2.5f);
	EXPECT_EQ(cryptanalysislib::midpoint(int16_t(-7), int16_t(2)), -3);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
