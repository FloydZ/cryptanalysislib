#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <string>

#include "algorithm/clamp.h"

using ::testing::InitGoogleTest;

/// `clamp` against `std::clamp`, the returned reference is one of the arguments
TEST(clamp, simple) {
	for (int v = -10; v <= 10; ++v) {
		for (int lo = -5; lo <= 5; ++lo) {
			for (int hi = lo; hi <= 6; ++hi) {
				EXPECT_EQ(cryptanalysislib::clamp(v, lo, hi), std::clamp(v, lo, hi)) << v << " " << lo << " " << hi;
				EXPECT_EQ(cryptanalysislib::clamp(v, hi, lo, std::greater<>()),
				          std::clamp(v, hi, lo, std::greater<>())) << v << " " << lo << " " << hi;
			}
		}
	}

	const double v = 1.5, lo = 0.0, hi = 1.0;
	EXPECT_EQ(&cryptanalysislib::clamp(v, lo, hi), &hi);
	EXPECT_EQ(&cryptanalysislib::clamp(lo, lo, hi), &lo);
	const double m = 0.5;
	EXPECT_EQ(&cryptanalysislib::clamp(m, lo, hi), &m);
}

/// unqualified calls are not ambiguous with `std::clamp`
TEST(clamp, unqualified) {
	using namespace cryptanalysislib;
	const std::string a = "b", lo = "c", hi = "e";
	EXPECT_EQ(clamp(a, lo, hi), "c");
}

/// usable in constant expressions
TEST(clamp, constexpr_clamp) {
	static_assert(cryptanalysislib::clamp(300, 0, 255) == 255);
	static_assert(cryptanalysislib::clamp(-3, 0, 255) == 0);
	static_assert(cryptanalysislib::clamp(17u, 0u, 255u) == 17u);
	EXPECT_EQ(cryptanalysislib::clamp(uint8_t(7), uint8_t(8), uint8_t(9)), 8);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
