#include <gtest/gtest.h>
#include <cstdint>
#include <list>
#include <numeric>
#include <vector>

#include "algorithm/numeric/iota.h"

using ::testing::InitGoogleTest;

/// `iota` against `std::iota`, wrap around of small unsigned types
TEST(iota, simple) {
	for (size_t n = 0; n <= 300; ++n) {
		std::vector<uint8_t> v(n), e(n);
		cryptanalysislib::iota(v.begin(), v.end(), uint8_t(250));
		std::iota(e.begin(), e.end(), uint8_t(250));
		EXPECT_EQ(v, e) << n;
	}

	std::vector<double> d(4);
	cryptanalysislib::iota(d.begin(), d.end(), 0.5);
	EXPECT_EQ(d, (std::vector<double>{0.5, 1.5, 2.5, 3.5}));

	// fill with iterators
	std::vector<int> x = {7, 8, 9};
	std::list<std::vector<int>::iterator> l(3);
	cryptanalysislib::iota(l.begin(), l.end(), x.begin());
	int s = 0;
	for (const auto it : l) { s = s * 10 + *it; }
	EXPECT_EQ(s, 789);
}

/// unqualified calls are not ambiguous with `std::iota`
TEST(iota, unqualified) {
	using namespace cryptanalysislib;
	std::vector<int> v(3);
	iota(v.begin(), v.end(), -1);
	EXPECT_EQ(v, (std::vector<int>{-1, 0, 1}));
}

/// usable in constant expressions
TEST(iota, constexpr_iota) {
	constexpr auto r = []() {
		int a[4] = {};
		cryptanalysislib::iota(a, a + 4, 3);
		return a[0] * 1000 + a[1] * 100 + a[2] * 10 + a[3];
	}();
	static_assert(r == 3456);
	EXPECT_EQ(r, 3456);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
