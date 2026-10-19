#include <gtest/gtest.h>
#include <cstdint>
#include <functional>
#include <iterator>
#include <list>
#include <numeric>
#include <vector>

#include "random.h"
#include "algorithm/numeric/adjacent_difference.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `adjacent_difference` against `std::adjacent_difference`, also inplace
TEST(adjacent_difference, simple) {
	for (size_t n = 0; n <= 200; ++n) {
		std::vector<uint32_t> a(n), out(n), e(n);
		for (auto &x : a) { x = uint32_t(rng()); }
		const auto r = cryptanalysislib::adjacent_difference(a.begin(), a.end(), out.begin());
		std::adjacent_difference(a.begin(), a.end(), e.begin());
		EXPECT_EQ(r, out.end());
		EXPECT_EQ(out, e) << n;

		cryptanalysislib::adjacent_difference(a.begin(), a.end(), a.begin());
		EXPECT_EQ(a, e) << n;
	}

	// operator order `op(current, previous)` and a list
	const std::list<int> l = {1, 4, 9, 16};
	std::vector<int> o;
	cryptanalysislib::adjacent_difference(l.begin(), l.end(), std::back_inserter(o),
	                                      [](const int cur, const int prev) { return 10 * cur + prev; });
	EXPECT_EQ(o, (std::vector<int>{1, 41, 94, 169}));
}

/// parallel `adjacent_difference` against `std::adjacent_difference`,
/// operator order and the sequential fallback
TEST(adjacent_difference, multithreaded) {
	for (const size_t n : {size_t(0), size_t(1), size_t(100), size_t(1u << 20u), size_t((1u << 20u) + 31u)}) {
		std::vector<uint32_t> a(n), out(n + 1, 7u), e(n + 1, 7u);
		for (auto &x : a) { x = uint32_t(rng()); }
		const auto r = cryptanalysislib::adjacent_difference(par_if(true), a.begin(), a.end(), out.begin());
		std::adjacent_difference(a.begin(), a.end(), e.begin());
		EXPECT_EQ(r, out.begin() + n);
		EXPECT_EQ(out, e) << n;

		const auto op = [](const uint32_t cur, const uint32_t prev) { return 3u * cur + prev; };
		cryptanalysislib::adjacent_difference(par_if(true), a.begin(), a.end(), out.begin(), op);
		std::adjacent_difference(a.begin(), a.end(), e.begin(), op);
		EXPECT_EQ(out, e) << n;
	}
}

/// usable in constant expressions
TEST(adjacent_difference, constexpr_adjacent_difference) {
	constexpr auto r = []() {
		int a[4] = {1, 3, 6, 10};
		cryptanalysislib::adjacent_difference(a, a + 4, a);
		return a[0] * 1000 + a[1] * 100 + a[2] * 10 + a[3];
	}();
	static_assert(r == 1234);
	EXPECT_EQ(r, 1234);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
