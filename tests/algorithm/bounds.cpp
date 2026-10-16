#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <forward_list>
#include <string>
#include <vector>

#include "random.h"
#include "algorithm/bounds.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `upper_bound`/`equal_range` against the `std::` versions for every value
/// around the range, duplicates included
TEST(bounds, all_values) {
	for (size_t n = 0; n <= 100; ++n) {
		std::vector<uint32_t> v(n);
		for (auto &x : v) { x = uint32_t(rng() % 20u); }
		std::sort(v.begin(), v.end());
		for (uint32_t x = 0; x <= 21; ++x) {
			EXPECT_EQ(cryptanalysislib::upper_bound(v.begin(), v.end(), x),
			          std::upper_bound(v.begin(), v.end(), x)) << n << " " << x;
			EXPECT_EQ(cryptanalysislib::equal_range(v.begin(), v.end(), x),
			          std::equal_range(v.begin(), v.end(), x)) << n << " " << x;
		}
	}
}

/// comparator, heterogeneous value type and forward iterators
TEST(bounds, comparator_forward_list) {
	std::vector<int> v = {9, 7, 7, 7, 4, 1};
	const auto r = cryptanalysislib::equal_range(v.begin(), v.end(), 7, std::greater<>());
	EXPECT_EQ(r.first - v.begin(), 1);
	EXPECT_EQ(r.second - v.begin(), 4);
	EXPECT_EQ(cryptanalysislib::upper_bound(v.begin(), v.end(), 4, std::greater<>()) - v.begin(), 5);

	struct P { int key; std::string name; };
	const std::vector<P> p = {{1, "a"}, {3, "b"}, {3, "c"}, {5, "d"}};
	struct C {
		bool operator()(const P &a, const int b) const { return a.key < b; }
		bool operator()(const int a, const P &b) const { return a < b.key; }
	};
	const auto rp = cryptanalysislib::equal_range(p.begin(), p.end(), 3, C{});
	EXPECT_EQ(rp.first - p.begin(), 1);
	EXPECT_EQ(rp.second - p.begin(), 3);
	EXPECT_EQ(cryptanalysislib::upper_bound(p.begin(), p.end(), 1, C{}) - p.begin(), 1);

	const std::forward_list<int> l = {1, 2, 2, 2, 3};
	const auto rl = cryptanalysislib::equal_range(l.begin(), l.end(), 2);
	EXPECT_EQ(std::distance(l.begin(), rl.first), 1);
	EXPECT_EQ(std::distance(l.begin(), rl.second), 4);
	EXPECT_EQ(std::distance(l.begin(), cryptanalysislib::upper_bound(l.begin(), l.end(), 0)), 0);
}

/// usable in constant expressions
TEST(bounds, constexpr_bounds) {
	constexpr auto r = []() {
		const int a[8] = {1, 2, 2, 2, 5, 6, 6, 9};
		const auto e = cryptanalysislib::equal_range(a, a + 8, 6);
		return int(e.first - a) * 100 + int(e.second - a) * 10 + int(cryptanalysislib::upper_bound(a, a + 8, 2) - a);
	}();
	static_assert(r == 574);
	EXPECT_EQ(r, 574);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
