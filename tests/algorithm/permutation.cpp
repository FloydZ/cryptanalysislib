#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <forward_list>
#include <list>
#include <random>
#include <string>
#include <vector>

#include "random.h"
#include "algorithm/permutation.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `is_permutation` against `std::is_permutation`: shuffled copies, one
/// changed element, different lengths, few distinct values
TEST(permutation, is_permutation) {
	for (size_t n = 0; n <= 60; ++n) {
		for (const uint32_t mod : {2u, 5u, 1000u}) {
			std::vector<uint32_t> a(n);
			for (auto &x : a) { x = uint32_t(rng() % mod); }
			auto b = a;
			std::shuffle(b.begin(), b.end(), std::mt19937(uint32_t(rng())));
			EXPECT_TRUE(cryptanalysislib::is_permutation(a.begin(), a.end(), b.begin())) << n;
			EXPECT_TRUE(cryptanalysislib::is_permutation(a.begin(), a.end(), b.begin(), b.end())) << n;

			if (n > 0) {
				auto c = b;
				c[rng() % n] += 1u;
				const bool e = std::is_permutation(a.begin(), a.end(), c.begin());
				EXPECT_EQ(cryptanalysislib::is_permutation(a.begin(), a.end(), c.begin()), e) << n;
				EXPECT_EQ(cryptanalysislib::is_permutation(c.begin(), c.end(), a.begin(), a.end()), e) << n;
				EXPECT_FALSE(cryptanalysislib::is_permutation(a.begin(), a.end(), b.begin(), b.end() - 1)) << n;
			}
		}
	}

	// predicate (equal modulo 10) and forward iterators
	const std::forward_list<int> l = {11, 2, 33, 2};
	const std::vector<int> v = {12, 3, 1, 22};
	const auto mod10 = [](const int x, const int y) { return (x % 10) == (y % 10); };
	EXPECT_TRUE(cryptanalysislib::is_permutation(l.begin(), l.end(), v.begin(), mod10));
	EXPECT_TRUE(cryptanalysislib::is_permutation(l.begin(), l.end(), v.begin(), v.end(), mod10));
	EXPECT_FALSE(cryptanalysislib::is_permutation(l.begin(), l.end(), v.begin(), v.end()));
}

/// `next_permutation`/`prev_permutation` against the `std::` versions: all
/// permutations of small multisets, in both directions
TEST(permutation, next_prev_permutation) {
	for (const std::vector<int> &start : {std::vector<int>{}, std::vector<int>{1},
	                                     std::vector<int>{1, 2, 3, 4, 5}, std::vector<int>{1, 1, 2, 2, 3},
	                                     std::vector<int>{4, 4, 4}, std::vector<int>{1, 2, 3, 4, 5, 6, 7}}) {
		auto v = start, e = start;
		size_t count = 0;
		while (true) {
			const bool r = cryptanalysislib::next_permutation(v.begin(), v.end());
			const bool re = std::next_permutation(e.begin(), e.end());
			ASSERT_EQ(r, re);
			ASSERT_EQ(v, e);
			count += 1;
			if (!r) { break; }
		}
		EXPECT_EQ(v, start); // wrapped around to the sorted permutation

		// backwards from the last permutation
		std::reverse(v.begin(), v.end()); e = v;
		size_t count2 = 0;
		while (true) {
			const bool r = cryptanalysislib::prev_permutation(v.begin(), v.end());
			const bool re = std::prev_permutation(e.begin(), e.end());
			ASSERT_EQ(r, re);
			ASSERT_EQ(v, e);
			count2 += 1;
			if (!r) { break; }
		}
		EXPECT_EQ(count, count2);
	}

	// comparator and bidirectional iterators
	std::list<char> l = {'c', 'b', 'a'};
	EXPECT_TRUE(cryptanalysislib::next_permutation(l.begin(), l.end(), std::greater<>()));
	EXPECT_EQ(l, (std::list<char>{'c', 'a', 'b'}));
	std::string s = "abc";
	EXPECT_FALSE(cryptanalysislib::prev_permutation(s.begin(), s.end()));
	EXPECT_EQ(s, "cba");
}

/// usable in constant expressions
TEST(permutation, constexpr_permutation) {
	constexpr auto r = []() {
		int a[4] = {1, 2, 3, 4};
		int n = 1;
		while (cryptanalysislib::next_permutation(a, a + 4)) { n += 1; }
		const int b[4] = {4, 1, 3, 2};
		return n * 10 + int(cryptanalysislib::is_permutation(a, a + 4, b));
	}();
	static_assert(r == 241);
	EXPECT_EQ(r, 241);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
