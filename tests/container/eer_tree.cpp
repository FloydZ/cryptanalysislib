#include <gtest/gtest.h>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "random.h"
#include "container/eer_tree.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

template<typename V>
static bool is_palindrome(const V &s, const size_t i, const size_t len) {
	for (size_t k = 0; k < len / 2; ++k) {
		if (s[i + k] != s[i + len - 1 - k]) { return false; }
	}
	return true;
}

/// after every symbol: the number of distinct palindromes, the number of
/// palindromic substrings and the longest palindromic suffix against brute force
template<typename T>
static void check(const std::vector<T> &s) {
	eer_tree<T> t;
	std::set<std::vector<T>> distinct;
	uint64_t total = 0;
	for (size_t n = 1; n <= s.size(); ++n) {
		const bool is_new = t.push_back(s[n - 1]);

		// palindromes ending at n-1
		size_t longest = 0;
		bool found_new = false;
		for (size_t i = 0; i < n; ++i) {
			if (is_palindrome(s, i, n - i)) {
				total += 1;
				longest = std::max(longest, n - i);
				found_new |= distinct.insert(std::vector<T>(s.begin() + i, s.begin() + n)).second;
			}
		}

		ASSERT_EQ(t.size(), n);
		ASSERT_EQ(is_new, found_new) << n;
		ASSERT_EQ(t.distinct_palindromes(), distinct.size()) << n;
		ASSERT_EQ(t.total_palindromes(), total) << n;
		ASSERT_EQ(t.longest_palindromic_suffix(), longest) << n;
	}
}

TEST(eer_tree, example) {
	eer_tree<char> t;
	for (const char c : std::string("abacaba")) { t.push_back(c); }
	EXPECT_EQ(t.distinct_palindromes(), 7u);
	EXPECT_EQ(t.longest_palindromic_suffix(), 7u);
	EXPECT_EQ(t.nr_nodes(), 9u);

	t.clear();
	EXPECT_EQ(t.size(), 0u);
	EXPECT_EQ(t.distinct_palindromes(), 0u);
	t.push_back('x');
	EXPECT_EQ(t.distinct_palindromes(), 1u);
}

TEST(eer_tree, random) {
	for (uint32_t sigma = 1; sigma <= 4; ++sigma) {
		for (uint32_t r = 0; r < 20; ++r) {
			std::vector<char> s(1 + rng() % 120);
			for (auto &c : s) { c = char('a' + rng() % sigma); }
			check(s);
		}
	}
}

TEST(eer_tree, uint8) {
	std::vector<uint8_t> s(200);
	for (auto &c : s) { c = uint8_t(rng() % 3u) * 100u; }
	check(s);
}

/// a long string, no fixed maximal length
TEST(eer_tree, long_string) {
	eer_tree<char> t;
	t.reserve(1u << 20);
	for (uint32_t i = 0; i < (1u << 20); ++i) { t.push_back('a'); }
	EXPECT_EQ(t.distinct_palindromes(), size_t(1u << 20));
	EXPECT_EQ(t.longest_palindromic_suffix(), size_t(1u << 20));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
