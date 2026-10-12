#include <gtest/gtest.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "algorithm/swap.h"

using ::testing::InitGoogleTest;

TEST(swap, scalars) {
	int a = 1, b = 2;
	cryptanalysislib::swap(a, b);
	EXPECT_EQ(a, 2);
	EXPECT_EQ(b, 1);

	uint64_t c = ~0ull, d = 0;
	cryptanalysislib::swap(c, d);
	EXPECT_EQ(c, 0u);
	EXPECT_EQ(d, ~0ull);
}

TEST(swap, move_only_and_strings) {
	std::unique_ptr<int> a(new int(1)), b(new int(2));
	cryptanalysislib::swap(a, b);
	EXPECT_EQ(*a, 2);
	EXPECT_EQ(*b, 1);

	std::string s = "first", t = "second";
	cryptanalysislib::swap(s, t);
	EXPECT_EQ(s, "second");
	EXPECT_EQ(t, "first");
}

TEST(swap, arrays) {
	int a[3] = {1, 2, 3}, b[3] = {4, 5, 6};
	cryptanalysislib::swap(a, b);
	EXPECT_EQ(a[0], 4); EXPECT_EQ(a[2], 6);
	EXPECT_EQ(b[0], 1); EXPECT_EQ(b[2], 3);
}

TEST(swap, iter_swap) {
	std::vector<int> v = {1, 2, 3};
	cryptanalysislib::iter_swap(v.begin(), v.begin() + 2);
	EXPECT_EQ(v[0], 3);
	EXPECT_EQ(v[2], 1);
}

TEST(swap, unqualified_with_std_visible) {
	// both cryptanalysislib::swap and std::swap are candidates: must not be ambiguous
	using namespace cryptanalysislib;
	using std::swap;
	std::string s = "a", t = "b";
	swap(s, t);
	EXPECT_EQ(s, "b");
	int x = 1, y = 2;
	swap(x, y);
	EXPECT_EQ(x, 2);
}

TEST(swap, swap_ranges) {
	for (size_t n = 0; n <= 20; ++n) {
		std::vector<int> a(n), b(n + 3);
		for (size_t i = 0; i < n; ++i) { a[i] = int(i); }
		for (size_t i = 0; i < n + 3; ++i) { b[i] = 100 + int(i); }
		const auto r = cryptanalysislib::swap_ranges(a.begin(), a.end(), b.begin());
		EXPECT_EQ(r - b.begin(), ptrdiff_t(n));
		for (size_t i = 0; i < n; ++i) { EXPECT_EQ(a[i], 100 + int(i)); EXPECT_EQ(b[i], int(i)); }
		for (size_t i = n; i < n + 3; ++i) { EXPECT_EQ(b[i], 100 + int(i)); }
	}

	std::string s[2] = {"a", "b"}, t[2] = {"c", "d"};
	cryptanalysislib::swap_ranges(s, s + 2, t);
	EXPECT_EQ(s[0], "c"); EXPECT_EQ(t[1], "b");
}

TEST(swap, exchange) {
	int a = 1;
	EXPECT_EQ(cryptanalysislib::exchange(a, 5), 1);
	EXPECT_EQ(a, 5);

	std::unique_ptr<int> p(new int(3));
	auto old = cryptanalysislib::exchange(p, nullptr);
	EXPECT_EQ(*old, 3);
	EXPECT_EQ(p, nullptr);

	constexpr int c = []() { int x = 7; const int o = cryptanalysislib::exchange(x, 9); return o * 10 + x; }();
	static_assert(c == 79);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
