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

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
