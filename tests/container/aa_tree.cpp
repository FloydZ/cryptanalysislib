#include <gtest/gtest.h>
#include <set>
#include <vector>

#include "random.h"
#include "container/aa_tree.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// random inserts and erases against std::set, invariants after every step
TEST(AATree, random_vs_std_set) {
	for (const uint64_t mod : {10ull, 1000ull, 1ull << 40}) {
		AATreeSet<uint64_t> t;
		std::set<uint64_t> e;
		for (size_t i = 0; i < 20000; ++i) {
			const uint64_t v = rng<uint64_t>() % mod;
			if (rng() % 3) {
				EXPECT_EQ(t.insert(v), e.insert(v).second);
			} else {
				EXPECT_EQ(t.erase(v), e.erase(v) == 1);
			}
			ASSERT_EQ(t.len(), e.size());
			if ((i % 1000) == 0) { ASSERT_TRUE(t.check()) << i << " " << mod; }
		}
		ASSERT_TRUE(t.check());
		for (uint64_t v = 0; v < 2000; ++v) {
			EXPECT_EQ(t.contains(v), e.count(v) == 1);
		}
		std::vector<uint64_t> in_order;
		t.for_each([&](const uint64_t v) { in_order.push_back(v); });
		EXPECT_EQ(in_order, std::vector<uint64_t>(e.begin(), e.end()));

		t.clear();
		EXPECT_TRUE(t.empty());
		EXPECT_TRUE(t.check());
	}
}

/// sorted inserts (the worst case of an unbalanced tree), then erase all
TEST(AATree, sorted) {
	AATreeSet<uint32_t> t;
	for (uint32_t i = 0; i < 100000; ++i) { EXPECT_TRUE(t.insert(i)); }
	EXPECT_TRUE(t.check());
	for (uint32_t i = 0; i < 100000; i += 2) { EXPECT_TRUE(t.erase(i)); }
	EXPECT_TRUE(t.check());
	EXPECT_EQ(t.len(), 50000u);
	for (uint32_t i = 1; i < 100000; i += 2) { EXPECT_TRUE(t.erase(i)); }
	EXPECT_TRUE(t.empty());
	EXPECT_TRUE(t.check());
}

/// custom ordering
TEST(AATree, greater) {
	AATreeSet<int, std::greater<int>> t;
	for (int i = 0; i < 10; ++i) { t.insert(i); }
	std::vector<int> v;
	t.for_each([&](int x) { v.push_back(x); });
	EXPECT_EQ(v, (std::vector<int>{9, 8, 7, 6, 5, 4, 3, 2, 1, 0}));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
