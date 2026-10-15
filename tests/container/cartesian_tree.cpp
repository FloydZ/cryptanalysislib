#include <gtest/gtest.h>
#include <set>
#include <vector>

#include "container/cartesian_tree.h"
#include "random.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// random inserts/erases against std::set, incl. kth and rank
TEST(cartesian_tree, random_vs_std_set) {
	for (const uint32_t mod : {20u, 2000u, 1u << 30}) {
		cartesian_tree<int64_t> t;
		std::set<int64_t> e;
		for (size_t i = 0; i < 20000; ++i) {
			const int64_t v = int64_t(rng() % mod);
			if (rng() % 3) {
				EXPECT_EQ(t.insert(v), e.insert(v).second);
			} else {
				EXPECT_EQ(t.erase(v), e.erase(v) == 1);
			}
		}
		ASSERT_EQ(t.size(), e.size());
		size_t k = 0;
		for (const int64_t v : e) {
			ASSERT_EQ(t.kth(k), v);
			ASSERT_EQ(t.rank(v), k);
			EXPECT_TRUE(t.contains(v));
			++k;
		}
		EXPECT_FALSE(t.contains(-1));
	}
}

/// sorted insertion stays balanced enough (no stack overflow on 10^6 keys)
TEST(cartesian_tree, sorted) {
	cartesian_tree<uint32_t> t;
	for (uint32_t i = 0; i < 1000000; ++i) { t.insert(i); }
	EXPECT_EQ(t.size(), 1000000u);
	EXPECT_EQ(t.kth(123456), 123456u);
	for (uint32_t i = 0; i < 1000000; i += 2) { t.erase(i); }
	EXPECT_EQ(t.kth(0), 1u);
}

/// constexpr evaluation (new/delete in a constant expression)
constexpr int constexpr_test() {
	cartesian_tree<int> t;
	for (int i = 9; i >= 0; --i) { t.insert(i * 3); }
	t.erase(12);
	return t.kth(4) * 100 + int(t.size());
}
static_assert(constexpr_test() == 1509);

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
