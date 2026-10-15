#include <gtest/gtest.h>
#include <cstdint>

#define LR_WITH_DUMB_METHODS
#include "container/left_right_array.h"
#include "random.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// every O(log n) query against its O(n) `*_dumb` reference, while indices
/// are set and freed at random
static void check_all(const left_right_array<> &a, const size_t n) {
	for (size_t k = 0; k <= n; ++k) {
		ASSERT_EQ(a.get_free_idx(k), a.get_free_idx_dumb(k)) << k;
		ASSERT_EQ(a.get_set_idx(k), a.get_set_idx_dumb(k)) << k;
	}
	for (size_t i = 0; i < n; ++i) {
		ASSERT_EQ(a.num_FLE(i), a.num_FLE_dumb(i)) << i;
		ASSERT_EQ(a.num_FLI(i), a.num_FLI_dumb(i)) << i;
		ASSERT_EQ(a.num_FRE(i), a.num_FRE_dumb(i)) << i;
		ASSERT_EQ(a.num_FRI(i), a.num_FRI_dumb(i)) << i;
		ASSERT_EQ(a.num_SLE(i), a.num_SLE_dumb(i)) << i;
		ASSERT_EQ(a.num_SLI(i), a.num_SLI_dumb(i)) << i;
		ASSERT_EQ(a.num_SRE(i), a.num_SRE_dumb(i)) << i;
		ASSERT_EQ(a.num_SRI(i), a.num_SRI_dumb(i)) << i;
	}
}

TEST(left_right_array, vs_dumb_methods) {
	for (const size_t n : {1ul, 2ul, 7ul, 64ul, 100ul}) {
		left_right_array<> a(n);
		check_all(a, n);
		for (size_t r = 0; r < 4 * n; ++r) {
			if ((rng() % 2) && a.num_free()) {
				const size_t k = rng() % a.num_free();
				const size_t e = a.get_free_idx_dumb(k);
				ASSERT_EQ(a.get_free_idx_chg(k), e);
				ASSERT_TRUE(a.is_set(e));
			} else if (a.num_set()) {
				const size_t k = rng() % a.num_set();
				const size_t e = a.get_set_idx_dumb(k);
				ASSERT_EQ(a.get_set_idx_chg(k), e);
				ASSERT_TRUE(a.is_free(e));
			}
			check_all(a, n);
		}
		a.set_all();
		EXPECT_EQ(a.num_set(), n);
		check_all(a, n);
		a.free_all();
		EXPECT_EQ(a.num_free(), n);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
