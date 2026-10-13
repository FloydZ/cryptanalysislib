#include <algorithm>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

#include "algorithm/shuffle.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

/// `random_shuffle` permutes the range, every position takes every value
TEST(shuffle, random_shuffle) {
	std::vector<uint32_t> v(10);
	uint32_t seen[10][10] = {};
	for (uint32_t r = 0; r < 2000; ++r) {
		for (uint32_t i = 0; i < 10; ++i) { v[i] = i; }
		cryptanalysislib::random_shuffle(v.begin(), v.end());
		std::vector<uint32_t> s = v;
		std::sort(s.begin(), s.end());
		for (uint32_t i = 0; i < 10; ++i) { ASSERT_EQ(s[i], i); seen[i][v[i]] += 1; }
	}
	for (uint32_t i = 0; i < 10; ++i) {
		for (uint32_t j = 0; j < 10; ++j) { EXPECT_GT(seen[i][j], 100u) << i << " " << j; }
	}

	// empty and single element ranges
	std::vector<uint32_t> e;
	cryptanalysislib::random_shuffle(e.begin(), e.end());
	std::vector<uint32_t> one = {7};
	cryptanalysislib::random_shuffle(one.begin(), one.end());
	EXPECT_EQ(one[0], 7u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
