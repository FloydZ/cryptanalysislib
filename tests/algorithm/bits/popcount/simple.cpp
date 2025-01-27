#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/bits/popcount.h"

using namespace cryptanalysislib::popcount;

using ::testing::InitGoogleTest;
using ::testing::Test;
constexpr size_t stack_size = 10;

TEST(Simple, uint32_t) {
	EXPECT_EQ(popcount(1u), (uint32_t)1u);
	EXPECT_EQ(popcount(1ul), (uint32_t)1u);
}

TEST(Memory, uint32_t) {
	uint32_t data[stack_size] = {0};
	EXPECT_EQ(popcount<uint32_t>(data, stack_size), (uint32_t)0u);

	data[0] = 1;
	EXPECT_EQ(popcount<uint32_t>(data, stack_size), (uint32_t)1u);
}

TEST(Memory, uint64_t) {
	uint64_t data[stack_size] = {0};
	EXPECT_EQ(popcount<uint64_t>(data, stack_size), (uint32_t)0u);

	data[0] = 1;
	EXPECT_EQ(popcount<uint64_t>(data, stack_size), (uint32_t)1u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
