#include <gtest/gtest.h>
#include <iostream>

#include "popcount/popcount.h"

using namespace cryptanalysislib::popcount;

using ::testing::InitGoogleTest;
using ::testing::Test;
constexpr size_t stack_size = 10;

TEST(Simple, uint32_t) {
	EXPECT_EQ(popcount(1u), 1);
	EXPECT_EQ(popcount(1ul), 1);
}

TEST(Memory, uint32_t) {
	uint32_t data[stack_size] = {0};
	EXPECT_EQ(popcount<uint32_t>(data, stack_size), 0);

	data[0] = 1;
	EXPECT_EQ(popcount<uint32_t>(data, stack_size), 1);
}

TEST(Memory, uint64_t) {
	uint64_t data[stack_size] = {0};
	EXPECT_EQ(popcount<uint64_t>(data, stack_size), 0);

	data[0] = 1;
	EXPECT_EQ(popcount<uint64_t>(data, stack_size), 1);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
