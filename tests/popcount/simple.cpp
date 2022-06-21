#include <gtest/gtest.h>
#include <iostream>

#include "popcount/x86.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr size_t stack_size = 10;

TEST(Simple, uint32_t) {
	EXPECT_EQ(popcount::count(1u), 1);
	EXPECT_EQ(popcount::count(1ul), 1);

	// fails
	// EXPECT_ANY_THROW(popcount::count(__uint128_t(1ull)));
}

TEST(Memory, uint32_t) {
	uint32_t data[stack_size] = {0};
	EXPECT_EQ(popcount::count(data, stack_size), 0);

	data[0] = 1;
	EXPECT_EQ(popcount::count(data, stack_size), 1);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
