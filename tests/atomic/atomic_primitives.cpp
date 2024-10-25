#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "atomic/atomic_primitives.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(primitives, cas) {
	uint32_t val = 0;
	CAS(&val, &val, 1);
	EXPECT_EQ(val, 1);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
