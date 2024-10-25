#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "atomic/atomic.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(futex, simple) {
	cryptanalysislib::atomic::futex f{0};
	EXPECT_EQ(f.up(), 0);
	EXPECT_EQ(f.down(), 0);
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
