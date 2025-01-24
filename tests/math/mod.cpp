#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "math/math.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


using namespace cryptanalysislib;

TEST(fastmod, rand) {
    using TypeParam = uint32_t;
    TypeParam a = 5;
	for (uint32_t i = 0; i < 100; ++i) {
        a = next_prime(a);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
