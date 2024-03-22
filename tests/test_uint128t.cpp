#include <gtest/gtest.h>
#include <iostream>

#include "uin128t.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(uint128_t, DoesNotLeak) {
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);

    return RUN_ALL_TESTS();
}
