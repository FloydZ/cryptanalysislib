#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "math/avg.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


using namespace cryptanalysislib::math;

template <typename T>
class AVG : public testing::Test {};

TYPED_TEST_SUITE_P(AVG);

TYPED_TEST_P(AVG, simple) {
	TypeParam a = 2;
	TypeParam b = 2;
	for (uint32_t i = 0; i < 100; ++i) {

        const TypeParam c1 = ceil_average(a, b);
        const TypeParam c2 = floor_average(a, b);

        EXPECT_EQ(c1, c2);
	}
}

REGISTER_TYPED_TEST_SUITE_P(AVG, simple);
using MyTypes = ::testing::Types<int, int16_t, int32_t, int64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, AVG, MyTypes);

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
