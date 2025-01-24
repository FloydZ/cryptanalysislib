#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "math/eea.h"
#include "math/math.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


using namespace cryptanalysislib;


template <typename T>
class EEA : public testing::Test {};

TYPED_TEST_SUITE_P(EEA);

TYPED_TEST_P(EEA, simple) {
	TypeParam a = 4;
	for (int i = 0; i < 100; ++i) {
		TypeParam c, d;
		a = next_prime(a+2);
		TypeParam b = next_prime(a+2);
		const auto e = eea<TypeParam >(c, d, a, b);
		EXPECT_EQ(e, 1);

		TypeParam t = c*a + d*b;
		EXPECT_EQ(e, t);
	}
}

REGISTER_TYPED_TEST_SUITE_P(EEA, simple);
using MyTypes = ::testing::Types<int, int16_t, int32_t, int64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, EEA, MyTypes);

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
