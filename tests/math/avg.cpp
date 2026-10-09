#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "math/math.h"
#include "random.h"

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

TYPED_TEST_P(AVG, random) {
	// reference in 128 bit: (x+y) cannot overflow, >> on a signed value is floor
	const TypeParam extremes[4] = {TypeParam(0), TypeParam(1), TypeParam(~TypeParam(0)),
	                               TypeParam(TypeParam(1) << (sizeof(TypeParam) * 8 - 1))};
	for (uint32_t i = 0; i < 100000; ++i) {
		TypeParam a = TypeParam(cryptanalysislib::rng<uint64_t>());
		TypeParam b = TypeParam(cryptanalysislib::rng<uint64_t>());
		if (i < 16) { a = extremes[i % 4]; b = extremes[i / 4]; }

		const __int128 s = __int128(a) + __int128(b);
		const __int128 fl = s >> 1;
		const __int128 ce = (s + 1) >> 1;
		EXPECT_EQ(__int128(floor_average(a, b)), fl);
		EXPECT_EQ(__int128(ceil_average(a, b)), ce);
	}
}

REGISTER_TYPED_TEST_SUITE_P(AVG, simple, random);
using MyTypes = ::testing::Types<int, int16_t, int32_t, int64_t, uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, AVG, MyTypes);

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
