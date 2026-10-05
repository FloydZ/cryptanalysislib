#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/equal.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

template <typename T>
class Equal : public testing::Test {};

TYPED_TEST_SUITE_P(Equal);

TYPED_TEST_P(Equal, simple) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in1; in1.resize(s);
    std::vector<TypeParam> in2; in2.resize(s);
    std::fill(in1.begin(), in1.end(), 1);
    std::fill(in2.begin(), in2.end(), 1);

    const auto d = cryptanalysislib::equal(in1.begin(), in1.end(), in2.begin());
    EXPECT_EQ(d, 0);
}

TYPED_TEST_P(Equal, multithreading) {
    constexpr static size_t s = 10000;
    std::vector<TypeParam> in1; in1.resize(s);
    std::vector<TypeParam> in2; in2.resize(s);
    std::fill(in1.begin(), in1.end(), 1);
    std::fill(in2.begin(), in2.end(), 1);

    const auto d = cryptanalysislib::equal(par_if(true), in1.begin(), in1.end(), in2.begin());
    EXPECT_EQ(d, 0);
}

REGISTER_TYPED_TEST_SUITE_P(Equal, simple, multithreading);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Equal, MyTypes);

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
