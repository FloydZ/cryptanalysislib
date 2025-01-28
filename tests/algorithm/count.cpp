#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/count.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

template <typename T>
class Count : public testing::Test {};

TYPED_TEST_SUITE_P(Count);

TYPED_TEST_P(Count, simple) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::count(in.begin(), in.end(), 1);
    EXPECT_EQ((size_t)d, s);
}

TYPED_TEST_P(Count, simd) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::internal::count_uXX_simd<TypeParam>(in.data(), s, 1);
    EXPECT_EQ((size_t)d, s);
}

TYPED_TEST_P(Count, multithreading) {
    constexpr static size_t s = 10000;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::count(par_if(true),in.begin(), in.end(), 1);
    EXPECT_EQ(d, s);
}


REGISTER_TYPED_TEST_SUITE_P(Count, simple, simd, multithreading);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Count, MyTypes);

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
