#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/accumulate.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(accumulate, int_) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::accumulate(in.begin(), in.end(), 0);
    EXPECT_EQ(d, s);
}

TEST(accumulate, int_simd_plus) {
    constexpr static size_t s = 100;
    using T = uint8_t;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::internal::accumulate_simd_int_plus<T>(in.data(), s, 0);
    EXPECT_EQ(d, s);
}

TEST(accumulate, int_multithreading) {
    constexpr static size_t s = 10000;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::accumulate(par_if(true),in.begin(), in.end(), 0);
    EXPECT_EQ(d, s);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
