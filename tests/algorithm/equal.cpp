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


TEST(equal, int_) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in1; in1.resize(s);
    std::vector<T> in2; in2.resize(s);
    std::fill(in1.begin(), in1.end(), 1);
    std::fill(in2.begin(), in2.end(), 1);

    const auto d = cryptanalysislib::equal(in1.begin(), in1.end(), in2.begin());
    EXPECT_EQ(d, 0);
}

TEST(equal, int_multithreading) {
    constexpr static size_t s = 10000;
    using T = int;
    std::vector<T> in1; in1.resize(s);
    std::vector<T> in2; in2.resize(s);
    std::fill(in1.begin(), in1.end(), 1);
    std::fill(in2.begin(), in2.end(), 1);

    const auto d = cryptanalysislib::equal(par_if(true), in1.begin(), in1.end(), in2.begin());
    EXPECT_EQ(d, 0);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
