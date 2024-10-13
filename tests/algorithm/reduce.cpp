#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/reduce.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using namespace cryptanalysislib;

TEST(reduce, int_) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::reduce(in.begin(), in.end(), 1);
    EXPECT_EQ(d, s+1);
}

TEST(reduce, int_multithreading) {
    constexpr static size_t s = 10000;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::reduce(par_if(true),in.begin(), in.end(), 1);
    EXPECT_EQ(d, s+1);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
