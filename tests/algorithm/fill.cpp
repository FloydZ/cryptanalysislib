#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/fill.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(fill, int_) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in; in.resize(s);
    cryptanalysislib::fill(in.begin(), in.end(), 1);
    for(size_t i = 0; i < s; i++) {
    	EXPECT_EQ(in[i], 1);
    }
}

TEST(fill, int_multithreading) {
    constexpr static size_t s = 10000;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    cryptanalysislib::fill(par_if(true),in.begin(), in.end(), 1);
    for(size_t i = 0; i < s; i++) {
    	EXPECT_EQ(in[i], 1);
    }
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
