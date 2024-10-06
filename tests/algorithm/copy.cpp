#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/copy.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(copy, simple) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::vector<T> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 2);


    cryptanalysislib::copy(in.begin(), in.end(), out.begin());

    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(in[i], out[i]);
    }
}


TEST(copy, multithreaded_single) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::vector<T> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 2);


    cryptanalysislib::copy(par_if(false), in.begin(), in.end(), out.begin());

    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(in[i], out[i]);
    }
}

TEST(copy, multithreaded_twothreads) {
    constexpr static size_t s = algorithmCopyConfig.min_size_per_thread*2;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::vector<T> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 2);


    cryptanalysislib::copy(par_if(true), in.begin(), in.end(), out.begin());

    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(in[i], out[i]);
    }
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
