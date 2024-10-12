#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/exclusive_scan.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


using namespace cryptanalysislib;

TEST(exclusive_scan, int_) {
	constexpr size_t s = 10;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::vector<T> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 0);

    cryptanalysislib::exclusive_scan(in.begin(), in.end(), out.begin(), 0);

	T acc = 0;
    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(out[i], acc);
    	acc += in[i];
    }
}


TEST(exclusive_scan, int_multithreading) {
	constexpr size_t s = 1<<16;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::vector<T> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 0);

    cryptanalysislib::exclusive_scan(par_if(true), in.begin(), in.end(), out.begin(), 0);

	T acc = 0;
    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(out[i], acc);
    	acc += in[i];
    }
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
