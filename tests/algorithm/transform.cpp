#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/transform.h"

#include "algorithm/accumulate.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using namespace cryptanalysislib;

TEST(transform, simple) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::vector<T> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 2);

    cryptanalysislib::transform(in.begin(), in.end(), out.begin(), [](T &a) {
    	return a;
    });

    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(in[i], out[i]);
    }
}

TEST(transform_reduce, simple) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in1; in1.resize(s);
    std::vector<T> in2; in2.resize(s);
    std::fill(in1.begin(), in1.end(), 1);
    std::fill(in2.begin(), in2.end(), 1);

    const auto a =cryptanalysislib::transform_reduce(in1.begin(), in1.end(),
    									in2.begin(), 0,
    									std::plus<T>(),
    									std::multiplies<T>());
	EXPECT_EQ(a, cryptanalysislib::accumulate(in1.begin(), in1.end(), 0));
}



TEST(transform_reduce, multithreaded_single) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in1; in1.resize(s);
    std::vector<T> in2; in2.resize(s);
    std::fill(in1.begin(), in1.end(), 1);
    std::fill(in2.begin(), in2.end(), 1);

    const auto a = cryptanalysislib::transform_reduce(par_if(false), in1.begin(), in1.end(), in2.begin(),
    									0,
    									std::plus<T>(),
    									std::multiplies<T>());
	EXPECT_EQ(a, cryptanalysislib::accumulate(in1.begin(), in1.end(), 0));
}

TEST(transform, multithreaded_twothreads) {
    constexpr static size_t s = 1000000;
    using T = int;
    std::vector<T> in1; in1.resize(s);
    std::vector<T> in2; in2.resize(s);
    std::fill(in1.begin(), in1.end(), 1);
    std::fill(in2.begin(), in2.end(), 1);


    const auto a = cryptanalysislib::transform_reduce(par_if(false), in1.begin(), in1.end(), in2.begin(),
    									0,
    									std::plus<T>(),
    									std::multiplies<T>());
	EXPECT_EQ(a, cryptanalysislib::accumulate(in1.begin(), in1.end(), 0));
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
