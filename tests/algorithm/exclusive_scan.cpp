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


template <typename T>
class Reduce : public testing::Test {};

TYPED_TEST_SUITE_P(Reduce);

TYPED_TEST_P(Reduce, simple) {
	constexpr size_t s = 10;
    std::vector<TypeParam> in; in.resize(s);
    std::vector<TypeParam> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 0);

    cryptanalysislib::exclusive_scan(in.begin(), in.end(), out.begin(), 0);

	TypeParam acc = 0;
    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(out[i], acc);
    	acc += in[i];
    }
}

TYPED_TEST_P(Reduce, multithreading) {
	constexpr size_t s = 1<<16;
    std::vector<TypeParam> in; in.resize(s);
    std::vector<TypeParam> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 0);

    cryptanalysislib::exclusive_scan(par_if(true), in.begin(), in.end(), out.begin(), 0);

	TypeParam acc = 0;
    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(out[i], acc);
    	acc += in[i];
    }
}

REGISTER_TYPED_TEST_SUITE_P(Reduce, simple, multithreading);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Reduce, MyTypes);

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
