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

template <typename T>
class Fill : public testing::Test {};

TYPED_TEST_SUITE_P(Fill);

TYPED_TEST_P(Fill, simple) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
    cryptanalysislib::fill(in.begin(), in.end(), 1);
    for(size_t i = 0; i < s; i++) {
    	EXPECT_EQ(in[i], 1);
    }
}

TYPED_TEST_P(Fill, multithreaded) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
    
    cryptanalysislib::fill(par_if(true),in.begin(), in.end(), 1);
    for(size_t i = 0; i < s; i++) {
    	EXPECT_EQ(in[i], 1);
    }
}


REGISTER_TYPED_TEST_SUITE_P(Fill, simple, multithreaded);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Fill, MyTypes);

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
