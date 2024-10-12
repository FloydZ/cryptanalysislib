#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "memory/memcmp.h"

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

    const auto d1 = cryptanalysislib::memcmp(in1.data(), in2.data(), s);
    EXPECT_EQ(d1, false);

    std::fill(in2.begin(), in2.end(), 0);
    const auto d2 = cryptanalysislib::memcmp(in1.data(), in2.data(), s);
    EXPECT_EQ(d2, true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
