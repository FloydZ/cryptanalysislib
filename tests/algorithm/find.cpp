#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/find.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(find, int_) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in; in.resize(s);
	std::fill(in.begin(), in.end(), 1);
    const auto r1 = cryptanalysislib::find(in.begin(), in.end(), 1);
	const auto t1 = std::distance(in.begin(), r1);
	EXPECT_EQ(t1, 0);

	const auto r2 = cryptanalysislib::find(in.begin(), in.end(), 0);
	const auto t2 = std::distance(in.begin(), r2);
	EXPECT_EQ(t2, s);
}

TEST(find, int_simd_) {
    constexpr static size_t s = 100;
    using T = uint32_t;
    std::vector<T> in; in.resize(s);
	std::fill(in.begin(), in.end(), 1);
	const auto t1 = cryptanalysislib::internal::find_uXX_simd<T>(in.data(), s, 1);
	EXPECT_EQ(t1, 0);

	const auto t2 = cryptanalysislib::internal::find_uXX_simd<T>(in.data(), s, 0);
	EXPECT_EQ(t2, s);
}

TEST(find, int_multithreading) {
    constexpr static size_t s = 10000;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto r1 = cryptanalysislib::find(par_if(true), in.begin(), in.end(), 1);
	const auto t1 = std::distance(in.begin(), r1);
	EXPECT_EQ(t1, 0);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
