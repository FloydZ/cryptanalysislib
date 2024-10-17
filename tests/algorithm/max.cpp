#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "random.h"
#include "algorithm/max.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;
using namespace cryptanalysislib;

TEST(max, simple) {
	constexpr size_t s = 100;
	using T = uint32_t;
	std::vector<T> d; d.resize(s);
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = cryptanalysislib::max(d.begin(), d.end());
	ASSERT_EQ(t, s-1);
}

TEST(max, simd_uint32_t) {
	constexpr size_t s = 100;
	auto d = new uint32_t [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = max_simd_uXX(d, s);
	ASSERT_EQ(t, s-1);

	delete[] d;
}

TEST(max, int_multithreading) {
    constexpr static size_t s = 100000;
    using T = uint32_t;
    std::vector<T> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = i; }

    const auto d = cryptanalysislib::max(par_if(true), in.begin(), in.end());
    EXPECT_EQ(d, s-1);
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
