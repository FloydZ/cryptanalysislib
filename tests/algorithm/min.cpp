#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "random.h"
#include "algorithm/min.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;
using namespace cryptanalysislib;

TEST(min, simple) {
	constexpr size_t s = 100;
	using T = uint32_t;
	std::vector<T> d; d.resize(s);
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = cryptanalysislib::min(d.begin(), d.end());
	ASSERT_EQ(t, 0);
}

TEST(min, simd_uint32_t) {
	constexpr size_t s = 100;
	auto d = new uint32_t [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = min_simd_uXX(d, s);
	ASSERT_EQ(t, 0);

	delete[] d;
}

TEST(min, int_multithreading) {
    constexpr static size_t s = 100000;
    using T = uint32_t;
    std::vector<T> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = i; }

    const auto d = cryptanalysislib::min(par_if(true), in.begin(), in.end());
    EXPECT_EQ(d, 0);
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
