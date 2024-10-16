#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "random.h"
#include "algorithm/argmax.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;
using namespace cryptanalysislib;


TEST(argmax, simd_uint32_t) {
	constexpr size_t s = 100;
	auto d = new uint32_t [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = argmax_simd_u32(d, s);
	ASSERT_EQ(t, s-1);


	for (size_t i = 0; i < s; ++i) { d[i] = rng(); }
    const size_t pos = rng(s);
    d[pos] = -1u;
    const size_t pos2 = argmax_simd_u32(d, s);
	ASSERT_EQ(pos, pos2);

	delete[] d;
}

TEST(argmax, simd_uint32_t_bl16) {
	constexpr size_t s = 100;
	auto d = new uint32_t [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = argmax_simd_u32_bl16(d, s);
	ASSERT_EQ(t, s-1);

	for (size_t i = 0; i < s; ++i) { d[i] = rng(); }
    const size_t pos = rng(s);
    d[pos] = -1u;
    const size_t pos2 = argmax_simd_u32(d, s);
	ASSERT_EQ(pos, pos2);

	delete[] d;
}

TEST(argmax, simd_uint32_t_bl32) {
	constexpr size_t s = 100;
	auto d = new uint32_t [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = argmax_simd_u32_bl32(d, s);
	ASSERT_EQ(t, s-1);

	for (size_t i = 0; i < s; ++i) { d[i] = rng(); }
    const size_t pos = rng(s);
    d[pos] = -1u;
    const size_t pos2 = argmax_simd_u32(d, s);
	ASSERT_EQ(pos, pos2);

	delete[] d;
}


TEST(argmax, int_multithreading) {
    constexpr static size_t s = 10000;
    using T = int;
    std::vector<T> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = i; }

    const auto d = cryptanalysislib::argmax(par_if(true), in.begin(), in.end());
    EXPECT_EQ(d, s-1);
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
