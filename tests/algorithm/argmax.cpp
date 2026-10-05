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

template <typename T>
class ArgMax : public testing::Test {};

TYPED_TEST_SUITE_P(ArgMax);

TYPED_TEST_P(ArgMax, simple) {
    constexpr static size_t s = 10000;
    std::vector<TypeParam> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = i; }

    const auto d = cryptanalysislib::argmax(in.begin(), in.end());
    EXPECT_EQ(d, s-1);
}

TYPED_TEST_P(ArgMax, multithreading) {
    constexpr static size_t s = 10000;
    std::vector<TypeParam> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = i; }

    const auto d = cryptanalysislib::argmax(par_if(true), in.begin(), in.end());
    EXPECT_EQ(d, s-1);
}

REGISTER_TYPED_TEST_SUITE_P(ArgMax, simple, multithreading);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, ArgMax, MyTypes);


TEST(argmax, simd_uint32_t_bl16) {
	constexpr size_t s = 100;
	auto d = new uint32_t [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = argmax_simd_u32_bl16(d, s);
	EXPECT_EQ(t, s-1);

	for (size_t i = 0; i < s; ++i) { d[i] = rng(); }
    const size_t pos = rng(s);
    d[pos] = -1u;
    const size_t pos2 = argmax_simd_u32(d, s);
	EXPECT_EQ(pos, pos2);

	delete[] d;
}

TEST(argmax, simd_uint32_t_bl32) {
	constexpr size_t s = 100;
	auto d = new uint32_t [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = argmax_simd_u32_bl32(d, s);
	EXPECT_EQ(t, s-1);

	for (size_t i = 0; i < s; ++i) { d[i] = rng(); }
    const size_t pos = rng(s);
    d[pos] = -1u;
    const size_t pos2 = argmax_simd_u32(d, s);
	EXPECT_EQ(pos, pos2);

	delete[] d;
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
