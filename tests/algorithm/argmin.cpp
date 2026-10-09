#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "random.h"
#include "algorithm/argmin.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;
using namespace cryptanalysislib;

template <typename T>
class ArgMin : public testing::Test {};

TYPED_TEST_SUITE_P(ArgMin);

TYPED_TEST_P(ArgMin, simple) {
    // keep all values distinct (no wrap-around for uint8_t)
    constexpr static size_t s = sizeof(TypeParam) == 1 ? 255 : 10000;
    std::vector<TypeParam> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = i; }

    const auto d = cryptanalysislib::argmin(in.begin(), in.end());
    EXPECT_EQ(d, 0);
}

TYPED_TEST_P(ArgMin, multithreading) {
    // keep all values distinct (no wrap-around for uint8_t)
    constexpr static size_t s = sizeof(TypeParam) == 1 ? 255 : 10000;
    std::vector<TypeParam> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = i; }

    const auto d = cryptanalysislib::argmin(par_if(true), in.begin(), in.end());
    EXPECT_EQ(d, 0);
}

REGISTER_TYPED_TEST_SUITE_P(ArgMin, simple, multithreading);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, ArgMin, MyTypes);

TEST(argmin, simd_uint32_t) {
	constexpr size_t s = 100;
	auto d = new uint32_t [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = internal::argmin_simd(d, s);
	EXPECT_EQ(t, 0u);


	for (size_t i = 0; i < s; ++i) { d[i] = rng(1, 38475983); }
    const size_t pos = rng(s);
    d[pos] = 0;
    const size_t pos2 = internal::argmin_simd(d, s);
	EXPECT_EQ(pos, pos2);

	delete[] d;
}

TEST(argmin, simd_uint32_t_bl16) {
	constexpr size_t s = 100;
	auto d = new uint32_t [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = internal::argmin_simd_bl16(d, s);
	EXPECT_EQ(t, 0u);

	for (size_t i = 0; i < s; ++i) { d[i] = rng(1, 38475983); }
    const size_t pos = rng(s);
    d[pos] = 0;
    const size_t pos2 = internal::argmin_simd(d, s);
	EXPECT_EQ(pos, pos2);

	delete[] d;
}

TEST(argmin, simd_uint32_t_bl32) {
	constexpr size_t s = 100;
	auto d = new uint32_t [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = internal::argmin_simd_bl32(d, s);
	EXPECT_EQ(t, 0);

	for (size_t i = 0; i < s; ++i) { d[i] = rng(1, 38475983); }
    const size_t pos = rng(s);
    d[pos] = 0;
    const size_t pos2 = internal::argmin_simd(d, s);
	EXPECT_EQ(pos, pos2);

	delete[] d;
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
