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
    constexpr static size_t s = 10000;
    std::vector<TypeParam> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = i; }

    const auto d = cryptanalysislib::argmin(in.begin(), in.end());
    EXPECT_EQ(d, s-1);
}

TYPED_TEST_P(ArgMin, multithreading) {
    constexpr static size_t s = 10000;
    std::vector<TypeParam> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = i; }

    const auto d = cryptanalysislib::argmin(par_if(true), in.begin(), in.end());
    EXPECT_EQ(d, s-1);
}

REGISTER_TYPED_TEST_SUITE_P(ArgMin, simple, multithreading);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, ArgMin, MyTypes);

TEST(argmin, simd_uint32_t) {
	constexpr size_t s = 100;
	auto d = new uint32_t [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = argmin_simd_u32(d, s);
	EXPECT_EQ(t, 0u);


	for (size_t i = 0; i < s; ++i) { d[i] = rng(1, 38475983); }
    const size_t pos = rng(s);
    d[pos] = 0;
    const size_t pos2 = argmin_simd_u32(d, s);
	EXPECT_EQ(pos, pos2);

	delete[] d;
}

TEST(argmin, simd_uint32_t_bl16) {
	constexpr size_t s = 100;
	auto d = new uint32_t [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = argmin_simd_u32_bl16(d, s);
	EXPECT_EQ(t, 0u);

	for (size_t i = 0; i < s; ++i) { d[i] = rng(1, 38475983); }
    const size_t pos = rng(s);
    d[pos] = 0;
    const size_t pos2 = argmin_simd_u32(d, s);
	EXPECT_EQ(pos, pos2);

	delete[] d;
}

TEST(argmin, simd_uint32_t_bl32) {
	constexpr size_t s = 100;
	auto d = new uint32_t [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = argmin_simd_u32_bl32(d, s);
	EXPECT_EQ(t, 0);

	for (size_t i = 0; i < s; ++i) { d[i] = rng(1, 38475983); }
    const size_t pos = rng(s);
    d[pos] = 0;
    const size_t pos2 = argmin_simd_u32(d, s);
	EXPECT_EQ(pos, pos2);

	delete[] d;
}

#ifdef USE_AVX2
TEST(argmin, avx2_int32_t) {
	constexpr size_t s = 100;
	auto d = new int32_t [s];
	for (size_t i = 1; i < s; ++i) { d[i] = i; }
	d[0] = 2;

	const auto t = argmin_avx2_i32(d, s);
	EXPECT_EQ(t, 1);

	delete[] d;
}
TEST(argmin, avx2_bl16_int32_t) {
	constexpr size_t s = 100;
	auto d = new int32_t [s];
	for (size_t i = 1; i < s; ++i) { d[i] = i; }
	d[0] = 2;

	const auto t = argmin_avx2_i32_bl16(d, s);
	EXPECT_EQ(t, 1);

	delete[] d;
}
TEST(argmin, avx2_bl32_uint32_t) {
	constexpr size_t s = 100;
	auto d = new int32_t [s];
	for (size_t i = 1; i < s; ++i) { d[i] = i; }
	d[0] = 2;

	const auto t = argmin_avx2_i32_bl32(d, s);
	EXPECT_EQ(t, 1);

	delete[] d;
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
