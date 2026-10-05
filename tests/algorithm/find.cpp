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


template <typename T>
class Find : public testing::Test {};

TYPED_TEST_SUITE_P(Find);

TYPED_TEST_P(Find, simple) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
	std::fill(in.begin(), in.end(), 1);
    const auto r1 = cryptanalysislib::find(in.begin(), in.end(), 1);
	const auto t1 = std::distance(in.begin(), r1);
	EXPECT_EQ(t1, 0);

	const auto r2 = cryptanalysislib::find(in.begin(), in.end(), 0);
	const auto t2 = std::distance(in.begin(), r2);
	EXPECT_EQ((size_t)t2, s);
}

TYPED_TEST_P(Find, simd) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
	std::fill(in.begin(), in.end(), 1);
	const auto t1 = cryptanalysislib::internal::find_uXX_simd<TypeParam>(in.data(), s, 1);
	EXPECT_EQ(t1, (TypeParam)0);

	const auto t2 = cryptanalysislib::internal::find_uXX_simd<TypeParam>(in.data(), s, 0);
	EXPECT_EQ(t2, s);
}

TYPED_TEST_P(Find, multithreading) {
    constexpr static size_t s = 10000;
    std::vector<TypeParam> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto r1 = cryptanalysislib::find(par_if(true), in.begin(), in.end(), 1);
	const auto t1 = std::distance(in.begin(), r1);
	EXPECT_EQ(t1, 0);
}

REGISTER_TYPED_TEST_SUITE_P(Find, simple, simd, multithreading);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Find, MyTypes);


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
