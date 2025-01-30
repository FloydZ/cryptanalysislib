#include "gtest/gtest.h"
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


template <typename T>
class Max : public testing::Test {};

TYPED_TEST_SUITE_P(Max);

TYPED_TEST_P(Max, simple) {
	constexpr size_t s = 100;
	std::vector<TypeParam> d; d.resize(s);
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = cryptanalysislib::max(d.begin(), d.end());
	EXPECT_EQ(t, s-1);
}

TYPED_TEST_P(Max, simd) {
	constexpr size_t s = 100;
	auto d = new TypeParam [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = max_simd_uXX(d, s);
	EXPECT_EQ(t, s-1);

	delete[] d;
}

TYPED_TEST_P(Max, simd_rng) {
	constexpr size_t s = 100;
    std::vector<TypeParam> d; d.resize(s);
	for (size_t i = 0; i < s; ++i) { d[i] = rand(); }

	const auto t = max_simd_uXX(d.data(), s);
    for (const auto &k : d) {
        EXPECT_GE(t, k);
    }
}

TYPED_TEST_P(Max, multithreading) {
	constexpr size_t b = sizeof(TypeParam)*8u - 1u;
    constexpr static size_t s = 1u<<b;
    std::vector<TypeParam> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = s - i - 1; }

    const auto d = cryptanalysislib::max(par_if(true), in.begin(), in.end());
    EXPECT_EQ(d, s-1);
}

TYPED_TEST_P(Max, multithreading_rnd) {
    constexpr static size_t s = 1u<<20;
    std::vector<TypeParam> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = rand(); }

    const auto d = cryptanalysislib::max(par_if(true), in.begin(), in.end());
    for (const auto &k : in) {
        EXPECT_GE(d, k);
    }
}

REGISTER_TYPED_TEST_SUITE_P(Max, simple, simd, simd_rng, multithreading, multithreading_rnd);
using MyTypes = ::testing::Types<uint8_t>;//, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Max, MyTypes);




int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
