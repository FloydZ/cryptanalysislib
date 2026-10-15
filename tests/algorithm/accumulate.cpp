#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/accumulate.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


template <typename T>
class Reduce : public testing::Test {};

TYPED_TEST_SUITE_P(Reduce);

TYPED_TEST_P(Reduce, simple) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
    std::fill(in.begin(), in.end(), (TypeParam)1);

    const auto d = cryptanalysislib::accumulate(in.begin(), in.end(), 0);
    EXPECT_EQ((size_t)d, s);
}
TYPED_TEST_P(Reduce, simd) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::internal::accumulate_simd_int_plus<TypeParam>(in.data(), s, 0);
    EXPECT_EQ(d, s);
}

TYPED_TEST_P(Reduce, multithreading) {
    constexpr static size_t s = 10000;
    std::vector<TypeParam> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::accumulate(par_if(true), in.begin(), in.end(), 0);
    // NOTE: the sum is computed in `TypeParam` (unlike std::accumulate, which
    // uses the type of `init`), so for uint8_t it wraps around: 10000 mod 256
    EXPECT_EQ(d, (TypeParam)s);
}

REGISTER_TYPED_TEST_SUITE_P(Reduce, simple, simd, multithreading);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Reduce, MyTypes);


/// `accumulate` with a custom operation (not only `+`)
TEST(Accumulate, binary_op) {
	std::vector<uint32_t> v = {3, 5, 7, 11};
	EXPECT_EQ(cryptanalysislib::accumulate(v.begin(), v.end(), 1u, [](const uint32_t a, const uint32_t b) { return a * b; }), 1155u);
	EXPECT_EQ(cryptanalysislib::accumulate(v.begin(), v.end(), 0u, [](const uint32_t a, const uint32_t b) { return a ^ b; }), 3u ^ 5u ^ 7u ^ 11u);
	// empty range: init
	EXPECT_EQ(cryptanalysislib::accumulate(v.begin(), v.begin(), 42u, [](const uint32_t a, const uint32_t b) { return a * b; }), 42u);
	// not commutative: left fold
	EXPECT_EQ(cryptanalysislib::accumulate(v.begin(), v.end(), 100u, [](const uint32_t a, const uint32_t b) { return a - b; }), 100u - 26u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
