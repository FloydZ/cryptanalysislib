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

template <typename T>
class Min : public testing::Test {};

TYPED_TEST_SUITE_P(Min);

TYPED_TEST_P(Min, simple) {
	constexpr size_t s = 100;
	std::vector<TypeParam> d; d.resize(s);
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = cryptanalysislib::min(d.begin(), d.end());
	EXPECT_EQ(t, 0);
}

TYPED_TEST_P(Min, simd) {
	constexpr size_t s = 100;
	auto d = new TypeParam [s];
	for (size_t i = 0; i < s; ++i) { d[i] = i; }

	const auto t = internal::min_simd_uXX(d, s);
	EXPECT_EQ(t, 0);

	delete[] d;
}

TYPED_TEST_P(Min, simd_rng) {
	constexpr size_t s = 100;
    std::vector<TypeParam> d; d.resize(s);
	for (size_t i = 0; i < s; ++i) { d[i] = rand(); }

	const auto t = internal::min_simd_uXX(d.data(), s);
    for (const auto &k : d) {
        EXPECT_LE(t, k);
    }
}

TYPED_TEST_P(Min, multithreading) {
	// capped at 2^20: `1 << (bits-1)` elements would be 2^63 for uint64_t
	constexpr size_t b = (sizeof(TypeParam)*8u - 1u) < 20u ? (sizeof(TypeParam)*8u - 1u) : 20u;
    constexpr static size_t s = 1ull<<b;
    std::vector<TypeParam> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = s - i - 1; }

    const auto d = cryptanalysislib::min(par_if(true), in.begin(), in.end());
    EXPECT_EQ(d, 0);
}

TYPED_TEST_P(Min, multithreading_rnd) {
    constexpr static size_t s = 1u<<20;
    std::vector<TypeParam> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = rand(); }

    const auto d = cryptanalysislib::min(par_if(true), in.begin(), in.end());
    for (const auto &k : in) {
        EXPECT_LE(d, k);
    }
}

TYPED_TEST_P(Min, scalar) {
	const TypeParam a = 3, b = 7;
	EXPECT_EQ(cryptanalysislib::min(a, b), a);
	EXPECT_EQ(cryptanalysislib::min(b, a), a);
	EXPECT_EQ(cryptanalysislib::min(a, a), a);
	// with std::min also visible the constrained overload is chosen (no ambiguity)
	using std::min;
	EXPECT_EQ(min(a, b), a);
}

REGISTER_TYPED_TEST_SUITE_P(Min, simple, simd, simd_rng, multithreading, multithreading_rnd, scalar);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Min, MyTypes);


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
