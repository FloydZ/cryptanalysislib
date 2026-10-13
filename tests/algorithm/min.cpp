#include <cstdint>
#include <algorithm>
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

/// min_element against std::min_element: sizes around the SIMD threshold,
/// few distinct values, so the minimum occurs repeatedly (first occurrence)
TYPED_TEST_P(Min, min_element) {
	for (size_t s = 1; s <= 300; ++s) {
		std::vector<TypeParam> in(s);
		for (auto &x : in) { x = TypeParam(rng() % 7u); }
		const auto e = std::min_element(in.begin(), in.end());
		EXPECT_EQ(cryptanalysislib::min_element(in.begin(), in.end()), e) << s;
		EXPECT_EQ(cryptanalysislib::min_element(in.begin(), in.end(), std::less<TypeParam>()), e) << s;
	}
	std::vector<TypeParam> empty;
	EXPECT_EQ(cryptanalysislib::min_element(empty.begin(), empty.end()), empty.end());
}

/// parallel min_element against std::min_element, the minimum placed
/// (several times) in different chunks
TYPED_TEST_P(Min, min_element_multithreading) {
	constexpr size_t s = 1u << 20u;
	std::vector<TypeParam> in(s);
	for (auto &x : in) { x = TypeParam(rng() % 100u + 10u); }
	for (const size_t pos : {size_t(0), s / 3, s / 2 + 1, s - 1}) {
		std::vector<TypeParam> v = in;
		v[pos] = TypeParam(1);
		v[s - 1] = TypeParam(1); // later duplicate: the first one wins
		EXPECT_EQ(cryptanalysislib::min_element(par_if(true), v.begin(), v.end()),
		          std::min_element(v.begin(), v.end())) << pos;
	}
}

REGISTER_TYPED_TEST_SUITE_P(Min, simple, simd, simd_rng, multithreading, multithreading_rnd, scalar, min_element, min_element_multithreading);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Min, MyTypes);


/// branchless min of non negative values (msb clear)
TEST(min, upos_min) {
	for (uint32_t i = 0; i < 100000; ++i) {
		const uint32_t a = rng<uint32_t>() >> 1, b = rng<uint32_t>() >> 1;
		ASSERT_EQ(cryptanalysislib::internal::upos_min<uint32_t>(a, b), std::min(a, b)) << a << " " << b;
		const uint64_t c = rng<uint64_t>() >> 1, d = rng<uint64_t>() >> 1;
		ASSERT_EQ(cryptanalysislib::internal::upos_min<uint64_t>(c, d), std::min(c, d)) << c << " " << d;
		const uint8_t e = rng<uint8_t>() >> 1, f = rng<uint8_t>() >> 1;
		ASSERT_EQ(cryptanalysislib::internal::upos_min<uint8_t>(e, f), std::min(e, f));
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
