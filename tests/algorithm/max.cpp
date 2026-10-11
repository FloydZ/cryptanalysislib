#include "gtest/gtest.h"
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "random.h"
#include "algorithm/max.h"

using ::testing::InitGoogleTest;
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

	const auto t = internal::max_simd_uXX(d, s);
	EXPECT_EQ(t, s-1);

	delete[] d;
}

TYPED_TEST_P(Max, simd_rng) {
	constexpr size_t s = 100;
    std::vector<TypeParam> d; d.resize(s);
	for (size_t i = 0; i < s; ++i) { d[i] = rand(); }

	const auto t = internal::max_simd_uXX(d.data(), s);
    for (const auto &k : d) {
        EXPECT_GE(t, k);
    }
}

TYPED_TEST_P(Max, multithreading) {
	// capped at 2^20: `1 << (bits-1)` elements would be 2^63 for uint64_t
	constexpr size_t b = (sizeof(TypeParam)*8u - 1u) < 20u ? (sizeof(TypeParam)*8u - 1u) : 20u;
    constexpr static size_t s = 1ull<<b;
    std::vector<TypeParam> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = s - i - 1; }

    const auto d = cryptanalysislib::max(par_if(true), in.begin(), in.end());
    EXPECT_EQ(d, s-1);
}

TYPED_TEST_P(Max, multithreading_rnd) {
    constexpr static size_t s = 1ull<<20;
    std::vector<TypeParam> in; in.resize(s);
	for (size_t i = 0; i < s; ++i) { in[i] = rand(); }

    const auto d = cryptanalysislib::max(par_if(true), in.begin(), in.end());
    for (const auto &k : in) {
        EXPECT_GE(d, k);
    }
}

TYPED_TEST_P(Max, scalar) {
	const TypeParam a = 3, b = 7;
	EXPECT_EQ(cryptanalysislib::max(a, b), b);
	EXPECT_EQ(cryptanalysislib::max(b, a), b);
	EXPECT_EQ(cryptanalysislib::max(a, a), a);
	// with std::max also visible the constrained overload is chosen (no ambiguity)
	using std::max;
	EXPECT_EQ(max(a, b), b);
}

/// max_element against std::max_element: sizes around the SIMD threshold,
/// random data with few distinct values, so the maximum occurs repeatedly
/// (the first occurrence must be returned)
TYPED_TEST_P(Max, max_element) {
	for (size_t s = 1; s <= 300; ++s) {
		std::vector<TypeParam> in(s);
		for (auto &x : in) { x = TypeParam(rng() % 7u); }
		const auto e = std::max_element(in.begin(), in.end());
		EXPECT_EQ(cryptanalysislib::max_element(in.begin(), in.end()), e) << s;
		EXPECT_EQ(cryptanalysislib::max_element(in.begin(), in.end(), std::less<TypeParam>()), e) << s;
	}
	std::vector<TypeParam> empty;
	EXPECT_EQ(cryptanalysislib::max_element(empty.begin(), empty.end()), empty.end());
}

/// parallel max_element against std::max_element, the maximum placed
/// (several times) in different chunks
TYPED_TEST_P(Max, max_element_multithreading) {
	constexpr size_t s = 1u << 20u;
	std::vector<TypeParam> in(s);
	for (auto &x : in) { x = TypeParam(rng() % 100u); }
	for (const size_t pos : {size_t(0), s / 3, s / 2 + 1, s - 1}) {
		std::vector<TypeParam> v = in;
		v[pos] = TypeParam(200);
		v[s - 1] = TypeParam(200); // later duplicate: the first one wins
		EXPECT_EQ(cryptanalysislib::max_element(par_if(true), v.begin(), v.end()),
		          std::max_element(v.begin(), v.end())) << pos;
	}
}

REGISTER_TYPED_TEST_SUITE_P(Max, simple, simd, simd_rng, multithreading, multithreading_rnd, scalar, max_element, max_element_multithreading);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Max, MyTypes);




int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
