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


/// parallel find_first_of / adjacent_find against std:: on random data,
/// with the (first) match placed in different chunks
TEST(Find, find_first_of_adjacent_find_parallel) {
	constexpr size_t s = 1u << 22; // above `min_size_per_thread`
	std::vector<uint32_t> in(s);
	for (size_t i = 0; i < s; ++i) { in[i] = 2u * (uint32_t)i + 1u; } // no equal neighbours
	const std::vector<uint32_t> cand = {0u, 4u, 8u};

	EXPECT_EQ(cryptanalysislib::find_first_of(par_if(true), in.begin(), in.end(), cand.begin(), cand.end()), in.end());
	EXPECT_EQ(cryptanalysislib::adjacent_find(par_if(true), in.begin(), in.end()), in.end());

	for (const size_t pos : {size_t(0), size_t(1), s / 3, s / 2 + 7, s - 2, s - 1}) {
		std::vector<uint32_t> v = in;
		v[pos] = 8u;
		if (pos + 1 < s) { v[pos + 1] = 8u; }
		auto e1 = std::find_first_of(v.begin(), v.end(), cand.begin(), cand.end());
		EXPECT_EQ(cryptanalysislib::find_first_of(par_if(true), v.begin(), v.end(), cand.begin(), cand.end()), e1) << pos;
		EXPECT_EQ(cryptanalysislib::find_first_of(v.begin(), v.end(), cand.begin(), cand.end()), e1) << pos;
		auto e2 = std::adjacent_find(v.begin(), v.end());
		EXPECT_EQ(cryptanalysislib::adjacent_find(par_if(true), v.begin(), v.end()), e2) << pos;
		EXPECT_EQ(cryptanalysislib::adjacent_find(v.begin(), v.end()), e2) << pos;
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
