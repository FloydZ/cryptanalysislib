#include <algorithm>
#include <cstdint>
#include <list>
#include <vector>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/count.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

template <typename T>
class Count : public testing::Test {};

TYPED_TEST_SUITE_P(Count);

TYPED_TEST_P(Count, simple) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::count(in.begin(), in.end(), 1);
    EXPECT_EQ((size_t)d, s);
}

TYPED_TEST_P(Count, simd) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::internal::count_uXX_simd<TypeParam>(in.data(), s, 1);
    EXPECT_EQ((size_t)d, s);
}

TYPED_TEST_P(Count, multithreading) {
    constexpr static size_t s = 10000;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::count(par_if(true),in.begin(), in.end(), 1);
    EXPECT_EQ(d, s);
}


REGISTER_TYPED_TEST_SUITE_P(Count, simple, simd, multithreading);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Count, MyTypes);

/// `count_if`, sequential and parallel, against std::count_if
TEST(Count, count_if) {
	for (const size_t s : {size_t(0), size_t(1), size_t(33), size_t(1u << 20)}) {
		std::vector<uint32_t> v(s);
		for (size_t i = 0; i < s; ++i) { v[i] = (uint32_t)(i * 2654435761u); }
		const auto odd = [](const uint32_t x) { return (x & 1u) != 0; };
		const auto e = std::count_if(v.begin(), v.end(), odd);
		EXPECT_EQ(cryptanalysislib::count_if(v.begin(), v.end(), odd), e) << s;
		EXPECT_EQ(cryptanalysislib::count_if(par_if(true), v.begin(), v.end(), odd), e) << s;
		EXPECT_EQ(cryptanalysislib::count_if(par_if(false), v.begin(), v.end(), odd), e) << s;
	}
}

/// non-contiguous and empty ranges (the SIMD kernel read a list as an array)
TEST(Count, list_and_empty) {
	const std::list<uint32_t> l = {1, 2, 3, 1, 1, 5};
	EXPECT_EQ(cryptanalysislib::count(l.begin(), l.end(), 1u), 3);
	EXPECT_EQ(cryptanalysislib::count(l.begin(), l.end(), 4u), 0);
	std::vector<uint32_t> e;
	EXPECT_EQ(cryptanalysislib::count(e.begin(), e.end(), 1u), 0);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
