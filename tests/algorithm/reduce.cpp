#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/reduce.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using namespace cryptanalysislib;

template <typename T>
class Reduce : public testing::Test {};

TYPED_TEST_SUITE_P(Reduce);

TYPED_TEST_P(Reduce, simple) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::reduce(in.begin(), in.end(), 1);
    EXPECT_EQ((size_t)d, s+1);
}

TYPED_TEST_P(Reduce, multithreading) {
    constexpr static size_t s = 10000;
    using T = int;
    std::vector<T> in; in.resize(s);
    std::fill(in.begin(), in.end(), 1);

    const auto d = cryptanalysislib::reduce(par_if(true),in.begin(), in.end(), 1);
    EXPECT_EQ(d, s+1);
}

REGISTER_TYPED_TEST_SUITE_P(Reduce, simple, multithreading);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Reduce, MyTypes);

/// all `reduce` overloads against std::reduce, and the `transform`s of reduce.h
TEST(Reduce, overloads) {
	for (const size_t s : {size_t(0), size_t(1), size_t(13), size_t(1u << 21)}) {
		std::vector<uint64_t> in(s);
		for (size_t i = 0; i < s; ++i) { in[i] = (i * 2654435761u) & 0xffffu; }
		const auto op = [](const uint64_t a, const uint64_t b) { return a ^ b; };
		EXPECT_EQ(cryptanalysislib::reduce(in.begin(), in.end()), std::reduce(in.begin(), in.end()));
		EXPECT_EQ(cryptanalysislib::reduce(par_if(true), in.begin(), in.end()), std::reduce(in.begin(), in.end()));
		EXPECT_EQ(cryptanalysislib::reduce(in.begin(), in.end(), uint64_t(9), op), std::reduce(in.begin(), in.end(), uint64_t(9), op));
		EXPECT_EQ(cryptanalysislib::reduce(par_if(true), in.begin(), in.end(), uint64_t(9), op), std::reduce(in.begin(), in.end(), uint64_t(9), op));
	}
}

TEST(Reduce, transform) {
	std::vector<uint32_t> a = {1, 2, 3, 4}, b = {10, 20, 30, 40}, o(5, 0);
	EXPECT_EQ(cryptanalysislib::transform(a.begin(), a.end(), o.begin(), [](uint32_t &x) { return x * x; }), o.begin() + 4);
	EXPECT_EQ(o, (std::vector<uint32_t>{1, 4, 9, 16, 0}));
	EXPECT_EQ(cryptanalysislib::transform(a.begin(), a.end(), b.begin(), o.begin(), [](uint32_t &x, uint32_t &y) { return x + y; }), o.begin() + 4);
	EXPECT_EQ(o, (std::vector<uint32_t>{11, 22, 33, 44, 0}));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
