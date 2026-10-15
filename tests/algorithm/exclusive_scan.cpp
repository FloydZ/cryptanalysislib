#include <cstdint>
#include <numeric>
#include <vector>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/exclusive_scan.h"

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
	constexpr size_t s = 10;
    std::vector<TypeParam> in; in.resize(s);
    std::vector<TypeParam> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 0);

    cryptanalysislib::exclusive_scan(in.begin(), in.end(), out.begin(), 0);

	TypeParam acc = 0;
    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(out[i], acc);
    	acc += in[i];
    }
}

TYPED_TEST_P(Reduce, multithreading) {
	constexpr size_t s = 1<<16;
    std::vector<TypeParam> in; in.resize(s);
    std::vector<TypeParam> out; out.resize(s);
    std::fill(in.begin(), in.end(), 1);
    std::fill(out.begin(), out.end(), 0);

    cryptanalysislib::exclusive_scan(par_if(true), in.begin(), in.end(), out.begin(), 0);

	TypeParam acc = 0;
    for (size_t i = 0; i < s; i++) {
        EXPECT_EQ(out[i], acc);
    	acc += in[i];
    }
}

/// all sizes around the SIMD block length, random data, against
/// `std::exclusive_scan`, out of place and inplace
TYPED_TEST_P(Reduce, simd_all_sizes) {
	for (size_t s = 0; s <= 300; ++s) {
		std::vector<TypeParam> in(s), out(s), e(s);
		for (auto &x : in) { x = TypeParam(rand()); }
		const TypeParam init = TypeParam(rand());
		std::exclusive_scan(in.begin(), in.end(), e.begin(), init);

		cryptanalysislib::exclusive_scan(in.begin(), in.end(), out.begin(), init);
		EXPECT_EQ(out, e) << s;

		cryptanalysislib::exclusive_scan(in.begin(), in.end(), in.begin(), init);
		EXPECT_EQ(in, e) << s;
	}
}

REGISTER_TYPED_TEST_SUITE_P(Reduce, simple, multithreading, simd_all_sizes);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Reduce, MyTypes);

/// all `exclusive_scan` overloads (op/no op, sequential/parallel) against std::
TEST(ExclusiveScan, overloads) {
	for (const size_t s : {size_t(0), size_t(1), size_t(13), size_t(1u << 20)}) {
		std::vector<uint64_t> in(s), e(s), o(s + 1, 0xdeadu);
		for (size_t i = 0; i < s; ++i) { in[i] = (i * 2654435761u) & 0xffu; }
		const auto op = [](const uint64_t a, const uint64_t b) { return a ^ (b * 3u); };

		std::exclusive_scan(in.begin(), in.end(), e.begin(), uint64_t(0));
		EXPECT_EQ(cryptanalysislib::exclusive_scan(in.begin(), in.end(), o.begin()), o.begin() + s);
		for (size_t i = 0; i < s; ++i) { ASSERT_EQ(o[i], e[i]) << i; }
		EXPECT_EQ(o[s], 0xdeadu);
		EXPECT_EQ(cryptanalysislib::exclusive_scan(par_if(true), in.begin(), in.end(), o.begin()), o.begin() + s);
		for (size_t i = 0; i < s; ++i) { ASSERT_EQ(o[i], e[i]) << i; }

		std::exclusive_scan(in.begin(), in.end(), e.begin(), uint64_t(7), op);
		EXPECT_EQ(cryptanalysislib::exclusive_scan(in.begin(), in.end(), o.begin(), uint64_t(7), op), o.begin() + s);
		for (size_t i = 0; i < s; ++i) { ASSERT_EQ(o[i], e[i]) << i; }
		// `op` is associative here (xor of a multiple): any chunking gives the same result
		const auto xop = [](const uint64_t a, const uint64_t b) { return a ^ b; };
		std::exclusive_scan(in.begin(), in.end(), e.begin(), uint64_t(7), xop);
		EXPECT_EQ(cryptanalysislib::exclusive_scan(par_if(true), in.begin(), in.end(), o.begin(), uint64_t(7), xop), o.begin() + s);
		for (size_t i = 0; i < s; ++i) { ASSERT_EQ(o[i], e[i]) << i; }
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
