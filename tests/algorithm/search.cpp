#include <algorithm>
#include <cstdint>
#include <algorithm>
#include <vector>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/search.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


template <typename T>
class Search : public testing::Test {};

TYPED_TEST_SUITE_P(Search);

TYPED_TEST_P(Search, simple) {
    constexpr static size_t s = 100;
    std::vector<TypeParam> in; in.resize(s);
	std::fill(in.begin(), in.end(), 1);
	const TypeParam ones[3] = {1, 1, 1};
	const TypeParam zero[1] = {0};
    const auto r1 = cryptanalysislib::search(in.begin(), in.end(), ones, ones + 3);
	EXPECT_EQ(r1 - in.begin(), 0);

	const auto r2 = cryptanalysislib::search(in.begin(), in.end(), zero, zero + 1);
	EXPECT_EQ((size_t)(r2 - in.begin()), s);

	in[s-1] = 0;
	const auto r3 = cryptanalysislib::search(in.begin(), in.end(), zero, zero + 1,
	    [](const TypeParam a, const TypeParam b) { return a == b; });
	EXPECT_EQ((size_t)(r3 - in.begin()), s-1);
}

TYPED_TEST_P(Search, simple_n) {
    constexpr static size_t n = 10;
    constexpr static size_t s = 1000;
    std::vector<TypeParam> in; in.resize(s);
	std::fill(in.begin(), in.end() - n, 2);
	std::fill(in.begin()+s-n, in.end(), 1);
    
    const auto r1 = cryptanalysislib::search_n(in.begin(), in.end(), n, 1);
	EXPECT_EQ((size_t)(r1 - in.begin()), s - n);

	const auto r2 = cryptanalysislib::search_n(in.begin(), in.end(), n, 0);
	EXPECT_EQ((size_t)(r2 - in.begin()), s);
}

//TYPED_TEST_P(Search, multithreading) {
//    constexpr static size_t s = 10000;
//    std::vector<TypeParam> in; in.resize(s);
//    std::fill(in.begin(), in.end(), 1);
//
//    const auto r1 = cryptanalysislib::search(par_if(true), in.begin(), in.end(), 1);
//	const auto t1 = std::distance(in.begin(), r1);
//	EXPECT_EQ(t1, 0);
//}

/// search_n (SIMD path) against std::search_n: few distinct values, so runs
/// of all lengths occur, also crossing block boundaries
TYPED_TEST_P(Search, search_n_random) {
	for (size_t len = 0; len <= 200; ++len) {
		std::vector<TypeParam> in(len);
		for (auto &x : in) { x = TypeParam(rand() % 3 == 0 ? 7 : 1); }
		for (const size_t count : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(5), size_t(9), size_t(17), size_t(40)}) {
			const auto e = std::search_n(in.begin(), in.end(), count, TypeParam(1));
			const auto r = cryptanalysislib::search_n(in.begin(), in.end(), count, TypeParam(1));
			ASSERT_EQ(r - in.begin(), e - in.begin()) << len << " " << count;
		}
	}
	// a long run crossing several blocks
	std::vector<TypeParam> in(1000, 0);
	std::fill(in.begin() + 333, in.begin() + 333 + 300, TypeParam(1));
	EXPECT_EQ(cryptanalysislib::search_n(in.begin(), in.end(), 300, TypeParam(1)) - in.begin(), 333);
	EXPECT_EQ(cryptanalysislib::search_n(in.begin(), in.end(), 301, TypeParam(1)), in.end());
}

REGISTER_TYPED_TEST_SUITE_P(Search, simple, simple_n, search_n_random);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, Search, MyTypes);


/// SIMD kernel and predicate overloads of search_n against std::search_n
TEST(Search, search_n_simd_and_predicate) {
	for (const size_t s : {size_t(0), size_t(1), size_t(33), size_t(1000), size_t(1u << 21)}) {
		std::vector<uint32_t> v(s);
		for (size_t i = 0; i < s; ++i) { v[i] = (i % 50u) < 7u ? 1u : (uint32_t)(i & 0xffu) | 2u; }
		if (s > 100) { for (size_t i = s - 30; i < s - 10; ++i) { v[i] = 1u; } }
		for (const size_t count : {size_t(0), size_t(1), size_t(7), size_t(8), size_t(20), size_t(21)}) {
			const auto e = std::search_n(v.begin(), v.end(), count, 1u);
			EXPECT_EQ(cryptanalysislib::internal::search_n_uXX_simd<uint32_t>(v.data(), s, count, 1u),
			          size_t(e - v.begin())) << s << " " << count;
			// predicate: equal modulo 2 (every odd value is a match of 1)
			const auto p = [](const uint32_t a, const uint32_t b) { return (a & 1u) == (b & 1u); };
			const auto ep = std::search_n(v.begin(), v.end(), count, 1u, p);
			EXPECT_EQ(cryptanalysislib::search_n(v.begin(), v.end(), count, 1u, p), ep) << s << " " << count;
			EXPECT_EQ(cryptanalysislib::search_n(par_if(true), v.begin(), v.end(), count, 1u, p), ep) << s << " " << count;
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
