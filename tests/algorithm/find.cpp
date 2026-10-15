#include <algorithm>
#include <cstdint>
#include <list>
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

/// predicate overloads (sequential and parallel) against std::
TEST(Find, predicate_overloads) {
	for (const size_t s : {size_t(0), size_t(1), size_t(17), size_t(1000), size_t(1u << 20)}) {
		std::vector<uint32_t> v(s);
		for (size_t i = 0; i < s; ++i) { v[i] = (uint32_t)(i % 97u); }
		for (const uint32_t t : {0u, 5u, 96u, 200u}) {
			const auto eq = [t](const uint32_t x) { return x == t; };
			const auto ge = [t](const uint32_t x) { return x >= t; };
			const auto lt = [t](const uint32_t x) { return x < t; };

			EXPECT_EQ(cryptanalysislib::find_if(v.begin(), v.end(), eq), std::find_if(v.begin(), v.end(), eq));
			EXPECT_EQ(cryptanalysislib::find_if(par_if(true), v.begin(), v.end(), ge), std::find_if(v.begin(), v.end(), ge));
			EXPECT_EQ(cryptanalysislib::find_if_not(v.begin(), v.end(), lt), std::find_if_not(v.begin(), v.end(), lt));
			EXPECT_EQ(cryptanalysislib::find_if_not(par_if(true), v.begin(), v.end(), lt), std::find_if_not(v.begin(), v.end(), lt));
		}
	}
}

TEST(Find, find_end) {
	const std::vector<uint32_t> v = {1, 2, 3, 1, 2, 3, 1, 2, 4};
	for (const std::vector<uint32_t> &n : std::vector<std::vector<uint32_t>>{{1, 2}, {1, 2, 3}, {2, 4}, {4}, {5}, {}, {1, 2, 3, 1, 2, 3, 1, 2, 4, 5}}) {
		EXPECT_EQ(cryptanalysislib::find_end(v.begin(), v.end(), n.begin(), n.end()),
		          std::find_end(v.begin(), v.end(), n.begin(), n.end()));
		// the predicate version, compares modulo 3
		const auto p = [](const uint32_t a, const uint32_t b) { return a % 3u == b % 3u; };
		EXPECT_EQ(cryptanalysislib::find_end(v.begin(), v.end(), n.begin(), n.end(), p),
		          std::find_end(v.begin(), v.end(), n.begin(), n.end(), p));
	}
}

TEST(Find, find_first_of_adjacent_find_predicate) {
	const auto p = [](const uint32_t a, const uint32_t b) { return a == b + 1u; };
	for (const size_t s : {size_t(0), size_t(1), size_t(2), size_t(100), size_t(1u << 21)}) {
		std::vector<uint32_t> v(s);
		for (size_t i = 0; i < s; ++i) { v[i] = 2u * (uint32_t)i; } // never v[i] == v[i+1] + 1
		for (const size_t pos : {size_t(0), s / 2, s ? s - 1 : 0}) {
			std::vector<uint32_t> w = v;
			if (pos + 1 < s) { w[pos] = 7u; w[pos + 1] = 6u; }
			const std::vector<uint32_t> cand = {6u, 1000u};
			EXPECT_EQ(cryptanalysislib::find_first_of(w.begin(), w.end(), cand.begin(), cand.end(), p),
			          std::find_first_of(w.begin(), w.end(), cand.begin(), cand.end(), p)) << s << " " << pos;
			EXPECT_EQ(cryptanalysislib::find_first_of(par_if(true), w.begin(), w.end(), cand.begin(), cand.end(), p),
			          std::find_first_of(w.begin(), w.end(), cand.begin(), cand.end(), p)) << s << " " << pos;
			EXPECT_EQ(cryptanalysislib::adjacent_find(w.begin(), w.end(), p),
			          std::adjacent_find(w.begin(), w.end(), p)) << s << " " << pos;
			EXPECT_EQ(cryptanalysislib::adjacent_find(par_if(true), w.begin(), w.end(), p),
			          std::adjacent_find(w.begin(), w.end(), p)) << s << " " << pos;
		}
	}
}

/// non-contiguous and empty ranges (the SIMD kernel read a list as an array)
TEST(Find, list_and_empty) {
	const std::list<uint32_t> l = {1, 2, 3, 1, 1, 5};
	const auto it = cryptanalysislib::find(l.begin(), l.end(), 5u);
	ASSERT_NE(it, l.end());
	EXPECT_EQ(*it, 5u);
	EXPECT_EQ(std::distance(l.begin(), it), 5);
	EXPECT_EQ(cryptanalysislib::find(l.begin(), l.end(), 4u), l.end());
	std::vector<uint32_t> e;
	EXPECT_EQ(cryptanalysislib::find(e.begin(), e.end(), 1u), e.end());
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
