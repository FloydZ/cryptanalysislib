#include <iterator>
#include <utility>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "random.h"
#include "algorithm/merge.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// merge against std::merge for many size combinations, with few distinct
/// values (duplicates across both inputs) and with disjoint value ranges
template<typename T>
static void check(const size_t n1, const size_t n2, const uint32_t mod) {
	std::vector<T> a(n1), b(n2), out(n1 + n2), e(n1 + n2);
	for (auto &x : a) { x = T(rng() % mod); }
	for (auto &x : b) { x = T(rng() % mod); }
	std::sort(a.begin(), a.end());
	std::sort(b.begin(), b.end());
	std::merge(a.begin(), a.end(), b.begin(), b.end(), e.begin());
	const auto it = cryptanalysislib::merge(a.begin(), a.end(), b.begin(), b.end(), out.begin());
	EXPECT_EQ(it, out.end());
	EXPECT_EQ(out, e) << n1 << " " << n2 << " " << mod;
}

TEST(merge, uint32_all_sizes) {
	for (size_t n1 = 0; n1 <= 40; ++n1) {
		for (size_t n2 = 0; n2 <= 40; ++n2) {
			check<uint32_t>(n1, n2, 5);
			check<uint32_t>(n1, n2, 1u << 30);
		}
	}
	check<uint32_t>(10000, 7, 1u << 30);
	check<uint32_t>(3, 10000, 1u << 30);
	check<uint32_t>(12345, 23456, 100);
	check<uint32_t>(12345, 23456, 0xffffffffu);
}

TEST(merge, other_types) {
	for (size_t n = 0; n <= 50; ++n) {
		check<uint8_t>(n, 50 - n, 7);
		check<uint64_t>(n, 2 * n, 1u << 30);
	}
}

TEST(merge, comparator) {
	std::vector<uint32_t> a = {9, 5, 1}, b = {8, 2}, out(5);
	cryptanalysislib::merge(a.begin(), a.end(), b.begin(), b.end(), out.begin(), std::greater<uint32_t>());
	EXPECT_EQ(out, (std::vector<uint32_t>{9, 8, 5, 2, 1}));
}

/// parallel merge against std::merge: duplicates, disjoint ranges, skewed sizes
TEST(merge, multithreading) {
	struct Case { size_t n1, n2; uint32_t mod; bool disjoint; };
	for (const Case c : {Case{1u << 20, 1u << 20, 7, false},
	                     Case{1u << 20, 1u << 20, 1u << 30, false},
	                     Case{1u << 20, 1u << 18, 1u << 30, true},
	                     Case{1u << 21, 5, 1u << 30, false},
	                     Case{3, 1u << 21, 1u << 30, false}}) {
		std::vector<uint32_t> a(c.n1), b(c.n2), out(c.n1 + c.n2), e(c.n1 + c.n2);
		for (auto &x : a) { x = uint32_t(rng() % c.mod); }
		for (auto &x : b) { x = uint32_t(rng() % c.mod) + (c.disjoint ? c.mod : 0u); }
		std::sort(a.begin(), a.end());
		std::sort(b.begin(), b.end());
		std::merge(a.begin(), a.end(), b.begin(), b.end(), e.begin());
		const auto it = cryptanalysislib::merge(par_if(true), a.begin(), a.end(), b.begin(), b.end(), out.begin());
		EXPECT_EQ(it, out.end());
		EXPECT_EQ(out, e) << c.n1 << " " << c.n2;
	}
}

/// stability: equal keys, elements of the first range must come first
TEST(merge, multithreading_stable) {
	constexpr size_t n = 1u << 19;
	std::vector<std::pair<uint32_t, uint32_t>> a(n), b(n), out(2 * n), e(2 * n);
	for (size_t i = 0; i < n; ++i) { a[i] = {uint32_t(i / 1000), 0}; b[i] = {uint32_t(i / 1000), 1}; }
	const auto comp = [](const auto &x, const auto &y) { return x.first < y.first; };
	std::merge(a.begin(), a.end(), b.begin(), b.end(), e.begin(), comp);
	cryptanalysislib::merge(par_if(true), a.begin(), a.end(), b.begin(), b.end(), out.begin(), comp);
	EXPECT_EQ(out, e);
}

/// co-rank: number of elements of `a` among the first d of the stable merge
TEST(merge, co_rank) {
	for (uint32_t it = 0; it < 200; ++it) {
		std::vector<uint32_t> a(rng<uint32_t>() % 50), b(rng<uint32_t>() % 50);
		for (auto &x : a) { x = rng<uint32_t>() % 10; }
		for (auto &x : b) { x = rng<uint32_t>() % 10; }
		std::sort(a.begin(), a.end());
		std::sort(b.begin(), b.end());
		// tag the elements of `a` to count them in the stable merge
		std::vector<std::pair<uint32_t, uint32_t>> ta, tb, m;
		for (const auto x : a) { ta.emplace_back(x, 0); }
		for (const auto x : b) { tb.emplace_back(x, 1); }
		std::merge(ta.begin(), ta.end(), tb.begin(), tb.end(), std::back_inserter(m),
		           [](const auto &x, const auto &y) { return x.first < y.first; });
		size_t ca = 0;
		for (size_t d = 0; d <= m.size(); ++d) {
			EXPECT_EQ(cryptanalysislib::internal::merge_co_rank(d, a.begin(), a.size(), b.begin(), b.size(), std::less<>()), ca) << d;
			if (d < m.size()) { ca += m[d].second == 0; }
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
