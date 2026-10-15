#include <iterator>
#include <utility>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <list>
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

/// `inplace_merge` against `std::inplace_merge`: all splits of small ranges,
/// few distinct values, and stability (tagged elements)
TEST(merge, inplace_merge) {
	for (size_t n = 0; n <= 60; ++n) {
		for (size_t k = 0; k <= n; ++k) {
			for (const uint32_t mod : {3u, 1000u}) {
				std::vector<std::pair<uint32_t, uint32_t>> v(n);
				for (size_t i = 0; i < n; ++i) { v[i] = {uint32_t(rng() % mod), uint32_t(i)}; }
				const auto comp = [](const auto &x, const auto &y) { return x.first < y.first; };
				std::sort(v.begin(), v.begin() + k, comp);
				std::sort(v.begin() + k, v.end(), comp);
				auto e = v;
				cryptanalysislib::inplace_merge(v.begin(), v.begin() + k, v.end(), comp);
				std::inplace_merge(e.begin(), e.begin() + k, e.end(), comp);
				ASSERT_EQ(v, e) << n << " " << k << " " << mod;
			}
		}
	}

	// large and skewed
	for (const size_t k : {size_t(1), size_t(1000), size_t(50000), size_t(99999)}) {
		std::vector<uint32_t> v(100000);
		for (auto &x : v) { x = uint32_t(rng()); }
		std::sort(v.begin(), v.begin() + k);
		std::sort(v.begin() + k, v.end());
		cryptanalysislib::inplace_merge(v.begin(), v.begin() + k, v.end());
		EXPECT_TRUE(std::is_sorted(v.begin(), v.end())) << k;
	}

	// bidirectional iterators and a comparator
	std::list<int> l = {9, 5, 1, 8, 4, 2};
	cryptanalysislib::inplace_merge(l.begin(), std::next(l.begin(), 3), l.end(), std::greater<>());
	EXPECT_EQ(l, (std::list<int>{9, 8, 5, 4, 2, 1}));
}

/// parallel `inplace_merge` against `std::inplace_merge`: stability (tagged
/// elements, few distinct keys), skewed splits, `<` on uint32_t (SIMD
/// merge) and the sequential fallback
TEST(merge, inplace_merge_multithreaded) {
	const size_t n = (1u << 20u) + 13u;
	for (const size_t k : {size_t(0), size_t(1), size_t(1000), n / 3, n - 1, n}) {
		std::vector<std::pair<uint32_t, uint32_t>> v(n);
		for (size_t i = 0; i < n; ++i) { v[i] = {uint32_t(rng() % 100u), uint32_t(i)}; }
		const auto comp = [](const auto &x, const auto &y) { return x.first < y.first; };
		std::sort(v.begin(), v.begin() + k, comp);
		std::sort(v.begin() + k, v.end(), comp);
		auto e = v;
		cryptanalysislib::inplace_merge(par_if(true), v.begin(), v.begin() + k, v.end(), comp);
		std::inplace_merge(e.begin(), e.begin() + k, e.end(), comp);
		ASSERT_EQ(v, e) << k;

		std::vector<uint32_t> w(n);
		for (auto &x : w) { x = uint32_t(rng()); }
		std::sort(w.begin(), w.begin() + k);
		std::sort(w.begin() + k, w.end());
		auto we = w;
		cryptanalysislib::inplace_merge(par_if(true), w.begin(), w.begin() + k, w.end());
		std::inplace_merge(we.begin(), we.begin() + k, we.end());
		ASSERT_EQ(w, we) << k;
	}

	std::vector<int> s = {1, 4, 6, 2, 3, 5};
	cryptanalysislib::inplace_merge(par_if(true), s.begin(), s.begin() + 3, s.end());
	EXPECT_EQ(s, (std::vector<int>{1, 2, 3, 4, 5, 6}));
}

/// usable in constant expressions
TEST(merge, constexpr_inplace_merge) {
	constexpr auto r = []() {
		int a[7] = {1, 4, 6, 2, 3, 5, 7};
		cryptanalysislib::inplace_merge(a, a + 3, a + 7);
		int s = 0;
		for (const int x : a) { s = s * 10 + x; }
		return s;
	}();
	static_assert(r == 1234567);
	EXPECT_EQ(r, 1234567);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
