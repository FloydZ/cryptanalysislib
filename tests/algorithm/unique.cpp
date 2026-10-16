#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <forward_list>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "random.h"
#include "algorithm/unique.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// `unique` against `std::unique`, few distinct values (long runs)
TEST(unique, random) {
	for (size_t n = 0; n <= 300; ++n) {
		for (const uint32_t mod : {1u, 2u, 5u, 1000u}) {
			std::vector<uint32_t> v(n), e;
			for (auto &x : v) { x = uint32_t(rng() % mod); }
			e = v;
			const auto r = cryptanalysislib::unique(v.begin(), v.end());
			const auto re = std::unique(e.begin(), e.end());
			ASSERT_EQ(r - v.begin(), re - e.begin()) << n << " " << mod;
			EXPECT_TRUE(std::equal(v.begin(), r, e.begin())) << n << " " << mod;
		}
	}
}

/// with a predicate, the first element of every run is kept
TEST(unique, predicate) {
	std::vector<int> v = {1, 3, 5, 2, 4, 7, 6, 8, 8, 9};
	const auto same_parity = [](const int a, const int b) { return (a % 2) == (b % 2); };
	const auto r = cryptanalysislib::unique(v.begin(), v.end(), same_parity);
	EXPECT_EQ(std::vector<int>(v.begin(), r), (std::vector<int>{1, 2, 7, 6, 9}));

	// move only elements and a forward list
	std::vector<std::unique_ptr<int>> u;
	for (const int x : {1, 1, 2, 2, 2, 3}) { u.emplace_back(std::make_unique<int>(x)); }
	const auto ru = cryptanalysislib::unique(u.begin(), u.end(), [](const auto &a, const auto &b) { return *a == *b; });
	ASSERT_EQ(ru - u.begin(), 3);
	EXPECT_EQ(*u[0], 1); EXPECT_EQ(*u[1], 2); EXPECT_EQ(*u[2], 3);

	std::forward_list<int> l = {4, 4, 1, 4, 4};
	const auto rl = cryptanalysislib::unique(l.begin(), l.end());
	EXPECT_EQ(std::distance(l.begin(), rl), 3);
}

/// `unique_copy` against `std::unique_copy`, also from an input iterator
TEST(unique, unique_copy) {
	for (size_t n = 0; n <= 300; ++n) {
		std::vector<uint32_t> v(n), out(n), e(n);
		for (auto &x : v) { x = uint32_t(rng() % 3u); }
		const auto r = cryptanalysislib::unique_copy(v.begin(), v.end(), out.begin());
		const auto re = std::unique_copy(v.begin(), v.end(), e.begin());
		ASSERT_EQ(r - out.begin(), re - e.begin()) << n;
		EXPECT_EQ(out, e) << n;
	}

	// the predicate compares with the last copied element, not the previous one
	std::vector<int> v = {1, 2, 3, 10, 11, 20}, out;
	cryptanalysislib::unique_copy(v.begin(), v.end(), std::back_inserter(out),
	                              [](const int a, const int b) { return b - a < 5; });
	EXPECT_EQ(out, (std::vector<int>{1, 10, 20}));

	std::istringstream in("a a b b b c a");
	std::vector<std::string> w;
	cryptanalysislib::unique_copy(std::istream_iterator<std::string>(in), std::istream_iterator<std::string>(),
	                              std::back_inserter(w));
	EXPECT_EQ(w, (std::vector<std::string>{"a", "b", "c", "a"}));
}

/// parallel `unique`/`unique_copy` against the `std::` versions: runs which
/// span several chunks (few distinct values, everything equal), an
/// equivalence predicate, uneven sizes and the sequential fallback
TEST(unique, multithreaded) {
	for (const size_t n : {size_t(0), size_t(1), size_t(1000), size_t(1u << 20u), size_t((1u << 20u) + 999u)}) {
		for (const uint32_t mod : {1u, 2u, 7u, 1u << 30u}) {
			std::vector<uint32_t> v(n);
			for (size_t i = 0; i < n; ++i) { v[i] = uint32_t(rng() % mod); }
			// long runs: repeat every value
			for (size_t i = 1; i < n; ++i) { if (rng() % 4u) { v[i] = v[i - 1]; } }

			std::vector<uint32_t> out(n), e(n);
			const auto rc = cryptanalysislib::unique_copy(par_if(true), v.begin(), v.end(), out.begin());
			const auto rec = std::unique_copy(v.begin(), v.end(), e.begin());
			ASSERT_EQ(rc - out.begin(), rec - e.begin()) << n << " " << mod;
			EXPECT_EQ(out, e) << n << " " << mod;

			auto w = v, we = v;
			const auto r = cryptanalysislib::unique(par_if(true), w.begin(), w.end());
			const auto re = std::unique(we.begin(), we.end());
			ASSERT_EQ(r - w.begin(), re - we.begin()) << n << " " << mod;
			EXPECT_TRUE(std::equal(w.begin(), r, we.begin())) << n << " " << mod;

			// equivalence predicate: same value / 1000
			const auto div = [](const uint32_t a, const uint32_t b) { return (a / 1000u) == (b / 1000u); };
			w = v; we = v;
			const auto rp = cryptanalysislib::unique(par_if(true), w.begin(), w.end(), div);
			const auto rep = std::unique(we.begin(), we.end(), div);
			ASSERT_EQ(rp - w.begin(), rep - we.begin()) << n << " " << mod;
			EXPECT_TRUE(std::equal(w.begin(), rp, we.begin())) << n << " " << mod;
			const auto rcp = cryptanalysislib::unique_copy(par_if(true), v.begin(), v.end(), out.begin(), div);
			const auto recp = std::unique_copy(v.begin(), v.end(), e.begin(), div);
			ASSERT_EQ(rcp - out.begin(), recp - e.begin()) << n << " " << mod;
			EXPECT_TRUE(std::equal(out.begin(), rcp, e.begin())) << n << " " << mod;
		}
	}

	// move only elements
	const size_t n = 1u << 17u;
	std::vector<std::unique_ptr<uint32_t>> u(n);
	for (size_t i = 0; i < n; ++i) { u[i] = std::make_unique<uint32_t>(uint32_t(i / 3u)); }
	const auto ru = cryptanalysislib::unique(par_if(true), u.begin(), u.end(), [](const auto &a, const auto &b) { return *a == *b; });
	ASSERT_EQ(size_t(ru - u.begin()), (n + 2u) / 3u);
	for (size_t i = 0; i < (n + 2u) / 3u; ++i) { ASSERT_EQ(*u[i], i) << i; }
}

/// both are usable in constant expressions
TEST(unique, constexpr_unique) {
	constexpr auto r = []() {
		int a[7] = {1, 1, 2, 3, 3, 3, 1}, b[7] = {};
		const auto e = cryptanalysislib::unique(a, a + 7);
		const auto f = cryptanalysislib::unique_copy(a, e, b);
		return int(e - a) * 100 + int(f - b) * 10 + b[3];
	}();
	static_assert(r == 441);
	EXPECT_EQ(r, 441);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
