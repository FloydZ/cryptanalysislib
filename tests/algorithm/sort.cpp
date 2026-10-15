#include <algorithm>
#include <vector>
#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <list>
#include <vector>

#include "random.h"
#include "algorithm/sort.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

TEST(sort, is_sorted_until) {
	for (size_t n = 0; n <= 100; ++n) {
		std::vector<uint32_t> v(n);
		for (size_t i = 0; i < n; ++i) { v[i] = uint32_t(i / 3); }
		if (n > 5) { v[n / 2] = 0; } // first unsorted position (if any)
		EXPECT_EQ(cryptanalysislib::is_sorted_until(v.begin(), v.end()), std::is_sorted_until(v.begin(), v.end())) << n;
		EXPECT_EQ(cryptanalysislib::is_sorted_until(v.begin(), v.end(), std::greater<>()),
		          std::is_sorted_until(v.begin(), v.end(), std::greater<>())) << n;
		EXPECT_EQ(cryptanalysislib::is_sorted(v.begin(), v.end()), std::is_sorted(v.begin(), v.end())) << n;
	}
	const std::list<int> l = {1, 2, 2, 5};
	EXPECT_TRUE(cryptanalysislib::is_sorted(l.begin(), l.end()));
}

TEST(sort, partial_sort) {
	for (size_t n = 0; n <= 200; n += 7) {
		for (size_t k = 0; k <= n; k += 3) {
			std::vector<uint32_t> v(n);
			for (auto &x : v) { x = uint32_t(rng() % 50u); }
			auto e = v;
			cryptanalysislib::partial_sort(v.begin(), v.begin() + k, v.end());
			std::partial_sort(e.begin(), e.begin() + k, e.end());
			EXPECT_TRUE(std::equal(v.begin(), v.begin() + k, e.begin())) << n << " " << k;

			std::sort(v.begin(), v.end()); std::sort(e.begin(), e.end());
			EXPECT_EQ(v, e) << "not a permutation " << n << " " << k;
		}
	}
}

/// sort by an extracted key (ska_sort) and partial_sort with a comparator
TEST(sort, extract_key_and_comparator) {
	struct P { uint32_t key, payload; };
	std::vector<P> v(1000);
	for (size_t i = 0; i < v.size(); ++i) { v[i] = P{(uint32_t)(i * 2654435761u) % 997u, (uint32_t)i}; }
	cryptanalysislib::sort(v.begin(), v.end(), [](const P &p) { return p.key; });
	EXPECT_TRUE(std::is_sorted(v.begin(), v.end(), [](const P &a, const P &b) { return a.key < b.key; }));

	std::vector<uint32_t> w(500), e;
	for (size_t i = 0; i < w.size(); ++i) { w[i] = (uint32_t)(i * 2654435761u); }
	e = w;
	cryptanalysislib::partial_sort(w.begin(), w.begin() + 37, w.end(), std::greater<>());
	std::sort(e.begin(), e.end(), std::greater<>());
	EXPECT_TRUE(std::equal(w.begin(), w.begin() + 37, e.begin()));
	EXPECT_TRUE(cryptanalysislib::is_sorted(w.begin(), w.begin() + 37, std::greater<>()));
	EXPECT_FALSE(cryptanalysislib::is_sorted(w.begin(), w.end(), std::greater<>()));
}

/// `stable_sort` against `std::stable_sort`: equal keys keep their order
TEST(sort, stable_sort) {
	struct P { uint32_t key, payload; };
	const auto by_key = [](const P &a, const P &b) { return a.key < b.key; };
	for (size_t n = 0; n <= 3000; n += (n < 100) ? 1 : 997) {
		std::vector<P> v(n);
		for (size_t i = 0; i < n; ++i) { v[i] = P{uint32_t(rng() % 10u), uint32_t(i)}; }
		auto e = v;
		cryptanalysislib::stable_sort(v.begin(), v.end(), by_key);
		std::stable_sort(e.begin(), e.end(), by_key);
		for (size_t i = 0; i < n; ++i) {
			ASSERT_EQ(v[i].key, e[i].key) << n << " " << i;
			ASSERT_EQ(v[i].payload, e[i].payload) << n << " " << i;
		}
	}

	std::vector<uint32_t> w(10000);
	for (auto &x : w) { x = uint32_t(rng()); }
	cryptanalysislib::stable_sort(w.begin(), w.end());
	EXPECT_TRUE(std::is_sorted(w.begin(), w.end()));
}

/// `partial_sort_copy` against `std::partial_sort_copy`: output smaller,
/// equal and larger than the input, also from a `std::list`
TEST(sort, partial_sort_copy) {
	for (size_t n = 0; n <= 150; n += 7) {
		for (size_t k = 0; k <= 160; k += 9) {
			std::vector<int32_t> in(n), out(k, -1), e(k, -1);
			for (auto &x : in) { x = int32_t(rng() % 50u) - 25; }
			const auto r = cryptanalysislib::partial_sort_copy(in.begin(), in.end(), out.begin(), out.end());
			const auto re = std::partial_sort_copy(in.begin(), in.end(), e.begin(), e.end());
			ASSERT_EQ(r - out.begin(), re - e.begin()) << n << " " << k;
			EXPECT_EQ(out, e) << n << " " << k;

			std::fill(out.begin(), out.end(), -1); std::fill(e.begin(), e.end(), -1);
			const std::list<int32_t> l(in.begin(), in.end());
			cryptanalysislib::partial_sort_copy(l.begin(), l.end(), out.begin(), out.end(), std::greater<>());
			std::partial_sort_copy(in.begin(), in.end(), e.begin(), e.end(), std::greater<>());
			EXPECT_EQ(out, e) << n << " " << k;
		}
	}
}

/// `nth_element`: every position for small sizes, few distinct values,
/// sorted/reversed inputs (pivot worst cases) and a comparator
TEST(sort, nth_element) {
	const auto check = [](std::vector<uint32_t> v, const size_t k, const char *what) {
		auto e = v;
		std::sort(e.begin(), e.end());
		cryptanalysislib::nth_element(v.begin(), v.begin() + k, v.end());
		if (k < v.size()) {
			ASSERT_EQ(v[k], e[k]) << what << " " << v.size() << " " << k;
			for (size_t i = 0; i < k; ++i) { ASSERT_LE(v[i], v[k]) << what << " " << i; }
			for (size_t i = k + 1; i < v.size(); ++i) { ASSERT_GE(v[i], v[k]) << what << " " << i; }
		}
		std::sort(v.begin(), v.end());
		ASSERT_EQ(v, e) << "not a permutation " << what << " " << v.size() << " " << k;
	};

	for (size_t n = 0; n <= 70; ++n) {
		for (size_t k = 0; k <= n; ++k) {
			std::vector<uint32_t> v(n);
			for (auto &x : v) { x = uint32_t(rng() % 1000u); }
			check(v, k, "random");
			for (auto &x : v) { x = uint32_t(rng() % 3u); }
			check(v, k, "few");
		}
	}

	for (const size_t n : {size_t(1000), size_t(65537)}) {
		std::vector<uint32_t> v(n);
		for (size_t i = 0; i < n; ++i) { v[i] = uint32_t(i); }
		for (const size_t k : {size_t(0), n / 3, n - 1}) {
			check(v, k, "sorted");
			std::vector<uint32_t> r(v.rbegin(), v.rend());
			check(r, k, "reversed");
			std::vector<uint32_t> c(n, 5);
			check(c, k, "constant");
			std::vector<uint32_t> o(n);
			for (size_t i = 0; i < n; ++i) { o[i] = uint32_t((i & 1u) ? i : n - i); } // organ pipe like
			check(o, k, "alternating");
		}
	}

	std::vector<int> w = {5, 1, 4, 2, 3};
	cryptanalysislib::nth_element(w.begin(), w.begin() + 1, w.end(), std::greater<>());
	EXPECT_EQ(w[1], 4);
}

/// parallel `stable_sort` against `std::stable_sort`: stability (tagged
/// elements, few distinct keys), `<` on uint32_t (SIMD merge), thread
/// counts which are not a power of two (odd number of runs), small inputs
TEST(sort, stable_sort_multithreaded) {
	struct P { uint32_t key, payload; };
	const auto by_key = [](const P &a, const P &b) { return a.key < b.key; };
	for (const size_t n : {size_t(0), size_t(1000), size_t(1u << 16u) + 3u, size_t(5u << 16u), size_t(1u << 20u) + 333u}) {
		std::vector<P> v(n);
		for (size_t i = 0; i < n; ++i) { v[i] = P{uint32_t(rng() % 1000u), uint32_t(i)}; }
		auto e = v;
		cryptanalysislib::stable_sort(par_if(true), v.begin(), v.end(), by_key);
		std::stable_sort(e.begin(), e.end(), by_key);
		for (size_t i = 0; i < n; ++i) {
			ASSERT_EQ(v[i].key, e[i].key) << n << " " << i;
			ASSERT_EQ(v[i].payload, e[i].payload) << n << " " << i;
		}

		std::vector<uint32_t> w(n);
		for (auto &x : w) { x = uint32_t(rng()); }
		auto we = w;
		cryptanalysislib::stable_sort(par_if(true), w.begin(), w.end());
		std::sort(we.begin(), we.end());
		EXPECT_EQ(w, we) << n;

		cryptanalysislib::stable_sort(par_if(true), w.begin(), w.end(), std::greater<>());
		std::sort(we.begin(), we.end(), std::greater<>());
		EXPECT_EQ(w, we) << n;
	}
}

/// parallel `partial_sort_copy` against `std::partial_sort_copy`: output
/// sizes from 0 to larger than the input
TEST(sort, partial_sort_copy_multithreaded) {
	const size_t n = (1u << 20u) + 77u;
	std::vector<int32_t> in(n);
	for (auto &x : in) { x = int32_t(rng() % 100000u) - 50000; }
	for (const size_t k : {size_t(0), size_t(1), size_t(100), size_t(5000), n / 2, n, n + 5}) {
		std::vector<int32_t> out(k, -1), e(k, -1);
		const auto r = cryptanalysislib::partial_sort_copy(par_if(true), in.begin(), in.end(), out.begin(), out.end());
		const auto re = std::partial_sort_copy(in.begin(), in.end(), e.begin(), e.end());
		ASSERT_EQ(r - out.begin(), re - e.begin()) << k;
		EXPECT_EQ(out, e) << k;
	}

	std::vector<int32_t> out(1000), e(1000);
	cryptanalysislib::partial_sort_copy(par_if(true), in.begin(), in.end(), out.begin(), out.end(), std::greater<>());
	std::partial_sort_copy(in.begin(), in.end(), e.begin(), e.end(), std::greater<>());
	EXPECT_EQ(out, e);
}

/// parallel `nth_element`: random, few distinct, sorted, reversed and
/// constant inputs, several positions
TEST(sort, nth_element_multithreaded) {
	const size_t n = (1u << 20u) + 5u;
	const auto check = [](std::vector<uint32_t> v, const size_t k, const char *what) {
		auto e = v;
		std::sort(e.begin(), e.end());
		cryptanalysislib::nth_element(par_if(true), v.begin(), v.begin() + k, v.end());
		ASSERT_EQ(v[k], e[k]) << what << " " << k;
		for (size_t i = 0; i < k; ++i) { ASSERT_LE(v[i], v[k]) << what << " " << i; }
		for (size_t i = k + 1; i < v.size(); ++i) { ASSERT_GE(v[i], v[k]) << what << " " << i; }
		std::sort(v.begin(), v.end());
		ASSERT_EQ(v, e) << "not a permutation " << what << " " << k;
	};

	std::vector<uint32_t> v(n);
	for (const size_t k : {size_t(0), size_t(12345), n / 2, n - 1}) {
		for (auto &x : v) { x = uint32_t(rng()); }
		check(v, k, "random");
		for (auto &x : v) { x = uint32_t(rng() % 5u); }
		check(v, k, "few");
		for (size_t i = 0; i < n; ++i) { v[i] = uint32_t(i); }
		check(v, k, "sorted");
		for (size_t i = 0; i < n; ++i) { v[i] = uint32_t(n - i); }
		check(v, k, "reversed");
		for (auto &x : v) { x = 7u; }
		check(v, k, "constant");
	}

	std::vector<int> w(n);
	for (auto &x : w) { x = int(rng() % 1000u); }
	auto we = w;
	cryptanalysislib::nth_element(par_if(true), w.begin(), w.begin() + 10, w.end(), std::greater<>());
	std::sort(we.begin(), we.end(), std::greater<>());
	EXPECT_EQ(w[10], we[10]);

	std::vector<int> s = {5, 1, 4};
	cryptanalysislib::nth_element(par_if(true), s.begin(), s.begin() + 1, s.end());
	EXPECT_EQ(s[1], 4);
}

/// usable in constant expressions
TEST(sort, constexpr_nth_partial_sort_copy) {
	constexpr auto r = []() {
		int a[20] = {}, b[3] = {};
		for (int i = 0; i < 20; ++i) { a[i] = (i * 7) % 20; }
		cryptanalysislib::partial_sort_copy(a, a + 20, b, b + 3);
		cryptanalysislib::nth_element(a, a + 13, a + 20);
		return a[13] * 1000 + b[0] * 100 + b[1] * 10 + b[2];
	}();
	static_assert(r == 13012);
	EXPECT_EQ(r, 13012);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
