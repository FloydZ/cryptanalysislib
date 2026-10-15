#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
#include <gtest/gtest.h>

#include "random.h"
#include "search/search.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;

/// sorted random arrays with duplicates, of all small sizes and some large ones
static std::vector<std::vector<uint64_t>> inputs() {
	std::vector<std::vector<uint64_t>> r;
	for (const size_t n : {size_t(0), size_t(1), size_t(2), size_t(3), size_t(7), size_t(64), size_t(1000)}) {
		for (const uint64_t mod : {4ull, 1000ull, 1ull << 40}) {
			std::vector<uint64_t> v(n);
			for (auto &x : v) { x = rng<uint64_t>() % mod; }
			std::sort(v.begin(), v.end());
			r.push_back(v);
		}
	}
	return r;
}

/// some keys: all elements, their neighbours, the extremes
static std::vector<uint64_t> keys(const std::vector<uint64_t> &v) {
	std::vector<uint64_t> k = {0, 1, 2, 3, ~0ull};
	for (const auto x : v) { k.push_back(x); k.push_back(x + 1); if (x) { k.push_back(x - 1); } }
	return k;
}

static size_t first_eq(const std::vector<uint64_t> &v, const uint64_t key) {
	const auto it = std::lower_bound(v.begin(), v.end(), key);
	return (it != v.end() && *it == key) ? size_t(it - v.begin()) : v.size();
}

static int cmp(const uint64_t &a, const uint64_t &b) { return (a > b) - (a < b); }

/// the FXT style searches (`bsearch*`, `idx_bsearch*`) against std::
TEST(search, fxt_bsearch) {
	for (const auto &v : inputs()) {
		const size_t n = v.size();
		std::vector<uint64_t> rv(v.rbegin(), v.rend());
		// the identity and a reversed index permutation of a reversed copy
		std::vector<uint64_t> id(n), rid(n);
		for (size_t i = 0; i < n; ++i) { id[i] = i; rid[i] = n - 1 - i; }
		for (const uint64_t k : keys(v)) {
			const size_t e = first_eq(v, k);
			const size_t g = size_t(std::lower_bound(v.begin(), v.end(), k) - v.begin());
			// first element <= k in the descending array
			const size_t l = size_t(std::lower_bound(rv.begin(), rv.end(), k, std::greater<>()) - rv.begin());
			EXPECT_EQ(bsearch(v.data(), n, k), e) << n << " " << k;
			EXPECT_EQ(bsearch(v.data(), n, k, cmp), e) << n << " " << k;
			EXPECT_EQ(bsearch_geq(v.data(), n, k), g) << n << " " << k;
			EXPECT_EQ(bsearch_geq(v.data(), n, k, cmp), g) << n << " " << k;
			EXPECT_EQ(bsearch_leq(rv.data(), n, k), l) << n << " " << k;
			EXPECT_EQ(bsearch_leq(rv.data(), (uint64_t)n, k, cmp), l) << n << " " << k;
			EXPECT_EQ(idx_bsearch(v.data(), n, id.data(), k), e) << n << " " << k;
			EXPECT_EQ(idx_bsearch(v.data(), n, id.data(), k, cmp), e) << n << " " << k;
			EXPECT_EQ(idx_bsearch_geq(v.data(), n, id.data(), k), g) << n << " " << k;
			EXPECT_EQ(idx_bsearch_geq(v.data(), n, id.data(), k, cmp), g) << n << " " << k;
			// f[x[i]] sorted although f itself is not: the reversed array with
			// the reversed index permutation
			EXPECT_EQ(idx_bsearch(rv.data(), n, rid.data(), k), e) << n << " " << k;
			EXPECT_EQ(idx_bsearch_geq(rv.data(), n, rid.data(), k), g) << n << " " << k;
		}
	}
}

TEST(search, bsearch_approx) {
	const std::vector<double> v = {0.5, 1.0, 1.25, 3.0, 7.5};
	EXPECT_EQ(bsearch_approx(v.data(), v.size(), 1.1, 0.2), 1u);
	EXPECT_EQ(bsearch_approx(v.data(), v.size(), 2.0, 0.1), v.size());
	EXPECT_EQ(bsearch_approx(v.data(), v.size(), 7.6, 0.2), 4u);
	EXPECT_EQ(bsearch_approx(v.data(), v.size(), 0.0, 0.5), 0u);
	EXPECT_EQ(bsearch_approx(v.data(), size_t(0), 1.0, 0.5), 0u);
	const auto dcmp = [](const double &a, const double &b) { return (a > b) - (a < b); };
	EXPECT_EQ(bsearch_approx<double>(v.data(), v.size(), 1.1, 0.2, +dcmp), 1u);
	EXPECT_EQ(bsearch_approx<double>(v.data(), v.size(), 2.0, 0.1, +dcmp), v.size());
}

/// the array searches returning the index of an equal element or -1
TEST(search, array_searches) {
	for (const auto &v0 : inputs()) {
		// unique keys: the returned index of these searches is any equal element
		std::vector<uint64_t> v = v0;
		v.erase(std::unique(v.begin(), v.end()), v.end());
		const size_t n = v.size();
		for (const uint64_t k : keys(v)) {
			const size_t e = first_eq(v, k);
			const size_t ee = e == n ? size_t(-1) : e;
			EXPECT_EQ(standard_binary_search(v.data(), n, k), ee) << n << " " << k;
			EXPECT_EQ(boundless_binary_search(v.data(), n, k), ee) << n << " " << k;
			EXPECT_EQ(doubletapped_binary_search(v.data(), n, k), ee) << n << " " << k;
			EXPECT_EQ(Khuong_bin_search(v.data(), n, k), ee) << n << " " << k;
			EXPECT_EQ(breaking_linear_search(v.data(), (uint64_t)n, k), (uint64_t)ee) << n << " " << k;
			EXPECT_EQ(internal::linear_search_scalar(v.data(), n, k), e) << n << " " << k;
		}
	}
}

/// Eytzinger layout (1-indexed, t[0] is a sentinel): returns the value of the
/// lower bound (the sentinel if there is none)
TEST(search, lower_bound_eytzinger_prefetch) {
	for (const size_t n : {size_t(1), size_t(2), size_t(5), size_t(100), size_t(1000)}) {
		std::vector<uint64_t> a(n);
		for (size_t i = 0; i < n; ++i) { a[i] = 2 * i + 1; }
		// big enough for the prefetches (16 * k) to stay in bounds
		std::vector<uint64_t> t(16 * (n + 1) + 64, 0);
		size_t idx = 0;
		auto build = [&](auto &&self, size_t k) -> void {
			if (k <= n) { self(self, 2 * k); t[k] = a[idx++]; self(self, 2 * k + 1); }
		};
		build(build, 1);
		t[0] = ~0ull; // no lower bound
		EXPECT_EQ(lower_bound_eytzinger_prefetch(t.data(), 0, uint64_t(5)), ~0ull);
		for (uint64_t x = 0; x <= 2 * n + 2; ++x) {
			const auto it = std::lower_bound(a.begin(), a.end(), x);
			const uint64_t e = it == a.end() ? ~0ull : *it;
			EXPECT_EQ(lower_bound_eytzinger_prefetch(t.data(), n, x), e) << n << " " << x;
		}
	}
}

/// iterator versions with a hash function / comparator
TEST(search, iterator_lower_bounds) {
	const auto h = [](const uint64_t &x) { return x; };
	const auto lt = [](const uint64_t &a, const uint64_t &b) { return a < b; };
	for (const auto &v : inputs()) {
		for (const uint64_t k : keys(v)) {
			const auto lb = std::lower_bound(v.begin(), v.end(), k);
			auto ub = std::upper_bound(v.begin(), v.end(), k);
			// last element equal to k, or end
			const auto last_eq = (ub != v.begin() && *(ub - 1) == k) ? ub - 1 : v.end();
			EXPECT_EQ(branchless_lower_bound(v.begin(), v.end(), k), lb) << v.size() << " " << k;
			EXPECT_EQ(cryptanalysislib::lower_bound(v.begin(), v.end(), k, h), lb) << v.size() << " " << k;
			EXPECT_EQ(cryptanalysislib::lower_bound(v.begin(), v.end(), k, lt), lb) << v.size() << " " << k;
			EXPECT_EQ(upper_bound_monobound_binary_search(v.begin(), v.end(), k, h), last_eq) << v.size() << " " << k;
			const auto first_eq_it = (lb != v.end() && *lb == k) ? lb : v.end();
			EXPECT_EQ(interpolation_search(v.begin(), v.end(), k, h), first_eq_it) << v.size() << " " << k;
			EXPECT_EQ(interpolation_search(v.begin(), v.end(), k), first_eq_it) << v.size() << " " << k;
			EXPECT_EQ(internal::interpolation_search_dispatch(v.begin(), v.end(), k, h), first_eq_it) << v.size() << " " << k;
			EXPECT_EQ(internal::interpolation_lower_bound(v.begin(), v.size(), k, h), size_t(lb - v.begin())) << v.size() << " " << k;
			EXPECT_EQ(linear_search(v.begin(), v.end(), k, h), first_eq_it) << v.size() << " " << k;
			EXPECT_EQ(linear_search(v.begin(), v.end(), k, lt), lb) << v.size() << " " << k;
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
