#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "random.h"
#include "search/convex.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// reference: strictly increasing then strictly decreasing, at least one of
/// both steps; returns the index of the maximum or 0
static uint64_t ref_strictly_convex(const std::vector<int> &f) {
	const uint64_t n = f.size();
	if (n <= 2) { return 0; }
	uint64_t m = 0;
	for (uint64_t i = 1; i < n; ++i) { if (f[i] > f[m]) { m = i; } }
	if (m == 0 || m == n - 1) { return 0; }
	for (uint64_t i = 1; i <= m; ++i) { if (!(f[i - 1] < f[i])) { return 0; } }
	for (uint64_t i = m + 1; i < n; ++i) { if (!(f[i - 1] > f[i])) { return 0; } }
	return m;
}

/// reference: some split point, weakly increasing before, weakly decreasing after
static bool ref_weakly_convex(const std::vector<int> &f) {
	const uint64_t n = f.size();
	for (uint64_t p = 0; p <= n; ++p) {
		bool ok = true;
		for (uint64_t i = 1; i < p; ++i) { ok &= f[i - 1] <= f[i]; }
		for (uint64_t i = std::max<uint64_t>(p, 1); i < n; ++i) { ok &= f[i - 1] >= f[i]; }
		if (ok) { return true; }
	}
	return false;
}

TEST(convex, examples) {
	constexpr static int a[] = {1, 3, 5, 4, 2};
	EXPECT_EQ(test_strictly_convex(a, 5), 2u);
	EXPECT_TRUE(is_strictly_convex(a, 5));
	EXPECT_FALSE(is_strictly_concave(a, 5));
	EXPECT_TRUE(is_weakly_convex(a, 5));
	EXPECT_EQ(convex_max_index(a, 5), 2u);

	const int b[] = {5, 3, 1, 2, 4};
	EXPECT_EQ(test_strictly_concave(b, 5), 2u);
	EXPECT_TRUE(is_weakly_concave(b, 5));
	EXPECT_EQ(concave_min_index(b, 5), 2u);

	// plateau: weakly, but not strictly convex
	const int c[] = {1, 3, 3, 2};
	EXPECT_FALSE(is_strictly_convex(c, 4));
	EXPECT_TRUE(is_weakly_convex(c, 4));

	// monotone: no maximum in the interior
	const int d[] = {1, 2, 3, 4};
	EXPECT_FALSE(is_strictly_convex(d, 4));
	EXPECT_TRUE(is_weakly_convex(d, 4));
	EXPECT_EQ(convex_max_index(d, 4), 3u);

	static_assert(convex_max_index(a, 5) == 2);
}

/// random sequences against the references (strict and weak, both directions)
TEST(convex, random) {
	for (uint32_t it = 0; it < 100000; ++it) {
		std::vector<int> f(rng() % 8);
		for (auto &x : f) { x = int(rng() % 4); }
		std::vector<int> g(f.size());
		for (size_t i = 0; i < f.size(); ++i) { g[i] = -f[i]; }

		ASSERT_EQ(test_strictly_convex(f.data(), f.size()), ref_strictly_convex(f));
		ASSERT_EQ(test_strictly_concave(g.data(), g.size()), ref_strictly_convex(f));
		ASSERT_EQ(is_weakly_convex(f.data(), f.size()), ref_weakly_convex(f));
		ASSERT_EQ(is_weakly_concave(g.data(), g.size()), ref_weakly_convex(f));
	}
}

/// O(log n) search on random strongly unimodal sequences
TEST(convex, max_min_index) {
	for (uint32_t it = 0; it < 10000; ++it) {
		const uint64_t n = 1 + rng() % 1000, peak = rng() % n;
		std::vector<int64_t> f(n);
		f[peak] = 0;
		for (uint64_t i = peak; i-- > 0;) { f[i] = f[i + 1] - 1 - int64_t(rng() % 5); }
		for (uint64_t i = peak + 1; i < n; ++i) { f[i] = f[i - 1] - 1 - int64_t(rng() % 5); }
		ASSERT_EQ(convex_max_index(f.data(), n), peak);
		for (auto &x : f) { x = -x; }
		ASSERT_EQ(concave_min_index(f.data(), n), peak);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
