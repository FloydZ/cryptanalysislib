#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>
#include <gtest/gtest.h>

#include "algorithm/shuffle.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

/// `random_shuffle` permutes the range, every position takes every value
TEST(shuffle, random_shuffle) {
	std::vector<uint32_t> v(10);
	uint32_t seen[10][10] = {};
	for (uint32_t r = 0; r < 2000; ++r) {
		for (uint32_t i = 0; i < 10; ++i) { v[i] = i; }
		cryptanalysislib::random_shuffle(v.begin(), v.end());
		std::vector<uint32_t> s = v;
		std::sort(s.begin(), s.end());
		for (uint32_t i = 0; i < 10; ++i) { ASSERT_EQ(s[i], i); seen[i][v[i]] += 1; }
	}
	for (uint32_t i = 0; i < 10; ++i) {
		for (uint32_t j = 0; j < 10; ++j) { EXPECT_GT(seen[i][j], 100u) << i << " " << j; }
	}

	// empty and single element ranges
	std::vector<uint32_t> e;
	cryptanalysislib::random_shuffle(e.begin(), e.end());
	std::vector<uint32_t> one = {7};
	cryptanalysislib::random_shuffle(one.begin(), one.end());
	EXPECT_EQ(one[0], 7u);
}

/// tiny generator with the range [3, 5]: three values per call
struct Gen3 {
	using result_type = uint32_t;
	uint64_t state;
	static constexpr result_type min() { return 3; }
	static constexpr result_type max() { return 5; }
	constexpr result_type operator()() {
		state = state * 6364136223846793005ull + 1442695040888963407ull;
		return 3u + uint32_t((state >> 33u) % 3u);
	}
};

/// `urbg_uniform`: generator range larger, equal and smaller than the bound
TEST(shuffle, urbg_uniform) {
	Gen3 g3{42};
	std::mt19937 g32(1);
	std::minstd_rand gm(2); // [1, 2^31 - 2]
	for (const uint64_t bound : {uint64_t(0), uint64_t(1), uint64_t(2), uint64_t(10), uint64_t(100)}) {
		std::vector<uint32_t> c3(bound + 1), c32(bound + 1);
		const uint32_t n = 2000u * uint32_t(bound + 1);
		for (uint32_t i = 0; i < n; ++i) {
			const uint64_t a = cryptanalysislib::internal::urbg_uniform(g3, bound);
			const uint64_t b = cryptanalysislib::internal::urbg_uniform(g32, bound);
			ASSERT_LE(a, bound);
			ASSERT_LE(b, bound);
			c3[a] += 1; c32[b] += 1;
		}
		// expected 2000 per value
		for (uint64_t v = 0; v <= bound; ++v) {
			EXPECT_GT(c3[v], 1700u) << bound << " " << v;
			EXPECT_LT(c3[v], 2300u) << bound << " " << v;
			EXPECT_GT(c32[v], 1700u) << bound << " " << v;
			EXPECT_LT(c32[v], 2300u) << bound << " " << v;
		}
	}

	// bound larger than the generator range: the top and the bottom bits are uniform
	constexpr uint64_t big = (1ull << 40u) - 1u;
	uint32_t top[4] = {}, low[4] = {};
	for (uint32_t i = 0; i < 40000; ++i) {
		const uint64_t a = cryptanalysislib::internal::urbg_uniform(gm, big);
		ASSERT_LE(a, big);
		top[a >> 38u] += 1; low[a & 3u] += 1;
	}
	for (uint32_t v = 0; v < 4; ++v) {
		EXPECT_GT(top[v], 9000u) << v; EXPECT_LT(top[v], 11000u) << v;
		EXPECT_GT(low[v], 9000u) << v; EXPECT_LT(low[v], 11000u) << v;
	}

	// full 64 bit range: the generator output is used directly
	std::mt19937_64 g64(3), h64(3);
	EXPECT_EQ(cryptanalysislib::internal::urbg_uniform(g64, uint64_t(-1)), h64());
}

/// `shuffle` permutes the range, every position takes every value, the
/// result only depends on the generator state
TEST(shuffle, shuffle) {
	std::mt19937 g(7);
	Gen3 g3{1};
	std::vector<uint32_t> v(10), w(10);
	uint32_t seen[10][10] = {}, seen3[10][10] = {};
	for (uint32_t r = 0; r < 2000; ++r) {
		for (uint32_t i = 0; i < 10; ++i) { v[i] = i; w[i] = i; }
		cryptanalysislib::shuffle(v.begin(), v.end(), g);
		cryptanalysislib::shuffle(w.begin(), w.end(), g3);
		std::vector<uint32_t> s = v, t = w;
		std::sort(s.begin(), s.end());
		std::sort(t.begin(), t.end());
		for (uint32_t i = 0; i < 10; ++i) {
			ASSERT_EQ(s[i], i); ASSERT_EQ(t[i], i);
			seen[i][v[i]] += 1; seen3[i][w[i]] += 1;
		}
	}
	for (uint32_t i = 0; i < 10; ++i) {
		for (uint32_t j = 0; j < 10; ++j) {
			EXPECT_GT(seen[i][j], 100u) << i << " " << j;
			EXPECT_GT(seen3[i][j], 100u) << i << " " << j;
		}
	}

	std::vector<uint32_t> a(1000), b;
	for (uint32_t i = 0; i < 1000; ++i) { a[i] = i; }
	b = a;
	cryptanalysislib::shuffle(a.begin(), a.end(), std::mt19937_64(5));
	std::mt19937_64 g5(5);
	cryptanalysislib::shuffle(b.begin(), b.end(), g5);
	EXPECT_EQ(a, b);

	std::vector<uint32_t> e;
	cryptanalysislib::shuffle(e.begin(), e.end(), g);
	std::vector<uint32_t> one = {7};
	cryptanalysislib::shuffle(one.begin(), one.end(), g);
	EXPECT_EQ(one[0], 7u);
}

/// usable in constant expressions
TEST(shuffle, constexpr_shuffle) {
	constexpr auto r = []() {
		uint32_t a[8] = {0, 1, 2, 3, 4, 5, 6, 7};
		Gen3 g{9};
		cryptanalysislib::shuffle(a, a + 8, g);
		uint32_t sum = 0, mask = 0;
		for (const uint32_t x : a) { sum += x; mask |= 1u << x; }
		return sum * 1000u + mask;
	}();
	static_assert(r == 28255u);
	EXPECT_EQ(r, 28255u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
