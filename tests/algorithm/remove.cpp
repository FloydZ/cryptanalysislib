#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "random.h"
#include "algorithm/remove.h"
#include "algorithm/remove.h" // include guard

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// remove against std::remove: all sizes around the SIMD block size, few
/// distinct values (many removals, also whole blocks), the result prefix and
/// the returned end must match
template<typename T>
static void check_remove(const size_t n, const uint32_t mod) {
	std::vector<T> v(n);
	for (auto &x : v) { x = T(rng() % mod); }
	auto e = v;
	const auto r = cryptanalysislib::remove(v.begin(), v.end(), T(1));
	const auto re = std::remove(e.begin(), e.end(), T(1));
	ASSERT_EQ(r - v.begin(), re - e.begin()) << n << " " << mod;
	EXPECT_TRUE(std::equal(v.begin(), r, e.begin())) << n << " " << mod;
}

TEST(remove, uint32) {
	for (size_t n = 0; n <= 300; ++n) {
		check_remove<uint32_t>(n, 2);
		check_remove<uint32_t>(n, 5);
		check_remove<uint32_t>(n, 1u << 20);
	}
	check_remove<uint32_t>(1u << 20, 3);
}

TEST(remove, other_types) {
	for (size_t n = 0; n <= 100; ++n) {
		check_remove<uint8_t>(n, 3);
		check_remove<uint64_t>(n, 3);
	}
}

TEST(remove, remove_if) {
	std::vector<uint32_t> v(1000);
	for (auto &x : v) { x = uint32_t(rng() % 10u); }
	auto e = v;
	const auto p = [](const uint32_t x) { return x < 4u; };
	const auto r = cryptanalysislib::remove_if(v.begin(), v.end(), p);
	const auto re = std::remove_if(e.begin(), e.end(), p);
	ASSERT_EQ(r - v.begin(), re - e.begin());
	EXPECT_TRUE(std::equal(v.begin(), r, e.begin()));
}

/// parallel remove/remove_if against std::: stability via (key, index) pairs
TEST(remove, multithreading) {
	constexpr size_t n = 1u << 21;
	for (const uint32_t mod : {2u, 7u, 1000u}) {
		std::vector<std::pair<uint32_t, uint32_t>> v(n);
		for (size_t i = 0; i < n; ++i) { v[i] = {uint32_t(rng() % mod), uint32_t(i)}; }
		auto e = v;
		const auto p = [](const auto &x) { return x.first == 1u; };
		const auto r = cryptanalysislib::remove_if(par_if(true), v.begin(), v.end(), p);
		const auto re = std::remove_if(e.begin(), e.end(), p);
		ASSERT_EQ(r - v.begin(), re - e.begin()) << mod;
		EXPECT_TRUE(std::equal(v.begin(), r, e.begin())) << mod;
	}

	// uint32_t: SIMD remove per chunk; also nothing removed / all removed
	for (const uint32_t mod : {4u, 1u << 30, 1u}) {
		std::vector<uint32_t> w(n);
		for (auto &x : w) { x = uint32_t(rng() % mod); }
		auto ew = w;
		const uint32_t val = mod == 1u ? 0u : 3u;
		const auto r = cryptanalysislib::remove(par_if(true), w.begin(), w.end(), val);
		const auto re = std::remove(ew.begin(), ew.end(), val);
		ASSERT_EQ(r - w.begin(), re - ew.begin()) << mod;
		EXPECT_TRUE(std::equal(w.begin(), r, ew.begin())) << mod;
	}

	// no element removed in the first chunks: no self move assignment of
	// non trivial types
	std::vector<std::vector<uint32_t>> vv(1u << 17, std::vector<uint32_t>{1, 2});
	vv.back() = {7};
	const auto rv = cryptanalysislib::remove_if(par_if(true), vv.begin(), vv.end(),
	                                            [](const auto &x) { return x.size() == 1; });
	ASSERT_EQ(rv - vv.begin(), (1u << 17) - 1u);
	for (auto it = vv.begin(); it != rv; ++it) { EXPECT_EQ(*it, (std::vector<uint32_t>{1, 2})); }
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
