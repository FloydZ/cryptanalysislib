#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <vector>

#include "random.h"
#include "algorithm/minmax.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// sequential and parallel minmax_element against std::minmax_element:
/// few distinct values, so min and max occur repeatedly
/// (first minimum, last maximum)
TEST(minmax_element, sequential) {
	for (size_t s = 0; s <= 100; ++s) {
		std::vector<uint32_t> v(s);
		for (auto &x : v) { x = uint32_t(rng() % 5u); }
		EXPECT_EQ(cryptanalysislib::minmax_element(v.begin(), v.end()),
		          std::minmax_element(v.begin(), v.end())) << s;
	}
}

TEST(minmax_element, multithreading) {
	constexpr size_t s = 1u << 21u;
	for (const uint32_t mod : {3u, 1000u, 1u << 30}) {
		std::vector<uint64_t> v(s);
		for (auto &x : v) { x = rng() % mod; }
		EXPECT_EQ(cryptanalysislib::minmax_element(par_if(true), v.begin(), v.end()),
		          std::minmax_element(v.begin(), v.end())) << mod;
		EXPECT_EQ(cryptanalysislib::minmax_element(par_if(true), v.begin(), v.end(), std::greater<uint64_t>()),
		          std::minmax_element(v.begin(), v.end(), std::greater<uint64_t>())) << mod;
	}
}

TEST(minmax, values) {
	EXPECT_EQ(cryptanalysislib::minmax({3, 1, 4, 1, 5}), std::make_pair(1, 5));
	const int a = 2, b = 1;
	EXPECT_EQ(cryptanalysislib::minmax(a, b).first, 1);
}

/// `minmax`/`minmax_element` with a comparator, the SIMD kernels
TEST(MinMax, comparator_and_simd) {
	const auto gt = [](const uint32_t a, const uint32_t b) { return a > b; };
	const uint32_t x = 3, y = 7;
	const auto p = cryptanalysislib::minmax(x, y, gt);
	EXPECT_EQ(p.first, 7u);
	EXPECT_EQ(p.second, 3u);
	const auto q = cryptanalysislib::minmax(x, x, gt);
	EXPECT_EQ(&q.first, &x);

	for (const size_t s : {size_t(0), size_t(1), size_t(2), size_t(31), size_t(1000)}) {
		std::vector<uint32_t> v(s);
		for (size_t i = 0; i < s; ++i) { v[i] = (uint32_t)(i * 2654435761u) % 1000u; }
		const auto e = std::minmax_element(v.begin(), v.end(), gt);
		const auto r = cryptanalysislib::minmax_element(v.begin(), v.end(), gt);
		EXPECT_EQ(r.first, e.first) << s;
		EXPECT_EQ(r.second, e.second) << s;
		if (s == 0) { continue; }

		const auto m = std::minmax_element(v.begin(), v.end());
		const auto t1 = cryptanalysislib::internal::simd_minmax_element(v.data(), v.data() + s);
		EXPECT_EQ(t1.first, *m.first) << s;
		EXPECT_EQ(t1.second, *m.second) << s;
		const auto t2 = cryptanalysislib::internal::minmax_element_simd(v.begin(), v.end());
		EXPECT_EQ(t2.first, *m.first) << s;
		EXPECT_EQ(t2.second, *m.second) << s;
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
