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

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
