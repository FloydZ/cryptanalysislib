#include <gtest/gtest.h>
#include <algorithm>
#include <cstdint>
#include <list>
#include <vector>

#include "algorithm/generate.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

// NOTE: always call `cryptanalysislib::generate`: an unqualified call on
// 	`std::` iterators also finds `std::generate` via ADL (ambiguous).

TEST(generate, simple) {
	for (const size_t n : {0ul, 1ul, 7ul, 100ul}) {
		std::vector<uint32_t> v(n), e(n);
		uint32_t c1 = 3, c2 = 3;
		cryptanalysislib::generate(v.begin(), v.end(), [&c1]() { return c1 += 5; });
		std::generate(e.begin(), e.end(), [&c2]() { return c2 += 5; });
		EXPECT_EQ(v, e);
	}
}

TEST(generate, forward_iterator) {
	std::list<int> l(10);
	int c = 0;
	cryptanalysislib::generate(l.begin(), l.end(), [&c]() { return c++; });
	int i = 0;
	for (const int x : l) { EXPECT_EQ(x, i++); }
}

TEST(generate_n, simple) {
	std::vector<uint64_t> v(20, 0);
	uint64_t c = 1;
	const auto it = cryptanalysislib::generate_n(v.begin(), 10, [&c]() { return c *= 2; });
	EXPECT_EQ(it, v.begin() + 10);
	for (size_t i = 0; i < 10; ++i) { EXPECT_EQ(v[i], 1ull << (i + 1)); }
	for (size_t i = 10; i < 20; ++i) { EXPECT_EQ(v[i], 0u); }

	// count == 0: nothing written, `first` returned
	EXPECT_EQ(cryptanalysislib::generate_n(v.begin(), 0, [&c]() { return c; }), v.begin());
	EXPECT_EQ(v[0], 2u);
}

/// constexpr evaluation
constexpr int sum_generated() {
	int a[5] = {0};
	int c = 0;
	cryptanalysislib::generate(a, a + 5, [&c]() { return ++c; });
	return a[0] + a[1] + a[2] + a[3] + a[4];
}
static_assert(sum_generated() == 15);

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
