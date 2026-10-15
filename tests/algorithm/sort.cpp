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

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
