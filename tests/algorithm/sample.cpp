#include <algorithm>
#include <cstdint>
#include <random>
#include <list>
#include <set>
#include <vector>
#include <gtest/gtest.h>

#include "algorithm/sample.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;

/// `sample` (sequential and parallel): min(n, size) distinct elements of the
/// population
TEST(sample, properties) {
	std::mt19937_64 g(42);
	for (const size_t s : {size_t(0), size_t(1), size_t(10), size_t(1000), size_t(1u << 20)}) {
		std::vector<uint32_t> v(s);
		for (size_t i = 0; i < s; ++i) { v[i] = (uint32_t)i; }
		for (const size_t n : {size_t(0), size_t(1), size_t(5), size_t(999), size_t(1u << 21)}) {
			const size_t k = std::min(n, s);
			std::vector<uint32_t> o(k + 1, 0xffffffffu);
			auto r = cryptanalysislib::sample(v.begin(), v.end(), o.begin(), n, g);
			EXPECT_EQ(size_t(r - o.begin()), k) << s << " " << n;
			EXPECT_EQ(std::set<uint32_t>(o.begin(), o.begin() + k).size(), k);
			EXPECT_EQ(o[k], 0xffffffffu);

			std::vector<uint32_t> p(k + 1, 0xffffffffu);
			auto rp = cryptanalysislib::sample(par_if(true), v.begin(), v.end(), p.begin(), n, g);
			EXPECT_EQ(size_t(rp - p.begin()), k) << s << " " << n;
			std::set<uint32_t> st(p.begin(), p.begin() + k);
			EXPECT_EQ(st.size(), k);
			for (const uint32_t x : st) { EXPECT_LT(x, s); }
			EXPECT_EQ(p[k], 0xffffffffu);
		}
	}
}

/// forward iterators (reservoir sampling): the returned iterator is the end
/// of the written sample, also if the whole population is taken
TEST(sample, forward_iterator) {
	std::mt19937_64 g(7);
	const std::list<uint32_t> l = {1, 2, 3, 4, 5};
	for (const size_t n : {size_t(0), size_t(2), size_t(5), size_t(9)}) {
		std::vector<uint32_t> o(6, 0xffffffffu);
		const auto r = cryptanalysislib::sample(l.begin(), l.end(), o.begin(), n, g);
		const size_t k = std::min<size_t>(n, 5);
		EXPECT_EQ(size_t(r - o.begin()), k) << n;
		EXPECT_EQ(std::set<uint32_t>(o.begin(), o.begin() + k).size(), k) << n;
		for (size_t i = 0; i < k; ++i) { EXPECT_GE(o[i], 1u); EXPECT_LE(o[i], 5u); }
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
