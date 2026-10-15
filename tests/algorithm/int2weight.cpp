#include <cstdint>
#include <set>
#include <vector>
#include <gtest/gtest.h>

#include "algorithm/int2weight.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;

/// every rank in [0, binom(n, wt)) maps to a distinct set of wt positions < n
TEST(int2weight, bijective) {
	for (const uint32_t n : {5u, 10u, 16u}) {
		for (const uint32_t wt : {1u, 2u, 3u}) {
			const uint64_t N = binom(n, wt);
			std::set<std::vector<uint32_t>> seen;
			std::set<uint64_t> seen_bits;
			for (uint64_t r = 0; r < N; ++r) {
				std::vector<uint32_t> w(wt, 0xffffffffu);
				int2weights(w, r, n, wt);
				std::set<uint32_t> s(w.begin(), w.end());
				ASSERT_EQ(s.size(), wt) << n << " " << wt << " " << r;
				for (const uint32_t p : w) { ASSERT_LT(p, n); }
				seen.insert(w);

				uint32_t raw[8] = {};
				int2weights(raw, r, n, wt, wt);
				EXPECT_TRUE(std::equal(w.begin(), w.end(), raw));

				uint64_t bits = 0;
				int2weight_bits(&bits, r, n, wt, wt);
				EXPECT_EQ(__builtin_popcountll(bits), (int)wt);
				uint64_t e = 0;
				for (const uint32_t p : w) { e |= 1ull << p; }
				EXPECT_EQ(bits, e);
				seen_bits.insert(bits);
			}
			EXPECT_EQ(seen.size(), N);
			EXPECT_EQ(seen_bits.size(), N);
		}
	}
}

/// `k` limits the number of written positions
TEST(int2weight, limit_k) {
	uint32_t w[4] = {99, 99, 99, 99};
	int2weights(w, 5u, 10, 3, 2);
	EXPECT_NE(w[0], 99u);
	EXPECT_NE(w[1], 99u);
	EXPECT_EQ(w[2], 99u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
