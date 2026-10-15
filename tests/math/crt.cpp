#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "math/math.h"
#include "random.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

// trial division
static bool is_prime_naive(const uint64_t n) {
	if (n < 2) { return false; }
	for (uint64_t d = 2; d * d <= n; d++) {
		if (n % d == 0) { return false; }
	}
	return true;
}

TEST(crt, brute_force) {
	for (int64_t n = 1; n <= 40; n++) {
		for (int64_t m = 1; m <= 40; m++) {
			int64_t l = n;
			while (l % m) { l += n; }   // lcm(n, m)
			for (int64_t a = -3; a < n; a++) {
				for (int64_t b = -3; b < m; b++) {
					int64_t x = -1;
					for (int64_t y = 0; y < l; y++) {
						if (pmod<int64_t>(y - a, n) == 0 && pmod<int64_t>(y - b, m) == 0) { x = y; break; }
					}
					const auto r = crt<int64_t>(a, n, b, m);
					if (x < 0) {
						EXPECT_EQ(r.second, -1);
					} else {
						EXPECT_EQ(r.first, x);
						EXPECT_EQ(r.second, l);
					}
				}
			}
		}
	}
}

TEST(crt, large) {
	// moduli up to 1e9 (documented limit)
	for (int i = 0; i < 10000; i++) {
		const int64_t n = 1 + (int64_t)rng<uint64_t>(1000000000ull);
		const int64_t m = 1 + (int64_t)rng<uint64_t>(1000000000ull);
		const int64_t x = (int64_t)rng<uint64_t>(1ull << 62);
		const auto r = crt<int64_t>(x % n, n, x % m, m);
		ASSERT_NE(r.second, -1);
		EXPECT_EQ(pmod<int64_t>(r.first - x, n), 0);
		EXPECT_EQ(pmod<int64_t>(r.first - x, m), 0);
		EXPECT_LT(r.first, r.second);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
