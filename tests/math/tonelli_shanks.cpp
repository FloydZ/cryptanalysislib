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

TEST(tonelli_shanks, small_primes) {
	for (uint64_t p = 3; p < 3000; p++) {
		if (!is_prime_naive(p)) { continue; }
		for (uint64_t n = 1; n < p; n++) {
			if (legendre<uint64_t>(n, p) != 1) { continue; }
			const uint64_t r = tonelli_shanks<uint64_t>(n, p);
			EXPECT_EQ(mulmod<uint64_t>(r, r, p), n) << n << " " << p;
		}
	}
}

TEST(tonelli_shanks, large_primes) {
	for (const uint64_t p : {2305843009213693951ull, 18446744073709551557ull, 998244353ull}) {
		for (int i = 0; i < 1000; i++) {
			const uint64_t x = 1 + rng<uint64_t>(p - 1);
			const uint64_t n = mulmod<uint64_t>(x, x, p);
			const uint64_t r = tonelli_shanks<uint64_t>(n, p);
			EXPECT_EQ(mulmod<uint64_t>(r, r, p), n);
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
