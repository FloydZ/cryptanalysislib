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

TEST(miller_rabin, small) {
	for (uint64_t n = 0; n < 200000; n++) {
		EXPECT_EQ(millerRabin<uint64_t>(n), is_prime_naive(n)) << n;
	}
}

TEST(miller_rabin, large) {
	// Carmichael numbers and large known primes / composites
	for (const uint64_t c : {561ull, 1105ull, 1729ull, 2465ull, 2821ull, 6601ull, 8911ull,
	                         3215031751ull, 4294967297ull /* 641 * 6700417 */,
	                         18446744073709551615ull /* 2^64-1 */}) {
		EXPECT_FALSE(millerRabin<uint64_t>(c)) << c;
	}
	for (const uint64_t p : {4294967291ull /* 2^32-5 */, 2305843009213693951ull /* 2^61-1 */,
	                         18446744073709551557ull /* 2^64-59 */, 1000000007ull, 998244353ull}) {
		EXPECT_TRUE(millerRabin<uint64_t>(p)) << p;
	}
	// products of two ~32-bit primes
	EXPECT_FALSE(millerRabin<uint64_t>(4294967291ull * 4294967279ull));
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
