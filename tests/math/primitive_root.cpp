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

TEST(primitive_root, small_primes) {
	for (uint64_t p = 3; p < 5000; p++) {
		if (!is_prime_naive(p)) { continue; }
		// brute force: smallest g whose multiplicative order is p-1
		uint64_t expected = 0;
		for (uint64_t g = 2; g < p && !expected; g++) {
			uint64_t x = 1, ord = 0;
			do { x = x * g % p; ord++; } while (x != 1);
			if (ord == p - 1) { expected = g; }
		}
		EXPECT_EQ(primitive_root<uint64_t>(p), expected) << p;
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
