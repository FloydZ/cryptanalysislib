#include <cstdint>
#include <gtest/gtest.h>

#include "math/math.h"
#include "math/lucas.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;

/// nCk mod p via Lucas' theorem against Pascal's triangle mod p
TEST(lucas, small_primes) {
	for (const uint64_t p : {2ull, 3ull, 5ull, 7ull, 13ull}) {
		uint64_t c[60][60] = {};
		for (uint32_t n = 0; n < 60; ++n) {
			c[n][0] = 1 % p;
			for (uint32_t k = 1; k <= n; ++k) { c[n][k] = (c[n - 1][k - 1] + c[n - 1][k]) % p; }
		}
		for (uint64_t n = 0; n < 60; ++n) {
			for (uint64_t k = 0; k <= n; ++k) {
				ASSERT_EQ(lucas<uint64_t>(n, k, p), c[n][k]) << n << " " << k << " " << p;
			}
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
