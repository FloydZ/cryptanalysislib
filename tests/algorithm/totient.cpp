#include <cstdint>
#include <numeric>
#include <gtest/gtest.h>

#include "algorithm/totient.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(totient, sieve) {
	const auto phi = cryptanalysislib::totient<uint32_t>(1000);
	ASSERT_EQ(phi.size(), 1000u);
	EXPECT_EQ(phi[1], 1u);
	for (uint32_t i = 2; i < 1000; ++i) {
		uint32_t e = 0;
		for (uint32_t j = 1; j <= i; ++j) { e += std::gcd(i, j) == 1; }
		ASSERT_EQ(phi[i], e) << i;
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
