#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/prefixsum.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(avx, prefixsum) {
	constexpr size_t s = 65;
	uint32_t *d1 = (uint32_t *)malloc(s * sizeof(uint32_t));
	uint32_t *d2 = (uint32_t *)malloc(s * sizeof(uint32_t));
	// for (uint32_t i = 0; i < s; i++) { d1[i] = fastrandombytes_uint64() % (1u << 8u); }
	for (uint32_t i = 0; i < s; i++) {
		d1[i] = i;
	}
	memcpy(d2, d1, s * sizeof(uint32_t));

	avx2_prefixsum_u32(d1, s);
	for (uint32_t i = 1; i < s; i++) {
		d2[i] += d2[i - 1];
	}

	for (uint32_t i = 0; i < s; i++) {
	EXPECT_EQ(d1[i], d2[i]);
	}
	free(d1); free(d2);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
