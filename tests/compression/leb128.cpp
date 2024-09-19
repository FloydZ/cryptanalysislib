#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "random.h"
#include "compression/compression.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr size_t limit = 1u << 12;

TEST(leb128, uint32_t_increasing) {
	using T = uint32_t;
	auto *in1 = (T 		 *)malloc(limit*sizeof(T));
	auto* out = (uint8_t *)malloc(limit*sizeof(T));
	uint8_t *pou = out;
	uint8_t *p2 = pou;
	for (size_t i = 0; i < limit; ++i) { in1[i] = i; }

	for (uint32_t i = 0; i < limit; i++) {
		const uint32_t n = leb128_encode<T>(out, in1[i]);
		out += n;
	}

	for (uint32_t i = 0; i < limit; i++) {
		const T t = leb128_decode<T>(&pou);
		EXPECT_EQ(in1[i], t);
	}

	free(in1); free(p2);
}

TEST(leb128, uint64_t_increasing) {
	using T = uint64_t;

	auto *in1 = (T 		 *)malloc(limit*sizeof(T));
	auto* out = (uint8_t *)malloc(limit*sizeof(T));
	uint8_t *pou = out;
	uint8_t *p2 = pou;
	for (size_t i = 0; i < limit; ++i) { in1[i] = 2*i; }

	for (uint32_t i = 0; i < limit; i++) {
		const uint32_t n = leb128_encode<T>(out, in1[i]);
		out += n;
	}

	for (uint32_t i = 0; i < limit; i++) {
		const T t = leb128_decode<T>(&pou);
		EXPECT_EQ(in1[i], t);
	}

	free(in1); free(p2);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
