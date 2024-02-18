#include <gtest/gtest.h>
#include <iostream>


using ::testing::InitGoogleTest;
using ::testing::Test;

using K = uint32_t;
using V = uint64_t;

#include "container/hashmap.h"
TEST(SimdHashMap, simple) {
	constexpr uint32_t l = 4;
	constexpr uint32_t bucket_size = 8;
	constexpr static SIMDHashMapConfig s{bucket_size, 1 << l, 0, l, 1};
	using HM = SIMDHashMap<s>;
	HM hm = HM{};

	// first insert via the non simd interface
	for (uint64_t i = 0; i < ((1u << l) * bucket_size); ++i) {
		hm.insert_simple(i, i + 1);
	}

	for (size_t i = 0; i < 1u << l; ++i) {
		EXPECT_EQ(hm.load(i), bucket_size);
	}
}

TEST(SimdHashMap, avxInsert) {
	constexpr uint32_t l = 4;
	constexpr uint32_t bucket_size = 8;
	constexpr static SIMDHashMapConfig s{bucket_size, 1 << l, 0, l, 1};
	using HM = SIMDHashMap<s>;
	HM hm = HM{};

	uint32x8_t data, index;
	for (uint64_t i = 0; i < ((1u << l) * bucket_size / 8); ++i) {
		for (uint32_t j = 0; j < 8; ++j) {
			data.v32[j] = i * 8 + j;
			index.v32[j] = i * 8 + j + 1;
		}

		hm.insert_simd(data, index);
	}

	for (size_t i = 0; i < 1u << l; ++i) {
		EXPECT_EQ(hm.load(i), bucket_size);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
