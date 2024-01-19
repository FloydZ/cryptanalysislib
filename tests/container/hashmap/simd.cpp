#include <gtest/gtest.h>
#include <iostream>

#include "container/hashmap.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using K = uint32_t;
using V = uint64_t;

/// TODO further testing
TEST(SimdHashMap, simple) {
	constexpr uint32_t l = 4;
	constexpr uint32_t bucket_size = 10;
	constexpr static SIMDHashMapConfig s{bucket_size, 1<<l, 0, l, 1};
	using HM = SIMDHashMap<s>;
	HM hm = HM{};

	// first insert via the non simd interface
	for (uint64_t i = 0; i < ((1u << l) * bucket_size); ++i) {
		hm.insert_simple(i, i+1);
	}

	//for (size_t i = 0; i < 1u << l; ++i) {
	//	EXPECT_EQ(hm.__internal_load_array[i], bucketsize);

	//	for (uint32_t j = 0; j < bucketsize; ++j) {
	//		EXPECT_EQ(hm.__internal_hashmap_array[i*bucketsize + j], (i+1) + j*(1u << l));

	//		const K k = H<l>(hm.__internal_hashmap_array[i*bucketsize + j] - 1);
	//		// std::cout << i << " " << j << " " << k << " " << hm.__internal_hashmap_array[i*bucketsize + j] - 1 << "\n";
	//		EXPECT_EQ(k, i);
	//	}
	//}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
