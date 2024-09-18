#include <gtest/gtest.h>
#include <iostream>

#include "container/hashmap.h"
#include "simd/simd.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using K = uint32_t;
using V = uint32_t;

TEST(HashMap, simple) {
	constexpr uint32_t l = 12;
	constexpr uint32_t nr_bytes = 32;
	constexpr static Hash<K, 0, l, 2> hashclass{};
	constexpr static SimpleCompressedHashMapConfig s = SimpleCompressedHashMapConfig{nr_bytes, 1u << l};
	using HM = SimpleCompressedHashMap<K, V, s, Hash<K, 0, l, 2>>;

	HM hm = HM{};

	for (uint64_t i = 0; i < ((1u << l) * nr_bytes); ++i) {
		hm.insert(i, i + 1);
	}

	/// TODO better testing
	for (size_t i = 0; i < 1u << l; ++i) {
	//	EXPECT_EQ(hm.__internal_load_array[i], bucketsize);

	//	for (uint32_t j = 0; j < bucketsize; ++j) {
	//		EXPECT_EQ(hm.__internal_hashmap_array[i * bucketsize + j], (i + 1) + j * (1u << l));

	//		const K k = hashclass(hm.__internal_hashmap_array[i * bucketsize + j] - 1);
	//		// std::cout << i << " " << j << " " << k << " " << hm.__internal_hashmap_array[i*bucketsize + j] - 1 << "\n";
	//		EXPECT_EQ(k, i);
	//	}
	}
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
