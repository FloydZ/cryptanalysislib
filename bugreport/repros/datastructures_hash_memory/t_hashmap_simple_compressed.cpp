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
	constexpr uint32_t nr_bytes = 64;
	constexpr static Hash<K, 0, l, 2> hashclass{};
	constexpr static SimpleCompressedHashMapConfig s = SimpleCompressedHashMapConfig{nr_bytes, 1u << l};
	using HM = SimpleCompressedHashMap<K, V, s, Hash<K, 0, l, 2>>;

	HM hm = HM{};
	hm.info();

	const size_t limit = (1u << (l-2));
	for (uint64_t i = 0; i < (limit * nr_bytes); i++) {
		hm.insert(i, i + 1);
	}

	for (uint64_t i = 0; i < limit; i++) {
		V *out = nullptr;
		uint32_t nr;
		hm.decompress(&out, nr, i);
		for (uint32_t j = 0; j < nr; ++j) {
			EXPECT_EQ(out[j], j*(1u<<l) + i + 1);
		}
	}
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
