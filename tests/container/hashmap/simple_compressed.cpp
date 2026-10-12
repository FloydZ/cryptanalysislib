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
	static_assert(HashMapAble<HM>);

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


/// decreasing and random values (negative deltas wrap around)
TEST(HashMap, decreasing_values) {
	constexpr uint32_t l = 4;
	constexpr uint32_t nr_bytes = 128;
	constexpr static SimpleCompressedHashMapConfig s = SimpleCompressedHashMapConfig{nr_bytes, 1u << l};
	using HM = SimpleCompressedHashMap<K, V, s, Hash<K, 0, l, 2>>;
	HM hm = HM{};
	const V values[] = {1000000, 5, 4000000000u, 0, 77, 76, 4294967295u, 1};
	for (const V v : values) { hm.insert(3, v); }
	V *out = nullptr;
	uint32_t nr = 0;
	hm.decompress(&out, nr, 3);
	ASSERT_EQ(nr, 8u);
	for (uint32_t j = 0; j < nr; ++j) { EXPECT_EQ(out[j], values[j]) << j; }
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
