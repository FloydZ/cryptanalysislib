#include <gtest/gtest.h>
#include <iostream>

#include "container/hashmap.h"
#include "simd/simd.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using K = uint32_t;
using V = uint64_t;

TEST(HashMap, simd) {
	constexpr uint32_t l = 12;
	constexpr uint32_t bucketsize = 10;
	constexpr static Hash<K, 0, l> hashclass{};
	constexpr static SimpleHashMapConfig s = SimpleHashMapConfig{bucketsize, 1u << l, 1};
	using HM = SimpleHashMap<K, V, s, Hash<K, 0, l>>;

	using SIMD = uint16x16_t;
	HM hm = HM{};

	for (uint64_t i = 0; i < ((1u << l) * bucketsize) / SIMD::LIMBS; ++i) {
		SIMD data, index;
		for (uint32_t j = 0; j < SIMD::LIMBS; ++j) {
			data[j] = i;
			index[j] = i + 1;
		}
		// TODO hm.insert_simd(data, index);
	}

	for (size_t i = 0; i < 1u << l; ++i) {
		EXPECT_EQ(hm.__internal_load_array[i], bucketsize);

		for (uint32_t j = 0; j < bucketsize; ++j) {
			EXPECT_EQ(hm.__internal_hashmap_array[i * bucketsize + j], (i + 1) + j * (1u << l));

			const K k = hashclass(hm.__internal_hashmap_array[i * bucketsize + j] - 1);
			// std::cout << i << " " << j << " " << k << " " << hm.__internal_hashmap_array[i*bucketsize + j] - 1 << "\n";
			EXPECT_EQ(k, i);
		}
	}
}

TEST(HashMap, simple) {
	constexpr uint32_t l = 12;
	constexpr uint32_t bucketsize = 10;
	constexpr static Hash<K, 0, l> hashclass{};
	constexpr static SimpleHashMapConfig s = SimpleHashMapConfig{bucketsize, 1u << l, 1};
	using HM = SimpleHashMap<K, V, s, Hash<K, 0, l>>;

	HM hm = HM{};

	for (uint64_t i = 0; i < ((1u << l) * bucketsize); ++i) {
		hm.insert(i, i + 1);
	}

	for (size_t i = 0; i < 1u << l; ++i) {
		EXPECT_EQ(hm.__internal_load_array[i], bucketsize);

		for (uint32_t j = 0; j < bucketsize; ++j) {
			EXPECT_EQ(hm.__internal_hashmap_array[i * bucketsize + j], (i + 1) + j * (1u << l));

			const K k = hashclass(hm.__internal_hashmap_array[i * bucketsize + j] - 1);
			// std::cout << i << " " << j << " " << k << " " << hm.__internal_hashmap_array[i*bucketsize + j] - 1 << "\n";
			EXPECT_EQ(k, i);
		}
	}
}

TEST(HashMap, multithreaded) {
	constexpr uint32_t l = 12;
	constexpr uint32_t bucketsize = 10;
	constexpr uint32_t threads = 2;
	constexpr static Hash<K, 0, l> hashclass{};
	constexpr static SimpleHashMapConfig s = SimpleHashMapConfig{bucketsize, 1u << l, threads};
	using HM = SimpleHashMap<K, V, s, Hash<K, 0, l>>;

	HM hm = HM{};
	hm.print();

#pragma omp parallel default(none) shared(l, hm) num_threads(threads)
	{
#pragma omp for
		for (uint64_t i = 0; i < ((1u << l) * bucketsize); ++i) {
			hm.insert(i, i + 1);
		}
	}


	for (size_t i = 0; i < 1u << l; ++i) {
		EXPECT_EQ(hm.__internal_load_array[i], bucketsize);
		for (uint32_t j = 0; j < bucketsize; ++j) {
			const K k = hashclass(hm.__internal_hashmap_array[i * bucketsize + j] - 1);
			EXPECT_EQ(k, i);
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
