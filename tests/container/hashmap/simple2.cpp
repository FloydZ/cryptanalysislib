#include <gtest/gtest.h>
#include <iostream>

#include "container/hashmap.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

using K = uint32_t;
using V = uint64_t;

TEST(HashMap2, simple) {
	constexpr uint32_t l = 12;
	constexpr uint32_t bucketsize = (64/sizeof(V)) - 1u;
	constexpr static Simple2HashMapConfig s = Simple2HashMapConfig{1u << l};
	using HM = Simple2HashMap<K, V, s, Hash<K, 0, l>>;

	HM hm = HM{};

	for (uint64_t i = 0; i < ((1u << l) * bucketsize); ++i) {
		hm.insert(i, i+1);
	}

	for (size_t i = 0; i < 1u << l; ++i) {
		EXPECT_EQ(hm.load(i), bucketsize);

		for (uint32_t j = 0; j < bucketsize; ++j) {
			EXPECT_EQ(hm.__array[i*(bucketsize + 1) + j], (i+1) + j*(1u << l));

			const K k = Hash<K, 0, l>::operator()(hm.__array[i*(bucketsize + 1) + j] - 1);
			// std::cout << i << " " << j << " " << k << " " << hm.__internal_hashmap_array[i*bucketsize + j] - 1 << "\n";
			EXPECT_EQ(k, i);
		}
	}
}

TEST(HashMap2, multithreaded) {
	constexpr uint32_t l = 12;
	constexpr uint32_t threads = 2;
	constexpr uint32_t bucketsize = (64/sizeof(V)) - 1u;
	constexpr static Simple2HashMapConfig s = Simple2HashMapConfig{1u << l, threads};
	using HM = Simple2HashMap<K, V, s, Hash<K, 0, l>>;

	HM hm = HM{};
	hm.info();

	#pragma omp parallel default(none) shared(hm, l) num_threads(threads)
	{
		#pragma omp for
		for (uint64_t i = 0; i < ((1u << l) * bucketsize); ++i) {
			hm.insert(i, i + 1);
		}
	}


	for (size_t i = 0; i < 1u << l; ++i) {
		EXPECT_EQ(hm.load(i), bucketsize);

		for (uint32_t j = 0; j < bucketsize; ++j) {
			const K k = Hash<K, 0, l>::operator()(hm.__array[i*(bucketsize + 1) + j] - 1);
			EXPECT_EQ(k, i);
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
