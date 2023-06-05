#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define TEST_BASE_LIST_SIZE (1u << 10u)

#include "helper.h"
#include "nn/nn.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

union U256i {
	__m256i v;
	uint32_t a[8];
	uint64_t b[4];
};

constexpr size_t LS = 1u << 10u;
constexpr static WindowedAVX2_Config global_config{256, 4, 20, 64, LS, 22, 16, 0, 512};

TEST(PopCountAVX2, uint32_t) {
	constexpr static WindowedAVX2_Config config{256, 4, 50, 64, 1u<<8, 12, 4, 0, 496};
	WindowedAVX2<config> algo{};
	__m256i a = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	//__m256i b = WindowedAVX2<config>::popcount_avx2_32(a);
	__m256i b = algo.popcount_avx2_32_old(a);

	U256i c = U256i {b};
	EXPECT_EQ(c.a[0], 0);
	EXPECT_EQ(c.a[1], 1);
	EXPECT_EQ(c.a[2], 1);
	EXPECT_EQ(c.a[3], 2);
	EXPECT_EQ(c.a[4], 1);
	EXPECT_EQ(c.a[5], 2);
	EXPECT_EQ(c.a[6], 2);
	EXPECT_EQ(c.a[7], 3);
}

TEST(PopCountAVX2, uint64_t) {
	constexpr static WindowedAVX2_Config config{256, 4, 50, 64, 1u<<8, 12, 4, 0, 496};
	WindowedAVX2<config> algo{};
	__m256i a = _mm256_setr_epi64x(0, 1, 2, 3);
	//__m256i b = algo.popcount_avx2_64(a);
	__m256i b = algo.popcount_avx2_64_old_v2(a);

	U256i c = U256i {b};
	EXPECT_EQ(c.b[0], 0);
	EXPECT_EQ(c.b[1], 1);
	EXPECT_EQ(c.b[2], 1);
	EXPECT_EQ(c.b[3], 2);

	for (size_t i = 0; i < 1000000; i++) {
		const uint64_t a1 = fastrandombytes_uint64();
		const uint64_t a2 = fastrandombytes_uint64();
		const uint64_t a3 = fastrandombytes_uint64();
		const uint64_t a4 = fastrandombytes_uint64();

		a = _mm256_setr_epi64x(a1, a2, a3, a4);
		b = algo.popcount_avx2_64_old_v2(a);

		c = U256i {b};
		EXPECT_EQ(c.b[0], __builtin_popcountll(a1));
		EXPECT_EQ(c.b[1], __builtin_popcountll(a2));
		EXPECT_EQ(c.b[2], __builtin_popcountll(a3));
		EXPECT_EQ(c.b[3], __builtin_popcountll(a4));
	}
}

TEST(Bruteforce, n32) {
	constexpr static WindowedAVX2_Config config{32, 1, 1, 32, LS, 10, 5, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_32(LS, LS);
	EXPECT_GT(algo.solutions_nr, 0);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, avx_32) {
	constexpr static WindowedAVX2_Config config{32, 1, 1, 32, LS, 10, 5, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_avx2_32(LS, LS);
	EXPECT_GT(algo.solutions_nr, 0);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, n64) {
	constexpr static WindowedAVX2_Config config{64, 1, 1, 64, LS, 10, 5, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_64(LS, LS);

	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, avx2_64) {
	constexpr static WindowedAVX2_Config config{64, 1, 1, 64, LS, 10, 5, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_avx2_64(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, avx2_64_1x1) {
	constexpr static WindowedAVX2_Config config{64, 1, 1, 64, LS, 10, 5, 0, 512};
	WindowedAVX2<config> algo{};

	for (uint32_t i = 0; i < 10; ++i) {
		algo.generate_random_instance();
		algo.bruteforce_avx2_64_1x1(LS, LS);
		EXPECT_EQ(algo.solutions_nr, 1);
		EXPECT_EQ(algo.all_solutions_correct(), true);
		algo.solutions_nr = 0;

		free(algo.L1);
		free(algo.L2);
		algo.L1 = nullptr;
		algo.L2 = nullptr;
	}
}

TEST(Bruteforce, avx2_64_uxv) {
	constexpr static WindowedAVX2_Config config{64, 1, 1, 64, LS, 10, 5, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_avx2_64_uxv<1,1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv<2,2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv<4,4>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv<8,8>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv<1,2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv<2,1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv<4,2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

}

// TODO not working
//TEST(Bruteforce, avx2_64_uxv_shuffle) {
//	constexpr static WindowedAVX2_Config config{64, 1, 1, LS, 32, 17, 0, 512};
//	WindowedAVX2<config> algo{};
//	algo.generate_random_instance();
//
//	algo.bruteforce_avx2_64_uxv_shuffle<1,1>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//	algo.solutions_nr = 0;
//
//	algo.bruteforce_avx2_64_uxv_shuffle<2,2>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//	algo.solutions_nr = 0;
//
//	algo.bruteforce_avx2_64_uxv_shuffle<4,4>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//	algo.solutions_nr = 0;
//
//	algo.bruteforce_avx2_64_uxv_shuffle<8,8>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//	algo.solutions_nr = 0;
//
//	algo.bruteforce_avx2_64_uxv_shuffle<1,2>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//	algo.solutions_nr = 0;
//
//	algo.bruteforce_avx2_64_uxv_shuffle<2,1>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//	algo.solutions_nr = 0;
//
//	algo.bruteforce_avx2_64_uxv_shuffle<4,2>(LS, LS);
//	EXPECT_EQ(algo.solutions_nr, 1);
//  EXPECT_EQ(algo.all_solutions_correct(), true);
//	algo.solutions_nr = 0;
//
//}

TEST(Bruteforce, n128) {
	constexpr static WindowedAVX2_Config config{128, 2, 1, 64, LS, 48, 32, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_128(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, avx_128) {
	constexpr static WindowedAVX2_Config config{128, 1, 1, 64, LS, 12, 6, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_avx2_128(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, avx_uxv_128) {
	constexpr static WindowedAVX2_Config config{128, 1, 1, 64, LS, 48, 6, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_avx2_128_32_2_uxv<1, 1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_128_32_2_uxv<2, 2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_128_32_2_uxv<4, 4>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_128_32_2_uxv<8, 8>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	// TODO
	//algo.bruteforce_avx2_128_32_2_uxv<1, 3>(LS, LS);
	//EXPECT_EQ(algo.solutions_nr, 1);
	//EXPECT_EQ(algo.all_solutions_correct(), true);
	//algo.solutions_nr = 0;

	//algo.bruteforce_avx2_128_32_2_uxv<3, 1>(LS, LS);
	//EXPECT_EQ(algo.solutions_nr, 1);
	//EXPECT_EQ(algo.all_solutions_correct(), true);
	//algo.solutions_nr = 0;
}

// takes to long
//TEST(Bruteforce, n256) {
//	constexpr static WindowedAVX2_Config config{256, 4, 1, 64, LS, 80, 50, 0, 512};
//	WindowedAVX2<config> algo{};
//	algo.generate_random_instance();
//
//
//	if constexpr (LS > 1u << 16) {
//		algo.bruteforce_256(LS, LS);
//		EXPECT_EQ(algo.solutions_nr, 1);
//		EXPECT_EQ(algo.all_solutions_correct(), true);
//	} else {
//		for (uint32_t i = 0; i < 1; ++i) {
//			algo.bruteforce_256(LS, LS);
//			EXPECT_EQ(algo.solutions_nr, 1);
//			EXPECT_EQ(algo.all_solutions_correct(), true);
//			algo.solutions_nr = 0;
//
//			free(algo.L1);
//			free(algo.L2);
//			algo.generate_random_instance();
//		}
//	}
//}

TEST(Bruteforce, avx_256) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, 64, LS, 80, 50, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	if constexpr (LS > 1u << 16) {
		algo.bruteforce_avx2_256(LS, LS);
		EXPECT_EQ(algo.solutions_nr, 1);
		EXPECT_EQ(algo.all_solutions_correct(), true);
	} else {
		for (uint32_t i = 0; i < 1; ++i) {
			algo.bruteforce_avx2_256(LS, LS);
			EXPECT_EQ(algo.solutions_nr, 1);
			EXPECT_EQ(algo.all_solutions_correct(), true);
			algo.solutions_nr = 0;

			free(algo.L1);
			free(algo.L2);
			algo.generate_random_instance();
		}
	}
}

TEST(Bruteforce, avx2_256_ux4) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, 64, LS, 80, 20, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_avx2_256_ux4<1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_256_ux4<2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_256_ux4<4>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_256_ux4<8>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}


TEST(Bruteforce, avx2_256_32_ux8) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, 64, LS, 25, 4, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_avx2_256_32_ux8<1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_256_32_ux8<2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_256_32_ux8<4>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_256_32_ux8<8>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}

TEST(Bruteforce, avx2_256_32_8x8) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, 64, LS, 25, 12, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_avx2_256_32_8x8(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}

TEST(Bruteforce, avx2_256_64_4x4) {
	constexpr static WindowedAVX2_Config config{256, 4, 1, 64, LS, 30, 16, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	if constexpr (LS > 1u << 16) {
		algo.bruteforce_avx2_256_64_4x4(LS, LS);
		EXPECT_EQ(algo.solutions_nr, 1);
		EXPECT_EQ(algo.all_solutions_correct(), true);
	} else {
		for (int i = 0; i < 10000; ++i) {
			algo.bruteforce_avx2_256_64_4x4(LS, LS);
			EXPECT_EQ(algo.solutions_nr, 1);
			EXPECT_EQ(algo.all_solutions_correct(), true);
			algo.solutions_nr = 0;

			free(algo.L1);
			free(algo.L2);
			algo.generate_random_instance();
		}
	}
}


TEST(NearestNeighborAVX, avx2_sort_nn_on64) {
	constexpr size_t LS = 1u << 18u;
	constexpr static WindowedAVX2_Config config{256, 4, 320, 64, LS, 22, 16, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	const uint64_t z = fastrandombytes_uint64();
	size_t e1 = algo.avx2_sort_nn_on64_simple<0>(LS, z, algo.L1);
	size_t e2 = algo.avx2_sort_nn_on64<0>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	e1 = algo.avx2_sort_nn_on64_simple<1>(LS, z, algo.L1);
	e2 = algo.avx2_sort_nn_on64<1>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	e1 = algo.avx2_sort_nn_on64_simple<2>(LS, z, algo.L1);
	e2 = algo.avx2_sort_nn_on64<2>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}


	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	e1 =algo.avx2_sort_nn_on64_simple<3>(LS, z, algo.L1);
	e2 =algo.avx2_sort_nn_on64<3>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}
}

TEST(NearestNeighborAVX, avx2_sort_nn_on_double64) {
	constexpr size_t LS = 1u << 18u;
	constexpr static WindowedAVX2_Config config{256, 4, 320, 64, LS, 22, 16, 0, 512};
	WindowedAVX2<config> algo1{};
	WindowedAVX2<config> algo2{};
	algo1.generate_random_instance();
	// once needed
	algo2.generate_random_instance();
	memcpy(algo2.L1, algo1.L1, LS*4*8);
	memcpy(algo2.L2, algo1.L2, LS*4*8);

	const uint64_t z = fastrandombytes_uint64();
	size_t e11 = algo1.avx2_sort_nn_on64_simple<0>(LS, z, algo1.L1);
	size_t e12 = algo1.avx2_sort_nn_on64_simple<0>(LS, z, algo1.L2);
	size_t e21=0, e22=0;
	algo2.avx2_sort_nn_on_double64<0, 8>(LS, LS, e21, e22, z);
	EXPECT_EQ(e11, e21);
	EXPECT_EQ(e12, e22);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo1.L1[i][j], algo2.L1[i][j]);
			ASSERT_EQ(algo1.L2[i][j], algo2.L2[i][j]);
		}
	}

	free(algo1.L1);
	free(algo1.L2);
	algo1.generate_random_instance();
	memcpy(algo2.L1, algo1.L1, LS*4*8);
	memcpy(algo2.L2, algo1.L2, LS*4*8);
	e21=0, e22=0;

	e11 = algo1.avx2_sort_nn_on64_simple<1>(LS, z, algo1.L1);
	e12 = algo1.avx2_sort_nn_on64_simple<1>(LS, z, algo1.L2);
	algo2.avx2_sort_nn_on_double64<1, 2>(LS, LS, e21, e22, z);
	EXPECT_EQ(e11, e21);
	EXPECT_EQ(e12, e22);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo1.L1[i][j], algo2.L1[i][j]);
			ASSERT_EQ(algo1.L2[i][j], algo2.L2[i][j]);
		}
	}

	free(algo1.L1);
	free(algo1.L2);
	algo1.generate_random_instance();
	memcpy(algo2.L1, algo1.L1, LS*4*8);
	memcpy(algo2.L2, algo1.L2, LS*4*8);
	e21=0, e22=0;

	e11 = algo1.avx2_sort_nn_on64_simple<2>(LS, z, algo1.L1);
	e12 = algo1.avx2_sort_nn_on64_simple<2>(LS, z, algo1.L2);
	algo2.avx2_sort_nn_on_double64<2, 4>(LS, LS, e21, e22, z);
	EXPECT_EQ(e11, e21);
	EXPECT_EQ(e12, e22);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo1.L1[i][j], algo2.L1[i][j]);
			ASSERT_EQ(algo1.L2[i][j], algo2.L2[i][j]);
		}
	}
}

TEST(NearestNeighborAVX, avx2_sort_nn_on32) {
	constexpr size_t LS = 1u << 18u;
	constexpr static WindowedAVX2_Config config{256, 8, 320, 32, LS, 10, 8, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	uint32_t z = fastrandombytes_uint64();
	size_t e1 = algo.avx2_sort_nn_on32_simple<0>(LS, z, algo.L1);
	size_t e2 = algo.avx2_sort_nn_on32<0>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	z = fastrandombytes_uint64();
	e1 = algo.avx2_sort_nn_on32_simple<1>(LS, z, algo.L1);
	e2 = algo.avx2_sort_nn_on32<1>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	z = fastrandombytes_uint64();
	e1 = algo.avx2_sort_nn_on32_simple<2>(LS, z, algo.L1);
	e2 = algo.avx2_sort_nn_on32<2>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}


	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	z = fastrandombytes_uint64();
	e1 =algo.avx2_sort_nn_on32_simple<3>(LS, z, algo.L1);
	e2 =algo.avx2_sort_nn_on32<3>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}
}

TEST(NearestNeighborAVX, avx2_sort_nn_on_double32) {
	constexpr size_t LS = 1u << 18u;
	constexpr static WindowedAVX2_Config config{256, 8, 320, 32, LS, 10, 12, 0, 512};
	WindowedAVX2<config> algo1{};
	WindowedAVX2<config> algo2{};
	algo1.generate_random_instance();
	// once needed
	algo2.generate_random_instance();
	memcpy(algo2.L1, algo1.L1, LS*4*8);
	memcpy(algo2.L2, algo1.L2, LS*4*8);

	const uint64_t z = fastrandombytes_uint64();
	size_t e11 = algo1.avx2_sort_nn_on32_simple<0>(LS, z, algo1.L1);
	size_t e12 = algo1.avx2_sort_nn_on32_simple<0>(LS, z, algo1.L2);
	size_t e21=0, e22=0;
	algo2.avx2_sort_nn_on_double32<0, 8>(LS, LS, e21, e22, z);
	EXPECT_EQ(e11, e21);
	EXPECT_EQ(e12, e22);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo1.L1[i][j], algo2.L1[i][j]);
			ASSERT_EQ(algo1.L2[i][j], algo2.L2[i][j]);
		}
	}

	free(algo1.L1);
	free(algo1.L2);
	algo1.generate_random_instance();
	memcpy(algo2.L1, algo1.L1, LS*4*8);
	memcpy(algo2.L2, algo1.L2, LS*4*8);
	e21=0, e22=0;

	e11 = algo1.avx2_sort_nn_on32_simple<1>(LS, z, algo1.L1);
	e12 = algo1.avx2_sort_nn_on32_simple<1>(LS, z, algo1.L2);
	algo2.avx2_sort_nn_on_double32<1, 2>(LS, LS, e21, e22, z);
	EXPECT_EQ(e11, e21);
	EXPECT_EQ(e12, e22);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo1.L1[i][j], algo2.L1[i][j]);
			ASSERT_EQ(algo1.L2[i][j], algo2.L2[i][j]);
		}
	}

	free(algo1.L1);
	free(algo1.L2);
	algo1.generate_random_instance();
	memcpy(algo2.L1, algo1.L1, LS*4*8);
	memcpy(algo2.L2, algo1.L2, LS*4*8);
	e21=0, e22=0;

	e11 = algo1.avx2_sort_nn_on32_simple<2>(LS, z, algo1.L1);
	e12 = algo1.avx2_sort_nn_on32_simple<2>(LS, z, algo1.L2);
	algo2.avx2_sort_nn_on_double32<2, 4>(LS, LS, e21, e22, z);
	EXPECT_EQ(e11, e21);
	EXPECT_EQ(e12, e22);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo1.L1[i][j], algo2.L1[i][j]);
			ASSERT_EQ(algo1.L2[i][j], algo2.L2[i][j]);
		}
	}
}


// k is not a multiple of 32
TEST(NearestNeighborAVX, avx2_sort_nn_on32_k) {
	constexpr size_t LS = 1u << 18u;
	constexpr static WindowedAVX2_Config config{256, 8, 320, 28, LS, 8, 4, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	uint32_t z = fastrandombytes_uint64();
	size_t e1 = algo.avx2_sort_nn_on32_simple<0>(LS, z, algo.L1);
	size_t e2 = algo.avx2_sort_nn_on32<0>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	z = fastrandombytes_uint64();
	e1 = algo.avx2_sort_nn_on32_simple<1>(LS, z, algo.L1);
	e2 = algo.avx2_sort_nn_on32<1>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			ASSERT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	z = fastrandombytes_uint64();
	e1 = algo.avx2_sort_nn_on32_simple<2>(LS, z, algo.L1);
	e2 = algo.avx2_sort_nn_on32<2>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}


	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	z = fastrandombytes_uint64();
	e1 =algo.avx2_sort_nn_on32_simple<3>(LS, z, algo.L1);
	e2 =algo.avx2_sort_nn_on32<3>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}
}

TEST(NearestNeighborAVX, avx2_sort_nn_on64_k) {
	constexpr size_t LS = 1u << 18u;
	constexpr static WindowedAVX2_Config config{256, 4, 320, 58, LS, 18, 16, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	const uint64_t z = fastrandombytes_uint64();
	size_t e1 = algo.avx2_sort_nn_on64_simple<0>(LS, z, algo.L1);
	size_t e2 = algo.avx2_sort_nn_on64<0>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	e1 = algo.avx2_sort_nn_on64_simple<1>(LS, z, algo.L1);
	e2 = algo.avx2_sort_nn_on64<1>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}

	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	e1 = algo.avx2_sort_nn_on64_simple<2>(LS, z, algo.L1);
	e2 = algo.avx2_sort_nn_on64<2>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}


	free(algo.L1);
	free(algo.L2);
	algo.generate_random_instance();
	memcpy(algo.L1, algo.L2, LS*4*8);

	e1 =algo.avx2_sort_nn_on64_simple<3>(LS, z, algo.L1);
	e2 =algo.avx2_sort_nn_on64<3>(LS, z, algo.L2);
	EXPECT_EQ(e1, e2);

	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < 4; ++j) {
			EXPECT_EQ(algo.L1[i][j], algo.L2[i][j]);
		}
	}
}


TEST(NearestNeighborAVX, MO640Params_n128_r2_64) {
	constexpr size_t LS = 1u << 14u;
	constexpr uint32_t nr_tries = 1000;
	uint32_t sols = 0;
	// 100%
	constexpr static WindowedAVX2_Config config64{128, 2, 250, 64, LS, 24, 11, 0, 600};

	WindowedAVX2<config64> algo64{};
	algo64.generate_random_instance();

	for (size_t i = 0; i < nr_tries; i++) {
		algo64.avx2_nn(LS, LS);
		sols += algo64.solutions_nr;
		algo64.solutions_nr = 0;

		free(algo64.L1);
		free(algo64.L2);
		algo64.generate_random_instance();
	}

	ASSERT_EQ(sols, nr_tries);
}

TEST(NearestNeighborAVX, MO640Params_n128_r2_64_masked) {
	constexpr size_t LS = 1u << 14u;
	constexpr uint32_t nr_tries = 1000;
	uint32_t sols = 0;

	// optimal: d=14, N=160
	//		100% success rate for d=17, N=160
	//		NOTE: that d=14 or d=17 doesn't matter for the runtime because below 600 elements are surviving
	//		NOTE: after the optimal d was found, I optimized the N
	// 87% runtime in Bruteforce
	//constexpr static WindowedAVX2_Config config64{128, 2, 180, 48, LS, 17, 11, 0, 600};

	// MUVH better
	// close to 100% correctners: rt 7-10s
	constexpr static WindowedAVX2_Config config64{128, 2, 1000, 48, LS, 15, 11, 0, 512};


	WindowedAVX2<config64> algo64{};
	algo64.generate_random_instance();

	for (size_t i = 0; i < nr_tries; i++) {
		algo64.avx2_nn(LS, LS);
		sols += algo64.solutions_nr;
		algo64.solutions_nr = 0;
	}

	ASSERT_EQ(sols, nr_tries);
}

TEST(NearestNeighborAVX, MO640Params_n128_r4_32) {
	constexpr size_t LS = 1u << 14u;

	// %77~80% correctness
	constexpr static WindowedAVX2_Config config2{128, 4, 20, 32, LS, 13, 11, 0, 1024};

	// Nearly 100%
	//constexpr static WindowedAVX2_Config config2{128, 4, 10, 32, LS, 14, 11, 0, 1024};
	WindowedAVX2<config2> algokek{};
	algokek.generate_random_instance();

	constexpr uint32_t nr_tries = 1000;
	uint32_t sols = 0;
	for (size_t i = 0; i < nr_tries; i++) {
		algokek.avx2_nn(LS, LS);
		sols += algokek.solutions_nr;
		algokek.solutions_nr = 0;

		free(algokek.L1);
		free(algokek.L2);
		algokek.generate_random_instance();
	}

	ASSERT_EQ(sols, nr_tries);
}


TEST(NearestNeighborAVX, MO1284Params_n256_r4) {
	//constexpr size_t LS = 1u << 18u;
	//constexpr static WindowedAVX2_Config config{256, 4, 300, 64, LS, 22, 16, 0, 512};
	//constexpr static WindowedAVX2_Config config{256, 4, 300, 64, LS, 20, 20, 0, 512};

	constexpr size_t LS = 1u << 20u;
	//constexpr static WindowedAVX2_Config config{256, 4, 100, 64, LS, 22, 20, 0, 5000};
	//constexpr static WindowedAVX2_Config config{256, 4, 300, 64, LS, 20, 20, 0, 512};
	//constexpr static WindowedAVX2_Config config{256, 4, 300, 64, LS, 20, 20, 0, 5000};

	// NN_LOWER all find all solutions
	//constexpr static WindowedAVX2_Config config{256, 4, 100, 64, LS, 25, 14, 0, 512}; //44s
	//constexpr static WindowedAVX2_Config config{256, 4, 150, 64, LS, 24, 14, 0, 512}; // 51s
	constexpr static WindowedAVX2_Config config{256, 4, 300, 64, LS, 23, 14, 0, 512}; //2

	// NN_EQUAL: (NOT correct I think)
	//constexpr static WindowedAVX2_Config config{256, 4, 300, 64, LS, 25, 14, 0, 512}; //
	//constexpr static WindowedAVX2_Config config{256, 4, 150, 64, LS, 24, 14, 0, 512}; // 51s

	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	constexpr uint32_t nr_tries = 10;
	uint32_t sols = 0;
	for (size_t i = 0; i < nr_tries; i++) {
		algo.avx2_nn(LS, LS);
		sols += algo.solutions_nr;
		algo.solutions_nr = 0;

		free(algo.L1);
		free(algo.L2);
		algo.generate_random_instance();
	}

	EXPECT_EQ(sols, nr_tries);
}

TEST(NearestNeighborAVX, MO431Params_n84_r3) {
	constexpr size_t LS = 1u << 14u;
	//constexpr static WindowedAVX2_Config config2{86, 3, 108, 32, LS, 12, 6, 0, 1024}; // %100
	//constexpr static WindowedAVX2_Config config2{86, 3, 60, 32, LS, 12, 6, 0, 1024}; // %100
	constexpr static WindowedAVX2_Config config2{86, 3, 10, 32, LS, 13, 6, 0, 700}; // %100
	//constexpr static WindowedAVX2_Config config2{86, 3, 50, 32, LS, 11, 6, 0, 1024};
	WindowedAVX2<config2> algokek{};
	algokek.generate_random_instance();

	constexpr uint32_t nr_tries = 1;
	uint32_t sols = 0;
	for (size_t i = 0; i < nr_tries; i++) {
		algokek.avx2_nn(LS, LS);
		sols += algokek.solutions_nr;
		algokek.solutions_nr = 0;

		free(algokek.L1);
		free(algokek.L2);
		algokek.generate_random_instance();
	}

	EXPECT_EQ(sols, nr_tries);
}

TEST(NearestNeighborAVX, Dev) {
	constexpr static WindowedAVX2_Config config{256, 4, 500, 64, 1u<<20, 21, 14, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();
	algo.run();
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	//algo.bench();
}

#ifdef __AVX512F__
TEST(Bruteforce, avx512_32_8x8) {
	constexpr static WindowedAVX2_Config config{32, 1, 1, 1, LS, 2, 4, 0, 512};
	WindowedAVX2<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_avx512_32_16x16(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}
#endif

int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
