#include <gtest/gtest.h>
#include <cstdint>

#define NN_CONFIG_SET
#define TEST_BASE_LIST_SIZE (1u << 10u)

#include "helper.h"
#include "nn/nn.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr size_t LS = 1u << 10u;
constexpr static NN_Config global_config{256, 4, 20, 64, LS, 22, 16, 0, 512};


TEST(Bruteforce, n32) {
	constexpr static NN_Config config{32, 1, 1, 32, LS, 10, 5, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_32(LS, LS);
	EXPECT_GT(algo.solutions_nr, 0);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

#ifdef USE_AVX2
TEST(Bruteforce, avx_32) {
	constexpr static NN_Config config{32, 1, 1, 32, LS, 10, 5, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_avx2_32(LS, LS);
	EXPECT_GT(algo.solutions_nr, 0);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}
#endif

TEST(Bruteforce, n64) {
	constexpr static NN_Config config{64, 1, 1, 64, LS, 10, 5, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_64(LS, LS);

	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

#ifdef USE_AVX2
TEST(Bruteforce, avx2_64) {
	constexpr static NN_Config config{64, 1, 1, 64, LS, 10, 5, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_avx2_64(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, avx2_64_1x1) {
	constexpr static NN_Config config{64, 1, 1, 64, LS, 10, 5, 0, 512};
	NN<config> algo{};

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
	constexpr static NN_Config config{64, 1, 1, 64, LS, 10, 5, 0, 512};
	NN<config> algo{};
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
TEST(Bruteforce, avx2_64_uxv_shuffle) {
	constexpr static NN_Config config{64, 1, 1, 64, LS, 32, 17, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_avx2_64_uxv_shuffle<1,1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv_shuffle<2,2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv_shuffle<4,4>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv_shuffle<8,8>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv_shuffle<1,2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv_shuffle<2,1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	algo.solutions_nr = 0;

	algo.bruteforce_avx2_64_uxv_shuffle<4,2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
  	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}
#endif

TEST(Bruteforce, n128) {
	constexpr static NN_Config config{128, 2, 1, 64, LS, 48, 32, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_128(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

#ifdef USE_AVX2
TEST(Bruteforce, avx_128) {
	constexpr static NN_Config config{128, 1, 1, 64, LS, 12, 6, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_avx2_128(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, avx_uxv_128) {
	constexpr static NN_Config config{128, 1, 1, 64, LS, 48, 6, 0, 512};
	NN<config> algo{};
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
}
#endif

// takes to long
TEST(Bruteforce, n256) {
	constexpr static NN_Config config{256, 4, 1, 64, LS, 80, 50, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();


	if constexpr (LS > 1u << 16) {
		algo.bruteforce_256(LS, LS);
		EXPECT_EQ(algo.solutions_nr, 1);
		EXPECT_EQ(algo.all_solutions_correct(), true);
	} else {
		for (uint32_t i = 0; i < 1; ++i) {
			algo.bruteforce_256(LS, LS);
			EXPECT_EQ(algo.solutions_nr, 1);
			EXPECT_EQ(algo.all_solutions_correct(), true);
			algo.solutions_nr = 0;

			free(algo.L1);
			free(algo.L2);
			algo.generate_random_instance();
		}
	}
}

#ifdef USE_AVX2
TEST(Bruteforce, avx_256) {
	constexpr static NN_Config config{256, 4, 1, 64, LS, 80, 50, 0, 512};
	NN<config> algo{};
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
	constexpr static NN_Config config{256, 4, 1, 64, LS, 80, 20, 0, 512};
	NN<config> algo{};
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
	constexpr static NN_Config config{256, 4, 1, 64, LS, 25, 4, 0, 512};
	NN<config> algo{};
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
	constexpr static NN_Config config{256, 4, 1, 64, LS, 25, 12, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_avx2_256_32_8x8(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}

TEST(Bruteforce, avx2_256_64_4x4) {
	constexpr static NN_Config config{256, 4, 1, 64, LS, 30, 16, 0, 512};
	NN<config> algo{};
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

TEST(Bruteforce, avx2_256_64_4x4_rearrange) {
	constexpr size_t LS = 652;
	constexpr static NN_Config config__{256, 4, 1, 64, LS, 10, 14, 0, 512};
	NN<config__> algo{};
	algo.generate_random_instance();
	algo.transpose(LS);
	algo.bruteforce_avx2_256_64_4x4_rearrange<LS>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}


TEST(NearestNeighborAVX, avx2_sort_nn_on64) {
	constexpr size_t LS = 1u << 18u;
	constexpr static NN_Config config{256, 4, 320, 64, LS, 22, 16, 0, 512};
	NN<config> algo{};
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
	constexpr static NN_Config config{256, 4, 320, 64, LS, 22, 16, 0, 512};
	NN<config> algo1{};
	NN<config> algo2{};
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
	constexpr static NN_Config config{256, 8, 320, 32, LS, 10, 8, 0, 512};
	NN<config> algo{};
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
	constexpr static NN_Config config{256, 8, 320, 32, LS, 10, 12, 0, 512};
	NN<config> algo1{};
	NN<config> algo2{};
	algo1.generate_random_instance();
	// once needed
	algo2.generate_random_instance();
	memcpy(algo2.L1, algo1.L1, LS*4*8);
	memcpy(algo2.L2, algo1.L2, LS*4*8);

	const uint64_t z = fastrandombytes_uint64();
	size_t e11 = algo1.avx2_sort_nn_on32_simple<0>(LS, z, algo1.L1);
	size_t e12 = algo1.avx2_sort_nn_on32_simple<0>(LS, z, algo1.L2);
	size_t e21=0, e22=0;
	algo2.avx2_sort_nn_on_double32<0, 7>(LS, LS, e21, e22, z);
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

TEST(NearestNeighborAVX, avx2_sort_nn_on_double32_allcorrect) {
	/// special test in which every elements should be marked as correct by the NN
	constexpr size_t LS = 1u << 12u;
	constexpr uint32_t dk = 10;
	constexpr static NN_Config config{256, 8, 1, 32, LS, dk, 12, 0, 512};
	NN<config> algo1{};
	algo1.generate_special_instance();

	uint32_t z;
	size_t e1=LS, e2=LS, new_e1=0, new_e2=0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	algo1.avx2_sort_nn_on_double32<0, 1>(e1, e2, new_e1, new_e2, z);
	EXPECT_EQ(new_e1, LS);
	EXPECT_EQ(new_e2, LS);
	new_e1 = 0; new_e2 = 0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	algo1.avx2_sort_nn_on_double32<0, 2>(e1, e2, new_e1, new_e2, z);
	EXPECT_EQ(new_e1, LS);
	EXPECT_EQ(new_e2, LS);
	new_e1 = 0; new_e2 = 0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	algo1.avx2_sort_nn_on_double32<0, 4>(e1, e2, new_e1, new_e2, z);
	EXPECT_EQ(new_e1, LS);
	EXPECT_EQ(new_e2, LS);

	new_e1 = 0; new_e2 = 0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	algo1.avx2_sort_nn_on_double32<1, 1>(e1, e2, new_e1, new_e2, z);
	EXPECT_EQ(new_e1, LS);
	EXPECT_EQ(new_e2, LS);
	new_e1 = 0; new_e2 = 0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	algo1.avx2_sort_nn_on_double32<1, 2>(e1, e2, new_e1, new_e2, z);
	EXPECT_EQ(new_e1, LS);
	EXPECT_EQ(new_e2, LS);
	new_e1 = 0; new_e2 = 0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	algo1.avx2_sort_nn_on_double32<1, 4>(e1, e2, new_e1, new_e2, z);
	EXPECT_EQ(new_e1, LS);
	EXPECT_EQ(new_e2, LS);
}

TEST(NearestNeighborAVX, avx2_sort_nn_on_32_allcorrect) {
	/// special test in which every elements should be marked as correct by the NN
	constexpr size_t LS = 1u << 12u;
	constexpr uint32_t dk = 10;
	constexpr static NN_Config config{256, 8, 1, 32, LS, dk, 12, 0, 512};
	NN<config> algo1{};
	algo1.generate_special_instance();

	uint32_t z;
	size_t e1=LS,new_e1=0;
	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on32<0>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);
	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on32<1>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);
	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on32<2>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);
	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on32<3>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);

	algo1.L1[fastrandombytes_uint64()%LS][0] = -1ull;
	algo1.L1[fastrandombytes_uint64()%LS][1] = -1ull;
	algo1.L1[fastrandombytes_uint64()%LS][2] = -1ull;
	algo1.L1[fastrandombytes_uint64()%LS][3] = -1ull;

	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on32<0>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on32<1>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on32<2>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
	z = fastrandombytes_weighted<uint32_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on32<3>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
}


TEST(NearestNeighborAVX, avx2_sort_nn_on_64_allcorrect) {
	/// special test in which every elements should be marked as correct by the NN
	constexpr size_t LS = 1u << 14u;
	constexpr uint32_t dk = 10;
	constexpr static NN_Config config{256, 4, 1, 64, LS, dk, 12, 0, 512};
	NN<config> algo1{};
	algo1.generate_special_instance();

	uint64_t z;
	size_t e1=LS,new_e1=0;
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on64<0>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on64<1>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on64<2>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on64<3>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS);

	algo1.L1[fastrandombytes_uint64()%LS][0] = -1ull;
	algo1.L1[fastrandombytes_uint64()%LS][1] = -1ull;
	algo1.L1[fastrandombytes_uint64()%LS][2] = -1ull;
	algo1.L1[fastrandombytes_uint64()%LS][3] = -1ull;

	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on64<0>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on64<1>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on64<2>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
	z = fastrandombytes_weighted<uint64_t>(dk);
	new_e1 = algo1.avx2_sort_nn_on64<3>(e1, z, algo1.L1);
	EXPECT_EQ(new_e1, LS-1);
}
// k is not a multiple of 32
TEST(NearestNeighborAVX, avx2_sort_nn_on32_k) {
	constexpr size_t LS = 1u << 18u;
	constexpr static NN_Config config{256, 8, 320, 28, LS, 8, 4, 0, 512};
	NN<config> algo{};
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
	constexpr static NN_Config config{256, 4, 320, 58, LS, 18, 16, 0, 512};
	NN<config> algo{};
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
	constexpr static NN_Config config64{128, 2, 250, 64, LS, 24, 11, 0, 600};

	NN<config64> algo64{};
	algo64.generate_random_instance();

	for (size_t i = 0; i < nr_tries; i++) {
		algo64.nn(LS, LS);
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
	//constexpr static NN_Config config64{128, 2, 180, 48, LS, 17, 11, 0, 600};

	// MUVH better
	// close to 100% correctners: rt 7-10s
	constexpr static NN_Config config64{128, 2, 1000, 48, LS, 15, 11, 0, 512};


	NN<config64> algo64{};
	algo64.generate_random_instance();

	for (size_t i = 0; i < nr_tries; i++) {
		algo64.nn(LS, LS);
		sols += algo64.solutions_nr;
		algo64.solutions_nr = 0;
	}

	ASSERT_EQ(sols, nr_tries);
}

TEST(NearestNeighborAVX, MO640Params_n128_r4_32) {
	constexpr size_t LS = 1u << 14u;

	// %77~80% correctness
	constexpr static NN_Config config2{128, 4, 20, 32, LS, 13, 11, 0, 1024};

	// Nearly 100%
	//constexpr static NN_Config config2{128, 4, 10, 32, LS, 14, 11, 0, 1024};
	NN<config2> algokek{};
	algokek.generate_random_instance();

	constexpr uint32_t nr_tries = 1000;
	uint32_t sols = 0;
	for (size_t i = 0; i < nr_tries; i++) {
		algokek.nn(LS, LS);
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
	//constexpr static NN_Config config{256, 4, 300, 64, LS, 22, 16, 0, 512};
	//constexpr static NN_Config config{256, 4, 300, 64, LS, 20, 20, 0, 512};

	constexpr size_t LS = 1u << 20u;
	//constexpr static NN_Config config{256, 4, 100, 64, LS, 22, 20, 0, 5000};
	//constexpr static NN_Config config{256, 4, 300, 64, LS, 20, 20, 0, 512};
	//constexpr static NN_Config config{256, 4, 300, 64, LS, 20, 20, 0, 5000};

	// NN_LOWER all find all solutions
	//constexpr static NN_Config config{256, 4, 100, 64, LS, 25, 14, 0, 512}; 	// 44s
	//constexpr static NN_Config config{256, 4, 150, 64, LS, 24, 14, 0, 512}; 	// 51s
	//constexpr static NN_Config config{256, 4, 300, 64, LS, 23, 14, 0, 512}; 	// 33.9s
	//constexpr static NN_Config config{256, 4, 330, 64, LS, 23, 14, 0, 10000}; 	// 35,5
	//constexpr static NN_Config config{256, 4, 600, 64, LS, 22, 14, 0, 512};
	//constexpr static NN_Config config{256, 4, 1146, 64, LS, 20, 14, 0, 1024};	// 6/10in 40s
	//constexpr static NN_Config config{256, 4, 15146, 64, LS, 19, 14, 0, 1024};	// 93s
	//constexpr static NN_Config config{256, 4, 4000, 64, LS, 19, 14, 0, 1024};//7/10 in 63

	// NOT WORKING: Theoretically the best configuration
	//constexpr static NN_Config config{256, 4, 5731, 64, LS, 22, 14, 0, 1000};
	//constexpr static NN_Config config{256, 8, 15, 32, LS, 13, 14, 0, 1024};
	//constexpr static NN_Config config{256, 4, 44, 64, LS, 31, 14, 0, 1024};


	// NN_LOWER find all solutions
	//constexpr static NN_Config config{256, 8, 100, 32, LS, 11, 14, 0, 1000}; // 69
	//constexpr static NN_Config config{256, 8, 250, 32, LS, 10, 14, 0, 512}; // 57s
	//constexpr static NN_Config config{256, 8, 250, 32, LS, 10, 14, 0, 1000}; // 30,1 27
	//constexpr static NN_Config config{256, 8, 220, 32, LS, 10, 14, 0, 1000}; // 9/10: 19.17
	//constexpr static NN_Config config{256, 8, 200, 32, LS, 10, 14, 0, 1000}; //10/10 in 11.61 9/10 in 19.16
	//constexpr static NN_Config config{256, 8, 180, 32, LS, 10, 14, 0, 1000}; // 9/10 in 13.273
	constexpr static NN_Config config{256, 8, 150, 32, LS, 10, 14, 0, 1000}; // 9/10 in 13.563
	//constexpr static NN_Config config{256, 8, 150, 32, LS, 10, 14, 0, 1000, 10}; // 10/10 in 7s,10s
	//constexpr static NN_Config config{256, 8, 150, 32, LS, 10, 14, 0, 1000, 8}; // 8.2
	//constexpr static NN_Config config{256, 8, 150, 32, LS, 10, 14, 0, 1000, 5}; // 7/10: 14s
	//constexpr static NN_Config config{256, 8, 120, 32, LS, 10, 14, 0, 1000}; // 10/10 in 5.3 8/10 in 11.7
	//constexpr static NN_Config config{256, 8, 500, 32, LS, 9, 14, 0, 512}; //

	// NN_EQUAL: (NOT correct I think)
	//constexpr static NN_Config config{256, 4, 1000, 64, LS, 23, 14, 0, 512}; // 3/10 and 109
	//constexpr static NN_Config config{256, 4, 150, 64, LS, 24, 14, 0, 512}; // 51s

	NN<config> algo{};
	config.print();

	/// if set to false, no solution will be inserted. Good to get worst case runtimes
	constexpr bool solution = true;
	algo.generate_random_instance(solution);
	constexpr uint32_t nr_tries = 10;
	uint32_t sols = 0;
	for (size_t i = 0; i < nr_tries; i++) {
		algo.nn(LS, LS);
		sols += algo.solutions_nr;
		algo.solutions_nr = 0;

		free(algo.L1);
		free(algo.L2);
		algo.generate_random_instance(solution);
	}

	EXPECT_EQ(sols, nr_tries);
}

TEST(NearestNeighborAVX, MO431Params_n84_r3) {
	constexpr size_t LS = 1u << 14u;
	//constexpr static NN_Config config2{86, 3, 108, 32, LS, 12, 6, 0, 1024}; // %100
	//constexpr static NN_Config config2{86, 3, 60, 32, LS, 12, 6, 0, 1024}; // %100
	constexpr static NN_Config config2{86, 3, 10, 32, LS, 13, 6, 0, 700}; // %100
	//constexpr static NN_Config config2{86, 3, 50, 32, LS, 11, 6, 0, 1024};
	NN<config2> algokek{};
	algokek.generate_random_instance();

	constexpr uint32_t nr_tries = 10;
	uint32_t sols = 0;
	for (size_t i = 0; i < nr_tries; i++) {
		algokek.nn(LS, LS);
		sols += algokek.solutions_nr;
		algokek.solutions_nr = 0;

		free(algokek.L1);
		free(algokek.L2);
		algokek.generate_random_instance();
	}

	EXPECT_EQ(sols, nr_tries);
}
#endif

#ifdef __AVX512F__
TEST(Bruteforce, avx512_32_8x8) {
	constexpr static NN_Config config{32, 1, 1, 1, LS, 2, 4, 0, 512};
	NN<config> algo{};
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
