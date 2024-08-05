#include <gtest/gtest.h>

#include "helper.h"
#include "nn/nn.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr size_t LS = 1u << 8u;

// TODO not working for ARM
#ifdef USE_AVX2
TEST(Bruteforce, simd_32) {
	constexpr static NN_Config config{32, 1, 1, 32, LS, 10, 5, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_simd_32(LS, LS);
	EXPECT_GT(algo.solutions_nr, 0);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, simd_64) {
	constexpr static NN_Config config{64, 1, 1, 64, LS, 10, 5, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_simd_64(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, simd_64_1x1) {
	constexpr static NN_Config config{64, 1, 1, 64, LS, 10, 5, 0, 512};
	NN<config> algo{};

	for (uint32_t i = 0; i < 10; ++i) {
		algo.generate_random_instance();
		algo.bruteforce_simd_64_1x1(LS, LS);
		EXPECT_EQ(algo.solutions_nr, 1);
		EXPECT_EQ(algo.all_solutions_correct(), true);
		algo.solutions_nr = 0;

		free(algo.L1);
		free(algo.L2);
		algo.L1 = nullptr;
		algo.L2 = nullptr;
	}
}

TEST(Bruteforce, simd_64_uxv) {
	constexpr static NN_Config config{64, 1, 1, 64, LS, 10, 5, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_simd_64_uxv<1,1>(LS, LS);
	EXPECT_GE(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_64_uxv<2,2>(LS, LS);
	EXPECT_GE(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_64_uxv<4,4>(LS, LS);
	EXPECT_GE(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_64_uxv<8,8>(LS, LS);
	EXPECT_GE(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_64_uxv<1,2>(LS, LS);
	EXPECT_GE(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_64_uxv<2,1>(LS, LS);
	EXPECT_GE(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_64_uxv<4,2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

}

TEST(Bruteforce, simd_64_uxv_shuffle) {
	constexpr static NN_Config config{64, 1, 1, 64, LS, 10, 10, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_simd_64_uxv_shuffle<1,1>(LS, LS);
	EXPECT_GE(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_64_uxv_shuffle<2,2>(LS, LS);
	EXPECT_GE(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_64_uxv_shuffle<4,4>(LS, LS);
	EXPECT_GE(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_64_uxv_shuffle<8,8>(LS, LS);
	EXPECT_GE(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_64_uxv_shuffle<1,2>(LS, LS);
	EXPECT_GE(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_64_uxv_shuffle<2,1>(LS, LS);
	EXPECT_GE(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_64_uxv_shuffle<4,2>(LS, LS);
	EXPECT_GE(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, simd_128) {
	constexpr static NN_Config config{128, 1, 1, 64, LS, 12, 6, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_simd_128(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, simd_uxv_128) {
	constexpr static NN_Config config{128, 1, 1, 64, LS, 48, 6, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_simd_128_32_2_uxv<1, 1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_128_32_2_uxv<2, 2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_128_32_2_uxv<4, 4>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
	// NOTE: dont use 8x8
}


TEST(Bruteforce, simd_256) {
	constexpr static NN_Config config{256, 4, 1, 64, LS, 80, 20, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();

	if constexpr (LS > (1u << 16)) {
		algo.bruteforce_simd_256(LS, LS);
		EXPECT_EQ(algo.solutions_nr, 1);
		EXPECT_EQ(algo.all_solutions_correct(), true);
	} else {
		for (uint32_t i = 0; i < 1; ++i) {
			algo.bruteforce_simd_256(LS, LS);
			EXPECT_EQ(algo.solutions_nr, 1);
			EXPECT_EQ(algo.all_solutions_correct(), true);
			algo.solutions_nr = 0;

			free(algo.L1);
			free(algo.L2);
			algo.generate_random_instance();
		}
	}
}

TEST(Bruteforce, simd_256_ux4) {
	constexpr static NN_Config config{256, 4, 1, 64, LS, 80, 20, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_simd_256_ux4<1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_256_ux4<2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_256_ux4<4>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_256_ux4<8>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}

TEST(Bruteforce, simd_256_32_ux8) {
	constexpr static NN_Config config{256, 4, 1, 64, LS, 25, 4, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();

	algo.bruteforce_simd_256_32_ux8<1>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_256_32_ux8<2>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_256_32_ux8<4>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;

	algo.bruteforce_simd_256_32_ux8<8>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
	algo.solutions_nr = 0;
}

TEST(Bruteforce, simd_256_64_4x4) {
	constexpr static NN_Config config{256, 4, 1, 64, LS, 30, 16, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();

	if constexpr (LS > (1u << 16)) {
		algo.bruteforce_simd_256_64_4x4(LS, LS);
		EXPECT_EQ(algo.solutions_nr, 1);
		EXPECT_EQ(algo.all_solutions_correct(), true);
	} else {
		for (size_t i = 0; i < 10; ++i) {
			algo.bruteforce_simd_256_64_4x4(LS, LS);
			EXPECT_EQ(algo.solutions_nr, 1);
			EXPECT_EQ(algo.all_solutions_correct(), true);
			algo.solutions_nr = 0;

			free(algo.L1);
			free(algo.L2);
			algo.generate_random_instance();
		}
	}
}

TEST(Bruteforce, simd_256_64_4x4_rearrange) {
	constexpr size_t LS = 652;
	constexpr static NN_Config config__{256, 4, 1, 64, LS, 10, 14, 0, 512, 0, 0, true};
	NN<config__> algo{};
	algo.generate_random_instance();
	algo.transpose(LS);
	algo.bruteforce_simd_256_64_4x4_rearrange<LS>(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}
#endif

int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
