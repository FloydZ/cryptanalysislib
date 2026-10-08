#include <gtest/gtest.h>

#include "helper.h"
#include "nn/nn.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr size_t LS = 1u << 8u;

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

		cryptanalysislib::aligned_free(algo.L1);
		cryptanalysislib::aligned_free(algo.L2);
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

			cryptanalysislib::aligned_free(algo.L1);
			cryptanalysislib::aligned_free(algo.L2);
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

			cryptanalysislib::aligned_free(algo.L1);
			cryptanalysislib::aligned_free(algo.L2);
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

/// moves the golden pair of `algo` to the given positions
template<typename A>
static void move_solution(A &algo, const size_t pl, const size_t pr) {
	for (uint32_t i = 0; i < A::ELEMENT_NR_LIMBS; i++) {
		std::swap(algo.L1[algo.solution_l][i], algo.L1[pl][i]);
		std::swap(algo.L2[algo.solution_r][i], algo.L2[pr][i]);
	}
	algo.solution_l = pl;
	algo.solution_r = pr;
}

/// list sizes, which are no multiple of the SIMD block sizes, and the
/// solution in the tails of the lists
TEST(Bruteforce, tails) {
	constexpr size_t LS2 = 1000;
	const size_t pos[][2] = {{LS2 - 1, LS2 - 1}, {LS2 - 1, 0}, {0, LS2 - 1}, {992, 993}};

	constexpr static NN_Config c128{128, 1, 1, 64, LS2, 48, 6, 0, 512};
	NN<c128> a128{};
	a128.generate_random_instance();
	constexpr static NN_Config c256{256, 4, 1, 64, LS2, 30, 16, 0, 512};
	NN<c256> a256{};
	a256.generate_random_instance();

	for (const auto &p: pos) {
		move_solution(a128, p[0], p[1]);
		a128.solutions_nr = 0;
		a128.bruteforce_simd_128_32_2_uxv<4, 4>(LS2, LS2);
		EXPECT_GE(a128.solutions_nr, 1);
		EXPECT_EQ(a128.all_solutions_correct(), true);

		a128.solutions_nr = 0;
		a128.bruteforce_simd_128_32_2_uxv<2, 4>(LS2, LS2);
		EXPECT_GE(a128.solutions_nr, 1);
		EXPECT_EQ(a128.all_solutions_correct(), true);

		move_solution(a256, p[0], p[1]);
		a256.solutions_nr = 0;
		a256.bruteforce_simd_256_64_4x4(LS2, LS2);
		EXPECT_GE(a256.solutions_nr, 1);
		EXPECT_EQ(a256.all_solutions_correct(), true);

		// the dispatcher with small lists of different sizes
		move_solution(a256, 3, 5);
		a256.solutions_nr = 0;
		a256.bruteforce(20, 6);
		EXPECT_GE(a256.solutions_nr, 1);
		EXPECT_EQ(a256.all_solutions_correct(), true);
	}
}

/// two solutions in the same pair of 8-blocks at the same rotation
TEST(Bruteforce, uxv_two_lanes) {
	constexpr size_t LS2 = 1000;
	constexpr static NN_Config c128{128, 1, 1, 64, LS2, 48, 6, 0, 512};
	NN<c128> a128{};
	a128.generate_random_instance();
	move_solution(a128, 0, 0);
	// second solution: lane 3 of the first blocks, rotation 0
	for (uint32_t i = 0; i < NN<c128>::ELEMENT_NR_LIMBS; i++) {
		a128.L1[3][i] = a128.L2[3][i];
	}

	const auto found = [&](const size_t l, const size_t r) {
		for (size_t i = 0; i < a128.solutions_nr; i++) {
			if (a128.solutions[i] == std::pair<size_t, size_t>{l, r}) { return true; }
		}
		return false;
	};

	a128.solutions_nr = 0;
	a128.bruteforce_simd_128_32_2_uxv<4, 4>(LS2, LS2);
	EXPECT_TRUE(found(0, 0));
	EXPECT_TRUE(found(3, 3));
	EXPECT_EQ(a128.all_solutions_correct(), true);

	a128.solutions_nr = 0;
	a128.bruteforce_simd_128_32_2_uxv<2, 4>(LS2, LS2);
	EXPECT_TRUE(found(0, 0));
	EXPECT_TRUE(found(3, 3));
	EXPECT_EQ(a128.all_solutions_correct(), true);
}

int main(int argc, char **argv) {
	rng_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
