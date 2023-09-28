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

TEST(Bruteforce, n64) {
	constexpr static NN_Config config{64, 1, 1, 64, LS, 10, 5, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_64(LS, LS);

	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

TEST(Bruteforce, n128) {
	constexpr static NN_Config config{128, 2, 1, 64, LS, 48, 32, 0, 512};
	NN<config> algo{};
	algo.generate_random_instance();
	algo.bruteforce_128(LS, LS);
	EXPECT_EQ(algo.solutions_nr, 1);
	EXPECT_EQ(algo.all_solutions_correct(), true);
}

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

int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
