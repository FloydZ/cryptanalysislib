#include <gtest/gtest.h>

#include "helper.h"
#include "nn/nn.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr size_t LS = 1u << 14u;

// only define this if you are developing MO paramters
//#define NN_DEV
#ifdef NN_DEV
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