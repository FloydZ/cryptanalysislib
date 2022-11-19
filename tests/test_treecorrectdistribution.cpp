#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>
#include <array>

#include "test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

const uint64_t TESTS = TESTSIZE;
const uint64_t BASELIST_SIZE = 10;
const double ERROR_FACTOR_HIGHER = 1.5;
const double ERROR_FACTOR_LOWER = 0.5;

#ifdef USE_FPLLL
///teste für tiefe 4
TEST(TreeTest, BuildTreeCheckDistributionOnBinaryD4) {
	fplll::ZZ_mat<kAryType> A(n, n);
	A.gen_uniform(1);

	Tree t{4, A, BASELIST_SIZE};

	std::vector<uint64_t> data;
	data.resize(TESTS);

	for (int i = 0; i < TESTS; ++i) {
		A.gen_uniform(2);

		Label target {};
		target.random();
		t.build_tree(target);

		data[i] = t[d + 1].get_load();
		// std::cout << "Resultsize: "<< i << " " << data[i] << "\n";
	}

	std::sort(data.begin(), data.end());
	auto median = data[TESTS/2];

	// std::cout << "Media: " << median << "\n";
	// std::cout << "data: "  << data << "\n";

	for (int i = 0; i < TESTS; ++i) {
		EXPECT_LE(data[i], (uint64_t(1)<<BASELIST_SIZE) * ERROR_FACTOR_HIGHER);
		EXPECT_GE(data[i], (uint64_t(1)<<BASELIST_SIZE) * ERROR_FACTOR_LOWER);
	}
}
#endif

int main(int argc, char **argv) {
	srand(0);
    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
