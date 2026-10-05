#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/random_index.h"
#include "algorithm/subsetsum.h"

#include "algorithm/subsetsum.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(SubSetSum, n32_d2) {
	constexpr uint32_t n = 32;
	constexpr uint64_t q = 1ul << n;
	constexpr static SSS instance{.n=n, .q=q, .bp=4, .l1=16, .l2=16};
	using S = HGJ<instance>;

	// using Value  = S::Value;
	using Label  = S::Label;
	using Matrix = S::Matrix;

	Matrix A; A.random();
	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	S s(A, target);
	s.run();
}

TEST(SubSetSum, n32_d3) {
	constexpr uint32_t n = 32;
	constexpr uint64_t q = 1ul << n;
	constexpr static SSS instance{.n=n, .q=q, .bp=2, .l1=11, .l2=11, .l3=10};
	using S = HGJ<instance>;

	// using Value  = S::Value;
	using Label  = S::Label;
	using Matrix = S::Matrix;

	Matrix A; A.random();
	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	S s(A, target);
	s.run();
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
