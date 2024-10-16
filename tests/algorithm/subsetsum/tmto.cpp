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

// TODO: simplify the enumerator interface to only need the lists as inputs and not the offset.

// TODO: PCS algorithm: remap the output of the tree as the next iT of the next iteration
//		flavor function: f = a*x+b mod p, p <= 2^k_upper1

// NOTE: random enumeration of the values
// NOTE: only two baselists are used
TEST(SubSetSum, n32_d2_baselists2) {
	// 15.841070074033095 7.999094705692274 6 10 0 0 100.0 2.0 0 120.0 14.06  1.77
	constexpr uint32_t n = 32ul;
	constexpr uint64_t q = (1ul << n); //4294967279

	using T 			= uint64_t;
	using Value     	= BinaryVector<n>;
	using Label    		= kAry_Type_T<q>;
	using Matrix 		= FqVector<T, n, q>;
	using Element		= Element_T<Value, Label, Matrix>;
	using List			= List_T<Element>;
	using Tree			= Tree_T<List>;

	Matrix A; A.random();
	constexpr uint32_t k_lower1=0, k_higher1=10, k_higher2=16;
	constexpr uint32_t p = 2;
	constexpr size_t baselist_size = sum_bc(n/2, p);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryListEnumerateMultiFullLength<List, n/2, p>;
	//using Enumerator = BinaryLexicographicEnumerator<List, n/2, p>;

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

    Tree t{1, A, 0};
	t.template join4lists_twolists_on_iT_hashmap_v2
			<k_lower1, k_higher1, k_higher1, k_higher2, Enumerator>
			(out, l1, l2, target);

	EXPECT_GT(out.load(), 0);
}

TEST(SubSetSum, n32_d2_rho) {
	constexpr uint32_t n = 32;
	constexpr uint64_t q = 1ul << n;
	constexpr static SSS instance{.n=n, .q=q, .bp=3, .l1=5, .l2=7};
	using S = sss_d2<instance>;

	using Value  = S::Value;
	using Label  = S::Label;
	using Matrix = S::Matrix;
	using Element= S::Element;
	using List   = S::List;
	using Tree   = S::Tree;

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
