#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/random_index.h"
#include "container/kAry_type.h"
#include "helper.h"
#include "matrix/matrix.h"
#include "tree.h"

#include "subsetsum.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// TODO: wrapper function for level2 hashmap with callback lambda for elements in last list.
//	- also include all these hashmap declaration in extra functions
//  - also simplify the enumerator interface to only need the lists as inputs and not the offset.

// NOTE: random enumeration of the values
TEST(SubSetSum, n32) {
	// 19.543010252924077 7.567052700200905 l1=5 l2=13 d1=0 d2=0 G=1 n1=2,nm1=0 0
	constexpr uint32_t n    = 32ul;
	constexpr uint64_t q    = (1ul << n);

	using T 			= uint64_t;
	using Value     	= BinaryVector<n>;
	using Label    		= kAry_Type_T<q>;
	using Matrix 		= FqVector<T, n, q, true>;
	using Element		= Element_T<Value, Label, Matrix>;
	using List			= List_T<Element>;
	using Tree			= Tree_T<List>;

	Matrix A; A.random();
	constexpr uint32_t k_lower1=0, k_higher1=13, k_higher2=18;
	constexpr uint32_t p = 2;
	constexpr size_t baselist_size = sum_bc(n/2, p);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryListEnumerateMultiFullLength<List, n/2, p>;
	//using Enumerator = BinaryLexicographicEnumerator<List, n/2, p>;

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	Tree::template join4lists_twolists_on_iT_hashmap_v2
			<k_lower1, k_higher1, k_higher1, k_higher2, Enumerator>
			(out, l1, l2, target, A);

	EXPECT_GT(out.load(), 0);
}

TEST(SubSetSum, n48) {
	// 22.707336747487684 12.775544757643935 2 20 0 0 1 3.0 0 2024.0 1.6311530334469542 3.90679931640625
	constexpr uint32_t n    = 48ul;
	constexpr uint64_t q    = (1ul << n);

	using T 			= uint64_t;
	using Value     	= BinaryVector<n>;
	using Label    		= kAry_Type_T<q>;
	using Matrix 		= FqVector<T, n, q, true>;
	using Element		= Element_T<Value, Label, Matrix>;
	using List			= List_T<Element>;
	using Tree			= Tree_T<List>;

	Matrix A; A.random();
	constexpr uint32_t k_lower1=0, k_higher1=20, k_higher2=22;
	constexpr uint32_t p = 3;
	constexpr size_t baselist_size = sum_bc(n/2, p);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryListEnumerateMultiFullLength<List, n/2, p>;
	//using Enumerator = BinaryLexicographicEnumerator<List, n/2, p>;

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	Tree::template join4lists_twolists_on_iT_hashmap_v2
			<k_lower1, k_higher1, k_higher1, k_higher2, Enumerator>
			(out, l1, l2, target, A);

	EXPECT_GT(out.load(), 0);
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
