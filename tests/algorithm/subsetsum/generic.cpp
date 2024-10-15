#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/random_index.h"
#include "container/kAry_type.h"
#include "helper.h"
#include "matrix/matrix.h"
#include "tree.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint32_t n    = 16ul;
constexpr uint32_t q    = (1ul << n);

using T 			= uint64_t;
//using Value     	= kAryContainer_T<T, n, 2>;
using Value     	= BinaryVector<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;

TEST(SubSetSum, generic) {
	Matrix A; A.random();
	// constexpr uint32_t k_lower1=0, k_higher1=n/3, k_higher2=2*n/3, k_higher3=n;
	constexpr uint32_t k_lower1=0, k_higher1=n/2, k_higher2=n;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};
	Label target; target.zero();
	std::vector<uint32_t> weights(n/2);
	generate_random_indices(weights, n);
	for (uint32_t i = 0; i < n/2; ++i) {
		Label::add(target, target, A[0][weights[i]]);
	}

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, 3>;
	Enumerator e{A};
	e.run(&l1, &l2, n/2);

	constexpr uint32_t depth = 2;
	//auto nr_iTs = [](const uint32_t i = 0 ){
	//	return (1ull << (depth - i -1ull)) - 1u;
	//};

	std::vector<Label> iTs(depth);
	std::vector<std::vector<Label>> sum_iTs(depth);
	for (uint32_t i = 0; i < depth - 1; ++i) {
		iTs[i].random(0, 1ull << n);
	}
	iTs[depth-1] = target;

	for (uint32_t ik = 0; ik < depth; ++ik) {
		sum_iTs[ik].resize(depth - ik);

		for (uint32_t i = 0; i < depth-ik; ++i) {
			sum_iTs[ik][i] = iTs[i+ik];
			for (uint32_t j = 1; j <= i; j++){
				Label::sub(sum_iTs[ik][i], sum_iTs[ik][i], iTs[i-j]);
			}
		}
	}


	Tree t(depth, A, l1, l2);
	t.join_stream_internal<0, k_lower1, k_higher1, k_higher2>
	        (t.lists[2], sum_iTs);
	t.join_stream_internal<1, k_lower1, k_higher1, k_higher2>
			(out, sum_iTs, false);


	std::cout << target << std::endl;
	uint32_t right=0;
	for(uint64_t i = 0; i < out.load(); ++i) {
		std::cout << out[i] << std::endl;
		// just for debugging, we are not filtering
		//if (out[i].value.popcnt() != n/2) { continue; }

		Label test_recalc1(0), test_recalc2(0), test_recalc3(0);
		A.mul(test_recalc3, out[i].value);
		// NOTE: the full length
		for (uint64_t j = 0; j < n; ++j) {
			if (out[i].value.get(j)) {
				test_recalc1 += A[0][j];
				Label::add(test_recalc2, test_recalc2, A[0][j]);
			}
		}

		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc2, 0, n));
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc3, 0, n));
		if (Label::cmp(out[i].label, target)) {
			right += 1;
		}
	}

	EXPECT_GT(right,0);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
