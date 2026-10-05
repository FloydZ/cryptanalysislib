#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/random_index.h"
#include "algorithm/subsetsum.h"
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


constexpr uint32_t n = 30u;
constexpr uint64_t q = (1ul << n);

using T 			= uint64_t;
using Value     	= BinaryVector<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;

TEST(SubSetSum, constexpr_join2lists_on_iT_hashmap_v2_multithreaded) {
	Matrix A; A.random();
	//NOTE: thats 2**16 elements, dont forget that
	constexpr uint32_t k_lower=0, k_higher=16;
	constexpr uint32_t p = 4;
	constexpr uint32_t nthreads = 2;
	constexpr uint32_t chunks = 4;

	constexpr size_t baselist_size = sum_bc(n/2, p);
	List out{1u<<8, chunks}, l1{baselist_size, chunks}, l2{baselist_size, chunks};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, 4>;
	Enumerator e{A};
	e.run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	Tree t{1, A, 0};
	t.join2lists_on_iT_v2
	    <k_lower, k_higher, 100, nthreads, chunks>
	    (par_if(true), out, l1, l2, target);


	auto right=true;
	int wrong=0;
	for(uint64_t i = 0; i < out.load(); ++i) {
		Label test_recalc1(0), test_recalc2(0), test_recalc3(0);
		A.mul(test_recalc3, out[i].value);
		// NOTE: the full length
		for (uint64_t j = 0; j < n; ++j) {
			if (out[i].value.get(j)) {
				test_recalc1 += A[0][j];
				Label::add(test_recalc2, test_recalc2, A[0][j]);
			}
		}

		// NOTE that we do not recalculate the label
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc2, k_lower, k_higher));
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc3, k_lower, k_higher));
		EXPECT_EQ(true, test_recalc1.is_equal(out[i].label, k_lower, k_higher));

		if (!(Label::cmp(out[i].label, target, k_lower, k_higher))) {
			right = false;
			wrong++;
		}

		out[i].recalculate_label(A);
		EXPECT_EQ(true, test_recalc1.is_equal(out[i].label, k_lower, k_higher));
	}


	Label el{};
	uint64_t num = 0;
	for (size_t i = 0; i < l1.load(); ++i) {
		for (size_t j = 0; j < l2.load(); ++j) {
			Label::add(el, l1[i].label, l2[j].label);
			if (el.is_equal(target, k_lower, k_higher)) {
				num += 1;
			}
		}
	}

	EXPECT_GT(out.load(), 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	if constexpr (n == 16) {
		EXPECT_GT(out.load(), 1u<<3);
		EXPECT_LT(out.load(), 1u<<7);
	}
	EXPECT_EQ(out.load(), num);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(nullptr));
	return RUN_ALL_TESTS();
}
