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

TEST(SubSetSum, join4lists) {
	Matrix A; A.random();
	constexpr uint64_t k_lower1=0, k_higher1=n/2;
	constexpr uint64_t k_lower2=n/2, k_higher2=n;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size}, l3{baselist_size}, l4{baselist_size};

	// not split enumeration (e.g. L1=L3, L2=L4)
	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.run(&l1, &l2, n/2);
	e.run(&l3, &l4, n/2);

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
		EXPECT_EQ(l3[i].is_correct(A), true);
		EXPECT_EQ(l4[i].is_correct(A), true);
	}

	Tree t{1, A, 0};
	// NOTE: the planted solution is only found with probability ~1-1/e for a
	// 	random intermediate target, hence the retries. `join4lists` modifies
	// 	the lists, so they are enumerated again.
	uint32_t right=0;
	for (uint32_t tries = 0; (tries < 32) && (right == 0); ++tries) {
	if (tries) { e.run(&l1, &l2, n/2); e.run(&l3, &l4, n/2); }
	t.join4lists(out, l1, l2, l3, l4, target,
	                 k_lower1, k_higher1, k_lower2, k_higher2, true);

	for(uint64_t i = 0; i < out.load(); ++i) {
		// just for debugging, we are not filtering
		if (out[i].value.popcnt() != n/2) {
			continue;
		}

		// first check thats is zero
		EXPECT_EQ(out[i].label.is_zero(k_lower1, k_higher2), true);

		Label test_recalc1(0), test_recalc2(0), test_recalc3(0);
		A.mul(test_recalc3, out[i].value);
		// NOTE: the full length
		for (uint64_t j = 0; j < n; ++j) {
			if (out[i].value.get(j)) {
				test_recalc1 += A[0][j];
				Label::add(test_recalc2, test_recalc2, A[0][j]);
			}
		}

		out[i].recalculate_label(A);
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc2, 0, n));
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc3, 0, n));
		EXPECT_EQ(true, test_recalc1.is_equal(out[i].label, 0, n));
		std::cout << out[i] << std::endl;

		// TODO not finished
		if (Label::cmp(out[i].label, target)) {
			right += 1;
		}
	}
	}

	EXPECT_GT(right,0);
}

TEST(SubSetSum, join4lists_on_iT_v2) {
	Matrix A; A.random();
	constexpr uint64_t k_lower1=0, k_higher1=n/2;
	constexpr uint64_t k_lower2=n/2, k_higher2=n;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	// constexpr size_t baselist_size = sum_bc(n/4, n/8);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size}, l3{baselist_size}, l4{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.run(&l1, &l2, n/2);
	e.run(&l3, &l4, n/2);

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
		EXPECT_EQ(l3[i].is_correct(A), true);
		EXPECT_EQ(l4[i].is_correct(A), true);
	}

	Tree t{1, A, 0};
	// NOTE: the planted solution is only found with probability ~1-1/e for a
	// 	random intermediate target, hence the retries.
	uint32_t right=0;
	for (uint32_t tries = 0; (tries < 32) && (right == 0); ++tries) {
	t.join4lists_on_iT_v2(out, l1, l2, l3, l4, target,
							  k_lower1, k_higher1, k_lower2, k_higher2);

	for(uint64_t i = 0; i < out.load(); ++i) {
		// just for debugging, we are not filtering
		if (out[i].value.popcnt() != n/2) {
			continue;
		}

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
		// out[i].recalculate_label(A);
		// NOTE: the label of an output is l1+l2+l3+l4 (mod q), but its value
		// 	is the XOR of the four parts. Both only agree if the parts have
		// 	disjoint supports, i.e. only then the output is a solution.
		const bool consistent = test_recalc1.is_equal(out[i].label, 0, n);
		std::cout << out[i] << std::endl;
		if (consistent && Label::cmp(out[i].label, target)) {
			right += 1;
		}
	}
	}

	EXPECT_GT(right,0);
}

TEST(SubSetSum, join4lists_on_iT_v2_constexpr) {
	Matrix A; A.random();
	constexpr uint64_t k_lower1=0, k_higher1=n/2;
	constexpr uint64_t k_lower2=n/2, k_higher2=n;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	// constexpr size_t baselist_size = sum_bc(n/4, n/8);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size}, l3{baselist_size}, l4{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.run(&l1, &l2, n/2);
	e.run(&l3, &l4, n/2);

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
		EXPECT_EQ(l3[i].is_correct(A), true);
		EXPECT_EQ(l4[i].is_correct(A), true);
	}

	Tree t{1, A, 0};
	// NOTE: the planted solution is only found with probability ~1-1/e for a
	// 	random intermediate target, hence the retries.
	uint32_t right=0;
	for (uint32_t tries = 0; (tries < 32) && (right == 0); ++tries) {
	t.template join4lists_on_iT_v2
			<k_lower1, k_higher1, k_lower2, k_higher2>
	        (out, l1, l2, l3, l4, target);

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
		EXPECT_EQ(l3[i].is_correct(A), true);
		EXPECT_EQ(l4[i].is_correct(A), true);
	}

	for(uint64_t i = 0; i < out.load(); ++i) {
		// just for debugging, we are not filtering
		if (out[i].value.popcnt() != n/2) {
			continue;
		}

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
		// NOTE: the label of an output is l1+l2+l3+l4 (mod q), but its value
		// 	is the XOR of the four parts. Both only agree if the parts have
		// 	disjoint supports, i.e. only then the output is a solution.
		const bool consistent = test_recalc1.is_equal(out[i].label, 0, n);

		//std::cout << out[i] << std::endl;
		//out[i].recalculate_label(A);
		//std::cout << out[i] << std::endl;
		//EXPECT_EQ(true, test_recalc1.is_equal(out[i].label, 0, n));

		if (consistent && Label::cmp(out[i].label, target)) {
			right += 1;
		}
	}
	}

	EXPECT_GT(right,0);
}

TEST(SubSetSum, join4lists_twolists_on_iT_v2) {
	Matrix A; A.random();
	constexpr uint64_t k_lower1=0, k_higher1=n/2;
	constexpr uint64_t k_lower2=n/2, k_higher2=n;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.run(&l1, &l2, n/2);

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	Tree t{1, A, 0};
	// NOTE: the planted solution is only found with probability ~1-1/e for a
	// 	random intermediate target, hence the retries.
	uint32_t right=0;
	for (uint32_t tries = 0; (tries < 32) && (right == 0); ++tries) {
	t.join4lists_twolists_on_iT_v2(out, l1, l2, target,
		k_lower1, k_higher1, k_lower2, k_higher2);

	for(uint64_t i = 0; i < out.load(); ++i) {
		// just for debugging, we are not filtering
		if (out[i].value.popcnt() != n/2) { continue; }

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
		// NOTE: the label of an output is l1+l2+l3+l4 (mod q), but its value
		// 	is the XOR of the four parts. Both only agree if the parts have
		// 	disjoint supports, i.e. only then the output is a solution.
		const bool consistent = test_recalc1.is_equal(out[i].label, 0, n);
		std::cout << out[i] << std::endl;
		if (consistent && Label::cmp(out[i].label, target)) {
			right += 1;
		}
	}
	}

	EXPECT_GT(right,0);
	EXPECT_GT(out.load(), 0);
}

TEST(SubSetSum, join4lists_twolists_on_iT_v2_constexpr) {
	Matrix A; A.random();
	constexpr uint64_t k_lower1=0, k_higher1=n/2;
	constexpr uint64_t k_lower2=n/2, k_higher2=n;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.run(&l1, &l2, n/2);

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	Tree t{1, A, 0};
	// NOTE: the planted solution is only found with probability ~1-1/e for a
	// 	random intermediate target, hence the retries.
	uint32_t right=0;
	for (uint32_t tries = 0; (tries < 32) && (right == 0); ++tries) {
	t.template join4lists_twolists_on_iT_v2
	        <k_lower1, k_higher1, k_lower2, k_higher2>
	        (out, l1, l2, target);

	for(uint64_t i = 0; i < out.load(); ++i) {
		// just for debugging, we are not filtering
		if (out[i].value.popcnt() != n/2) { continue; }

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
		// NOTE: the label of an output is l1+l2+l3+l4 (mod q), but its value
		// 	is the XOR of the four parts. Both only agree if the parts have
		// 	disjoint supports, i.e. only then the output is a solution.
		const bool consistent = test_recalc1.is_equal(out[i].label, 0, n);
		std::cout << out[i] << std::endl;
		if (consistent && Label::cmp(out[i].label, target)) {
			right += 1;
		}
	}
	}

	EXPECT_GT(right,0);
}

TEST(SubSetSum, constexpr_join4lists_on_iT_hashmap_v2) {
	Matrix A; A.random();
	constexpr uint32_t k_lower1=0, k_higher1=8;
	constexpr uint32_t k_lower2=8, k_higher2=16;
	constexpr uint32_t p = n/4;

	constexpr size_t baselist_size = sum_bc(n/2, p);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, p>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	using D = typename Label::DataType;
	using E = std::pair<size_t, size_t>;
	// NOTE: you need to choose the `bucketsize` correctly.
	constexpr static SimpleHashMapConfig simpleHashMapConfigL0 {
			100, 1ul<<(k_higher1-k_lower1), 1
	};
	constexpr static SimpleHashMapConfig simpleHashMapConfigL1 {
			100, 1ul<<(k_higher2-k_lower2), 1
	};
	using HML0 = SimpleHashMap<D, size_t, simpleHashMapConfigL0, Hash<D, k_lower1, k_higher1, 2>>;
	using HML1 = SimpleHashMap<D,      E, simpleHashMapConfigL1, Hash<D, k_lower2, k_higher2, 2>>;
	HML0 hml0{};
	HML1 hml1{};

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	Tree t{1, A, 0};
	t.template join4lists_twolists_on_iT_hashmap_v2
			<k_lower1, k_higher1, k_lower2, k_higher2>
			(out, l1, l2, hml0, hml1, target);


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
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc2, k_lower1, k_higher2));
		EXPECT_EQ(true, test_recalc1.is_equal(test_recalc3, k_lower1, k_higher2));
		// NOTE: the label of an output is l1+l2+l1'+l2' (mod q), but its value
		// 	is the XOR of the four parts. Both only agree if the parts have
		// 	disjoint supports, i.e. only then the output is a solution.

		if (!(Label::cmp(out[i].label, target, k_lower1, k_higher2))) {
			right = false;
			wrong++;
		}
	}

	EXPECT_GT(out.load(), 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	if constexpr (n == 16) {
		EXPECT_GT(out.load(), 1u<<3);
		EXPECT_LT(out.load(), 1u<<7);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
