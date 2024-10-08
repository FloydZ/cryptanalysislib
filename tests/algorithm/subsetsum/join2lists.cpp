#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/random_index.h"
#include "container/kAry_type.h"
#include "helper.h"
#include "matrix/matrix.h"
#include "tree.h"

// test include
#include "subsetsum.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// TODO non power of two prime extension are not really supported.
// The problem is, that for a normal prime, carries occur
constexpr uint32_t n = 16u;
constexpr uint64_t q = (1ul << n);

using T 			= uint64_t;
//using Value     	= kAryPackedContainer_T<T, n, 2>;
using Value     	= BinaryVector<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;

TEST(SubSetSum, join2lists) {
	Matrix A; A.random();
	const uint64_t k_lower=0, k_higher=8;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	// compute the number of collisions via the simple quadratic algorithm
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

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
	}

	Tree t{1, A, 0};
	t.join2lists(out, l1, l2, target, k_lower, k_higher, true);

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);

		// revert the changes of the right list
		Element l2_tmp = l2[i];
		l2_tmp.label.neg();
		l2_tmp.label += target;
		EXPECT_EQ(l2_tmp.is_correct(A), true);
	}

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0; i < out.load(); ++i) {
		EXPECT_EQ(out[i].label.is_zero(k_lower, k_higher), true);

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

	EXPECT_GT(out.load(), 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	if constexpr (n == 16) {
		EXPECT_GT(out.load(), 1u << 3);
		EXPECT_LT(out.load(), 1u << 7);
	}
	EXPECT_EQ(out.load(), num);
}

TEST(SubSetSum, constexpr_join2lists) {
	Matrix A; A.random();
	constexpr uint64_t k_lower=0, k_higher=8;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	// compute the number of collisions via the simple quadratic algorithm
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

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
	}

	Tree::template join2lists<k_lower, k_higher>(out, l1, l2, target, true);

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);

		// revert the changes of the right list
		Element l2_tmp = l2[i];
		l2_tmp.label.neg();
		l2_tmp.label += target;
		EXPECT_EQ(l2_tmp.is_correct(A), true);
	}

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0; i < out.load(); ++i) {
		EXPECT_EQ(out[i].label.is_zero(k_lower, k_higher), true);

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

	EXPECT_GT(out.load(), 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	if constexpr (n == 16) {
		EXPECT_GT(out.load(), 1u << 3);
		EXPECT_LT(out.load(), 1u << 7);
	}
	EXPECT_EQ(out.load(), num);
}

TEST(SubSetSum, join2lists_on_iT) {
	//  `on_iT` does not alter the lists, except for sorting.
	Matrix A; A.random();
	const uint64_t k_lower=0, k_higher=8;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
	}

	Tree::join2lists_on_iT(out, l1, l2, target, k_lower, k_higher);

	// make sure that the elements are still correct
	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
	}

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
		EXPECT_GT(out.load(), 1u << 3);
		EXPECT_LT(out.load(), 1u << 7);
	}
	EXPECT_EQ(out.load(), num);
}

TEST(SubSetSum, join2lists_on_iT_v2) {
	Matrix A; A.random();
	const uint64_t k_lower=0, k_higher=8;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
	}

	Tree::join2lists_on_iT_v2(out, l1, l2, target, k_lower, k_higher);

	// make sure that the elements are still correct
	for (size_t i = 0; i < baselist_size; ++i) {
		EXPECT_EQ(l1[i].is_correct(A), true);
		EXPECT_EQ(l2[i].is_correct(A), true);
	}

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
		EXPECT_GT(out.load(), 1u << 3);
		EXPECT_LT(out.load(), 1u << 7);
	}
	EXPECT_EQ(out.load(), num);
}

TEST(SubSetSum, constexpr_join2lists_on_iT_v2) {
	Matrix A; A.random();
	constexpr uint32_t k_lower=0, k_higher=8;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	l2.template sort_level<k_lower, k_higher>();
	Tree::template join2lists_on_iT_v2<k_lower, k_higher>
			(out, l1, l2, target);

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
		EXPECT_GT(out.load(), 1u << 3);
		EXPECT_LT(out.load(), 1u << 7);
	}
	EXPECT_EQ(out.load(), num);
}

TEST(SubSetSum, constexpr_join2lists_on_iT_hashmap_v2) {
	Matrix A; A.random();
	constexpr uint32_t k_lower=0, k_higher=8;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List out{1u<<8}, l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	using D = typename Label::DataType;
	// NOTE: you need to choose the `bucketsize` correctly.
	constexpr static SimpleHashMapConfig simpleHashMapConfig{
	    100, 1ul<<(k_higher-k_lower), 1
	};
	using HM = SimpleHashMap<D, size_t, simpleHashMapConfig, Hash<D, k_lower, k_higher, 2>>;
	HM hm{};

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	Tree::template join2lists_on_iT_hashmap_v2
	        <k_lower, k_higher>
	        (out, l1, l2, hm, target);


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

TEST(SubSetSum, constexpr_join2lists_on_iT_hashmap_hashmap_v2) {
	/// tests the function which outputs hashmap
	Matrix A; A.random();
	// note: as the elements of the out hashmap are shifted down
	// dont set `k_higher` = n
	constexpr uint32_t k_lower=0, k_higher=8;

	constexpr size_t baselist_size = sum_bc(n/2, n/4);
	List l1{baselist_size}, l2{baselist_size};

	using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
	Enumerator e{A};
	e.template run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
			(&l1, &l2, n/2);

	using D = typename Label::DataType;
	// NOTE: you need to choose the `bucketsize` correctly.
	constexpr static SimpleHashMapConfig simpleHashMapConfig{
			100, 1ul<<(k_higher-k_lower), 1
	};
	using HMIn = SimpleHashMap<D, size_t, simpleHashMapConfig, Hash<D, k_lower, k_higher, 2>>;
	using HMOut= SimpleHashMap<D, std::pair<size_t, size_t>, simpleHashMapConfig, Hash<D, k_lower+8, k_higher+8, 2>>;
	HMIn hmIn{};
	HMOut hmOut{};

	Label target;
	std::vector<uint32_t> weights(n/2);
	generate_subsetsum_instance(target, weights, A, n);

	Tree::template join2lists_on_iT_hashmap_v2
			<k_lower, k_higher>
			(hmOut, l1, l2, hmIn, target);

	size_t sols = 0, wrong = 0;
	Element te;
	for (size_t i = 0; i < HMOut::nrbuckets; ++i) {
		for (uint32_t j = 0; j < hmOut.load_without_hash(i); ++j) {
			sols += 1;
			const size_t a1 = hmOut[i*HMOut::bucketsize + j].first;
			const size_t a2 = hmOut[i*HMOut::bucketsize + j].second;
			ASSERT(a1 < l1.load());
			ASSERT(a2 < l2.load());

			Element::add(te, l1[a1], l2[a2]);
			ASSERT(te.label.is_equal(target, k_lower, k_higher));
			// EXPECT_EQ(true, target.is_equal(out[i].label, k_lower, k_higher));
		}
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

	EXPECT_EQ(0, wrong);
	EXPECT_EQ(sols, num);
	EXPECT_GT(sols, 0);
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(nullptr));
	return RUN_ALL_TESTS();
}
