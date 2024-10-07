#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "helper.h"
#include "tree.h"
#include "matrix/fq_matrix.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint32_t n    = 8;
constexpr uint32_t k    = 8;
constexpr uint32_t q    = 3;

using T 			= uint8_t;
using Matrix 		= FqMatrix<T, n, k, q>;
using Value     	= FqNonPackedVector<n, q, T>;
using Label    		= FqNonPackedVector<k, q, T>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;

TEST(TreeTest, join2lists) {
	size_t basesize = 9;
	Matrix A; A.identity();

	const std::vector<uint32_t> ta{{0, n}};
	uint32_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, 0, ta);

	List out{1u<<basesize}, l1{0}, l2{0};
	l1.random(1u << basesize, A);
	l2.random(1u << basesize, A);

	Label target {};
	target.zero();
	target.random();

	Tree::join2lists(out, l1, l2, target, ta);

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0;i < out.load();++i) {
		out[i].recalculate_label(A);
		if (!(Label::cmp(out[i].label, target, k_lower, k_higher))) {
			right = false;
			wrong++;
		}
	}

	EXPECT_GT(out.load(), 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	EXPECT_GT(out.load(),1u<<3);
	EXPECT_LT(out.load(),1u<<7);
}

TEST(TreeTest, sort_level_with_target) {
	size_t basesize = 8;
	Matrix A; A.identity();

	const std::vector<uint32_t> ta{{0, n}};

	List out1{1u<<basesize}, out2{1u<<basesize}, l1{0}, l2{0};
	l1.random(1u << basesize, A);
	l2.random(1u << basesize, A);
	List l22 = l2;
	Label target {}; target.random();

	for (size_t i = 0; i < l2.load(); ++i) {
		const bool b = l2[i].is_equal(l22[i]);
		EXPECT_EQ(b, true);
	}

	for (size_t s = 0; s < l2.load(); ++s) {
		Label::add(l2[s].label, l2[s].label, target);
	}

	l2.sort_level(0, n);
	l22.sort_level(0, n, target);
	for (size_t i = 0; i < l2.load(); ++i) {
		Label::sub(l2[i].label, l2[i].label, target);
		const bool b = l2[i].is_equal(l22[i]);
		EXPECT_EQ(b, true);
	}
}

TEST(TreeTest, join2lists_on_iT) {
	size_t basesize = 8;
	Matrix A; A.identity();

	const std::vector<uint32_t> ta{{0, n}};

	List out1{1u<<basesize}, out2{1u<<basesize}, l1{0}, l2{0};
	l1.random(1u << basesize, A);
	l2.random(1u << basesize, A);
	List l22 = l2;
	Label target {}; target.random();


	// NOTE: this is a little hacky. `join2lists` alters l2 in a way which is
	// not recoverable by `join2lists_on_iT`. Thus, the order of function calls
	// does matter here.
	l1.sort_level(ta[0], ta[1]);
	Tree::join2lists_on_iT(out2, l1, l22, target, ta[0], ta[1]);
	Tree::join2lists(out1, l1, l2, target, ta);

	// check loads
	EXPECT_GT(out1.load(), 0);
	EXPECT_GT(out2.load(), 0);

	EXPECT_GT(out1.load(), 1u<<2);
	EXPECT_LT(out1.load(), 1u<<7);
	EXPECT_EQ(out1.load(), out2.load());

	// check if l2 is correctly sorted
	for (size_t i = 0; i < l2.load(); ++i) {
		Label::sub(l2[i].label, l2[i].label, target);
		const bool b = l2[i].is_equal(l22[i]);
		EXPECT_EQ(b, true);
	}

	// check correct output
	for (size_t i = 0; i < out1.load(); ++i) {
		const bool b = out1[i].value.is_equal(out2[i].value);
		EXPECT_EQ(b, true);
	}
}

TEST(TreeTest, join4lists) {
	uint32_t basesize = 8u;
	Matrix A;
	A.identity();

	const std::vector<uint32_t> ta{{0, n/2, n}};
	uint32_t k_lower=0, k_higher=0;

	List out{1u<<12}, l1{0}, l2{0}, l3{0}, l4{0};
	l1.random(1u << basesize, A);
	l2.random(1u << basesize, A);
	l3.random(1u << basesize, A);
	l4.random(1u << basesize, A);

	Label target; target.random();

	Tree::join4lists(out, l1, l2, l3, l4, target, ta);

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0;i < out.load();++i) {
		out[i].recalculate_label(A);

		for (uint32_t j = 0; j < 2; ++j) {
			translate_level(&k_lower, &k_higher, j, ta);
			if (!(Label::cmp(out[i].label, target, k_lower, k_higher))) {
				right = false;
				wrong++;
			}
		}
	}

	EXPECT_GT(out.load(), 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	EXPECT_GT(out.load(),1u<<9);
	EXPECT_LT(out.load(),1u<<11);
}

TEST(TreeTest, join4lists_with2lists) {
	unsigned int basesize = 10;
	Matrix A;
	A.identity();

	const std::vector<uint32_t> ta{{0, n/2, n}};
	uint32_t k_lower=0, k_higher=0;

	List out{1u<<basesize}, l1{0}, l2{0}, l3{0}, l4{0};
	l1.random(1u << basesize, A);
	l2.random(1u << basesize, A);

	Label target {};
	target.zero();
	target.random();

	Tree::streamjoin4lists_twolists(out, l1, l2, target, ta);

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0;i < out.load();++i) {
		// std::cout << out[i];
		out[i].recalculate_label(A);
		// std::cout << out[i];

		for (int j = 0; j < 2; ++j) {
			translate_level(&k_lower, &k_higher, j, ta);
			if (!(Label::cmp(out[i].get_label(), target, k_lower, k_higher))) {
				right = false;
				wrong++;
			}
		}
	}

	EXPECT_GT(out.load(), 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	EXPECT_GT(out.load(),1u<<9);
	EXPECT_LT(out.load(),1u<<11);
}


int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
    return RUN_ALL_TESTS();
}
