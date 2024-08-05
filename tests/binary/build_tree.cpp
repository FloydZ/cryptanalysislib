#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "element.h"
#include "matrix/matrix.h"
#include "list/list.h"
#include "matrix/matrix.h"
#include "tree.h"


constexpr uint32_t n  = 10;

using BinaryValue     = BinaryContainer<n>;
using BinaryLabel     = BinaryContainer<n>;
using BinaryMatrix    = FqMatrix<uint64_t, n, n, 2>;
using BinaryElement   = Element_T<BinaryValue, BinaryLabel, BinaryMatrix>;
using BinaryList      = List_T<BinaryElement>;
using BinaryTree      = Tree_T<BinaryList>;

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(TreeTest, join2lists) {
	unsigned int basesize = 15;
	BinaryMatrix A;
	A.identity();

	const std::vector<uint64_t> ta{{0, n}};
	uint64_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, 0, ta);

	BinaryList out{1u<<basesize}, l1{0}, l2{0};
	l1.generate_base_random(1u << basesize, A);
	l2.generate_base_random(1u << basesize, A);

	BinaryLabel target {};
	target.zero();
	target.random();

	BinaryTree::join2lists(out, l1, l2, target, ta);

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0;i < out.load();++i) {
		// std::cout << out[i].get_label();
		out[i].recalculate_label(A);
		if (!(BinaryLabel::cmp(out[i].get_label(), target, k_lower, k_higher))) {
			right = false;
			wrong++;
		}
	}

	EXPECT_GT(out.load(), 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	EXPECT_GT(out.load(),1u<<9);
	EXPECT_LT(out.load(),1u<<11);
}

TEST(TreeTest, join4lists) {
	uint32_t basesize = 10u;
	BinaryMatrix A;
	A.identity();

	const std::vector<uint64_t> ta{{0, n/2, n}};
	uint64_t k_lower=0, k_higher=0;

	BinaryList out{1u<<12}, l1{0}, l2{0}, l3{0}, l4{0};
	l1.generate_base_random(1u << basesize, A);
	l2.generate_base_random(1u << basesize, A);
	l3.generate_base_random(1u << basesize, A);
	l4.generate_base_random(1u << basesize, A);

	BinaryLabel target {};
	target.zero();
	target.random();

	BinaryTree::streamjoin4lists(out, l1, l2, l3, l4, target, ta);

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0;i < out.load();++i) {
		// std::cout << out[i];
		out[i].recalculate_label(A);
		// std::cout << out[i];

		for (uint32_t j = 0; j < 2; ++j) {
			translate_level(&k_lower, &k_higher, j, ta);
			if (!(BinaryLabel::cmp(out[i].get_label(), target, k_lower, k_higher))) {
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
	BinaryMatrix A;
	A.identity();

	const std::vector<uint64_t> ta{{0, n/2, n}};
	uint64_t k_lower=0, k_higher=0;

	BinaryList out{1u<<basesize}, l1{0}, l2{0}, l3{0}, l4{0};
	l1.generate_base_random(1u << basesize, A);
	l2.generate_base_random(1u << basesize, A);

	BinaryLabel target {};
	target.zero();
	target.random();

	BinaryTree::streamjoin4lists_twolists(out, l1, l2, target, ta);

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0;i < out.load();++i) {
		// std::cout << out[i];
		out[i].recalculate_label(A);
		// std::cout << out[i];

		for (int j = 0; j < 2; ++j) {
			translate_level(&k_lower, &k_higher, j, ta);
			if (!(BinaryLabel::cmp(out[i].get_label(), target, k_lower, k_higher))) {
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

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif
