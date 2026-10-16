#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "element.h"
#include "matrix/matrix.h"
#include "list/list.h"
#include "matrix/matrix.h"
#include "tree.h"


constexpr uint32_t n  = 10;

using BinaryValue     = BinaryVector<n>;
using BinaryLabel     = BinaryVector<n>;
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

	const std::vector<uint32_t> ta{{0, n}};
	uint32_t k_lower, k_higher;
	translate_level(&k_lower, &k_higher, 0, ta);

	BinaryList out{1u<<basesize}, l1{0}, l2{0};
	l1.random(1u << basesize, A);
	l2.random(1u << basesize, A);

	BinaryLabel target {};
	target.zero();
	target.random();

	BinaryTree t{1, A, 0};
	t.join2lists(out, l1, l2, target, ta);

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
	// NOTE: 2^15 * 2^15 pairs, matching on n=10 bits: 2^20 expected
	// 	(was: 2^9 < out < 2^11)
	EXPECT_GT(out.load(),1u<<19);
	EXPECT_LT(out.load(),1u<<21);
}

TEST(TreeTest, join4lists) {
	uint32_t basesize = 8u;
	BinaryMatrix A;
	A.identity();

	const std::vector<uint32_t> ta{{0, n/2, n}};
	uint32_t k_lower=0, k_higher=0;

	BinaryList out{1u<<12}, l1{0}, l2{0}, l3{0}, l4{0};
	l1.random(1u << basesize, A);
	l2.random(1u << basesize, A);
	l3.random(1u << basesize, A);
	l4.random(1u << basesize, A);

	BinaryLabel target {};
	target.zero();
	target.random();

	BinaryTree t{1, A, 0};
	t.join4lists(out, l1, l2, l3, l4, target, ta);

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
	// NOTE: 2^8 * 2^8 pairs per side matching on n/2=5 bits: 2^11 each,
	// 	then 2^11 * 2^11 matching on the other 5 bits: 2^17 expected
	EXPECT_GT(out.load(),1u<<16);
	EXPECT_LT(out.load(),1u<<18);
}

TEST(TreeTest, join4lists_with2lists) {
	unsigned int basesize = 8;
	BinaryMatrix A;
	A.identity();

	const std::vector<uint32_t> ta{{0, n/2, n}};
	uint32_t k_lower=0, k_higher=0;

	BinaryList out{1u<<basesize}, l1{0}, l2{0}, l3{0}, l4{0};
	l1.random(1u << basesize, A);
	l2.random(1u << basesize, A);

	BinaryLabel target {};
	target.zero();
	target.random();

	BinaryTree t{1, A, 0};
	t.streamjoin4lists_twolists(out, l1, l2, target, ta);

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
	// NOTE: the lists `L3`, `L4` are simulated by `L1`, `L2`: as in
	// 	`join4lists` 2^17 outputs are expected
	EXPECT_GT(out.load(),1u<<16);
	EXPECT_LT(out.load(),1u<<18);
}

TEST(TreeTest, join8lists) {
	constexpr uint32_t basesize = 5u;
	BinaryMatrix A;
	A.identity();

	const std::vector<uint32_t> ta{{0, 4, 7, n}};

	BinaryList out{1u<<17};
	std::vector<BinaryList> L(8, BinaryList{0});
	for (auto &l : L) { l.random(1u << basesize, A); }

	BinaryLabel target {};
	target.zero();
	target.random();

	BinaryTree t{1, A, 0};
	t.join8lists(out, L, target, ta);

	uint32_t wrong = 0;
	for (uint64_t i = 0; i < out.load(); ++i) {
		// NOTE: the labels are the target
		if (!BinaryLabel::cmp(out[i].label, target, 0, n)) { wrong++; }
		out[i].recalculate_label(A);
		if (!BinaryLabel::cmp(out[i].label, target, 0, n)) { wrong++; }
	}

	EXPECT_EQ(0, wrong);
	// NOTE: 2^20 4-tuples per side matching on 4+7 bits: 2^9 each, then
	// 	2^18 pairs matching on the last 3 bits: 2^15 expected
	// NOTE: the base lists only have 2^5 elements, so the number of outputs
	// 	varies a lot: about 1 in 200 seeds is below 2^14
	EXPECT_GT(out.load(),1u<<13);
	EXPECT_LT(out.load(),1u<<17);
}

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif
