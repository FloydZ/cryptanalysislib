#include <gtest/gtest.h>
#include <iostream>
#include <cstdint>

#include "binary.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

// TODO more limbs n>64 not working
//TEST(TreeTest, join2lists) {
//	unsigned int basesize = 15;//15
//	mzd_t *A_ = mzd_init(n, n);
//	Matrix_T<mzd_t *> A((mzd_t *) A_);
//	A.gen_identity(n);
//
//	const std::vector<uint64_t> ta{{0, n}};
//	uint64_t k_lower, k_higher;
//	translate_level(&k_lower, &k_higher, 0, ta);
//
//	BinaryList out{0}, l1{0}, l2{0};
//	l1.generate_base_random(1u << basesize, A);
//	l2.generate_base_random(1u << basesize, A);
//
//	BinaryLabel target {};
//	target.zero();
//	target.random();
//
//	BinaryTree::join2lists(out, l1, l2, target, ta);
//
//	auto right=true;
//	int wrong=0;
//	for(uint64_t i = 0;i < out.get_load();++i) {
//		// std::cout << out[i].get_label();
//		out[i].recalculate_label(A);
//		if (!(BinaryLabel::cmp(out[i].get_label(), target, k_lower, k_higher))) {
//			right = false;
//			wrong++;
//		}
//	}
//
//	EXPECT_GT(out.get_load(), 0);
//	EXPECT_EQ(0, wrong);
//	EXPECT_EQ(right, true);
//	EXPECT_GT(out.get_load(),1u<<9);
//	EXPECT_LT(out.get_load(),1u<<11);
//}
//
//TEST(TreeTest, join4lists) {
//	unsigned int basesize = 10;
//	mzd_t *A_ = mzd_init(n, n);
//	Matrix_T<mzd_t *> A((mzd_t *) A_);
//	A.gen_identity(n);
//
//	const std::vector<uint64_t> ta{{0, n/2, n}};
//	uint64_t k_lower=0, k_higher=0;
//
//	BinaryList out{0}, l1{0}, l2{0}, l3{0}, l4{0};
//	l1.generate_base_random(1u << basesize, A);
//	l2.generate_base_random(1u << basesize, A);
//	l3.generate_base_random(1u << basesize, A);
//	l4.generate_base_random(1u << basesize, A);
//
//	BinaryLabel target {};
//	target.zero();
//	target.random();
//
//	BinaryTree::streamjoin4lists(out, l1, l2, l3, l4, target, ta);
//
//	auto right=true;
//	int wrong=0;
//	for(uint64_t i = 0;i < out.get_load();++i) {
//		// std::cout << out[i];
//		out[i].recalculate_label(A);
//		// std::cout << out[i];
//
//		for (int j = 0; j < 2; ++j) {
//			translate_level(&k_lower, &k_higher, j, ta);
//			if (!(BinaryLabel::cmp(out[i].get_label(), target, k_lower, k_higher))) {
//				right = false;
//				wrong++;
//			}
//		}
//	}
//
//	EXPECT_GT(out.get_load(), 0);
//	EXPECT_EQ(0, wrong);
//	EXPECT_EQ(right, true);
//	EXPECT_GT(out.get_load(),1u<<9);
//	EXPECT_LT(out.get_load(),1u<<11);
//}
//
//TEST(TreeTest, join4lists_with2lists) {
//	unsigned int basesize = 10;
//	mzd_t *A_ = mzd_init(n, n);
//	Matrix_T<mzd_t *> A((mzd_t *) A_);
//	A.gen_identity(n);
//
//	const std::vector<uint64_t> ta{{0, n/2, n}};
//	uint64_t k_lower=0, k_higher=0;
//
//	BinaryList out{0}, l1{0}, l2{0}, l3{0}, l4{0};
//	l1.generate_base_random(1u << basesize, A);
//	l2.generate_base_random(1u << basesize, A);
//
//	BinaryLabel target {};
//	target.zero();
//	target.random();
//
//	BinaryTree::streamjoin4lists_twolists(out, l1, l2, target, ta);
//
//	auto right=true;
//	int wrong=0;
//	for(uint64_t i = 0;i < out.get_load();++i) {
//		// std::cout << out[i];
//		out[i].recalculate_label(A);
//		// std::cout << out[i];
//
//		for (int j = 0; j < 2; ++j) {
//			translate_level(&k_lower, &k_higher, j, ta);
//			if (!(BinaryLabel::cmp(out[i].get_label(), target, k_lower, k_higher))) {
//				right = false;
//				wrong++;
//			}
//		}
//	}
//
//	EXPECT_GT(out.get_load(), 0);
//	EXPECT_EQ(0, wrong);
//	EXPECT_EQ(right, true);
//	EXPECT_GT(out.get_load(),1u<<9);
//	EXPECT_LT(out.get_load(),1u<<11);
//}

/* TODO not wroking
TEST(TreeTest, join8lists) {
	unsigned int basesize = 5;
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *) A_);
	A.gen_identity(n);

	const std::vector<uint64_t> ta{{0, n/4, n/2, G_n}};
	uint64_t k_lower=0, k_higher=0;

	BinaryList out{0};
	std::vector<BinaryList> L;
	for (int k = 0; k < 8; k++) {
		BinaryList L_i{0};
		L_i.generate_base_random(1u << basesize, A);
		L.push_back(L_i);
	}

	BinaryLabel target {};
	target.zero();
	target.random();

	BinaryTree::streamjoin8lists(out, L, target, ta);

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0;i < out.get_load();++i) {
		std::cout << out[i];
		out[i].recalculate_label(A);
		std::cout << out[i];

		for (int j = 0; j < 2; ++j) {
			translate_level(&k_lower, &k_higher, j, ta);
			if (!(BinaryLabel::cmp(out[i].get_label(), target, k_lower, k_higher))) {
				right = false;
				wrong++;
			}
		}
	}

	EXPECT_GT(out.get_load(), 0);
	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	EXPECT_GT(out.get_load(),1u<<9);
	EXPECT_LT(out.get_load(),1u<<11);
}
*/

// TODO not working because using not binary tree
/*
TEST(TreeTest, BuildTreeTest1) {
	const uint64_t d=1;
	unsigned int basesize=10;
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.gen_identity(n);

//    t[0].set_load(0);
//    t[1].set_load(0);
//    t[0].generate_base(1u << 5u, A);
//    t[1].generate_base(1u << 5u, A);
//    t[0].sort_level(0);
//    t[1].sort_level(0);
//    t.join_stream(0);
	__level_translation_array[1]=5;
	__level_translation_array[2]=10;
	__level_translation_array[3]=15;
	__level_translation_array[4]=20;

	BinaryTree t{d, A, basesize, __level_translation_array};
	uint64_t k_lower, k_higher;

	BinaryLabel target {};
	target.zero();
	target.random();
	t.build_tree(target);

	//std::cout<<"target is "<<target<<"\n";
	//std::cout << "bla" << t[0] << "\n";
	//std::cout<<"\nResultsize: "<<t[d+1].get_load();

	auto right=true;
	int wrong=0;
	for(uint64_t i = 0;i<t[d+1].get_load();++i) {
		t[d+1][i].recalculate_label(A);
		//std::cout<<"\n"<<t[d+1][i];
		for(int j =0;j<d;++j) {
			translate_level(&k_lower, &k_higher, j, __level_translation_array);

			if (!(BinaryLabel::cmp(t[d + 1][i].get_label(), target, k_lower, k_higher))) {
				right = false;
				std::cout << "\n" << t[d + 1][i].get_label();
				wrong++;
			}
		}
	}

	EXPECT_EQ(0, wrong);
	EXPECT_EQ(right, true);
	EXPECT_GT(t[d+1].get_load(),1u<<(basesize-1));
	EXPECT_LT(t[d+1].get_load(),1u<<(basesize+1));
}

TEST(TreeTest, RestoreBaselists) {
	const uint64_t d=4;
	unsigned int basesize=10;
	fplll::ZZ_mat<Label_Type> A_(n, n);
	//A.gen_uniform(4);
	//A.fill(0);
	A_.gen_identity(n);
	const Matrix_T<fplll::ZZ_mat<Label_Type>> A{A_};

	__level_translation_array[1]=5;
	__level_translation_array[2]=10;
	__level_translation_array[3]=15;
	__level_translation_array[4]=20;

	Tree t{d, A, basesize, __level_translation_array};
	Tree t2{d, A, basesize, __level_translation_array};

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].generate_base_random(1u<<basesize, A);
	t[1].generate_base_random(1u<<basesize, A);
	t[0].sort_level(-1);
	t[1].sort_level(-1);
	t2[0].generate_base_random(1u<<basesize, A);
	t2[1].generate_base_random(1u<<basesize, A);
	Label target {};
	target.random();
	for(uint64_t i=0;i<t[0].get_load();++i) {
		t2[0][i] = t[0][i];
		t2[1][i] = t[1][i];
	}
	t.build_tree(target, false);
	uint64_t wrong1=0;
	uint64_t wrong2=0;
	bool correct=true;
	t[0].sort_level(-1);
	t[1].sort_level(-1);
	for(uint64_t i=0;i<t[0].get_load();++i) {
		if (!(t2[0][i].is_equal(t[0][i], -1))) {
			correct = false;
			wrong1++;
		}
		if(!(t2[1][i].is_equal(t[1][i],-1))){
			correct = false;
			wrong2++;
		}
	}

	EXPECT_EQ(correct,true);
	EXPECT_EQ(wrong1,0);
	EXPECT_EQ(wrong2,0);
}

TEST(TreeTest, RestoreLabelSingleList) {
	const uint64_t d=4;
	unsigned int basesize=10;

	fplll::ZZ_mat<Label_Type> A_(n, n);
	//A_.gen_uniform(4);
	//A_.fill(0);
	A_.gen_identity(n);
	const Matrix_T<fplll::ZZ_mat<Label_Type>> A{A_};

	__level_translation_array[1]=5;
	__level_translation_array[2]=10;
	__level_translation_array[3]=15;
	__level_translation_array[4]=20;

	Tree t{d, A, basesize, __level_translation_array};

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].generate_base_random(1u<<basesize, A);
	t[1].generate_base_random(1u<<basesize, A);

	Label target {};
	target.random();
	Element tmp {};
	t.build_tree(target, false);

	uint64_t wrong=0;
	bool correct=true;
	for(uint64_t i=0;i<t[d+1].get_load();++i) {
		tmp.get_label()=t[d+1][i].get_label();
		t[d+1][i].recalculate_label(A_);

		if (!(t[d+1][i].is_equal(tmp, -1))) {
			correct = false;
			wrong++;
		}

	}

	EXPECT_EQ(correct,true);
	EXPECT_EQ(wrong,0);
}


TEST(TreeTest, RestoreLabelTwoLists) {
	const uint64_t d=4;
	unsigned int basesize=10;
	fplll::ZZ_mat<Label_Type> A_(n, n);
	//A_.gen_uniform(4);
	//A_.fill(0);
	A_.gen_identity(n);
	const Matrix_T<fplll::ZZ_mat<Label_Type>> A{A_};

	__level_translation_array[1]=5;
	__level_translation_array[2]=10;
	__level_translation_array[3]=15;
	__level_translation_array[4]=20;

	Tree t{d, A, basesize, __level_translation_array};

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].generate_base_random(1u<<basesize, A);
	t[1].generate_base_random(1u<<basesize, A);

	Label target {};
	target.random();
	Element tmp {};
	t.build_tree(target, false,true);

	uint64_t wrong=0;
	bool correct=true;
	for (int a=0;a<2;++a)
		for(uint64_t i=0;i<t[d+a].get_load();++i) {
			tmp.get_label()=t[d+a][i].get_label();
			t[d+a][i].recalculate_label(A_);
			if (!(t[d+a][i].is_equal(tmp, -1))) {
				correct = false;
				wrong++;
			}
		}

	EXPECT_EQ(correct,true);
	EXPECT_EQ(wrong,0);
}
*/

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif
