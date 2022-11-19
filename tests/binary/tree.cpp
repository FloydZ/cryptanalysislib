#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "./binary.h"
#include "tree.h"

#include "m4ri/m4ri.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

#if 0 // TODO, irgendwas dan der api hat sich getan

TEST(BinaryTreeTest, CrossProdoct) {
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.fill(0);
	BinaryList L1{0}, L2{0}, out{0};

	constexpr uint64_t size = 100;
	L1.set_load(size);
	L2.set_load(size);
	L1.generate_base_random(size, A);
	L2.generate_base_random(size, A);

	BinaryTree::cross_product(out, L1, L2, 0, n/2, n);
}

TEST(BinaryTreeTest, JoinForLevelOne) {
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.fill(0);

	BinaryTree t{1, A, 10, __level_translation_array};

    t[0].generate_base_random(1u << 10u, A);
    t[1].generate_base_random(1u << 10u, A);
    t.join_stream(0);

    EXPECT_EQ(1u << 20u, t[2].get_load());
}


TEST(BinaryTreeTest, JoinForLevelTwo) {
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.fill(0);

	BinaryTree t{2, A, 10, __level_translation_array};

    t[0].generate_base_random(1u << 2u, A);
    t[1].generate_base_random(1u << 2u, A);
    t.join_stream(0);
    t.join_stream(1);
    EXPECT_EQ(1u << 8u, t[3].get_load());
}


TEST(BinaryTreeTest, JoinForLevelThree) {
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.fill(0);

	BinaryTree t{3, A, 10, __level_translation_array};

    t[0].generate_base_random(1u << 2u, A);
    t[1].generate_base_random(1u << 2u, A);
    t.join_stream(0);
    t.join_stream(1);
    t.join_stream(2);
    EXPECT_EQ(1u << 16u, t[4].get_load());
}


TEST(BinaryTreeTest, JoinRandomListsLevel0) {
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.fill(0);
	A.gen_uniform(2);

	BinaryTree t{2, A, 4u, __level_translation_array};

    t[0].generate_base_random(1u << 4u, A);
    t[1].generate_base_random(1u << 4u, A);
	// std::cout << t[0];
    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);

    uint64_t num = 0;
    for (int i = 0; i < t[0].get_load(); ++i) {
        for (int j = 0; j < t[1].get_load(); ++j) {
            if (t[0][i].is_equal(t[1][j], 0)) {
                num++;
            }
        }
    }
    t.join_stream(0);
	// std::cout << num << "\n";
    EXPECT_NE(0, num);
    EXPECT_EQ(t[2].get_load(), num);
}


TEST(BinaryTreeTest, JoinRandomListsLevel1) {
	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.fill(0);
	A.gen_uniform(2);

	BinaryTree t{2, A, 10, __level_translation_array};

    t[0].generate_base_random(1u << 6u, A);
    t[1].generate_base_random(1u << 6u, A);
    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);
    t.join_stream(0);

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << 6u, A);
    t[1].generate_base_random(1u << 6u, A);

    t[2].sort_level(1, __level_translation_array);
    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);

	BinaryElement tmp;
    const uint64_t k_lower = 0;
	const uint64_t k_upper = tmp.label_size();

	uint64_t num = 0;
	BinaryElement el{};
    for (int i = 0; i < t[0].get_load(); ++i)
        for (int j = 0; j < t[1].get_load(); ++j) {
            if (t[0][i].is_equal(t[1][j], 0)) {
                BinaryElement::add(el, t[0][i], t[1][j], k_lower, k_upper);

                for (int o = 0; o < t[2].get_load(); ++o)
                    if (el.is_equal(t[2][o], 1))
                        num++;

            }
        }

    t.join_stream(1);

	EXPECT_NE(0, num);
	EXPECT_EQ(t[3].get_load(), num);
}


TEST(BinaryTreeTest, JoinRandomListsLevel2) {
	unsigned int base_size = 6u;

	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.fill(0);
	A.gen_uniform(2);

	BinaryTree t{3, A, 10, __level_translation_array};

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);

    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);
    t.join_stream(0);

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);

    t[2].sort_level(1, __level_translation_array);
    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);
    t.join_stream(1);

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);
    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);
    t[3].sort_level(2, __level_translation_array);

	BinaryElement tmp;
	const uint64_t k_lower = 0;
	const uint64_t k_upper = tmp.label_size();

    uint64_t num = 0;
	BinaryElement el{};
	BinaryElement el2{};
    for (int i = 0; i < t[0].get_load(); ++i)
        for (int j = 0; j < t[1].get_load(); ++j) {
            if (t[0][i].is_equal(t[1][j], 0)) {
	            BinaryElement::add(el, t[0][i], t[1][j]);
                for (int o = 0; o < t[2].get_load(); ++o) {
                    if (el.is_equal(t[2][o], 1)) {
	                    BinaryElement::add(el2, el, t[2][o]);
                        for (int r = 0; r < t[3].get_load(); ++r)
                            if (el2.is_equal(t[3][r], 2)) {
                                num++;
                            }
                    }
                }
            }
        }
    t.join_stream(2);

	EXPECT_NE(0, num);
    EXPECT_EQ(t[4].get_load(), num);
}

TEST(BinaryTreeTest, JoinRandomListsLevel3) {
	unsigned int base_size = 5u;

	mzd_t *A_ = mzd_init(n, n);
	Matrix_T<mzd_t *> A((mzd_t *)A_);
	A.fill(0);
	A.gen_uniform(2);

	BinaryTree t{4, A, 10, __level_translation_array};

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);
    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);
    t.join_stream(0);

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);

    t[2].sort_level(1, __level_translation_array);
    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);
    t.join_stream(1);

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);
    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);
    t[3].sort_level(2, __level_translation_array);
    t.join_stream(2);

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);
    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);
    t[4].sort_level(3, __level_translation_array);

	BinaryElement tmp;
	const uint64_t k_lower = 0;
	const uint64_t k_upper = tmp.label_size();

    uint64_t num = 0;
	BinaryElement el{};
	BinaryElement el2{};
	BinaryElement el3{};
    for (int i = 0; i < t[0].get_load(); ++i)
        for (int j = 0; j < t[1].get_load(); ++j) {
            if (t[0][i].is_equal(t[1][j], 0)) {
	            BinaryElement::add(el, t[0][i], t[1][j]);
                for (int o = 0; o < t[2].get_load(); ++o) {
                    if (el.is_equal(t[2][o], 1)) {
	                    BinaryElement::add(el2, el, t[2][o]);
                        for (int r = 0; r < t[3].get_load(); ++r)
                            if (el2.is_equal(t[3][r], 2)) {
	                            BinaryElement::add(el3, el2, t[3][r]);
                                for (int w = 0; w < t[4].get_load(); ++w) {
                                    if (el3.is_equal(t[4][w], 3)) {
                                        num++;
                                    }
                                }
                            }
                    }
                }
            }
        }
    t.join_stream(3);

	EXPECT_NE(0, num);
	EXPECT_EQ(t[5].get_load(), num);
}


//
//TEST(BinaryTreeTest, BuildTreeCheckDistributionOnBinaryD4) {
//	const uint64_t d=4;
//	Tree_T t{d};
//	fplll::ZZ_mat<Label_Type> A(n, n);
//	A.gen_uniform(2);
//
//	__level_translation_array[1]=5;
//	__level_translation_array[2]=10;
//	__level_translation_array[3]=15;
//	__level_translation_array[4]=20;
//	__level_translation_array[5]=25;
//	__level_translation_array[6]=30;
//	__level_translation_array[7]=35;
//	Label target {};
//	target.zero();
//	target.random();
//	std::cout<<"target is "<<target<<"\n";
//	t.build_tree(target,A);
//
//	std::cout<<"\nResultsize: "<<t[d+1].get_load();
//	auto right=true;
//	int wrong=0;
//	for(uint64_t i = 0;i<t[d+1].get_load();++i) {
//		t[d+1][i].recalculate_label(A);
//		//std::cout<<"\n"<<t[d+1][i];
//		for(int j =0;j<d;++j)
//			if(!(Label::cmp(t[d+1][i].get_label(),target,j, __level_translation_array))) {
//				right = false;
//				std::cout<<"\n"<<t[d+1][i].get_label();
//				wrong++;
//				break;
//			}
//
//	}
//	std::cout<<"\nWrong results: "<<wrong<<"\n";
//
//	EXPECT_EQ(0, wrong);
//	EXPECT_EQ(right, true);
//}

#endif

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif
