#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "test.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

#ifdef USE_FPLLL
TEST(TreeTest, JoinForLevelOne) {
	fplll::ZZ_mat<kAryType> A_(n, n);
	A_.fill(0);
	Matrix_T<fplll::ZZ_mat<kAryType>> A{A_};

	Tree t{1, A, 10, __level_translation_array};

    t[0].generate_base_random(1u << 10u, A);
    t[1].generate_base_random(1u << 10u, A);
    t.join_stream(0);

    EXPECT_EQ(1u << 20u, t[2].get_load());
}

TEST(TreeTest, JoinForLevelTwo) {
	fplll::ZZ_mat<kAryType> A_(n, n);
	A_.fill(0);
	Matrix_T<fplll::ZZ_mat<kAryType>> A{A_};

    Tree t{2, A, 10, __level_translation_array};

    t[0].generate_base_random(1u << 2u, A);
    t[1].generate_base_random(1u << 2u, A);
    t.join_stream(0);
    t.join_stream(1);
    EXPECT_EQ(1u << 8u, t[3].get_load());
}

TEST(TreeTest, JoinForLevelThree) {
	fplll::ZZ_mat<kAryType> A_(n, n);
	A_.fill(0);
	Matrix_T<fplll::ZZ_mat<kAryType>> A{A_};

    Tree t{3, A, 10, __level_translation_array};

    t[0].generate_base_random(1u << 2u, A);
    t[1].generate_base_random(1u << 2u, A);
    t.join_stream(0);
    t.join_stream(1);
    t.join_stream(2);
    EXPECT_EQ(1u << 16u, t[4].get_load());
}

TEST(TreeTest, JoinRandomListsLevel0) {
	fplll::ZZ_mat<kAryType> A_(n, n);
	//A_.fill(0);
	A_.gen_uniform(2);
	Matrix_T<fplll::ZZ_mat<kAryType>> A{A_};

    Tree t{2, A, 4u, __level_translation_array};

    t[0].generate_base_random(1u << 4u, A);
    t[1].generate_base_random(1u << 4u, A);
	// std::cout << t[0];
    t[0].sort_level(0);
    t[1].sort_level(0);

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

TEST(TreeTest, JoinRandomListsLevel1) {
	fplll::ZZ_mat<kAryType> A_(n, n);
	//A_.fill(0);
	A_.gen_uniform(2);
	Matrix_T<fplll::ZZ_mat<kAryType>> A{A_};

	Tree t{2, A, 10, __level_translation_array};

    t[0].generate_base_random(1u << 4u, A);
    t[1].generate_base_random(1u << 4u, A);
    t[0].sort_level(0);
    t[1].sort_level(0);
    t.join_stream(0);

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << 4u, A);
    t[1].generate_base_random(1u << 4u, A);

    t[2].sort_level(1);
    t[0].sort_level(0);
    t[1].sort_level(0);

    uint64_t num = 0;
    Element el{};
    for (int i = 0; i < t[0].get_load(); ++i)
        for (int j = 0; j < t[1].get_load(); ++j) {
            if (t[0][i].is_equal(t[1][j], 0)) {
                Element::add(el, t[0][i], t[1][j]);

                for (int o = 0; o < t[2].get_load(); ++o)
                    if (el.is_equal(t[2][o], 1))
                        num++;

            }
        }

    t.join_stream(1);

	EXPECT_NE(0, num);
	EXPECT_EQ(t[3].get_load(), num);
}

TEST(TreeTest, JoinRandomListsLevel2) {
	fplll::ZZ_mat<kAryType> A_(n, n);
	//A_.fill(0);
	unsigned int base_size = 4u;
	A_.gen_uniform(2);
	Matrix_T<fplll::ZZ_mat<kAryType>> A{A_};

    Tree t{3, A, 10, __level_translation_array};

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);

    t[0].sort_level(0);
    t[1].sort_level(0);
    t.join_stream(0);

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);

    t[2].sort_level(1);
    t[0].sort_level(0);
    t[1].sort_level(0);
    t.join_stream(1);

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);
    t[0].sort_level(0);
    t[1].sort_level(0);
    t[3].sort_level(2);

    uint64_t num = 0;
    Element el{};
    Element el2{};
    for (int i = 0; i < t[0].get_load(); ++i)
        for (int j = 0; j < t[1].get_load(); ++j) {
            if (t[0][i].is_equal(t[1][j], 0)) {
                Element::add(el, t[0][i], t[1][j]);
                for (int o = 0; o < t[2].get_load(); ++o) {
                    if (el.is_equal(t[2][o], 1)) {
                        Element::add(el2, el, t[2][o]);
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

TEST(TreeTest, JoinRandomListsLevel3) {
	fplll::ZZ_mat<kAryType> A_(n, n);
	//A_.fill(0);
	unsigned int base_size = 3u;
	A_.gen_uniform(2);
	Matrix_T<fplll::ZZ_mat<kAryType>> A{A_};

    Tree t{4, A, 10, __level_translation_array};

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);
    t[0].sort_level(0);
    t[1].sort_level(0);
    t.join_stream(0);

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);

    t[2].sort_level(1);
    t[0].sort_level(0);
    t[1].sort_level(0);
    t.join_stream(1);

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);
    t[0].sort_level(0);
    t[1].sort_level(0);
    t[3].sort_level(2);
    t.join_stream(2);

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << base_size, A);
    t[1].generate_base_random(1u << base_size, A);
    t[0].sort_level(0);
    t[1].sort_level(0);
    t[4].sort_level(3);

    uint64_t num = 0;
    Element el{};
    Element el2{};
    Element el3{};
    for (int i = 0; i < t[0].get_load(); ++i)
        for (int j = 0; j < t[1].get_load(); ++j) {
            if (t[0][i].is_equal(t[1][j], 0)) {
                Element::add(el, t[0][i], t[1][j]);
                for (int o = 0; o < t[2].get_load(); ++o) {
                    if (el.is_equal(t[2][o], 1)) {
                        Element::add(el2, el, t[2][o]);
                        for (int r = 0; r < t[3].get_load(); ++r)
                            if (el2.is_equal(t[3][r], 2)) {
                                Element::add(el3, el2, t[3][r]);
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
//TEST(TreeTest, BuildTreeCheckDistributionOnBinaryD4) {
//	const uint64_t d=4;
//	Tree_T t{d};
//	fplll::ZZ_mat<kAryType> A(n, n);
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


int main(int argc, char **argv) {
    __level_translation_array[0]=0;
    __level_translation_array[1]=2;
    __level_translation_array[2]=4;
    __level_translation_array[3]=6;
    __level_translation_array[4]=8;

    InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
    return RUN_ALL_TESTS();
}
