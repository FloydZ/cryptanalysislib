#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "tree.h"

constexpr uint32_t n  = 20;
using BinaryValue     = BinaryContainer<n>;
using BinaryLabel     = BinaryContainer<n>;
using BinaryMatrix    = FqMatrix<uint64_t, n, n, 2, true>;
using BinaryElement   = Element_T<BinaryValue, BinaryLabel, BinaryMatrix>;
using BinaryList      = List_T<BinaryElement>;
using BinaryTree      = Tree_T<BinaryList>;

static std::vector<uint64_t> __level_translation_array{{0, 5, 10, 15, n}};
static std::vector<std::vector<uint8_t>> __level_filter_array{{ {{4,0,0}}, {{1,0,0}}, {{1,0,0}}, {{0,0,0}} }};


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(BinaryTreeTest, CrossProdoct) {
	BinaryMatrix A;
	A.fill(0);
	constexpr uint64_t size = 100;
	BinaryList L1{0}, L2{0}, out{size*size};

	L1.generate_base_random(size, A);
	L2.generate_base_random(size, A);

	BinaryTree::cross_product(out, L1, L2, 0, n/2, n);
}

TEST(BinaryTreeTest, JoinForLevelOne) {
	BinaryMatrix A;
	A.fill(0);

	BinaryTree t{1, A, 10, __level_translation_array, __level_filter_array};

    t[0].generate_base_random(1u << 10u, A);
    t[1].generate_base_random(1u << 10u, A);
    t.join_stream(0);

    EXPECT_EQ(1u << 20u, t[2].load());
}

TEST(BinaryTreeTest, JoinForLevelTwo) {
	BinaryMatrix A;
	A.fill(0);

	BinaryTree t{2, A, 10, __level_translation_array, __level_filter_array};

    t[0].generate_base_random(1u << 2u, A);
    t[1].generate_base_random(1u << 2u, A);
    t.join_stream(0);
    t.join_stream(1);
    EXPECT_EQ(1u << 8u, t[3].load());
}

TEST(BinaryTreeTest, JoinForLevelThree) {
	BinaryMatrix A;
	A.fill(0);

	BinaryTree t{3, A, 10, __level_translation_array, __level_filter_array};

    t[0].generate_base_random(1u << 2u, A);
    t[1].generate_base_random(1u << 2u, A);
    t.join_stream(0);
    t.join_stream(1);
    t.join_stream(2);
    EXPECT_EQ(1u << 16u, t[4].load());
}

TEST(BinaryTreeTest, JoinRandomListsLevel0) {
	BinaryMatrix A;
	A.random();

	constexpr size_t size = 12u;

	static std::vector<uint64_t> __level_translation_array{{0, n}};
	BinaryTree t{2, A, size, __level_translation_array, __level_filter_array};

    t[0].generate_base_random(1u << size, A);
    t[1].generate_base_random(1u << size, A);
    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);

    uint64_t num = 0;
    for (size_t i = 0; i < t[0].load(); ++i) {
        for (size_t j = 0; j < t[1].load(); ++j) {
            if (t[0][i].is_equal(t[1][j])) {
                num++;
            }
        }
    }

    t.join_stream(0);
    EXPECT_NE(0, num);
    EXPECT_EQ(t[2].load(), num);
}

//TEST(BinaryTreeTest, JoinRandomListsLevel1) {
//	BinaryMatrix A;
//	A.random();
//
//	constexpr size_t size = 10;
//	static std::vector<uint64_t> __level_translation_array{{0, n/2, n}};
//	BinaryTree t{2, A, size, __level_translation_array, __level_filter_array};
//
//    t[0].generate_base_random(1u << size, A);
//    t[1].generate_base_random(1u << size, A);
//    t[0].sort_level(0, __level_translation_array);
//    t[1].sort_level(0, __level_translation_array);
//    t.join_stream(0);
//
//    t[0].set_load(0);
//    t[1].set_load(0);
//    t[0].generate_base_random(1u << size, A);
//    t[1].generate_base_random(1u << size, A);
//
//    t[2].sort_level(1, __level_translation_array);
//    t[0].sort_level(0, __level_translation_array);
//    t[1].sort_level(0, __level_translation_array);
//
//	BinaryElement tmp;
//    const uint64_t k_lower = 0;
//	const uint64_t k_upper = tmp.label_size();
//
//	uint64_t num = 0;
//	BinaryElement el{};
//    for (size_t i = 0; i < t[0].load(); ++i) {
//		for (size_t j = 0; j < t[1].load(); ++j) {
//			if (t[0][i].is_equal(t[1][j], 0)) {
//				BinaryElement::add(el, t[0][i], t[1][j], k_lower, k_upper);
//
//				for (size_t o = 0; o < t[2].load(); ++o) {
//					if (el.is_equal(t[2][o], 1)) {
//						num++;
//					}
//				}
//			}
//		}
//	}
//
//    t.join_stream(1);
//
//	EXPECT_NE(0, num);
//	EXPECT_EQ(t[3].load(), num);
//}
//
//TEST(BinaryTreeTest, JoinRandomListsLevel2) {
//	unsigned int base_size = 6u;
//
//	BinaryMatrix A;
//	A.fill(0);
//	A.random();
//
//	BinaryTree t{3, A, 10, __level_translation_array, __level_filter_array};
//
//    t[0].set_load(0);
//    t[1].set_load(0);
//    t[0].generate_base_random(1u << base_size, A);
//    t[1].generate_base_random(1u << base_size, A);
//
//    t[0].sort_level(0, __level_translation_array);
//    t[1].sort_level(0, __level_translation_array);
//    t.join_stream(0);
//
//    t[0].set_load(0);
//    t[1].set_load(0);
//    t[0].generate_base_random(1u << base_size, A);
//    t[1].generate_base_random(1u << base_size, A);
//
//    t[2].sort_level(1, __level_translation_array);
//    t[0].sort_level(0, __level_translation_array);
//    t[1].sort_level(0, __level_translation_array);
//    t.join_stream(1);
//
//    t[0].set_load(0);
//    t[1].set_load(0);
//    t[0].generate_base_random(1u << base_size, A);
//    t[1].generate_base_random(1u << base_size, A);
//    t[0].sort_level(0, __level_translation_array);
//    t[1].sort_level(0, __level_translation_array);
//    t[3].sort_level(2, __level_translation_array);
//
//	BinaryElement tmp;
//	const uint64_t k_lower = 0;
//	const uint64_t k_upper = tmp.label_size();
//
//    uint64_t num = 0;
//	BinaryElement el{};
//	BinaryElement el2{};
//    for (size_t i = 0; i < t[0].load(); ++i)
//        for (size_t j = 0; j < t[1].load(); ++j) {
//            if (t[0][i].is_equal(t[1][j], 0)) {
//	            BinaryElement::add(el, t[0][i], t[1][j]);
//                for (size_t o = 0; o < t[2].load(); ++o) {
//                    if (el.is_equal(t[2][o], 1)) {
//	                    BinaryElement::add(el2, el, t[2][o]);
//                        for (size_t r = 0; r < t[3].load(); ++r)
//                            if (el2.is_equal(t[3][r], 2)) {
//                                num++;
//                            }
//                    }
//                }
//            }
//        }
//    t.join_stream(2);
//
//	EXPECT_NE(0, num);
//    EXPECT_EQ(t[4].load(), num);
//}
//
//TEST(BinaryTreeTest, JoinRandomListsLevel3) {
//	unsigned int base_size = 5u;
//
//	BinaryMatrix A;
//	A.random();
//
//	BinaryTree t{4, A, 10, __level_translation_array, __level_filter_array};
//
//    t[0].set_load(0);
//    t[1].set_load(0);
//    t[0].generate_base_random(1u << base_size, A);
//    t[1].generate_base_random(1u << base_size, A);
//    t[0].sort_level(0, __level_translation_array);
//    t[1].sort_level(0, __level_translation_array);
//    t.join_stream(0);
//
//    t[0].set_load(0);
//    t[1].set_load(0);
//    t[0].generate_base_random(1u << base_size, A);
//    t[1].generate_base_random(1u << base_size, A);
//
//    t[2].sort_level(1, __level_translation_array);
//    t[0].sort_level(0, __level_translation_array);
//    t[1].sort_level(0, __level_translation_array);
//    t.join_stream(1);
//
//    t[0].set_load(0);
//    t[1].set_load(0);
//    t[0].generate_base_random(1u << base_size, A);
//    t[1].generate_base_random(1u << base_size, A);
//    t[0].sort_level(0, __level_translation_array);
//    t[1].sort_level(0, __level_translation_array);
//    t[3].sort_level(2, __level_translation_array);
//    t.join_stream(2);
//
//    t[0].set_load(0);
//    t[1].set_load(0);
//    t[0].generate_base_random(1u << base_size, A);
//    t[1].generate_base_random(1u << base_size, A);
//    t[0].sort_level(0, __level_translation_array);
//    t[1].sort_level(0, __level_translation_array);
//    t[4].sort_level(3, __level_translation_array);
//
//	BinaryElement tmp;
//	const uint64_t k_lower = 0;
//	const uint64_t k_upper = tmp.label_size();
//
//    uint64_t num = 0;
//	BinaryElement el{};
//	BinaryElement el2{};
//	BinaryElement el3{};
//    for (size_t i = 0; i < t[0].load(); ++i)
//        for (size_t j = 0; j < t[1].load(); ++j) {
//            if (t[0][i].is_equal(t[1][j], 0)) {
//	            BinaryElement::add(el, t[0][i], t[1][j]);
//                for (size_t o = 0; o < t[2].load(); ++o) {
//                    if (el.is_equal(t[2][o], 1)) {
//	                    BinaryElement::add(el2, el, t[2][o]);
//                        for (size_t r = 0; r < t[3].load(); ++r)
//                            if (el2.is_equal(t[3][r], 2)) {
//	                            BinaryElement::add(el3, el2, t[3][r]);
//                                for (size_t w = 0; w < t[4].load(); ++w) {
//                                    if (el3.is_equal(t[4][w], 3)) {
//                                        num++;
//                                    }
//                                }
//                            }
//                    }
//                }
//            }
//        }
//    t.join_stream(3);
//
//	EXPECT_NE(0, num);
//	EXPECT_EQ(t[5].load(), num);
//}

#ifndef EXTERNAL_MAIN
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
#endif
