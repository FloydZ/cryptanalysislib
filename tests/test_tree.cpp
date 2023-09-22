#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <cstdint>

#include "helper.h"
#include "kAry_type.h"
#include "tree.h"
#include "matrix/fq_matrix.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr uint32_t n    = 15;
constexpr uint32_t k    = 15;
constexpr uint32_t q    = 3;

using T 			= uint8_t;
using Matrix 		= FqMatrix<T, n, k, q>;
using Value     	= kAryContainer_T<T, n, q>;
using Label    		= kAryContainer_T<T, k, q>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;

static std::vector<uint64_t> __level_translation_array{{0, 5, 10, 15, n}};
static std::vector<std::vector<uint8_t>> __level_filter_array{{ {{4,0,0}}, {{1,0,0}}, {{1,0,0}}, {{0,0,0}} }};

TEST(TreeTest, JoinForLevelOne) {
	Matrix A;
	A.fill(0);

	Tree t{1, A, 10, __level_translation_array, __level_filter_array};

    t[0].generate_base_random(1u << 10u, A);
    t[1].generate_base_random(1u << 10u, A);
    t.join_stream(0);

    EXPECT_EQ(1u << 20u, t[2].load());
}

TEST(TreeTest, JoinForLevelTwo) {
	Matrix A;
	A.fill(0);

    Tree t{2, A, 10, __level_translation_array, __level_filter_array};

    t[0].generate_base_random(1u << 2u, A);
    t[1].generate_base_random(1u << 2u, A);
    t.join_stream(0);
    t.join_stream(1);
    EXPECT_EQ(1u << 8u, t[3].load());
}

TEST(TreeTest, JoinForLevelThree) {
	Matrix A;
	A.fill(0);
    Tree t{3, A, 10, __level_translation_array, __level_filter_array};

    t[0].generate_base_random(1u << 2u, A);
    t[1].generate_base_random(1u << 2u, A);
    t.join_stream(0);
    t.join_stream(1);
    t.join_stream(2);
    EXPECT_EQ(1u << 16u, t[4].load());
}

TEST(TreeTest, JoinRandomListsLevel0) {
	Matrix A;
	A.random();

	static std::vector<uint64_t> __level_translation_array{{0, n}};
    Tree t{2, A, 4u, __level_translation_array, __level_filter_array};

    t[0].generate_base_random(1u << 12u, A);
    t[1].generate_base_random(1u << 12u, A);
    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);

    uint64_t num = 0;
    for (size_t i = 0; i < t[0].load(); ++i) {
        for (size_t j = 0; j < t[1].load(); ++j) {
            if (t[0][i].is_equal(t[1][j], 0)) {
                num++;
            }
        }
    }
    t.join_stream(0);
    EXPECT_NE(0, num);
    EXPECT_EQ(t[2].load(), num);
}

TEST(TreeTest, JoinRandomListsLevel1) {
	Matrix A;
	A.random();

	static std::vector<uint64_t> __level_translation_array{{0, n/2, n}};
	Tree t{2, A, 11, __level_translation_array, __level_filter_array};

    t[0].generate_base_random(1u << 12u, A);
    t[1].generate_base_random(1u << 12u, A);
    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);
    t.join_stream(0);

    t[0].set_load(0);
    t[1].set_load(0);
    t[0].generate_base_random(1u << 12u, A);
    t[1].generate_base_random(1u << 12u, A);

    t[2].sort_level(1, __level_translation_array);
    t[0].sort_level(0, __level_translation_array);
    t[1].sort_level(0, __level_translation_array);

    uint64_t num = 0;
    Element el{};
    for (size_t i = 0; i < t[0].load(); ++i) {
		for (size_t j = 0; j < t[1].load(); ++j) {
			if (t[0][i].is_equal(t[1][j], 0)) {
				Element::add(el, t[0][i], t[1][j]);

				for (size_t o = 0; o < t[2].load(); ++o) {
					if (el.is_equal(t[2][o], 1)) {
						num++;
					}
				}
			}
		}
	}

    t.join_stream(1);

	EXPECT_NE(0, num);
	EXPECT_EQ(t[3].load(), num);
}

//TEST(TreeTest, JoinRandomListsLevel2) {
//	Matrix A;
//	A.random();
//
//	constexpr size_t base_size = 12;
//	static std::vector<uint64_t> __level_translation_array{{0, n/3, 2*n/3, n}};
//    Tree t{3, A, 10, __level_translation_array, __level_filter_array};
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
//    uint64_t num = 0;
//    Element el{};
//    Element el2{};
//    for (size_t i = 0; i < t[0].load(); ++i) {
//		for (size_t j = 0; j < t[1].load(); ++j) {
//			if (t[0][i].is_equal(t[1][j], 0)) {
//				Element::add(el, t[0][i], t[1][j]);
//				for (size_t o = 0; o < t[2].load(); ++o) {
//					if (el.is_equal(t[2][o], 1)) {
//						Element::add(el2, el, t[2][o]);
//						for (size_t r = 0; r < t[3].load(); ++r) {
//							if (el2.is_equal(t[3][r], 2)) {
//								num++;
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//    t.join_stream(2);
//
//	EXPECT_NE(0, num);
//    EXPECT_EQ(t[4].load(), num);
//}
//
//TEST(TreeTest, JoinRandomListsLevel3) {
//	Matrix A;
//	A.random();
//
//	constexpr size_t base_size = 6;
//	static std::vector<uint64_t> __level_translation_array{{0, n/4, n/2, 3*n/4, n}};
//    Tree t{4, A, 10, __level_translation_array, __level_filter_array};
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
//    uint64_t num = 0;
//    Element el{};
//    Element el2{};
//    Element el3{};
//    for (size_t i = 0; i < t[0].load(); ++i) {
//		for (size_t j = 0; j < t[1].load(); ++j) {
//			if (t[0][i].is_equal(t[1][j], 0)) {
//				Element::add(el, t[0][i], t[1][j]);
//				for (size_t o = 0; o < t[2].load(); ++o) {
//					if (el.is_equal(t[2][o], 1)) {
//						Element::add(el2, el, t[2][o]);
//						for (size_t r = 0; r < t[3].load(); ++r) {
//							if (el2.is_equal(t[3][r], 2)) {
//								Element::add(el3, el2, t[3][r]);
//								for (size_t w = 0; w < t[4].load(); ++w) {
//									if (el3.is_equal(t[4][w], 3)) {
//										num++;
//									}
//								}
//							}
//						}
//					}
//				}
//			}
//		}
//	}
//
//    t.join_stream(3);
//
//	EXPECT_NE(0, num);
//	EXPECT_EQ(t[5].load(), num);
//}

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
