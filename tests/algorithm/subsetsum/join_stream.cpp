#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/random_index.h"
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
using Value     	= BinaryContainer<n>;
using Label    		= kAry_Type_T<q>;
using Matrix 		= FqVector<T, n, q, true>;
using Element		= Element_T<Value, Label, Matrix>;
using List			= List_T<Element>;
using Tree			= Tree_T<List>;

// unused ignore
static std::vector<std::vector<uint8_t>> __level_filter_array{{ {{4,0,0}}, {{1,0,0}}, {{1,0,0}}, {{0,0,0}} }};

TEST(SubSetSum, JoinRandomListsLevel0) {
	Matrix A;
	A.random();

	static std::vector<uint64_t> tbl{{0, n}};
	Tree t{2, A, 10u, tbl, __level_filter_array};

	t[0].random(1u << 8u, A);
	t[1].random(1u << 8u, A);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);

	uint64_t num = 0;
	for (size_t i = 0; i < t[0].load(); ++i) {
		for (size_t j = 0; j < t[1].load(); ++j) {
			if (t[0][i].is_equal(t[1][j], tbl[0], tbl[1])) {
				num++;
			}
		}
	}

	t.join_stream(0);
	EXPECT_NE(0, num);
	EXPECT_EQ(t[2].load(), num);
}

// NOTE: takes very long
TEST(SubSetSum, JoinRandomListsLevel1) {
	Matrix A;
	A.random();

	static std::vector<uint64_t> tbl{{0, n/2, n}};
	Tree t{2, A, 11, tbl, __level_filter_array};

	t[0].random(1u << 12u, A);
	t[1].random(1u << 12u, A);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t.join_stream(0);

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << 12u, A);
	t[1].random(1u << 12u, A);

	t[2].sort_level(1, tbl);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);

	uint64_t num = 0;
	Element el{};
	for (size_t i = 0; i < t[0].load(); ++i) {
		for (size_t j = 0; j < t[1].load(); ++j) {
			if (t[0][i].is_equal(t[1][j], tbl[0], tbl[1])) {
				Element::add(el, t[0][i], t[1][j]);

				for (size_t o = 0; o < t[2].load(); ++o) {
					if (el.is_equal(t[2][o], tbl[1], tbl[2])) {
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

TEST(TreeTest, JoinRandomListsLevel2) {
	Matrix A;
	A.random();

	constexpr size_t base_size = 5;
	static std::vector<uint64_t> tbl{{0, n/3, 2*n/3, n}};
	Tree t{3, A, base_size, tbl, __level_filter_array};

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);

	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t.join_stream(0);

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);

	t[2].sort_level(1, tbl);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t.join_stream(1);

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t[3].sort_level(2, tbl);

	uint64_t num = 0;
	Element el{};
	Element el2{};
	for (size_t i = 0; i < t[0].load(); ++i) {
		for (size_t j = 0; j < t[1].load(); ++j) {
			if (t[0][i].is_equal(t[1][j], tbl[0], tbl[1])) {
				Element::add(el, t[0][i], t[1][j]);
				for (size_t o = 0; o < t[2].load(); ++o) {
					if (el.is_equal(t[2][o], tbl[1], tbl[2])) {
						Element::add(el2, el, t[2][o]);
						for (size_t r = 0; r < t[3].load(); ++r) {
							if (el2.is_equal(t[3][r], tbl[2], tbl[3])) {
								num++;
							}
						}
					}
				}
			}
		}
	}

	t.join_stream(2);

	EXPECT_NE(0, num);
	EXPECT_EQ(t[4].load(), num);
}

TEST(SubSetSum, JoinRandomListsLevel3) {
	Matrix A;
	A.random();

	constexpr size_t base_size = 5;
	static std::vector<uint64_t> tbl{{0, n/4, n/2, 3*n/4, n}};
	Tree t{4, A, 10, tbl, __level_filter_array};

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t.join_stream(0);

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);

	t[2].sort_level(1, tbl);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t.join_stream(1);

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t[3].sort_level(2, tbl);
	t.join_stream(2);

	t[0].set_load(0);
	t[1].set_load(0);
	t[0].random(1u << base_size, A);
	t[1].random(1u << base_size, A);
	t[0].sort_level(0, tbl);
	t[1].sort_level(0, tbl);
	t[4].sort_level(3, tbl);

	uint64_t num = 0;
	Element el{};
	Element el2{};
	Element el3{};
	for (size_t i = 0; i < t[0].load(); ++i) {
		for (size_t j = 0; j < t[1].load(); ++j) {
			if (t[0][i].is_equal(t[1][j], tbl[0], tbl[1])) {
				Element::add(el, t[0][i], t[1][j]);
				for (size_t o = 0; o < t[2].load(); ++o) {
					if (el.is_equal(t[2][o], tbl[1], tbl[2])) {
						Element::add(el2, el, t[2][o]);
						for (size_t r = 0; r < t[3].load(); ++r) {
							if (el2.is_equal(t[3][r], tbl[2], tbl[3])) {
								Element::add(el3, el2, t[3][r]);
								for (size_t w = 0; w < t[4].load(); ++w) {
									if (el3.is_equal(t[4][w], tbl[3], tbl[4])) {
										num++;
									}
								}
							}
						}
					}
				}
			}
		}
	}

	t.join_stream(3);

	EXPECT_NE(0, num);
	EXPECT_EQ(t[5].load(), num);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	ident();
	random_seed(time(NULL));
	return RUN_ALL_TESTS();
}
