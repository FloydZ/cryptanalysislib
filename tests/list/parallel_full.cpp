#include <gtest/gtest.h>
#include <iostream>

#include "container/fq_vector.h"
#include "list/parallel_full.h"
#include "matrix/matrix.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr uint32_t k = 100;
constexpr uint32_t n = 100;
constexpr uint32_t q = 5;
constexpr size_t list_size = 1u << 10;

using MatrixT = uint64_t;
using Matrix = FqMatrix<MatrixT, n, k, q>;
using Value = FqPackedVector<MatrixT, k, q>;
using Label = FqPackedVector<MatrixT, n, q>;
using Element = Element_T<Value, Label, Matrix>;
using List = Parallel_List_FullElement_T<Element>;

TEST(List1, random) {
	List L{list_size};
	L.random();
	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L[i].is_zero(), false);
	}
}

TEST(List1, sort) {
	List L{list_size};
	L.random();
	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L[i].is_zero(), false);
	}

	L.sort(0, list_size);
	for (size_t i = 0; i < list_size - 1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1]), false);
	}

	L.zero();

	for (size_t i = 0; i < list_size - 1; ++i) {
		EXPECT_EQ(L[i].is_zero(), true);
	}
}

TEST(List1, hash_sort) {
	List L{list_size};
	L.random();
	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L[i].is_zero(), false);
	}

	auto ls = [](const Element &e) -> uint64_t {
		return e.label_ptr(0);
	};

	L.sort(ls, 0, list_size);

	for (size_t i = 0; i < list_size - 1; ++i) {
		EXPECT_EQ(L[i].label_ptr(0) <= L[i + 1].label_ptr(0), true);
	}

	L.zero();

	for (size_t i = 0; i < list_size - 1; ++i) {
		EXPECT_EQ(L[i].is_zero(), true);
	}
}

TEST(List1, sort_level_same_limb) {
	List L{list_size};
	L.random();
	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L[i].is_zero(), false);
	}

	constexpr uint32_t lower = 2;
	constexpr uint32_t upper = 31;
	L.sort_level(lower, upper);

	for (size_t i = 0; i < list_size - 1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], lower, upper), false);
	}

	L.zero();

	for (size_t i = 0; i < list_size - 1; ++i) {
		EXPECT_EQ(L[i].is_zero(), true);
	}
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
