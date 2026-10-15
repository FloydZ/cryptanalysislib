#include <gtest/gtest.h>
#include <iostream>

#include "container/fq_vector.h"
#include "list/parallel_full.h"
#include "matrix/matrix.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr uint32_t k = 100;
constexpr uint32_t n = 100;
constexpr uint32_t q = 2;
constexpr size_t list_size = 1u << 10;

using MatrixT = uint64_t;
using Matrix = FqMatrix<MatrixT, n, k, q>;
using Value = FqPackedVector<k, q, MatrixT>;
using Label = FqPackedVector<n, q, MatrixT>;
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
		EXPECT_EQ(L[i].is_lower(L[i + 1]), true);
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
}

TEST(List1, sort_level_same_limb) {
	List L{list_size};
	L.random();
	for (size_t i = 0; i < list_size; ++i) {
		EXPECT_EQ(L[i].is_zero(), false);
	}

	constexpr uint32_t lower = 0;
	constexpr uint32_t upper = 31;
	L.sort_level(lower, upper);

	for (size_t i = 0; i < list_size - 1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], lower, upper), true);
	}
}

/// q = 4, the sorted range crosses a limb
TEST(List4, sort_level) {
	constexpr uint32_t q4 = 4, n4 = 40;
	using Matrix4 = FqMatrix<MatrixT, n4, n4, q4>;
	using Value4 = FqPackedVector<n4, q4, MatrixT>;
	using Label4 = FqPackedVector<n4, q4, MatrixT>;
	using Element4 = Element_T<Value4, Label4, Matrix4>;
	using List4 = Parallel_List_FullElement_T<Element4>;

	List4 L{list_size};
	L.random();

	// crosses the limb boundary at coordinate 32
	constexpr uint32_t lower = 20, upper = 44 - 4;
	L.sort_level(lower, upper);
	for (size_t i = 0; i < list_size - 1; ++i) {
		EXPECT_LE(L[i].label.hash(lower, upper), L[i + 1].label.hash(lower, upper));
		EXPECT_FALSE(L[i + 1].is_lower(L[i], lower, upper));
	}

	for (size_t i = 0; i < list_size; i += 17) {
		const auto [s, e] = L.search_boundaries(L[i], lower, upper, 1);
		ASSERT_NE(s, uint64_t(-1));
		EXPECT_LE(s, i);
		EXPECT_LT(i, e);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
