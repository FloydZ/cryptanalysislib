#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>

#include "container/fq_vector.h"
#include "list/list.h"
#include "matrix/fq_matrix.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

#define THREADS 2

constexpr uint32_t k = 70;
constexpr uint32_t n = 100;
constexpr uint32_t q = 5;

constexpr size_t LS = 100;

using MatrixT = uint8_t;
using Matrix 		= FqMatrix<MatrixT, n, k, q>;
using Value 		= kAryContainer_T<MatrixT, k, q>;
using Label 		= kAryContainer_T<MatrixT, n, q>;
using Element       = Element_T<Value, Label, Matrix>;
using List 			= List_T<Element>;

/// if this test fails, something really bad is going on
TEST(List, simple) {
	List L{LS, 1};
}

TEST(List, copy) {
	List L{LS, 1}, L2{LS, 1};

	Matrix m;
	m.identity();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.generate_base_random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.is_correct(m), true);

	L2 = L;
	EXPECT_EQ(L2.load(), LS);
	EXPECT_EQ(L2.size(), LS);
	EXPECT_EQ(L2.is_correct(m), true);
}

TEST(List, base_random) {
	List L{LS, 1};
	Matrix m;
	m.identity();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.generate_base_random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	EXPECT_EQ(L.is_correct(m), true);
	for (size_t i = 0; i < LS; ++i) {
		for (uint32_t j = 0; j < std::min(k, n); ++j) {
			EXPECT_EQ(L[i].get_label(j) == L[i].get_value(j), true);
		}
	}
}


TEST(List, sort_level) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.generate_base_random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(0, 20);
	EXPECT_EQ(L.is_correct(m), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, 20), true);
	}
}

TEST(List, search_level) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.generate_base_random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(0, 20);
	EXPECT_EQ(L.is_correct(m), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, 20), true);
	}

	for (size_t i = 0; i < LS; ++i) {
		const size_t pos = L.search_level(L[i], 0, 20);
		EXPECT_EQ(pos, i);
	}
}

TEST(List, search) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.generate_base_random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(0, 20);
	EXPECT_EQ(L.is_correct(m), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, 20), true);
	}

	for (size_t i = 0; i < LS; ++i) {
		const size_t pos = L.search(L[i]);
		EXPECT_EQ(pos, i);
	}
}

TEST(List, search_boundaries) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.generate_base_random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(0, 20);
	EXPECT_EQ(L.is_correct(m), true);
	for (size_t i = 0; i < LS-1; ++i) {
		EXPECT_EQ(L[i].is_lower(L[i + 1], 0, 20), true);
	}

	for (size_t i = 0; i < LS; ++i) {
		const auto pos = L.search_boundaries(L[i], 0, 20);
		EXPECT_EQ(pos.first, i);
		EXPECT_EQ(pos.second, i + 1);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
