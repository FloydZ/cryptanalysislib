#include <gtest/gtest.h>
#include <iostream>

#include "container/fq_vector.h"
#include "matrix/matrix.h"
#include "list/parallel.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

#define THREADS 2
constexpr size_t LS = 100;
constexpr uint32_t k = 70;
constexpr uint32_t n = 100;
constexpr uint32_t q = 5;

using MatrixT = uint8_t;
using Matrix 		= FqMatrix<MatrixT, n, k, q>;
using Value 		= FqNonPackedVector<k, q, MatrixT>;
using Label 		= FqNonPackedVector<n, q, MatrixT>;
using Element       = Element_T<Value, Label, Matrix>;
using List 			= Parallel_List_T<Element>;

// if this test fails, something bad is going on
TEST(List1, simple) {
	List L{LS, 1};
}

TEST(List1, copy) {
	List L{LS, 1}, L2{LS, 1};

	Matrix m;
	m.identity();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	EXPECT_EQ(L.is_correct(m), true);
}

TEST(List1, sort) {
	List L{LS, 1};

	Matrix m;
	m.identity();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	EXPECT_EQ(L.is_correct(m), true);

	L.sort();
	// NOT valid anymore EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_sorted(), true);

}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
