#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>

#include "container/fq_vector.h"
#include "list/list.h"
#include "matrix/matrix.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

#define THREADS 2

constexpr size_t LS = 1u<<8u;

#define K 20u
#define N 100u
#define Q 8u

#define ListName uint8_kAC_kAC
#define MatrixT uint8_t
#define Matrix 	FqMatrix<MatrixT, N, K, Q>
#define Value 	kAryContainer_T<MatrixT, K, Q>
#define Label 	kAryContainer_T<MatrixT, N, Q>
#define Element Element_T<Value, Label, Matrix>
#define List 	List_T<Element>
#include "test_list.h"
#undef ListName
#undef MatrixT
#undef Matrix
#undef Value
#undef Label
#undef Element
#undef List

#define ListName uint8_kPAC_kPAC
#define MatrixT uint8_t
#define Matrix 	FqMatrix<MatrixT, N, K, Q, true>
#define Value 	kAryPackedContainer_T<MatrixT, K, Q>
#define Label 	kAryPackedContainer_T<MatrixT, N, Q>
#define Element Element_T<Value, Label, Matrix>
#define List 	List_T<Element>
#include "test_list.h"
#undef ListName
#undef MatrixT
#undef Matrix
#undef Value
#undef Label
#undef Element
#undef List

#define ListName binary_64
#define MatrixT uint64_t
#define Matrix 	FqMatrix<MatrixT, N, K, 2, true>
#define Value 	BinaryContainer<K>
#define Label 	BinaryContainer<N>
#define Element Element_T<Value, Label, Matrix>
#define List 	List_T<Element>
#include "test_list.h"
#undef ListName
#undef MatrixT
#undef Matrix
#undef Value
#undef Label
#undef Element
#undef List



#undef K
#undef N
#undef Q

// NOTE DO NOT REMOVE: SPECIAL TEST FOR KARY
#define K 20u
#define Q 255u
#define N (bits_log2(Q))
#define ListName uint8_kPAC_kA
#define MatrixT uint8_t
#define Matrix 	FqVector<MatrixT, K, Q, true>
#define Value 	BinaryContainer<K>
#define Label 	kAry_Type_T<Q>
#define Element Element_T<Value, Label, Matrix>
#define List 	List_T<Element>

TEST(ListName, kAry_search) {
	constexpr size_t qbits = bits_log2(Q);
	constexpr size_t LS = 1u<<(qbits);
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(0, qbits);
	EXPECT_EQ(L.is_correct(m), true);


	// as q is so small they are all  equal
	for (size_t i = 0; i < LS; ++i) {
		const size_t pos1 = L.linear_search(L[i]);
		const size_t pos2 = L.binary_search(L[i]);
		// DOES not work if the image space is to small
		// const size_t pos3 = L.template interpolation_search<0, qbits>(L[i]);
		ASSERT_LE(pos1, i);
		ASSERT_LE(pos2, i);
		// ASSERT_LE(pos3, i);
	}
}

TEST(ListName, sort_level) {
	const uint32_t k_lower = 0, k_higher = N;
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.sort_level(k_lower, k_higher);

	ASSERT(L.is_sorted(k_lower, k_higher));
	EXPECT_EQ(L.is_correct(m), true);
}

TEST(ListName, sort_level_constexpr) {
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	L.template sort_level<0, N>();
	EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_correct(m), true);
}


TEST(ListName, sort_level_target) {
	const uint32_t k_lower = 0, k_higher=N;
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	Element rnd;
	rnd.random();

	L.sort_level(0, N, rnd.label);
	EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_sorted(rnd.label, false, k_lower, k_higher), true);
}

TEST(Keke, sort_level_target_constexpr) {
	const uint32_t k_lower = 0, k_higher=N;
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);

	Element rnd;
	rnd.random();

	L.template sort_level<0, N>(rnd.label);
	EXPECT_EQ(L.is_correct(m), true);
	EXPECT_EQ(L.is_sorted(rnd.label, false, k_lower, k_higher), true);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
