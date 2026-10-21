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
#define Value 	FqNonPackedVector<K, Q, MatrixT>
#define Label 	FqNonPackedVector<N, Q, MatrixT>
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
#define Value 	FqPackedVector<K, Q, MatrixT>
#define Label 	FqPackedVector<N, Q, MatrixT>
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
#define Value 	BinaryVector<K>
#define Label 	BinaryVector<N>
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
#define N (ceil_log2(Q))
#define ListName uint8_kPAC_kA
#define MatrixT uint8_t
#define Matrix 	FqVector<MatrixT, K, Q, true>
#define Value 	BinaryVector<K>
#define Label 	kAry_Type_T<Q>
#define Element Element_T<Value, Label, Matrix>
#define List 	List_T<Element>

#ifdef __cpp_explicit_this_parameter
TEST(ListName, list_view) {
	constexpr size_t qbits = ceil_log2(Q);
	constexpr size_t LS = 1u<<(qbits);
	List L{LS, 1};
	Matrix m;
	m.random();
	EXPECT_EQ(L.size(), LS);
	EXPECT_EQ(L.load(), 0);
	L.random(LS, m);
	EXPECT_EQ(L.load(), LS);
	EXPECT_EQ(L.size(), LS);


	constexpr size_t si = 10;
	constexpr size_t s = 10;
	const auto v = L.view(s, s+si);

	// as q is so small they are all  equal
	for (size_t i = 0; i < si; ++i) {
		const bool eq = v[i].is_equal(L[s+i]);
		EXPECT_EQ(eq, true);
	}

	std::cout << v;
}
#endif


TEST(ListName, kAry_search) {
	constexpr size_t qbits = ceil_log2(Q);
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
		EXPECT_LE(pos1, i);
		EXPECT_LE(pos2, i);
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

	EXPECT_TRUE(L.is_sorted(k_lower, k_higher));
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
	const uint32_t k_lower = 0, k_higher= N;
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
	const uint32_t k_lower = 0, k_higher= N;
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

/// `add_and_append(..., tid)` appends into the chunk `tid` of the list, a full
/// chunk does not overwrite the other chunks
TEST(ListName, add_and_append_tid) {
	constexpr uint32_t threads = 2;
	List L{LS, threads};
	Matrix m;
	m.random();
	Element e1, e2, e;
	e1.random(m); e2.random(m);
	Element::add(e, e1, e2, 0, N, -1u);

	// both overloads, both chunks
	for (uint32_t tid = 0; tid < threads; ++tid) {
		L.add_and_append(e1, e2, 0, N, -1u, false, tid);
		L.template add_and_append<0, N, -1u, false>(e1, e2, tid);
		EXPECT_EQ(L.load(tid), 2u) << tid;
		for (size_t i = L.start_pos(tid); i < L.start_pos(tid) + L.load(tid); ++i) {
			EXPECT_TRUE(L[i].label.is_equal(e.label)) << tid << " " << i;
			EXPECT_TRUE(L[i].value.is_equal(e.value)) << tid << " " << i;
		}
	}

	// fill chunk 1 (and more): chunk 0 is untouched
	Element z;
	z.zero();
	for (size_t i = 0; i < L.size(1) + 3; ++i) {
		L.add_and_append(z, z, 0, N, -1u, false, 1);
	}
	EXPECT_EQ(L.load(1), L.size(1));
	EXPECT_EQ(L.load(0), 2u);
	for (size_t i = L.start_pos(0); i < L.start_pos(0) + L.load(0); ++i) {
		EXPECT_TRUE(L[i].label.is_equal(e.label)) << i;
	}
	EXPECT_EQ(L.size(), LS);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
