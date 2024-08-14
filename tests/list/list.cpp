#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>

#include "container/fq_vector.h"
#include "list/list.h"
#include "matrix/matrix.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

#define THREADS 2

constexpr size_t LS = 6;

#define K 20u
#define N 20u
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

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
