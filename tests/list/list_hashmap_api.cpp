#include <gtest/gtest.h>
#include <iostream>
#include <omp.h>

#include "list/list.h"
#include "container/hashmap.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr size_t LS = 1u<<8u;

#define K 20u
#define N 100u
#define Q 8u

using MatrixT = uint8_t;
using Matrix= FqMatrix<MatrixT, N, K, Q>;
using Value = FqNonPackedVector<K, Q, MatrixT>;
using Label = FqNonPackedVector<N, Q, MatrixT>;
using Element= Element_T<Value, Label, Matrix>;
using List = List_T<Element>;

TEST(ListName, list_view) {
	List L{LS, 1};
	Matrix m;
	m.random();
	L.random(LS, m);

}
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
