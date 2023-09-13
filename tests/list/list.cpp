#include <gtest/gtest.h>
#include <iostream>

#include "container/fq_vector.h"
#include "list/list.h"
#include "matrix/fq_matrix.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr uint32_t k = 100;
constexpr uint32_t n = 100;
constexpr uint32_t q = 5;

using MatrixT = uint8_t;
using Matrix 		= FqMatrix<MatrixT, n, k, q>;
using Value 		= kAryContainer_T<MatrixT, k, q>;
using Label 		= kAryContainer_T<MatrixT, n, q>;
using Element       = Element_T<Value, Label, Matrix>;
using List 			= List_T<Element>;


TEST(List1, simple) {
	List L{100,1};
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
