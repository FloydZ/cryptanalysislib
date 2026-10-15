#include <gtest/gtest.h>
#include <iostream>

#include "container/fq_vector.h"
#include "matrix/matrix.h"
#include "list/parallel_index.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr uint32_t nri = 100;

constexpr uint32_t k = 100;
constexpr uint32_t n = 100;
constexpr uint32_t q = 5;

using MatrixT = uint8_t;
using Matrix 		= FqMatrix<MatrixT, n, k, q>;
using Value 		= FqNonPackedVector<k, q, MatrixT>;
using Label 		= FqNonPackedVector<n, q, MatrixT>;
using Element       = Element_T<Value, Label, Matrix>;
using List 			= Parallel_List_IndexElement_T<Element, nri>;


TEST(List1, simple) {
	List L{100,1};
}

TEST(List1, sort) {
	constexpr size_t size = 1u << 10;
	for (const uint32_t threads : {1u, 4u}) {
		List L{size, threads};
		for (size_t i = 0; i < size; ++i) {
			L.__data[i].first.random();
			L.__data[i].second.fill(0);
			L.__data[i].second[0] = i;
		}

		const auto copy = L.__data;
		L.sort();

		for (uint32_t tid = 0; tid < threads; ++tid) {
			const size_t s = L.start_pos(tid), e = L.end_pos(tid);
			for (size_t i = s; i + 1 < e; ++i) {
				EXPECT_FALSE(L.__data[i + 1].first.is_lower(L.__data[i].first));
			}

			// the indices moved together with their labels and stayed
			// within their segment
			for (size_t i = s; i < e; ++i) {
				const uint32_t idx = L.__data[i].second[0];
				EXPECT_GE(idx, s);
				EXPECT_LT(idx, e);
				EXPECT_TRUE(L.__data[i].first.is_equal(copy[idx].first));
			}
		}
	}
}

TEST(List1, zero_element) {
	List L{16, 1};
	L.__data[3].first.random();
	L.__data[3].second.fill(7);
	L.zero_element(3);
	EXPECT_TRUE(L.__data[3].first.is_zero());
	for (const auto x : L.__data[3].second) { EXPECT_EQ(x, 0); }
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
