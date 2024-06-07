#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "memory/memory.h"


using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

using T = uint8_t;
constexpr size_t size = 1000;


TEST(Memory, copy) {
	T *a1 = (T *) calloc(size, sizeof(T));
	T *a2 = (T *) malloc(size * sizeof(T));
	for (size_t i = 0; i < size; ++i) {
		a2[i] = i;
	}

	cryptanalysislib::memcpy(a2, a1, size);
	for (size_t i = 0; i < size; ++i) {
		EXPECT_EQ(a2[i], 0);
	}

	free(a1);
	free(a2);
}

// TODO stacksmashing on a avx2 machine
// TEST(Memory, set) {
// 	T *a1 = (T *) calloc(size, sizeof(T));
// 	T a = 1;
//
// 	cryptanalysislib::memset(a1, a, size);
// 	for (size_t i = 0; i < size; ++i) {
// 		EXPECT_EQ(a1[i], a);
// 	}
//
// 	free(a1);
// }

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
