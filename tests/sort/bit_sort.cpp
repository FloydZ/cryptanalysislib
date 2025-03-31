#include <gtest/gtest.h>
#include <cstdint>

#include "sort/bit_sort.h"

using ::testing::InitGoogleTest;

TEST(bit_sort, uint32_t) {
	constexpr size_t s = 1u << 10u;
	using T = uint32_t;
	T *data = (T *)malloc(sizeof(T) * s);
	for (uint32_t i = 0; i < s; i++) { data[i] = i; }

	free(data);
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
