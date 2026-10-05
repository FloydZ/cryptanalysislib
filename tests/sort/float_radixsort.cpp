#include <gtest/gtest.h>
#include <cstdint>

#include "sort/float_radixsort.h"

using ::testing::InitGoogleTest;

TEST(pluggable_sort, std_sort_u8) {
    constexpr size_t size = 1u<<10u;
    float *values = new float[size];
    for (size_t i = 0; i < size; i++) {
        values[i] = (float)(rand() - 16384);
    }

    RadixSort rs;
    uint32_t *sorted = rs.Sort(values, size).GetIndices();
    for (uint32_t i = 1; i < size; i++) {
        EXPECT_LE(values[sorted[i-1]], values[sorted[i]]);
    }
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
