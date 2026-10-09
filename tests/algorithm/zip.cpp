#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "algorithm/zip.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(zip, uint8_t_) {
    using TypeParam = uint8_t;
    using TypeParam2 = uint16_t;
    // all lengths around the SIMD widths (16 on NEON, 32 on AVX2)
    for (size_t s = 0; s <= 100; s++) {
        TypeParam d1[100], d2[100];
        TypeParam2 out[100];
        for (uint32_t i = 0; i < s; i++) {
            // NOTE: different values, so swapped inputs are detected
            d1[i] = i; d2[i] = 255 - 3*i;
        }

        zip_u8(out, d1, d2, s);

        for (size_t i = 0; i < s; i++) {
            const TypeParam2 t = d1[i] | (((TypeParam2)d2[i]) << (sizeof(TypeParam)*8));
            EXPECT_EQ(out[i], t);
        }
    }
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
