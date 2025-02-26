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
    constexpr static size_t s = 1u << 10;
    TypeParam d1[s], d2[s];
    TypeParam2 out[s];
    for (uint32_t i = 0; i < s; i++) {
        d1[i] = i; d2[i] = i;
    }

    zip_u8(out, d1, d2, s);

    for (size_t i = 0; i < s; i++) {
        const TypeParam2 t = d1[i] | (((TypeParam2)d2[i]) << (sizeof(TypeParam)*8));
        EXPECT_EQ(out[i], t);
    
    }
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
