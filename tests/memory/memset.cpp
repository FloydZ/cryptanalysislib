#include <algorithm>
#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "memory/memcmp.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(equal, int_) {
    constexpr static size_t s = 100;
    using T = int;
    std::vector<T> in1; in1.resize(s);
    std::vector<T> in2; in2.resize(s);
    std::fill(in1.begin(), in1.end(), 1);
    std::fill(in2.begin(), in2.end(), 1);

    const auto d1 = cryptanalysislib::memcmp(in1.data(), in2.data(), s);
    EXPECT_EQ(d1, false);

    std::fill(in2.begin(), in2.end(), 0);
    const auto d2 = cryptanalysislib::memcmp(in1.data(), in2.data(), s);
    EXPECT_EQ(d2, true);
}

#ifdef USE_AVX2
/// every length and every position of a single differing byte
template<typename S>
static void memcmp_simd_check() {
    constexpr size_t max = 200;
    uint8_t a[max], b[max];
    for (size_t i = 0; i < max; ++i) { a[i] = b[i] = (uint8_t)(i * 7u + 3u); }

    for (size_t n = 0; n < max; ++n) {
        // unaligned start
        EXPECT_FALSE(cryptanalysislib::memcmp_simd_u8<S>(a + 1, b + 1, n - (n > 0)));
        EXPECT_FALSE(cryptanalysislib::memcmp_simd_u8<S>(a, b, n));
        for (size_t p = 0; p < n; ++p) {
            b[p] ^= 0x10;
            EXPECT_TRUE(cryptanalysislib::memcmp_simd_u8<S>(a, b, n));
            b[p] ^= 0x10;
        }
    }
}

TEST(memcmp, simd) {
    memcmp_simd_check<uint64x4_t>();
    memcmp_simd_check<uint8x32_t>();
    memcmp_simd_check<uint32x8_t>();
#ifdef USE_AVX512F
    memcmp_simd_check<uint64x8_t>();
#endif
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
