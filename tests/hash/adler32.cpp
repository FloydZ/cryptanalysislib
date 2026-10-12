#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>

#include "hash/hash.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(Hash, simple) {

}

#ifdef USE_AVX2
TEST(Adler32, avx) {
	constexpr static size_t size = 1024;
	auto *data = (uint8_t *)malloc(size);
	rng(data, size);

	for (uint32_t i = 64; i < size; ++i) {
		const uint32_t t1 = adler32(0, data, i);
		const uint32_t t2 = avx2_adler32(0, data, i);
		EXPECT_EQ(t1, t2);
	}

	free(data);
}
#endif

/// all implementations and the dispatcher agree for all lengths 0..1024 and
/// two start values (incl. the scalar reference from zlib's definition)
TEST(Adler32, all_versions) {
	constexpr static size_t size = 1024;
	auto *data = (uint8_t *)malloc(size);
	rng(data, size);

	for (const uint32_t val : {1u, 0x12345678u}) {
		for (size_t i = 0; i <= size; ++i) {
			// reference: the definition, modulo 65521
			uint32_t a = val & 0xffffu, b = val >> 16u;
			for (size_t j = 0; j < i; ++j) { a = (a + data[j]) % 65521u; b = (b + a) % 65521u; }
			const uint32_t e = (b << 16u) | a;
			EXPECT_EQ(adler32(val, data, i), e) << i;
			EXPECT_EQ(adler32_dispatch(val, data, i), e) << i;
#ifdef USE_AVX2
			EXPECT_EQ(avx2_adler32(val, data, i), e) << i;
#endif
#ifdef USE_AVX512F
			EXPECT_EQ(avx512_adler32(val, data, i), e) << i;
#endif
		}
	}
	free(data);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
    return RUN_ALL_TESTS();
}

