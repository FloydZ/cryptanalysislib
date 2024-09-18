#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "compression/deflate.h"
#include "compression/inflate.h"
#include "math/math.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;


TEST(deflate, dev) {
	for (int i = 1; i < 1000; ++i) {
		const auto k1 = sdefl_ilog2(sdefl_npow2(i) >> 2);;
		std::cout << k1 << " " << i << std::endl;
	}
}


TEST(deflate, random_simple) {
	constexpr size_t s = 1u << 14;
	uint8_t *data = (uint8_t *) malloc(s);
	uint8_t *comp = (uint8_t *) calloc(s * 2, 1);
	uint8_t *decomp = (uint8_t *) calloc(s, 1);

	for (size_t t = 0; t < 123; t++) {
		fastrandombytes(data, s);
		struct sdefl sdefl;

		for (uint32_t lvl = SDEFL_LVL_MIN; lvl < SDEFL_LVL_MAX; ++lvl) {
			const size_t len = sdeflate(&sdefl, comp, data, (int) s, lvl);
			const size_t n = sinflate(decomp, (int) s, comp, len);

			const int same = memcmp(data, decomp, (size_t) s);
			if (((size_t) n != s) || same) {
			}

			EXPECT_EQ(s, n);
			EXPECT_EQ(same, 0);
		}
	}

	free(data);
	free(comp);
	free(decomp);
}

TEST(deflate, string_simple) {
	constexpr size_t s = 1u << 14;
	uint8_t *data = (uint8_t *) malloc(s);
	uint8_t *comp = (uint8_t *) calloc(s * 2, 1);
	uint8_t *decomp = (uint8_t *) calloc(s, 1);
	const char alphabet[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz1234567890 ,.";
	for (size_t t = 0; t < 123; t++) {
		for (size_t i = 0; i < s; i++) {
			data[i] = alphabet[fastrandombytes_uint64(sizeof(alphabet))];
		}
		struct sdefl sdefl;

		for (uint32_t lvl = SDEFL_LVL_MIN; lvl < SDEFL_LVL_MAX; ++lvl) {
			const size_t len = sdeflate(&sdefl, comp, data, (int) s, lvl);
			const size_t n = sinflate(decomp, (int) s, comp, len);

			const int same = memcmp(data, decomp, (size_t) s);
			if (((size_t) n != s) || same) {
			}

			EXPECT_EQ(s, n);
			EXPECT_EQ(same, 0);
		}
	}

	free(data);
	free(comp);
	free(decomp);
}


int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
