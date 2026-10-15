#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "random.h"
#include "compression/horizontalpacking32.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

TEST(horizontalpacking32, size) {
	EXPECT_EQ(horizontalpacked_size(0, 7), 0u);
	EXPECT_EQ(horizontalpacked_size(1, 7), 1u);
	EXPECT_EQ(horizontalpacked_size(8, 7), 7u);
	EXPECT_EQ(horizontalpacked_size(128, 32), 512u);
}

/// the bit layout: value i in bits [i*b, (i+1)*b), lsb first
TEST(horizontalpacking32, layout) {
	const uint32_t in[3] = {0b101, 0b011, 0b111};
	uint8_t out[2];
	horizontalpack32(in, 3, 3, out);
	// bits: 101 | 011 << 3 | 111 << 6
	EXPECT_EQ(out[0], uint8_t(0b11011101));
	EXPECT_EQ(out[1], uint8_t(0b1));
}

/// pack and unpack (SIMD and scalar) for all bit widths and sizes around the
/// blocks of 128 values; the packed buffer has the exact size (ASan)
TEST(horizontalpacking32, roundtrip) {
	for (uint32_t b = 0; b <= 32; ++b) {
		const uint64_t mask = (b == 32) ? 0xffffffffull : ((1ull << b) - 1ull);
		for (const size_t n : {0ul, 1ul, 7ul, 127ul, 128ul, 129ul, 255ul, 256ul, 257ul, 1000ul, 4096ul}) {
			std::vector<uint32_t> v(n), o1(n + 1, 0xdeadbeef), o2(n + 1, 0xdeadbeef);
			for (auto &x : v) { x = uint32_t(rng()); }

			uint8_t *p = new uint8_t[horizontalpacked_size(n, b) + (n == 0 || b == 0)];
			horizontalpack32(v.data(), n, b, p);
			horizontalunpack32(p, n, b, o1.data());
			horizontalunpack32_scalar(p, n, b, o2.data());
			delete[] p;

			for (size_t i = 0; i < n; ++i) {
				ASSERT_EQ(o1[i], uint32_t(v[i] & mask)) << b << " " << n << " " << i;
				ASSERT_EQ(o2[i], uint32_t(v[i] & mask)) << b << " " << n << " " << i;
			}
			// nothing written behind the output
			EXPECT_EQ(o1[n], 0xdeadbeefu);
			EXPECT_EQ(o2[n], 0xdeadbeefu);
		}
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
