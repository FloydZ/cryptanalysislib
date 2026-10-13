#include <cstdint>
#include <gtest/gtest.h>

#include "algorithm/bits/bin2naf.h"
#include "algorithm/bits/bsr.h"
#include "algorithm/bits/clz.h"
#include "algorithm/bits/ffs.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib::algorithm;

/// non-adjacent form: x = np - nm, no two adjacent nonzero digits, np & nm == 0
TEST(bits, bin2naf) {
	for (uint64_t x = 0; x < (1u << 16); ++x) {
		uint64_t np, nm;
		bit::bin2naf(np, nm, x);
		ASSERT_EQ(bit::naf2bin(np, nm), x) << x;
		ASSERT_EQ(np & nm, 0u) << x;
		const uint64_t nz = np | nm;
		ASSERT_EQ(nz & (nz >> 1), 0u) << x;

		bit::bin2sbin(np, nm, x);
		ASSERT_EQ(bit::naf2bin(np, nm), x) << x;
		ASSERT_EQ(np & nm, 0u) << x;
	}
	// the NAF of 7 = 8 - 1
	uint32_t np, nm;
	bit::bin2naf<uint32_t>(np, nm, 7u);
	EXPECT_EQ(np, 8u);
	EXPECT_EQ(nm, 1u);
}

TEST(bits, clz_bsr_ffs) {
	EXPECT_EQ(clz<uint32_t>(0), 32u);
	EXPECT_EQ(clz<uint64_t>(0), 64u);
	EXPECT_EQ(clz<uint8_t>(0), 8u);
	EXPECT_EQ(clz<uint8_t>(1), 7u);
	EXPECT_EQ(clz<uint16_t>(0x100), 7u);
	EXPECT_EQ(clz<int32_t>(-1), 0u);
	EXPECT_EQ(cryptanalysislib::ffs<uint32_t>(0), 0u);
	EXPECT_EQ(cryptanalysislib::ffs<int8_t>(-128), 8u);
	for (uint32_t i = 0; i < 64; ++i) {
		const uint64_t x = 1ull << i;
		EXPECT_EQ(clz<uint64_t>(x), 63u - i);
		EXPECT_EQ(bsr<uint64_t>(x), i);
		EXPECT_EQ(bsr<uint64_t>(x | 1u), i);
		EXPECT_EQ(cryptanalysislib::ffs<uint64_t>(x), i + 1u);
		EXPECT_EQ(cryptanalysislib::ffs<uint64_t>(x | (1ull << 63)), i + 1u);
		if (i < 32) {
			EXPECT_EQ(clz<uint32_t>(uint32_t(x)), 31u - i);
			EXPECT_EQ(bsr<uint32_t>(uint32_t(x)), i);
			EXPECT_EQ(cryptanalysislib::ffs<uint32_t>(uint32_t(x)), i + 1u);
		}
	}
	const unsigned __int128 h = (unsigned __int128)1 << 100;
	EXPECT_EQ(cryptanalysislib::ffs<unsigned __int128>(h), 101u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
