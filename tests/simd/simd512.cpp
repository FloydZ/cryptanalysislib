#include <cstdint>
#include <cstdio>
#include <gtest/gtest.h>
#include <iostream>

#include "helper.h"
#include "random.h"
#include "simd/simd.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

#ifdef USE_AVX512F
#ifdef USE_AVX512BW
/// the free operators also exist for the signed types
TEST(int32x16_t, operators) {
	const int32x16_t a = int32x16_t::set1(-3), b = int32x16_t::set1(5);
	const int32x16_t c = (a + b) * b - a;
	for (uint32_t i = 0; i < 16; ++i) { EXPECT_EQ(c[i], 13); }
	int32x16_t e = a;
	e ^= b;
	EXPECT_EQ(e[3], -3 ^ 5);

	EXPECT_EQ((int8x64_t::set1(1) << 2)[7], 4);
	EXPECT_EQ((int16x32_t::set1(6) & int16x32_t::set1(3))[1], 2);
	EXPECT_EQ((int64x8_t::set1(6) | int64x8_t::set1(3))[2], 7);
	EXPECT_EQ((~int64x8_t::set1(0))[0], -1);
}

TEST(uint16x32_t, all_equal) {
	alignas(64) uint16_t a[32];
	for (uint32_t r = 0; r < 1000; ++r) {
		const uint16_t v = uint16_t(rng());
		for (uint32_t i = 0; i < 32; ++i) { a[i] = v; }
		EXPECT_TRUE(uint16x32_t::all_equal(uint16x32_t::load(a)));
		const uint32_t p = rng() % 32u;
		a[p] ^= uint16_t(1u << (rng() % 16u));
		EXPECT_FALSE(uint16x32_t::all_equal(uint16x32_t::load(a)));
	}

	constexpr bool c = uint16x32_t::all_equal(uint16x32_t::set1(7));
	static_assert(c);
}
#endif

TEST(AVX512, hadd_histogram) {
	alignas(64) uint64_t a[8], o[8];
	alignas(64) uint32_t b[16], p[16];
	for (uint32_t r = 0; r < 1000; ++r) {
		for (uint32_t i = 0; i < 8; ++i) { a[i] = rng(); }
		for (uint32_t i = 0; i < 16; ++i) { b[i] = uint32_t(rng()); }
		// small nibble alphabet, so that the histogram is not mostly 1
		if (r & 1u) {
			for (uint32_t i = 0; i < 8; ++i) { a[i] &= 0x3333333333333333ull; }
			for (uint32_t i = 0; i < 16; ++i) { b[i] &= 0x33333333u; }
		}
		const uint8_t nib = uint8_t(rng() % 4u);
		const uint64x8_t va = uint64x8_t::load(a);
		const uint32x16_t vb = uint32x16_t::load(b);

		auto ref64 = [&](const uint64_t x, const uint32_t bits) {
			uint64_t s = 0;
			for (uint32_t k = 0; k < 64; k += bits) { s += (x >> k) & ((1ull << bits) - 1ull); }
			return s;
		};
		auto nibs = [&](const uint64_t x, const uint32_t n) {
			uint64_t c = 0;
			for (uint32_t k = 0; k < n; ++k) { c += ((x >> (4u*k)) & 0xFu) == nib; }
			return c;
		};

		uint64x8_t::store(o, uint64x8_t::hadd_epu8(va));  for (uint32_t i = 0; i < 8; ++i) { EXPECT_EQ(o[i], ref64(a[i], 8)); }
		uint64x8_t::store(o, uint64x8_t::hadd_epu16(va)); for (uint32_t i = 0; i < 8; ++i) { EXPECT_EQ(o[i], ref64(a[i], 16)); }
		uint64x8_t::store(o, uint64x8_t::hadd_epu32(va)); for (uint32_t i = 0; i < 8; ++i) { EXPECT_EQ(o[i], ref64(a[i], 32)); }
		uint64x8_t::store(o, uint64x8_t::histogram_epi4(va, nib)); for (uint32_t i = 0; i < 8; ++i) { EXPECT_EQ(o[i], nibs(a[i], 16)); }

		uint32x16_t::store(p, uint32x16_t::hadd_epu8(vb));  for (uint32_t i = 0; i < 16; ++i) { EXPECT_EQ(p[i], ref64(b[i], 8)); }
		uint32x16_t::store(p, uint32x16_t::hadd_epu16(vb)); for (uint32_t i = 0; i < 16; ++i) { EXPECT_EQ(p[i], ref64(b[i], 16)); }
		uint32x16_t::store(p, uint32x16_t::histogram_epi4(vb, nib)); for (uint32_t i = 0; i < 16; ++i) { EXPECT_EQ(p[i], nibs(b[i], 8)); }
	}
}

TEST(uint8x64_t, TTrandom) {
	uint8x64_t t1 = uint8x64_t::random();

	uint32_t atleast_one_not_zero = false;
	for (uint32_t i = 0; i < 64; ++i) {
		if (t1.v8[i] > 0) {
			atleast_one_not_zero = true;
			//	break;
		}
	}

	EXPECT_EQ(atleast_one_not_zero, true);
}

TEST(uint8x64_t, set1) {
	uint8x64_t t1 = uint8x64_t::set1(0);
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t1.v8[i], 0);
	}

	uint8x64_t t2 = uint8x64_t::set1(1);
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t2.v8[i], 1);
	}
}

TEST(uint8x64_t, set) {
	uint32_t pos = 53;
	uint8x64_t t1 = uint8x64_t::set(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	for (uint32_t i = 0; i < 64; ++i) {
		if (i == pos) {
			EXPECT_EQ(t1.v8[i], 1);
			continue;
		}
		EXPECT_EQ(t1.v8[i], 0);
	}

	uint8x64_t t2 = uint8x64_t::setr(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
	for (uint32_t i = 0; i < 64; ++i) {
		if (i == (63 - pos)) {
			EXPECT_EQ(t2.v8[i], 1);
			continue;
		}
		EXPECT_EQ(t2.v8[i], 0);
	}
}

TEST(uint8x64_t, unalinged_load) {
	uint8_t data[64] = {0};

	uint8x64_t t1 = uint8x64_t::unaligned_load(data);
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t1.v8[i], 0u);
	}
}

TEST(uint8x64_t, alinged_load) {
	alignas(256) uint8_t data[64] = {0};
	uint8x64_t t1 = uint8x64_t::aligned_load(data);
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t1.v8[i], 0u);
	}
}

TEST(uint8x64_t, unalinged_store) {
	uint8x64_t t1 = uint8x64_t::random();
	uint8_t data[64] = {0};

	uint8x64_t::unaligned_store(data, t1);
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t1.v8[i], data[i]);
	}
}

TEST(uint8x64_t, alinged_store) {
	uint8x64_t t1 = uint8x64_t::random();
	alignas(256) uint8_t data[64] = {0};

	uint8x64_t::aligned_store(data, t1);
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t1.v8[i], data[i]);
	}
}
TEST(uint8x64_t, logic) {
	const uint8x64_t t1 = uint8x64_t::set1(0);
	const uint8x64_t t2 = uint8x64_t::set1(1);
	uint8x64_t t3 = uint8x64_t::set1(2);

	t3 = t1 + t2;
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t3.v8[i], 1);
	}

	t3 = t2 - t1;
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t3.v8[i], 1);
	}

	t3 = t2 - t2;
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t3.v8[i], 0);
	}

	t3 = t1 ^ t2;
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t3.v8[i], 1);
	}

	t3 = t1 | t2;
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t3.v8[i], 1);
	}

	t3 = t1 & t2;
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t3.v8[i], 0);
	}

	t3 = ~t1;
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t3.v8[i], uint8_t(-1u));
	}

	t3 = uint8x64_t::mullo(t1, t2);
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t3.v8[i], 0);
	}

	t3 = uint8x64_t::slli(t1, 1);
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t3.v8[i] , 0);
	}

	t3 = uint8x64_t::slli(t2, 1);
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t3.v8[i] , 2);
	}
}

TEST(uint8x64_t, slri) {
	for (uint8_t j = 0; j < 8; j++) {
		const uint8x64_t t1 = uint8x64_t::set1(1u << j);
		const uint8x64_t t2 = uint8x64_t::srli(t1, j);
		for (uint32_t i = 0; i < 64; ++i) {
			EXPECT_EQ(t2.v8[i], 1);
		}
	}

	/// special case for j = 8
	const uint8x64_t t1 = uint8x64_t::set1((1u << 7u) - 1u);
	const uint8x64_t t2 = uint8x64_t::srli(t1, 8);
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t2.v8[i], 0);
	}
}


TEST(uint8x64_t, popcnt) {
	auto t1 = uint8x64_t::set1(4);
	auto t2 = uint8x64_t::popcnt(t1);
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t2.v8[i], 1);
	}


	// NOTE: leading zeros of 0b00000100. Was 2, the trailing zeros, which
	// 	the former implementation computed (now `tzcnt`).
	auto t3 = uint8x64_t::lzcnt(t1);
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t3.v8[i], 5);
	}

	auto t4 = uint8x64_t::tzcnt(t1);
	for (uint32_t i = 0; i < 64; ++i) {
		EXPECT_EQ(t4.v8[i], 2);
	}
}

#endif
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
