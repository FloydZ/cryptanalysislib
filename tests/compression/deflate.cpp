#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <vector>

#include "../../src/compression/deflate.h"
#include "../../src/compression/inflate.h"
#include "../../src/math/math.h"

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
		rng(data, s);
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
			data[i] = alphabet[rng(sizeof(alphabet))];
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


/// all levels (incl. `SDEFL_LVL_MAX`), sizes around the raw block size (65535)
/// and the block size (256 KiB), random/text/run data, both formats
TEST(deflate, levels_sizes_formats) {
	auto s = std::make_unique<struct sdefl>();
	for (const size_t n : {1ul, 2ul, 3ul, 4ul, 5ul, 100ul, 65535ul, 65536ul, 262144ul, 262145ul, 300000ul}) {
		for (uint32_t kind = 0; kind < 3; ++kind) {
			std::vector<uint8_t> data(n), comp(sdefl_bound(int(n))), decomp(n);
			for (size_t i = 0; i < n; ++i) {
				data[i] = kind == 0 ? uint8_t(rng())
				        : kind == 1 ? uint8_t("abcdefgh "[rng() % 9])
				                    : uint8_t(i / 1000);
			}

			for (int lvl = SDEFL_LVL_MIN; lvl <= SDEFL_LVL_MAX; ++lvl) {
				const int len = sdeflate(s.get(), comp.data(), data.data(), int(n), lvl);
				ASSERT_LE(len, sdefl_bound(int(n)));
				ASSERT_EQ(sinflate(decomp.data(), int(n), comp.data(), len), int(n)) << n << " " << lvl;
				ASSERT_EQ(decomp, data) << n << " " << kind << " " << lvl;

				// zlib format: header 0x78 0x01, adler32 at the end
				const int zlen = zsdeflate(s.get(), comp.data(), data.data(), int(n), lvl);
				ASSERT_LE(zlen, sdefl_bound(int(n)));
				EXPECT_EQ(comp[0], 0x78u);
				EXPECT_EQ(comp[1], 0x01u);
				ASSERT_EQ(zsinflate(decomp.data(), int(n), comp.data(), zlen), int(n)) << n << " " << lvl;
				ASSERT_EQ(decomp, data) << n << " " << kind << " " << lvl;
			}
		}
	}
}

/// truncated, corrupted and random inputs and too small outputs: the result
/// may be garbage, but no access outside of the buffers (exactly sized, for ASan)
TEST(inflate, corrupted_input) {
	auto s = std::make_unique<struct sdefl>();
	const size_t n = 5000;
	std::vector<uint8_t> data(n), full(sdefl_bound(int(n)));
	for (size_t i = 0; i < n; ++i) { data[i] = uint8_t("abcdefgh "[rng() % 9]); }

	for (uint32_t it = 0; it < 3000; ++it) {
		const int lvl = int(rng() % (SDEFL_LVL_MAX + 1));
		const bool zlib = rng() & 1u;
		const int len = zlib ? zsdeflate(s.get(), full.data(), data.data(), int(n), lvl)
		                     : sdeflate(s.get(), full.data(), data.data(), int(n), lvl);

		// truncate and flip some bits
		const size_t clen = 1 + rng() % size_t(len);
		uint8_t *c = new uint8_t[clen];
		memcpy(c, full.data(), clen);
		const uint32_t flips = rng() % 4;
		for (uint32_t f = 0; f < flips; ++f) { c[rng() % clen] ^= uint8_t(1u << (rng() % 8)); }

		const size_t cap = 1 + rng() % (n + 10);
		uint8_t *o = new uint8_t[cap];
		const int r = zlib ? zsinflate(o, int(cap), c, int(clen)) : sinflate(o, int(cap), c, int(clen));
		EXPECT_LE(r, int(cap));
		delete[] o;
		delete[] c;
	}

	// random garbage
	for (uint32_t it = 0; it < 3000; ++it) {
		const size_t clen = 1 + rng() % 200, cap = 1 + rng() % 1000;
		uint8_t *c = new uint8_t[clen], *o = new uint8_t[cap];
		for (size_t i = 0; i < clen; ++i) { c[i] = uint8_t(rng()); }
		EXPECT_LE(sinflate(o, int(cap), c, int(clen)), int(cap));
		delete[] o;
		delete[] c;
	}
}

/// compressible data must compress, higher levels must not be much worse
TEST(deflate, ratio) {
	auto s = std::make_unique<struct sdefl>();
	const size_t n = 1u << 16;
	std::vector<uint8_t> data(n), comp(sdefl_bound(int(n)));
	for (size_t i = 0; i < n; ++i) { data[i] = uint8_t("abcdefgh "[rng() % 9]); }
	const int l0 = sdeflate(s.get(), comp.data(), data.data(), int(n), SDEFL_LVL_MIN);
	const int l8 = sdeflate(s.get(), comp.data(), data.data(), int(n), SDEFL_LVL_MAX);
	// 9 symbols: about log2(9) = 3.17 bits per byte
	EXPECT_LT(l0, int(n / 2));
	EXPECT_LE(l8, l0);
}

TEST(deflate, empty) {
	uint8_t data[1] = {0}, comp[64], decomp[1];
	for (int lvl = SDEFL_LVL_MIN; lvl <= SDEFL_LVL_MAX; ++lvl) {
		struct sdefl sdefl;
		const int len = sdeflate(&sdefl, comp, data, 0, lvl);
		// at least the final block: BFINAL=1, BTYPE=00, LEN=0, NLEN=0xFFFF
		EXPECT_GT(len, 0);
		EXPECT_EQ(comp[0] & 1u, 1u);
		EXPECT_EQ(0, sinflate(decomp, 1, comp, len));
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
