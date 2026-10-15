#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "random.h"
#include "crypto/sha3.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

static std::string hex(const uint8_t *d, const size_t n) {
	static const char *h = "0123456789abcdef";
	std::string s;
	for (size_t i = 0; i < n; ++i) { s += h[d[i] >> 4]; s += h[d[i] & 15]; }
	return s;
}

template<size_t N>
static std::string hex(const std::array<uint8_t, N> &d) {
	return hex(d.data(), N);
}

constexpr static const char *m448 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";

// source: https://www.di-mgt.com.au/sha_testvectors.html (NIST FIPS 202 examples)
TEST(sha3, vectors) {
	EXPECT_EQ(hex(sha3_224("")), "6b4e03423667dbb73b6e15454f0eb1abd4597f9a1b078e3f5b5a6bc7");
	EXPECT_EQ(hex(sha3_224("abc")), "e642824c3f8cf24ad09234ee7d3c766fc9a3a5168d0c94ad73b46fdf");
	EXPECT_EQ(hex(sha3_256("")), "a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a");
	EXPECT_EQ(hex(sha3_256("abc")), "3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532");
	EXPECT_EQ(hex(sha3_256(m448)), "41c0dba2a9d6240849100376a8235e2c82e1b9998a999e21db32dd97496d3376");
	EXPECT_EQ(hex(sha3_384("")), "0c63a75b845e4f7d01107d852e4c2485c51a50aaaa94fc61995e71bbee983a2ac3713831264adb47fb6bd1e058d5f004");
	EXPECT_EQ(hex(sha3_384("abc")), "ec01498288516fc926459f58e2c6ad8df9b473cb0fc08c2596da7cf0e49be4b298d88cea927ac7f539f1edf228376d25");
	EXPECT_EQ(hex(sha3_512("")), "a69f73cca23a9ac5c8b567dc185a756e97c982164fe25859e0d1dcc1475c80a615b2123af1f5f94c11e3e9402c3ac558f500199d95b6d3e301758586281dcd26");
	EXPECT_EQ(hex(sha3_512("abc")), "b751850b1a57168a5693cd924b6b096e08f621827444f70d884f5d0240d2712e10e116e9192af3c91a7ec57647e3934057340b4cf408d5a56592f8274eec53f0");
}

/// the rate boundary: 135, 136, 137 bytes for SHA3-256 (rate 136)
TEST(sha3, rate_boundary) {
	std::vector<uint8_t> m(400);
	for (size_t i = 0; i < m.size(); ++i) { m[i] = uint8_t(i); }
	for (const size_t n : {135u, 136u, 137u, 272u, 400u}) {
		// streaming byte by byte equals at once
		sha3_256_t s;
		for (size_t i = 0; i < n; ++i) { s.update(m.data() + i, 1); }
		EXPECT_EQ(s.digest(), sha3_256(m.data(), n)) << n;
	}
}

// source: SHAKE128/256 of the empty message (FIPS 202 / NIST examples)
TEST(sha3, shake) {
	uint8_t out[512];
	shake128(out, 32, nullptr, 0);
	EXPECT_EQ(hex(out, 32), "7f9c2ba4e88f827d616045507605853ed73b8093f6efbc88eb1a6eacfa66ef26");
	shake256(out, 64, nullptr, 0);
	EXPECT_EQ(hex(out, 64), "46b9dd2b0ba88d13233b3feb743eeb243fcd52ea62b81b82b50c27646ed5762fd75dc4ddd8c0f200cb05019d67b592f6fc821c49479ab48640292eacb3b7c4be");

	// squeezing in pieces (across the rate) equals squeezing at once
	shake128(out, 512, (const uint8_t *)"abc", 3);
	shake128_t s;
	s.update("abc");
	uint8_t p[512];
	size_t i = 0;
	while (i < 512) {
		const size_t k = std::min<size_t>(1 + rng() % 100, 512 - i);
		s.squeeze(p + i, k);
		i += k;
	}
	EXPECT_EQ(hex(p, 512), hex(out, 512));
}

TEST(sha3, constexpr_) {
	constexpr auto d = sha3_256("abc");
	static_assert(d[0] == 0x3a && d[31] == 0x32);
	constexpr auto e = []() {
		std::array<uint8_t, 4> o{};
		shake256(o.data(), 4, nullptr, 0);
		return o;
	}();
	static_assert(e[0] == 0x46 && e[3] == 0x2b);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
