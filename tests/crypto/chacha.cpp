#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "random.h"
#include "crypto/chacha.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

static std::string hex(const uint8_t *d, const size_t n) {
	static const char *h = "0123456789abcdef";
	std::string s;
	for (size_t i = 0; i < n; ++i) { s += h[d[i] >> 4]; s += h[d[i] & 15]; }
	return s;
}

constexpr static std::array<uint8_t, 32> key = []() {
	std::array<uint8_t, 32> k{};
	for (uint32_t i = 0; i < 32; ++i) { k[i] = uint8_t(i); }
	return k;
}();

// source: RFC 8439 2.3.2 (block function)
TEST(chacha, rfc8439_block) {
	constexpr uint8_t nonce[12] = {0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00};
	chacha20 c(key.data(), nonce, 1);
	const auto b = c.block();
	EXPECT_EQ(hex(b.data(), 64),
	          "10f1e7e4d13b5915500fdd1fa32071c4c7d1f4c733c068030422aa9ac3d46c4e"
	          "d2826446079faa0914c2d705d98b02a2b5129cd1de164eb9cbd083e8a2503c4e");
}

// source: RFC 8439 2.4.2 (encryption, "sunscreen")
TEST(chacha, rfc8439_encryption) {
	constexpr uint8_t nonce[12] = {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00};
	const std::string p = "Ladies and Gentlemen of the class of '99: If I could offer you only one tip for the future, sunscreen would be it.";
	std::vector<uint8_t> c(p.size()), d(p.size());
	chacha20 e(key.data(), nonce, 1);
	e.crypt(c.data(), (const uint8_t *)p.data(), p.size());
	EXPECT_EQ(hex(c.data(), c.size()),
	          "6e2e359a2568f98041ba0728dd0d6981e97e7aec1d4360c20a27afccfd9fae0b"
	          "f91b65c5524733ab8f593dabcd62b3571639d624e65152ab8f530c359f0861d8"
	          "07ca0dbf500d6a6156a38e088a22b65e52bc514d16ccf806818ce91ab7793736"
	          "5af90bbf74a35be6b40b8eedf2785e42874d");

	// decryption, in place and in pieces
	d = c;
	chacha20 f(key.data(), nonce, 1);
	size_t i = 0;
	while (i < d.size()) {
		const size_t k = std::min<size_t>(1 + rng() % 40, d.size() - i);
		f.crypt(d.data() + i, d.data() + i, k);
		i += k;
	}
	EXPECT_EQ(std::string(d.begin(), d.end()), p);
}

TEST(chacha, constexpr_and_rounds) {
	constexpr auto b = []() {
		constexpr uint8_t nonce[12] = {0x00, 0x00, 0x00, 0x09, 0x00, 0x00, 0x00, 0x4a, 0x00, 0x00, 0x00, 0x00};
		chacha20 c(key.data(), nonce, 1);
		return c.block();
	}();
	static_assert(b[0] == 0x10 && b[63] == 0x4e);

	// different number of rounds, different key streams
	uint8_t n[12] = {0}, s8[64], s12[64], s20[64];
	chacha8(key.data(), n).keystream(s8, 64);
	chacha12(key.data(), n).keystream(s12, 64);
	chacha20(key.data(), n).keystream(s20, 64);
	EXPECT_NE(memcmp(s8, s12, 64), 0);
	EXPECT_NE(memcmp(s12, s20, 64), 0);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
