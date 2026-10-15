#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>
#include <string>

#include "helper.h"
#include "random.h"
#include "crypto/md5.h"

using ::testing::InitGoogleTest;

using namespace cryptanalysislib::crypto;

/// \return the digest as lower case hex string
static std::string hex(const std::array<uint8_t, 16> &d) {
	static const char *digits = "0123456789abcdef";
	std::string r;
	for (const auto b : d) { r += digits[b >> 4u]; r += digits[b & 0xfu]; }
	return r;
}

// usable at compile time
static_assert(md5("abc")[0] == 0x90 && md5("abc")[15] == 0x72);

/// RFC 1321, appendix A.5
TEST(MD5, rfc1321) {
	EXPECT_EQ(hex(md5("")), "d41d8cd98f00b204e9800998ecf8427e");
	EXPECT_EQ(hex(md5("a")), "0cc175b9c0f1b6a831c399e269772661");
	EXPECT_EQ(hex(md5("abc")), "900150983cd24fb0d6963f7d28e17f72");
	EXPECT_EQ(hex(md5("message digest")), "f96b697d7cb7938d525a2f31aaf161d0");
	EXPECT_EQ(hex(md5("abcdefghijklmnopqrstuvwxyz")), "c3fcd3d76192e4007dfb496cca67e13b");
	EXPECT_EQ(hex(md5("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789")),
	          "d174ab98d277d9f5a5611c2c9f419d9f");
	EXPECT_EQ(hex(md5("12345678901234567890123456789012345678901234567890123456789012345678901234567890")),
	          "57edf4a22be3c955ac49da2e2107b67a");
}

/// the pointer and the string_view interface agree, for lengths around
/// the block boundaries (55, 56, 63, 64, ...)
TEST(MD5, pointer_vs_string) {
	std::string s;
	for (uint32_t n = 0; n < 200; ++n) {
		EXPECT_EQ(md5(s), md5((const uint8_t *)s.data(), s.size()));
		s += char('a' + (n % 26));
	}
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
