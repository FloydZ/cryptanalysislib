#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <gtest/gtest.h>

#include "crypto/md5.h"
#include "crypto/sha1.h"
#include "crypto/sha3/keccak.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

static std::string hex(const uint8_t *d, const size_t n) {
	static const char *h = "0123456789abcdef";
	std::string s;
	for (size_t i = 0; i < n; ++i) { s += h[d[i] >> 4]; s += h[d[i] & 15]; }
	return s;
}

/// the pointer/length overload of md5, over several blocks (RFC 1321)
TEST(crypto, md5_pointer_length) {
	const std::string s = "12345678901234567890123456789012345678901234567890123456789012345678901234567890";
	const auto r = cryptanalysislib::crypto::internal::md5((const uint8_t *)s.data(), s.size());
	EXPECT_EQ(hex(r.data(), 16), "57edf4a22be3c955ac49da2e2107b67a");
	const std::string t = "abcdefghijklmnopqrstuvwxyz";
	const auto r2 = cryptanalysislib::crypto::internal::md5(t.data(), t.size());
	EXPECT_EQ(hex(r2.data(), 16), "c3fcd3d76192e4007dfb496cca67e13b");
	EXPECT_EQ(cryptanalysislib::crypto::internal::md5_rotl(0x80000001u, 1), 0x00000003u);
}

/// the std::array<std::byte> overload of sha1
TEST(crypto, sha1_array) {
	constexpr std::array<std::byte, 3> abc = {std::byte{'a'}, std::byte{'b'}, std::byte{'c'}};
	constexpr auto r = cryptanalysislib::sha1(abc);
	uint8_t b[20];
	for (size_t i = 0; i < 20; ++i) { b[i] = uint8_t(r[i]); }
	EXPECT_EQ(hex(b, 20), "a9993e364706816aba3e25717850c26c9cd0d89d");
	static_assert(cryptanalysislib::internal::choose(0xf0u, 0xccu, 0xaau) == ((0xf0u & 0xccu) | (~0xf0u & 0xaau)));
	static_assert(cryptanalysislib::internal::majority(0xf0u, 0xccu, 0xaau) == ((0xf0u & 0xccu) | (0xf0u & 0xaau) | (0xccu & 0xaau)));
	static_assert(cryptanalysislib::internal::parity(0xf0u, 0xccu, 0xaau) == (0xf0u ^ 0xccu ^ 0xaau));
}

/// Keccak-f[1600] on the all-zero state: the first lanes of the known answer
TEST(crypto, keccak_f) {
	cryptanalysislib::hash::keccak::state_1600 s{};
	cryptanalysislib::hash::keccak::keccak_f(s);
	EXPECT_EQ(s[0], 0xF1258F7940E1DDE7ull);
	EXPECT_EQ(s[1], 0x84D5CCF933C0478Aull);
	EXPECT_EQ(s[2], 0xD598261EA65AA9EEull);
	cryptanalysislib::hash::keccak::keccak_f(s);
	EXPECT_EQ(s[0], 0x2D5C954DF96ECB3Cull);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
