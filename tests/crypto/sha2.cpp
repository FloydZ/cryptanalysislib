#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "random.h"
#include "crypto/sha2.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

template<size_t N>
static std::string hex(const std::array<uint8_t, N> &d) {
	static const char *h = "0123456789abcdef";
	std::string s;
	for (const uint8_t x : d) { s += h[x >> 4]; s += h[x & 15]; }
	return s;
}

constexpr static const char *m448 = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
constexpr static const char *m896 = "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";

// source: https://www.di-mgt.com.au/sha_testvectors.html (NIST FIPS 180-4 examples)
TEST(sha2, sha224) {
	EXPECT_EQ(hex(sha224("")), "d14a028c2a3a2bc9476102bb288234c415a2b01f828ea62ac5b3e42f");
	EXPECT_EQ(hex(sha224("abc")), "23097d223405d8228642a477bda255b32aadbce4bda0b3f7e36c9da7");
	EXPECT_EQ(hex(sha224(m448)), "75388b16512776cc5dba5da1fd890150b0c6455cb4f58b1952522525");
	EXPECT_EQ(hex(sha224(m896)), "c97ca9a559850ce97a04a96def6d99a9e0e0e2ab14e6b8df265fc0b3");
}

TEST(sha2, sha256) {
	EXPECT_EQ(hex(sha256("")), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
	EXPECT_EQ(hex(sha256("abc")), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
	EXPECT_EQ(hex(sha256(m448)), "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
	EXPECT_EQ(hex(sha256(m896)), "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");
}

TEST(sha2, sha384) {
	EXPECT_EQ(hex(sha384("")), "38b060a751ac96384cd9327eb1b1e36a21fdb71114be07434c0cc7bf63f6e1da274edebfe76f65fbd51ad2f14898b95b");
	EXPECT_EQ(hex(sha384("abc")), "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7");
	EXPECT_EQ(hex(sha384(m448)), "3391fdddfc8dc7393707a65b1b4709397cf8b1d162af05abfe8f450de5f36bc6b0455a8520bc4e6f5fe95b1fe3c8452b");
	EXPECT_EQ(hex(sha384(m896)), "09330c33f71147e83d192fc782cd1b4753111b173b3b05d22fa08086e3b0f712fcc7c71a557e2db966c3e9fa91746039");
}

TEST(sha2, sha512) {
	EXPECT_EQ(hex(sha512("")), "cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
	EXPECT_EQ(hex(sha512("abc")), "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
	EXPECT_EQ(hex(sha512(m448)), "204a8fc6dda82f0a0ced7beb8e08a41657c16ef468b228a8279be331a703c33596fd15c13b1b07f9aa1d3bea57789ca031ad85c7a71dd70354ec631238ca3445");
	EXPECT_EQ(hex(sha512(m896)), "8e959b75dae313da8cf4f72814fc143f8f7779c6eb9f7fa17299aeadb6889018501d289e4900f7e4331b99dec4b5433ac7d329eeb6dd26545e96e55b874be909");
}

/// one million times 'a', fed in uneven chunks
TEST(sha2, million_a) {
	const std::string chunk(997, 'a');
	sha256_t s256;
	sha512_t s512;
	size_t n = 0;
	while (n < 1000000) {
		const size_t k = std::min(chunk.size(), size_t(1000000) - n);
		s256.update((const uint8_t *)chunk.data(), k);
		s512.update((const uint8_t *)chunk.data(), k);
		n += k;
	}
	EXPECT_EQ(hex(s256.digest()), "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
	EXPECT_EQ(hex(s512.digest()), "e718483d0ce769644e2e42c7bc15b4638e1f98b13b2044285632a803afa973ebde0ff244877ea60a4cb0432ce577c31beb009c5c2c49aa2e4eadb217ad8cc09b");
}

/// streaming in random chunks equals hashing at once
TEST(sha2, streaming) {
	for (uint32_t r = 0; r < 50; ++r) {
		std::vector<uint8_t> m(rng() % 1000);
		for (auto &x : m) { x = uint8_t(rng()); }
		sha384_t s;
		size_t i = 0;
		while (i < m.size()) {
			const size_t k = std::min(size_t(rng() % 200), m.size() - i);
			s.update(m.data() + i, k);
			i += k;
		}
		EXPECT_EQ(s.digest(), sha384(m.data(), m.size()));
	}
}

TEST(sha2, constexpr_) {
	constexpr auto d = sha256("abc");
	static_assert(d[0] == 0xba && d[31] == 0xad);
	constexpr auto e = sha512(m896);
	static_assert(e[0] == 0x8e && e[63] == 0x09);
	constexpr uint8_t b[3] = {'a', 'b', 'c'};
	constexpr auto f = sha224(b, 3);
	static_assert(f[0] == 0x23 && f[27] == 0xa7);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
