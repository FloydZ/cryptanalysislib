#include <gtest/gtest.h>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "random.h"
#include "algorithm/bits/base2.h"

using ::testing::InitGoogleTest;
using namespace cryptanalysislib;

/// reference: one char per bit, the most significant bit first
static std::string reference_encode(const std::vector<uint8_t> &v) {
	std::string s;
	for (const uint8_t b : v) {
		for (int j = 7; j >= 0; --j) { s.push_back(((b >> j) & 1u) ? '1' : '0'); }
	}
	return s;
}

TEST(base2, hello_world) {
	const char *e = "010010000110010101101100011011000110111100100000010101110110111101110010011011000110010000100001";
	uint8_t out[12];
	base2_decode(out, e, 12);
	EXPECT_EQ(std::string((const char *)out, 12), "Hello World!");

	char s[96];
	base2_encode(s, (const uint8_t *)"Hello World!", 12);
	EXPECT_EQ(std::string(s, 96), std::string(e));
}

/// all sizes around the 8 byte blocks; the buffers have the exact size, so
/// ASan detects out of bounds accesses of the tails
TEST(base2, random) {
	for (size_t n = 0; n <= 100; ++n) {
		std::vector<uint8_t> v(n);
		for (auto &x : v) { x = uint8_t(rng()); }
		const std::string e = reference_encode(v);

		std::vector<char> s(8*n + (n == 0));
		base2_encode(s.data(), v.data(), n);
		EXPECT_EQ(std::string(s.data(), 8*n), e) << n;

		std::vector<uint8_t> w(n + (n == 0), 0xa5);
		base2_decode(w.data(), e.data(), n);
		w.resize(n);
		EXPECT_EQ(w, v) << n;
	}
}

TEST(base2, all_bytes) {
	for (uint32_t b = 0; b < 256; ++b) {
		const std::vector<uint8_t> v{uint8_t(b)};
		const std::string e = reference_encode(v);
		EXPECT_EQ(internal::base2_decode_word(*(const uint64_t *)e.data()), b);
		const uint64_t w = internal::base2_encode_word(uint8_t(b));
		EXPECT_EQ(std::string((const char *)&w, 8), e);
	}
}

TEST(base2, constexpr_) {
	constexpr auto d = []() {
		std::array<uint8_t, 2> out{};
		base2_decode(out.data(), "0100100001101001", 2);
		return out;
	}();
	static_assert(d[0] == 'H' && d[1] == 'i');

	constexpr auto e = []() {
		std::array<char, 16> out{};
		const uint8_t in[2] = {'H', 'i'};
		base2_encode(out.data(), in, 2);
		return out;
	}();
	static_assert(e[0] == '0' && e[1] == '1' && e[4] == '1' && e[7] == '0' && e[15] == '1');
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	rng_seed(time(NULL));
	return RUN_ALL_TESTS();
}
