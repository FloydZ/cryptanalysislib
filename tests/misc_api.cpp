#include <cstdint>
#include <numeric>
#include <string_view>
#include <gtest/gtest.h>

#include "container/array.h"
#include "container/hashmap.h"
#include "graph/digraph.h"
#include "log.h"
#include "simd/simd.h"
#include "string/stringliteral.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

/// the fxt bit helpers of `digraph`
TEST(misc, digraph_helpers) {
	using namespace cryptanalysislib::internal::digraph;
	EXPECT_FALSE(one_bit_q(0));
	EXPECT_TRUE(one_bit_q(1ull << 63));
	EXPECT_FALSE(one_bit_q(3));

	// Zeckendorf: no two adjacent ones, sum of the Fibonacci numbers is k
	uint64_t fib[92] = {1, 2};
	for (int i = 2; i < 92; ++i) { fib[i] = fib[i - 1] + fib[i - 2]; }
	for (uint64_t k = 0; k < 5000; ++k) {
		const uint64_t r = bin2fibrep(k);
		ASSERT_EQ(r & (r >> 1), 0u) << k;
		uint64_t s = 0;
		for (int i = 0; i < 64; ++i) { if ((r >> i) & 1u) { s += fib[i]; } }
		ASSERT_EQ(s, k);
	}

	for (uint32_t n = 1; n <= 64; n += 7) {
		for (uint32_t r = 0; r < n; r += 3) {
			const uint64_t mask = n == 64 ? ~0ull : ((1ull << n) - 1u);
			const uint64_t x = 0x0123456789abcdefull & mask;
			const uint64_t e = r ? (((x << r) | (x >> (n - r))) & mask) : x;
			EXPECT_EQ(bit_rotate_left(x, r, n), e) << n << " " << r;
		}
	}

	// all combinations of 3 out of 8 bits in colex order
	uint64_t x = first_comb(3);
	EXPECT_EQ(x, 7u);
	uint32_t cnt = 1;
	while (true) {
		const uint64_t y = next_colex_comb(x);
		if (y >= (1u << 8)) { break; }
		ASSERT_GT(y, x);
		ASSERT_EQ(__builtin_popcountll(y), 3);
		x = y; cnt += 1;
	}
	EXPECT_EQ(cnt, 56u);

	EXPECT_TRUE(is_parenword(0b0011, 4));  // "(())" read from bit 0: 1,1,0,0
	EXPECT_TRUE(is_parenword(0b0101, 4));  // "()()"
	EXPECT_FALSE(is_parenword(0b1010, 4)); // ")(" first
	EXPECT_FALSE(is_parenword(0b0111, 4));

	const uint32_t a[] = {1, 3, 3, 8, 10};
	for (uint32_t v = 0; v < 12; ++v) {
		const uint64_t r = bsearch(a, 5, v);
		const auto it = std::lower_bound(a, a + 5, v);
		EXPECT_EQ(r, (it != a + 5 && *it == v) ? uint64_t(it - a) : 5u) << v;
	}
}

TEST(misc, string_literal) {
	using cryptanalysislib::internal::StringLiteral;
	constexpr StringLiteral a("abc"), b("abd"), c("abc");
	constexpr StringLiteral d("ab");
	static_assert(a == c);
	static_assert(a != b);
	static_assert(a != d);
	static_assert((a <=> b) < 0);
	static_assert((a <=> c) == 0);
	EXPECT_EQ(a.string_view(), std::string_view("abc"));
	EXPECT_TRUE((d <=> a) < 0); // shorter first
}

TEST(misc, swar_internal) {
	namespace si = cryptanalysislib::swar::internal;
	EXPECT_EQ(si::ctz(uint32_t(8)), 3u);
	EXPECT_EQ(si::ctz(uint32_t(0)), 32u);
	EXPECT_EQ(si::clz(uint16_t(1)), 15u);
	EXPECT_EQ(si::clz(uint64_t(0)), 64u);
	EXPECT_EQ(si::popcount(uint64_t(0xf0f0)), 8u);
	EXPECT_EQ(si::popcount(uint8_t(0xff)), 8u);
}

/// lane-wise division by an invariant: all numerators and divisors (8 bit),
/// and random ones (16 and 32 bit)
TEST(misc, udiv_invariant) {
	using namespace cryptanalysislib::internal;
	using V8 = vec_t<uint8_t, 16>;
	for (uint32_t d = 1; d < 256; ++d) {
		for (uint32_t base = 0; base < 256; base += 16) {
			V8 a;
			for (uint32_t i = 0; i < 16; ++i) { a[i] = uint8_t(base + i); }
			const V8 q = udiv_invariant<uint8_t, 16>(a, uint8_t(d));
			for (uint32_t i = 0; i < 16; ++i) { ASSERT_EQ(q[i], uint8_t((base + i) / d)) << base + i << " " << d; }
		}
	}
	using V32 = vec_t<uint32_t, 4>;
	for (uint32_t r = 0; r < 100000; ++r) {
		const uint32_t d = 1u + uint32_t(cryptanalysislib::rng() % 0xffffffffu);
		V32 a;
		for (uint32_t i = 0; i < 4; ++i) { a[i] = uint32_t(cryptanalysislib::rng()); }
		a[0] = 0xffffffffu;
		const V32 q = udiv_invariant<uint32_t, 4>(a, d);
		for (uint32_t i = 0; i < 4; ++i) { ASSERT_EQ(q[i], a[i] / d) << a[i] << " " << d; }
	}
	using V16 = vec_t<uint16_t, 8>;
	for (uint32_t d = 1; d < 65536; d += 37) {
		V16 a;
		for (uint32_t i = 0; i < 8; ++i) { a[i] = uint16_t(cryptanalysislib::rng()); }
		a[0] = 0xffffu;
		const V16 q = udiv_invariant<uint16_t, 8>(a, uint16_t(d));
		for (uint32_t i = 0; i < 8; ++i) { ASSERT_EQ(q[i], uint16_t(a[i] / d)) << a[i] << " " << d; }
	}
}

TEST(misc, logging_to_string) {
	using namespace cryptanalysislib::logging;
	EXPECT_STREQ(to_string(LoggingLevel::debug), "debug");
	EXPECT_STREQ(to_string(LoggingLevel::info), "info");
	EXPECT_STREQ(to_string(LoggingLevel::warning), "warning");
	EXPECT_STREQ(to_string(LoggingLevel::error), "error");
}

TEST(misc, const_array_transform) {
	using namespace cryptanalysislib;
	constexpr const_array<int, 3> a{1, 2, 3};
	constexpr const_array<int, 2> b{10, 20};
	constexpr auto c = transform(a, b, [](int x, int y) { return x * y; });
	static_assert(c.size() == 2);
	static_assert(c[0] == 10 && c[1] == 40);
	constexpr auto d = transform(a, [](int x) { return x + 1; });
	static_assert(d[2] == 4);
}

TEST(misc, growth_policy_mod) {
	using namespace cryptanalysislib::hh::detail;
	for (const std::size_t h : {0ull, 1ull, 12345ull, ~0ull}) {
		EXPECT_EQ(mod<0>(h), h % PRIMES[0]);
		EXPECT_EQ(mod<5>(h), h % PRIMES[5]);
		EXPECT_EQ(MOD_PRIME[7](h), h % PRIMES[7]);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
