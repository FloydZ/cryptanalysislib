#include <cmath>
#include <cstdint>
#include <numeric>
#include <gtest/gtest.h>

#include "math/math.h"
#include "math/gcd.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace cryptanalysislib;

TEST(math, abs_feq) {
	EXPECT_EQ(math::abs(-5), 5);
	EXPECT_EQ(math::abs(5u), 5u);
	EXPECT_EQ(math::abs(-2.5), 2.5);
	EXPECT_TRUE(math::feq(1.0, 1.0 + 1e-16));
	EXPECT_FALSE(math::feq(1.0, 1.0 + 1e-10));
	EXPECT_TRUE(math::feq(1e20, 1e20 * (1. + 1e-16)));
	// |a - b| for a, b with the msb clear
	for (uint32_t i = 0; i < 100000; ++i) {
		const uint32_t a = rng<uint32_t>() >> 1, b = rng<uint32_t>() >> 1;
		ASSERT_EQ(math::abs_branchless<uint32_t>(a, b), a > b ? a - b : b - a) << a << " " << b;
		const int64_t c = rng<uint64_t>() >> 2, d = rng<uint64_t>() >> 2;
		ASSERT_EQ(math::abs_branchless<int64_t>(c, d), c > d ? c - d : d - c);
		const uint16_t e = rng<uint16_t>() >> 1, f = rng<uint16_t>() >> 1;
		ASSERT_EQ(math::abs_branchless<uint16_t>(e, f), uint16_t(e > f ? e - f : f - e)) << e << " " << f;
		const uint64_t g = rng<uint64_t>() >> 1, h = rng<uint64_t>() >> 1;
		ASSERT_EQ(math::abs_branchless<uint64_t>(g, h), g > h ? g - h : h - g) << g << " " << h;
		const int32_t k = rng<uint32_t>() >> 2, l = rng<uint32_t>() >> 2;
		ASSERT_EQ(math::abs_branchless<int32_t>(k, l), k > l ? k - l : l - k) << k << " " << l;
	}
	// all 8 bit pairs with the msb clear
	for (uint32_t a = 0; a < 128; ++a) {
		for (uint32_t b = 0; b < 128; ++b) {
			ASSERT_EQ(math::abs_branchless<uint8_t>(uint8_t(a), uint8_t(b)), uint8_t(a > b ? a - b : b - a)) << a << " " << b;
		}
	}
}

TEST(math, cceil) {
	for (const float f : {0.f, 1.f, 1.5f, -1.5f, -2.f, 1e6f + 0.5f, 0.25f, -0.25f}) {
		EXPECT_EQ(math::cceil(f), (int32_t)std::ceil(f)) << f;
	}
	for (const double f : {0., 1., 1.5, -1.5, -2., 3e10 + 0.5, 0.25, -0.25}) {
		EXPECT_EQ(math::cceil(f), (int64_t)std::ceil(f)) << f;
	}
}

TEST(math, gcd_variants) {
	for (uint32_t i = 0; i < 10000; ++i) {
		const uint64_t a = rng<uint64_t>() % 100000, b = rng<uint64_t>() % 100000;
		const uint64_t e = std::gcd(a, b);
		EXPECT_EQ(internal::gcd_recursive_v0<uint64_t>(a, b), e) << a << " " << b;
		EXPECT_EQ(internal::gcd_recursive_v1<uint64_t>(a, b), e) << a << " " << b;
		EXPECT_EQ(internal::gcd_recursive_v2<uint64_t>(a, b), e) << a << " " << b;
		EXPECT_EQ(internal::gcd_non_recursive_v1<uint64_t>(a, b), e) << a << " " << b;
		EXPECT_EQ(internal::gcd_non_recursive_v2<uint64_t>(a, b), e) << a << " " << b;
		EXPECT_EQ(internal::gcd_binary<uint64_t>(a, b), e) << a << " " << b;
		const int32_t c = (int32_t)(rng<uint32_t>() % 2000) - 1000, d = (int32_t)(rng<uint32_t>() % 2000) - 1000;
		EXPECT_EQ(internal::gcd_binary<int32_t>(c, d), std::gcd(c, d)) << c << " " << d;
	}
	EXPECT_EQ(internal::gcd_binary<uint64_t>(~0ull, 3ull), 3u);
}

TEST(math, bc_helpers) {
	EXPECT_EQ(internal::bc_gcd(12, 18), 6u);
	EXPECT_EQ(internal::bc_gcd(0, 7), 7u);
	EXPECT_EQ(internal::bc_gcd(7, 0), 7u);
	// (a * n) / d for d | a * n, without overflow of a * n
	EXPECT_EQ(internal::bc_mul_div(10, 6, 4), 15u);
	EXPECT_EQ(internal::bc_mul_div(1ull << 62, 6, 4), 3ull << 61);
	EXPECT_EQ(internal::bc_mul_div(0, 6, 4), 0u);
}

TEST(math, ipow_v2) {
	for (uint64_t a = 0; a < 20; ++a) {
		uint64_t e = 1;
		for (uint32_t x = 0; x < 15; ++x) {
			EXPECT_EQ(math::ipow_v2(a, x), e) << a << " " << x;
			e *= a;
		}
	}
	EXPECT_EQ(math::ipow_v2(2.0, 10u), 1024.0);
}

TEST(math, log_internal) {
	for (const double x : {0.5, 1.0, 2.0, 10.0, 1000.0}) {
		EXPECT_NEAR(math::internal::log(x, 0.0), std::log(x), 1e-12 * (1 + std::abs(std::log(x)))) << x;
		EXPECT_NEAR(math::internal::log_iter(x, std::log(x)), std::log(x), 1e-12) << x;
	}
	for (const double x : {1e-10, 1e-3, 0.1, 0.3}) {
		EXPECT_NEAR(math::internal::logGT(x), std::log(x), 1e-10 * (1 + std::abs(std::log(x)))) << x;
	}
	for (const double x : {2.0, 1024.0, 1e10, 1e100}) {
		EXPECT_NEAR(math::internal::logLT(x), std::log(x), 1e-10 * (1 + std::abs(std::log(x)))) << x;
	}
}

TEST(math, sqrt_cbrt_guess) {
	for (const double x : {0., 1e-6, 0.5, 2., 1e6, 1e100}) {
		EXPECT_NEAR(math::sqrt(x, x > 1 ? x : 1.), std::sqrt(x), 1e-12 * (1 + std::sqrt(x))) << x;
		EXPECT_NEAR(math::cbrt(x, x > 1 ? x : 1.), std::cbrt(x), 1e-12 * (1 + std::cbrt(x))) << x;
	}
	EXPECT_NEAR(math::cbrt(-8., -1.), -2., 1e-12);
}

TEST(math, round_up_to_power_of_two) {
	EXPECT_EQ(math::round_up_to_power_of_two(0), 1u);
	EXPECT_EQ(math::round_up_to_power_of_two(1), 1u);
	EXPECT_EQ(math::round_up_to_power_of_two(3), 4u);
	EXPECT_EQ(math::round_up_to_power_of_two(1024), 1024u);
	EXPECT_EQ(math::round_up_to_power_of_two(1025), 2048u);
	EXPECT_EQ(math::round_up_to_power_of_two((1ull << 40) + 1), 1ull << 41);
}

TEST(math, complex_scalar_ops) {
	const math::complex<double> a(2., -4.);
	const auto b = a * 0.5;
	const auto c = 3. * a;
	const auto d = a / 2.;
	EXPECT_EQ(b.real(), 1.); EXPECT_EQ(b.imag(), -2.);
	EXPECT_EQ(c.real(), 6.); EXPECT_EQ(c.imag(), -12.);
	EXPECT_EQ(d.real(), 1.); EXPECT_EQ(d.imag(), -2.);
}

/// the 128 bit products and the signed fastmod/fastdiv of `math::internal`
TEST(math, mod_internal) {
	using namespace math::internal;
	for (uint32_t i = 0; i < 100000; ++i) {
		const uint64_t x = rng<uint64_t>(), y = rng<uint64_t>();
		const uint32_t d = rng<uint32_t>();
		EXPECT_EQ(mul128_u32(x, d), uint64_t(((__uint128_t)x * d) >> 64));
		EXPECT_EQ(mul128_from_u64(x, y), uint64_t(((__uint128_t)x * y) >> 64));
		const __uint128_t z = ((__uint128_t)rng<uint64_t>() << 64) | rng<uint64_t>();
		// (z * y) >> 128 = the high 64 bits of the 192 bit product
		const __uint128_t lo = (z & ~0ull) * (__uint128_t)y, hi = (z >> 64) * (__uint128_t)y;
		EXPECT_EQ(mul128_u64(z, y), uint64_t((hi + (lo >> 64)) >> 64));

		int32_t sd = (int32_t)(rng<uint32_t>() % 2000001) - 1000000;
		if (sd == 0) { sd = 7; }
		const int32_t a = (int32_t)rng<uint32_t>();
		const uint64_t M = computeM_s32(sd);
		const int32_t pd = sd < 0 ? -sd : sd;
		EXPECT_EQ(fastmod_s32(a, M, pd), a % pd) << a << " " << sd;
		EXPECT_EQ(fastdiv_s32(a, M, sd), a / sd) << a << " " << sd;

		const uint32_t ud = 1u + (rng<uint32_t>() % 100000u), n = rng<uint32_t>();
		EXPECT_EQ(is_divisible(n, computeM_u32(ud)), n % ud == 0) << n << " " << ud;
		EXPECT_TRUE(is_divisible(n - (n % ud), computeM_u32(ud)));
	}
	for (const int32_t d : {1, 2, 3, 8, -1, -2, -3, -7, 1 << 30, -(1 << 30), 2147483647, -2147483647}) {
		for (const int32_t a : {0, 1, -1, 5, -5, 2147483647, -2147483647 - 1}) {
			if (d == -1 && a == -2147483647 - 1) { continue; }
			const int32_t pd = d < 0 ? -d : d;
			EXPECT_EQ(fastmod_s32(a, computeM_s32(d), pd), a % pd) << a << " " << d;
			EXPECT_EQ(fastdiv_s32(a, computeM_s32(d), d), a / d) << a << " " << d;
		}
	}
}

TEST(math, submod_mod_pow) {
	for (uint32_t i = 0; i < 10000; ++i) {
		const uint64_t m = 1u + (rng<uint64_t>() >> 1);
		const uint64_t a = rng<uint64_t>() % m, b = rng<uint64_t>() % m;
		EXPECT_EQ(submod(a, b, m), uint64_t(((__uint128_t)a + m - b) % m));
		const uint64_t e = rng<uint64_t>() % 100;
		uint64_t r = 1 % m;
		for (uint64_t k = 0; k < e; ++k) { r = uint64_t((__uint128_t)r * a % m); }
		EXPECT_EQ(mod_pow<uint64_t>(a, e, m), r);
	}
	EXPECT_EQ(mod_pow<int64_t>(-2, 3, 7), 6);
	EXPECT_EQ(mod_pow<uint32_t>(3, 0, 1), 0u);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
