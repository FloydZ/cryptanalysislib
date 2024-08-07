#include <gtest/gtest.h>

#include "helper.h"
#include "math/math.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr size_t N = 64u;
using T = uint64_t;
using BigInt = big_int<N, T>;

TEST(prime, simple) {
	constexpr bool b1 = is_prime(3);
	EXPECT_EQ(b1, true);

	constexpr bool b2 = is_prime(17);
	EXPECT_EQ(b2, true);

	bool b3 = is_prime(18361375334787046697ull);
	EXPECT_EQ(b3, true);


	constexpr bool b4 = is_prime(4);
	EXPECT_EQ(b4, false);

	constexpr bool b5 = is_prime(98127398717);
	EXPECT_EQ(b5, false);

	constexpr size_t np1 = next_prime(2);
	EXPECT_EQ(np1, 2ull);
	constexpr size_t np2 = next_prime(5);
	EXPECT_EQ(np2, 5ull);
	constexpr size_t np3 = next_prime(10);
	EXPECT_EQ(np3, 11ull);
	constexpr size_t np4 = next_prime(33);
	EXPECT_EQ(np4, 37ull);
	size_t np5 = next_prime(18361375334787046695ull);
	EXPECT_EQ(np5, 18361375334787046697ull);
}


TEST(abs, simple) {
	EXPECT_NEAR(cryptanalysislib::math::fabs(-1.1), 1.1, 0.00001);
	EXPECT_NEAR(cryptanalysislib::math::fabs(-1.1f), 1.1f, 0.00001);
}

TEST(cceil, simple) {
	EXPECT_EQ(cryptanalysislib::math::cceil(1.1), 2);
	EXPECT_EQ(cryptanalysislib::math::cceil(-1.1), -1);
}

TEST(floor, simple) {
	EXPECT_EQ(cryptanalysislib::math::floor(1.1), 1);
	EXPECT_EQ(cryptanalysislib::math::floor(-1.1), -2);
}

TEST(round, simple) {
	EXPECT_EQ(cryptanalysislib::math::round(1.1), 1);
	EXPECT_EQ(cryptanalysislib::math::round(-1.1), -1);
}

TEST(entropy, simple) {
	//EXPECT_EQ(std::isnan(HH(1.1)), true);
	//EXPECT_EQ(std::isnan(HH(1.0)), true);
	//EXPECT_EQ(std::isnan(HH(0.)), true);
	EXPECT_DOUBLE_EQ(cryptanalysislib::math::HH(1.0), 0.0);
	EXPECT_DOUBLE_EQ(cryptanalysislib::math::HH(0.0), 0.0);
	// EXPECT_DOUBLE_EQ(HH(0.5), 1.0);
}

TEST(exp, simple) {
	EXPECT_EQ(cryptanalysislib::math::exp(1), 2);
	EXPECT_DOUBLE_EQ(cryptanalysislib::math::exp(1.), cryptanalysislib::math::internal::e());
}

TEST(ipow, simple) {
	EXPECT_EQ(cryptanalysislib::math::ipow(1., 2), 1);
	EXPECT_EQ(cryptanalysislib::math::ipow(2., 2), 4);
}

TEST(mod, simple) {
	constexpr uint32_t mod = 718293;
	for (uint32_t i = 0; i < 1u<<14; ++i) {
		const uint32_t a = fastrandombytes_uint64();
		const uint32_t b = fastrandombytes_uint64();
		const uint32_t c = fastmod<mod>(a+b);
		EXPECT_EQ(c, (a+b)%mod);
	}
}

TEST(low, simple) {
	EXPECT_DOUBLE_EQ(cryptanalysislib::math::log(cryptanalysislib::math::internal::e()), 1);
	EXPECT_DOUBLE_EQ(cryptanalysislib::math::log2(2.), 1);
	EXPECT_DOUBLE_EQ(cryptanalysislib::math::log2(4.), 2);
	EXPECT_DOUBLE_EQ(cryptanalysislib::math::log2(8.), 3);
}

TEST(root, simple) {
	EXPECT_EQ(cryptanalysislib::math::sqrt(4.), 2.);
	EXPECT_EQ(cryptanalysislib::math::sqrt(9), 3);

	EXPECT_EQ(cryptanalysislib::math::cbrt(27), 3);
	EXPECT_DOUBLE_EQ(cryptanalysislib::math::cbrt(27.), 3.);
	EXPECT_DOUBLE_EQ(cryptanalysislib::math::cbrt(27.), 3);

	// TODO
	// NOT WORKING: EXPECT_EQ(kthrt(27., 2), 3);
}

#ifdef USE_AVX512
TEST(bc, avx512) {
	EXPECT_EQ(simd_binom(10, 2), bc(10, 2));
}
#endif

TEST(big_int, simple) {
	BigInt one = BigInt{1};
	BigInt zero = BigInt{};
	EXPECT_EQ(one[0], 1ul);
	for (size_t i = 1; i < N; ++i) {
		EXPECT_EQ(one[i], 0ul);
	}

	EXPECT_EQ(one == zero, false);
	EXPECT_EQ(one == one, true);

	/// IMPORTANT: that's is autocasting the result down from a
	/// big_int<N + 1> to a big_int<N>
	BigInt tmp = one + zero;
	EXPECT_EQ(tmp == one, true);
}

TEST(big_int, binom) {
	const auto b1 = binomial<1, T, 1u, 1u>();
	EXPECT_EQ(b1[0], 1);
	const auto b2 = binomial<2, T, 2u, 2u>();
	EXPECT_EQ(b2[0], 1);

	const auto b3 = binomial<1, T, 256, 2u>();
	EXPECT_EQ(b3[0], 32640);

	// Number is too big:w
	constexpr auto b4 = binomial<1, T, 1024, 7u>();
	EXPECT_EQ(b4[0], 229479463334370304);
}
int main(int argc, char **argv) {
	random_seed(time(NULL));
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
