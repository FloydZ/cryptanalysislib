#include <gtest/gtest.h>

#include "helper.h"
#include "math/math.h"
#include "random.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

constexpr size_t N = 64u;
using T = uint64_t;
using BigInt = big_int<N, T>;


TEST(abs, simple) {
	EXPECT_NEAR(fabs(-1.1), 1.1, 0.00001);
	EXPECT_NEAR(fabs(-1.1f), 1.1f, 0.00001);
}

TEST(cceil, simple) {
	EXPECT_EQ(cceil(1.1), 2);
	EXPECT_EQ(cceil(-1.1), -1);
}

TEST(floor, simple) {
	EXPECT_EQ(floor(1.1), 1);
	EXPECT_EQ(floor(-1.1), -2);
}

TEST(round, simple) {
	EXPECT_EQ(round(1.1), 1);
	EXPECT_EQ(round(-1.1), -1);
}

TEST(entropy, simple) {
	//EXPECT_EQ(std::isnan(HH(1.1)), true);
	//EXPECT_EQ(std::isnan(HH(1.0)), true);
	//EXPECT_EQ(std::isnan(HH(0.)), true);
	EXPECT_DOUBLE_EQ(HH(1.0), 0.0);
	EXPECT_DOUBLE_EQ(HH(0.0), 0.0);
	// EXPECT_DOUBLE_EQ(HH(0.5), 1.0);
}

TEST(exp, simple) {
	EXPECT_EQ(exp(1), 2);
	EXPECT_DOUBLE_EQ(exp(1.), e());
}

TEST(ipow, simple) {
	EXPECT_EQ(ipow(1., 2), 1);
	EXPECT_EQ(ipow(2., 2), 4);
}

TEST(low, simple) {
	EXPECT_DOUBLE_EQ(log(e()), 1);
	EXPECT_DOUBLE_EQ(log2(2.), 1);
	EXPECT_DOUBLE_EQ(log2(4.), 2);
	EXPECT_DOUBLE_EQ(log2(8.), 3);
	EXPECT_DOUBLE_EQ(log2(16.), 4);
}

TEST(root, simple) {
	EXPECT_EQ(sqrt(4.), 2.);
	EXPECT_EQ(sqrt(9), 3);

	EXPECT_EQ(cbrt(27), 3);
	EXPECT_DOUBLE_EQ(cbrt(27.), 3.);
	EXPECT_DOUBLE_EQ(cbrt(27.), 3);

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
