#include <gtest/gtest.h>
#include <iostream>
#include <cstdio>

#include "hash/hash.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

TEST(Hash, bits) {
	using T = uint64_t;
	T d[2] = {-1ull, -1ull};
	T r = Hash<T, 0, 10, 2>::hash(d);
	T mask = (1u<<10u) -1u;
	EXPECT_EQ(r, mask);

	r = Hash<T, 60, 70, 2>::hash(d);
	mask = (1u<<10u) -1u;
	EXPECT_EQ(r, mask);
}

TEST(Hash, base_q) {
	// q = 3: 2 bits per digit, digits 0..7 = 1,2,0,1,2,2,0,1
	using T = uint64_t;
	const T d[2] = {0b01'00'10'10'01'00'10'01ull, 0};
	EXPECT_EQ((Hash<T, 0, 4, 3>::hash(d)), 1u + 2u*3u + 0u*9u + 1u*27u);
	EXPECT_EQ((Hash<T, 2, 5, 3>::hash(d)), 0u + 1u*3u + 2u*9u);
	EXPECT_EQ((Hash<T, 2, 5, 3>::hash(d[0])), 0u + 1u*3u + 2u*9u);

	// digits 31 and 32 lie in different limbs
	const T e[2] = {2ull << 62u, 1ull};
	EXPECT_EQ((Hash<T, 31, 33, 3>::hash(e)), 2u + 1u*3u);
}

TEST(Hash, extract) {
	using T = uint64_t;
	const T d[2] = {-1ull, -1ull};
	EXPECT_EQ((extract<T, 60, 70>(d)), (1u << 10u) - 1u);

	const uint32_t e[2] = {0xABCD1234u, 0x5678EF01u};
	// bits [20, 32) = 0xABC from e[0], bits [32, 50) = the low 18 bits of e[1]
	EXPECT_EQ((extract<uint32_t, 20, 50>(e)), 0xABCu | ((0x5678EF01u & 0x3FFFFu) << 12u));

	// flip: [a|c] -> [c|a]
	const T f[1] = {0b1111'000ull};
	EXPECT_EQ((extract<T, 0, 7, 3>(f)), 0b0001111ull);
	EXPECT_EQ((extract<T, 60, 70, 63>(d)), (1u << 10u) - 1u);
}

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
	ident();
    return RUN_ALL_TESTS();
}
