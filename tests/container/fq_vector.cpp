#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "container/fq_vector.h"
#include "container/fq_packed_vector.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(test, simple) {
	using K3 = kAryPackedContainer_T<uint64_t, 100, 3>;
	using K5 = kAryPackedContainer_T<uint64_t, 100, 5>;
	auto t31 = K3{};
	auto t32 = K3{};
	auto t33 = K3{};
	K3::add_only_weight_partly<0, 2>(t33, t31, t32);
	
}

constexpr uint32_t n = 4;
constexpr uint32_t q = 4;
using K4 = kAryContainer_T<uint8_t, n, q>;

TEST(F4, random) {
	K4 t = K4();
	t.random();

	for (uint32_t i = 0; i < n; i++){
    	EXPECT_LE(t.get(i), q);
	}
}

TEST(F4, mod_T) {
	EXPECT_EQ(K4::mod_T<uint32_t>(4), 0);
	EXPECT_EQ(K4::mod_T<uint32_t>(0), 0);
	EXPECT_EQ(K4::mod_T<uint32_t>(1), 1);
	EXPECT_EQ(K4::mod_T<uint32_t>(2), 2);
	EXPECT_EQ(K4::mod_T<uint32_t>(3), 3);
}

TEST(F4, add_T) {
	EXPECT_EQ(K4::add_T<uint32_t>(4, 0), 0);
	EXPECT_EQ(K4::add_T<uint32_t>(0, 0), 0);
	EXPECT_EQ(K4::add_T<uint32_t>(1, 0), 1);
	EXPECT_EQ(K4::add_T<uint32_t>(2, 0), 2);
	EXPECT_EQ(K4::add_T<uint32_t>(3, 0), 3);
	
	EXPECT_EQ(K4::add_T<uint32_t>((4u << 8u) ^ 4, 0), (0u << 8u) ^0);
	EXPECT_EQ(K4::add_T<uint32_t>((0u << 8u) ^ 0, 0), (0u << 8u) ^0);
	EXPECT_EQ(K4::add_T<uint32_t>((1u << 8u) ^ 1, 0), (1u << 8u) ^1);
	EXPECT_EQ(K4::add_T<uint32_t>((2u << 8u) ^ 2, 0), (2u << 8u) ^2);
	EXPECT_EQ(K4::add_T<uint32_t>((3u << 8u) ^ 3, 0), (3u << 8u) ^3);
	

	EXPECT_EQ(K4::add_T<uint32_t>(4, 1), 1);
	EXPECT_EQ(K4::add_T<uint32_t>(0, 1), 1);
	EXPECT_EQ(K4::add_T<uint32_t>(1, 1), 2);
	EXPECT_EQ(K4::add_T<uint32_t>(2, 1), 3);
	EXPECT_EQ(K4::add_T<uint32_t>(3, 1), 0);

	EXPECT_EQ(K4::add_T<uint32_t>((4u << 8u) ^ 4, 1), (0u << 8u) ^ 1);
	EXPECT_EQ(K4::add_T<uint32_t>((0u << 8u) ^ 0, 1), (0u << 8u) ^ 1);
	EXPECT_EQ(K4::add_T<uint32_t>((1u << 8u) ^ 1, 1), (1u << 8u) ^ 2);
	EXPECT_EQ(K4::add_T<uint32_t>((2u << 8u) ^ 2, 1), (2u << 8u) ^ 3);
	EXPECT_EQ(K4::add_T<uint32_t>((3u << 8u) ^ 3, 1), (3u << 8u) ^ 0);


	EXPECT_EQ(K4::add_T<uint64_t>(4, 0), 0);
	EXPECT_EQ(K4::add_T<uint64_t>(0, 0), 0);
	EXPECT_EQ(K4::add_T<uint64_t>(1, 0), 1);
	EXPECT_EQ(K4::add_T<uint64_t>(2, 0), 2);
	EXPECT_EQ(K4::add_T<uint64_t>(3, 0), 3);
	
	EXPECT_EQ(K4::add_T<uint64_t>((4u << 8u) ^ 4, 0), (0u << 8u) ^0);
	EXPECT_EQ(K4::add_T<uint64_t>((0u << 8u) ^ 0, 0), (0u << 8u) ^0);
	EXPECT_EQ(K4::add_T<uint64_t>((1u << 8u) ^ 1, 0), (1u << 8u) ^1);
	EXPECT_EQ(K4::add_T<uint64_t>((2u << 8u) ^ 2, 0), (2u << 8u) ^2);
	EXPECT_EQ(K4::add_T<uint64_t>((3u << 8u) ^ 3, 0), (3u << 8u) ^3);
	

	EXPECT_EQ(K4::add_T<uint64_t>(4, 1), 1);
	EXPECT_EQ(K4::add_T<uint64_t>(0, 1), 1);
	EXPECT_EQ(K4::add_T<uint64_t>(1, 1), 2);
	EXPECT_EQ(K4::add_T<uint64_t>(2, 1), 3);
	EXPECT_EQ(K4::add_T<uint64_t>(3, 1), 0);

	EXPECT_EQ(K4::add_T<uint64_t>((4u << 8u) ^ 4, 1), (0u << 8u) ^ 1);
	EXPECT_EQ(K4::add_T<uint64_t>((0u << 8u) ^ 0, 1), (0u << 8u) ^ 1);
	EXPECT_EQ(K4::add_T<uint64_t>((1u << 8u) ^ 1, 1), (1u << 8u) ^ 2);
	EXPECT_EQ(K4::add_T<uint64_t>((2u << 8u) ^ 2, 1), (2u << 8u) ^ 3);
	EXPECT_EQ(K4::add_T<uint64_t>((3u << 8u) ^ 3, 1), (3u << 8u) ^ 0);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
