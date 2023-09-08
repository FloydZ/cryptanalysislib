#include <gtest/gtest.h>
#include <iostream>
#include <stdint.h>

#include "container/fq_vector.h"
#include "container/fq_packed_vector.h"
#include "simd/simd.h"

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

constexpr uint32_t n = 127;
constexpr uint32_t q = 4;
using K4 = kAryContainer_T<uint8_t, n, q>;
using K7 = kAryContainer_T<uint8_t, n, 7>;

TEST(F4, random) {
	K4 t = K4();
	t.random();
	t.print();
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

TEST(F4, sub_T) {
	EXPECT_EQ(K4::sub_T<uint64_t>(4, 0), 0);
	EXPECT_EQ(K4::sub_T<uint64_t>(0, 0), 0);
	EXPECT_EQ(K4::sub_T<uint64_t>(1, 0), 1);
	EXPECT_EQ(K4::sub_T<uint64_t>(2, 0), 2);
	EXPECT_EQ(K4::sub_T<uint64_t>(3, 0), 3);

	EXPECT_EQ(K4::sub_T<uint64_t>(4, 1), 3);
	EXPECT_EQ(K4::sub_T<uint64_t>(0, 1), 3);
	EXPECT_EQ(K4::sub_T<uint64_t>(1, 1), 0);
	EXPECT_EQ(K4::sub_T<uint64_t>(2, 1), 1);
	EXPECT_EQ(K4::sub_T<uint64_t>(3, 1), 2);

	EXPECT_EQ(K4::sub_T<uint32_t>(4, 0), 0);
	EXPECT_EQ(K4::sub_T<uint32_t>(0, 0), 0);
	EXPECT_EQ(K4::sub_T<uint32_t>(1, 0), 1);
	EXPECT_EQ(K4::sub_T<uint32_t>(2, 0), 2);
	EXPECT_EQ(K4::sub_T<uint32_t>(3, 0), 3);

	EXPECT_EQ(K4::sub_T<uint32_t>(4, 1), 3);
	EXPECT_EQ(K4::sub_T<uint32_t>(0, 1), 3);
	EXPECT_EQ(K4::sub_T<uint32_t>(1, 1), 0);
	EXPECT_EQ(K4::sub_T<uint32_t>(2, 1), 1);
	EXPECT_EQ(K4::sub_T<uint32_t>(3, 1), 2);

	EXPECT_EQ(K4::sub_T<uint32_t>(4, 2), 2);
	EXPECT_EQ(K4::sub_T<uint32_t>(0, 2), 2);
	EXPECT_EQ(K4::sub_T<uint32_t>(1, 2), 3);
	EXPECT_EQ(K4::sub_T<uint32_t>(2, 2), 0);
	EXPECT_EQ(K4::sub_T<uint32_t>(3, 2), 1);

	EXPECT_EQ(K4::sub_T<uint8_t>(4, 0), 0);
	EXPECT_EQ(K4::sub_T<uint8_t>(0, 0), 0);
	EXPECT_EQ(K4::sub_T<uint8_t>(1, 0), 1);
	EXPECT_EQ(K4::sub_T<uint8_t>(2, 0), 2);
	EXPECT_EQ(K4::sub_T<uint8_t>(3, 0), 3);

	EXPECT_EQ(K4::sub_T<uint8_t>(4, 1), 3);
	EXPECT_EQ(K4::sub_T<uint8_t>(0, 1), 3);
	EXPECT_EQ(K4::sub_T<uint8_t>(1, 1), 0);
	EXPECT_EQ(K4::sub_T<uint8_t>(2, 1), 1);
	EXPECT_EQ(K4::sub_T<uint8_t>(3, 1), 2);

	EXPECT_EQ(K4::sub_T<uint8_t>(4, 2), 2);
	EXPECT_EQ(K4::sub_T<uint8_t>(0, 2), 2);
	EXPECT_EQ(K4::sub_T<uint8_t>(1, 2), 3);
	EXPECT_EQ(K4::sub_T<uint8_t>(2, 2), 0);
	EXPECT_EQ(K4::sub_T<uint8_t>(3, 2), 1);

	EXPECT_EQ(K4::sub_T<uint32_t>((4u << 8u) ^ 4, 0), (0u << 8u) ^0);
	EXPECT_EQ(K4::sub_T<uint32_t>((0u << 8u) ^ 0, 0), (0u << 8u) ^0);
	EXPECT_EQ(K4::sub_T<uint32_t>((1u << 8u) ^ 1, 0), (1u << 8u) ^1);
	EXPECT_EQ(K4::sub_T<uint32_t>((2u << 8u) ^ 2, 0), (2u << 8u) ^2);
	EXPECT_EQ(K4::sub_T<uint32_t>((3u << 8u) ^ 3, 0), (3u << 8u) ^3);

	EXPECT_EQ(K4::sub_T<uint32_t>((4u << 8u) ^ 4, (1u << 8u) ^ 0), (3u << 8u) ^ 0);
	EXPECT_EQ(K4::sub_T<uint32_t>((0u << 8u) ^ 0, (1u << 8u) ^ 0), (3u << 8u) ^ 0);
	EXPECT_EQ(K4::sub_T<uint32_t>((1u << 8u) ^ 1, (1u << 8u) ^ 0), (0u << 8u) ^ 1);
	EXPECT_EQ(K4::sub_T<uint32_t>((2u << 8u) ^ 2, (1u << 8u) ^ 0), (1u << 8u) ^ 2);
	EXPECT_EQ(K4::sub_T<uint32_t>((3u << 8u) ^ 3, (1u << 8u) ^ 0), (2u << 8u) ^ 3);
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

TEST(F4, mul_T) {
	EXPECT_EQ(K4::mul_T<uint32_t>(4, 0), 0);
	EXPECT_EQ(K4::mul_T<uint32_t>(0, 0), 0);
	EXPECT_EQ(K4::mul_T<uint32_t>(1, 0), 0);
	EXPECT_EQ(K4::mul_T<uint32_t>(2, 0), 0);
	EXPECT_EQ(K4::mul_T<uint32_t>(3, 0), 0);

	EXPECT_EQ(K4::mul_T<uint32_t>(4, 1), 0);
	EXPECT_EQ(K4::mul_T<uint32_t>(0, 1), 0);
	EXPECT_EQ(K4::mul_T<uint32_t>(1, 1), 1);
	EXPECT_EQ(K4::mul_T<uint32_t>(2, 1), 2);
	EXPECT_EQ(K4::mul_T<uint32_t>(3, 1), 3);

	EXPECT_EQ(K4::mul_T<uint32_t>(4, 2), 0);
	EXPECT_EQ(K4::mul_T<uint32_t>(0, 2), 0);
	EXPECT_EQ(K4::mul_T<uint32_t>(1, 2), 2);
	EXPECT_EQ(K4::mul_T<uint32_t>(2, 2), 0);
	EXPECT_EQ(K4::mul_T<uint32_t>(3, 2), 2);
	
	EXPECT_EQ(K4::mul_T<uint32_t>(4, 3), 0);
	EXPECT_EQ(K4::mul_T<uint32_t>(0, 3), 0);
	EXPECT_EQ(K4::mul_T<uint32_t>(1, 3), 3);
	EXPECT_EQ(K4::mul_T<uint32_t>(2, 3), 2);
	EXPECT_EQ(K4::mul_T<uint32_t>(3, 3), 1);
}


TEST(F4, add256_T) {
	const uint8x32_t t0 = uint8x32_t::set1(0);
	const uint8x32_t t1 = uint8x32_t::set1(1);
	const uint8x32_t t2 = uint8x32_t::set1(2);
	const uint8x32_t t3 = uint8x32_t::set1(3);
	
	EXPECT_EQ(K4::add256_T(t0, t0), t0);
	EXPECT_EQ(K4::add256_T(t0, t1), t1);
	EXPECT_EQ(K4::add256_T(t0, t2), t2);
	EXPECT_EQ(K4::add256_T(t0, t3), t3);
	EXPECT_EQ(K4::add256_T(t1, t2), t3);
	EXPECT_EQ(K4::add256_T(t1, t3), t0);
	EXPECT_EQ(K4::add256_T(t2, t1), t3);
	EXPECT_EQ(K4::add256_T(t3, t1), t0);
	EXPECT_EQ(K4::add256_T(t1, t1), t2);
	EXPECT_EQ(K4::add256_T(t2, t2), t0);
	EXPECT_EQ(K4::add256_T(t3, t3), t2);
	EXPECT_EQ(K4::add256_T(t3, t1), t0);
	EXPECT_EQ(K4::add256_T(t3, t2), t1);
	EXPECT_EQ(K4::add256_T(t2, t3), t1);
}

TEST(F4, mul256_T) {
	const uint8x32_t t0 = uint8x32_t::set1(0);
	const uint8x32_t t1 = uint8x32_t::set1(1);
	const uint8x32_t t2 = uint8x32_t::set1(2);
	const uint8x32_t t3 = uint8x32_t::set1(3);
	
	EXPECT_EQ(K4::mul256_T(t0, t0), t0);
	EXPECT_EQ(K4::mul256_T(t0, t1), t0);
	EXPECT_EQ(K4::mul256_T(t0, t2), t0);
	EXPECT_EQ(K4::mul256_T(t0, t3), t0);
	EXPECT_EQ(K4::mul256_T(t1, t2), t2);
	EXPECT_EQ(K4::mul256_T(t1, t3), t3);
	EXPECT_EQ(K4::mul256_T(t2, t1), t2);
	EXPECT_EQ(K4::mul256_T(t3, t1), t3);
	EXPECT_EQ(K4::mul256_T(t1, t1), t1);
	EXPECT_EQ(K4::mul256_T(t2, t2), t0);
	EXPECT_EQ(K4::mul256_T(t3, t3), t1);
	EXPECT_EQ(K4::mul256_T(t3, t1), t3);
	EXPECT_EQ(K4::mul256_T(t3, t2), t2);
	EXPECT_EQ(K4::mul256_T(t2, t3), t2);
}

TEST(F4, mod) {
	K4 t1 = K4();
	K4 t2 = K4();
	for (uint32_t i = 0; i < n; i++){
		t1.set(i, fastrandombytes_uint64());
	}

	K4::mod(t2, t1);
	for (uint32_t i = 0; i < n; i++){
		EXPECT_LE(t2.get(i), q);
	}
}

TEST(F4, add) {
	K4 t1 = K4();
	K4 t2 = K4();
	K4 t3 = K4();
	t3.zero();

	t1.random();
	t2.random();
	//for (uint32_t i = 0; i < n; i++){
	//	t1.set(i, fastrandombytes_uint64());
	//	t2.set(i, fastrandombytes_uint64());
	//}

	K4::add(t3, t2, t1);
	for (uint32_t i = 0; i < n; i++){
		EXPECT_EQ(t3.get(i), (t1.get(i) + t2.get(i)) % q);
	}
}

TEST(F4, sub) {
	K4 t1 = K4();
	K4 t2 = K4();
	K4 t3 = K4();
	t3.zero();
	t1.random();
	t2.random();


	K4::sub(t3, t1, t2);
	for (uint32_t i = 0; i < n; i++){
		EXPECT_EQ(t3.get(i), K4::sub_T<uint8_t>(t1.get(i), t2.get(i)));
	}
}

TEST(F4, mul) {
	K4 t1 = K4();
	K4 t2 = K4();
	K4 t3 = K4();
	t3.zero();
	t1.random();
	t2.random();

	//for (uint32_t i = 0; i < n; i++){
	//	t1.set(i, fastrandombytes_uint64());
	//	t2.set(i, fastrandombytes_uint64());
	//}

	K4::mul(t3, t1, t2);
	for (uint32_t i = 0; i < n; i++){
		EXPECT_EQ(t3.get(i), (t1.get(i) * t2.get(i)) % q);
	}
}

TEST(F4, scalar) {
	K4 t1 = K4();
	K4 t3 = K4();
	t3.zero();
	t1.random();
	const uint8_t t2 = rand();
	//for (uint32_t i = 0; i < n; i++){
	//	t1.set(i, fastrandombytes_uint64());
	//	t2.set(i, fastrandombytes_uint64());
	//}
	K4::scalar<uint8_t >(t3, t1, t2);
	for (uint32_t i = 0; i < n; i++){
		EXPECT_EQ(t3.get(i), (t1.get(i) * t2) % q);
	}
}




TEST(F7, mod_T) {
	EXPECT_EQ(K7::mod_T<uint32_t>(0), 0);
	EXPECT_EQ(K7::mod_T<uint32_t>(1), 1);
	EXPECT_EQ(K7::mod_T<uint32_t>(2), 2);
	EXPECT_EQ(K7::mod_T<uint32_t>(3), 3);
	EXPECT_EQ(K7::mod_T<uint32_t>(4), 4);
	EXPECT_EQ(K7::mod_T<uint32_t>(5), 5);
	EXPECT_EQ(K7::mod_T<uint32_t>(6), 6);
	EXPECT_EQ(K7::mod_T<uint32_t>(7), 0);
	EXPECT_EQ(K7::mod_T<uint32_t>(8), 1);
}
int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
