#include <gtest/gtest.h>
#include <iostream>
#include <stdint.h>

#include "container/fq_vector.h"
#include "container/fq_packed_vector.h"
#include "simd/simd.h"

using ::testing::InitGoogleTest;
using ::testing::Test;

TEST(test, simple) {
	using K3 = FqPackedVector<uint64_t, 100, 3>;
	// using K5 = kAryPackedContainer_T<uint64_t, 100, 5>;
	auto t31 = K3{};
	auto t32 = K3{};
	auto t33 = K3{};
	K3::add_only_weight_partly<0, 2>(t33, t31, t32);
	
}

constexpr uint32_t n = 127;
using K4 = FqNonPackedVector<uint8_t, n, 4>;
using K5 = FqNonPackedVector<uint8_t, n, 5>;
using K7 = FqNonPackedVector<uint8_t, n, 7>;

/// A field for which no optimized implementation exists
using KGeneric = FqNonPackedVector<uint8_t, n, 9>;

#define NR_TESTS (1u << 4u)

#define NAME uint8_K4
#define PRIME 4
#define T uint8_t
#define K FqNonPackedVector<T, n, PRIME>
#include "test_fqvector.h"
#undef PRIME
#undef T
#undef K
#undef NAME

#define NAME uint64_K4
#define PRIME 4
#define T uint64_t
#define K kAryContainer_T<T, n, PRIME>
#include "test_fqvector.h"
#undef PRIME
#undef T
#undef K
#undef NAME

#define NAME uint8_K11
#define PRIME 11
#define T uint8_t
#define K kAryContainer_T<T, n, PRIME>
#include "test_fqvector.h"
#undef PRIME
#undef T
#undef K
#undef NAME

#define NAME uint8_K255
#define PRIME 255
#define T uint8_t
#define K kAryContainer_T<T, n, PRIME>
#include "test_fqvector.h"
#undef PRIME
#undef T
#undef K
#undef NAME








TEST(FGeneric, mod_T) {
	EXPECT_EQ(KGeneric::mod_T<uint32_t>(4), 4);
	EXPECT_EQ(KGeneric::mod_T<uint32_t>(0), 0);
	EXPECT_EQ(KGeneric::mod_T<uint32_t>(1), 1);
	EXPECT_EQ(KGeneric::mod_T<uint32_t>(2), 2);
	EXPECT_EQ(KGeneric::mod_T<uint32_t>(3), 3);
}


TEST(F4, random) {
	K4 t = K4();
	t.random();
	t.print();
	for (uint32_t i = 0; i < n; i++){
    	EXPECT_LE(t.get(i), 4);
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
		t1.set(rng(), i);
	}

	K4::mod(t2, t1);
	for (uint32_t i = 0; i < n; i++){
		EXPECT_LE(t2.get(i), 4);
	}
}

TEST(F4, add) {
	K4 t1 = K4();
	K4 t2 = K4();
	K4 t3 = K4();
	t3.zero();

	t1.random();
	t2.random();

	K4::add(t3, t1, t2);
	for (uint32_t i = 0; i < n; i++) {
		EXPECT_EQ(t3.get(i), (t1.get(i) + t2.get(i)) % 4);
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

	K4::mul(t3, t1, t2);
	for (uint32_t i = 0; i < n; i++){
		EXPECT_EQ(t3.get(i), (t1.get(i) * t2.get(i)) % 4);
	}
}

TEST(F4, scalar) {
	K4 t1 = K4();
	K4 t3 = K4();
	t3.zero();
	t1.random();
	const uint8_t t2 = rand() % 4;
	K4::scalar<uint8_t >(t3, t1, t2);
	for (uint32_t i = 0; i < n; i++) {
		EXPECT_EQ(t3.get(i), (t1.get(i) * t2) % 4);
	}
}




TEST(F7, mod_T) {
	const auto tmp = K7::mod_T<uint32_t>( 0);
	EXPECT_EQ(tmp, 0);
	EXPECT_EQ(K7::mod_T<uint32_t>( 1), 1);
	EXPECT_EQ(K7::mod_T<uint32_t>( 2), 2);
	EXPECT_EQ(K7::mod_T<uint32_t>( 3), 3);
	EXPECT_EQ(K7::mod_T<uint32_t>( 4), 4);
	EXPECT_EQ(K7::mod_T<uint32_t>( 5), 5);
	EXPECT_EQ(K7::mod_T<uint32_t>( 6), 6);
	EXPECT_EQ(K7::mod_T<uint32_t>( 7), 0);
	EXPECT_EQ(K7::mod_T<uint32_t>( 8), 1);
	EXPECT_EQ(K7::mod_T<uint32_t>( 9), 2);
	EXPECT_EQ(K7::mod_T<uint32_t>(10), 3);
	EXPECT_EQ(K7::mod_T<uint32_t>(11), 4);
	EXPECT_EQ(K7::mod_T<uint32_t>(12), 5);
	EXPECT_EQ(K7::mod_T<uint32_t>(13), 6);
	EXPECT_EQ(K7::mod_T<uint32_t>(14), 0);

	/// mult edge cases
	EXPECT_EQ(K7::mod_T<uint32_t>(49), 0);
	EXPECT_EQ(K7::mod_T<uint32_t>(48), 6);
}

TEST(F7, add_T) {
	EXPECT_EQ(K7::add_T<uint32_t>(7, 0), 0);
	EXPECT_EQ(K7::add_T<uint32_t>(0, 0), 0);
	EXPECT_EQ(K7::add_T<uint32_t>(1, 0), 1);
	EXPECT_EQ(K7::add_T<uint32_t>(2, 0), 2);
	EXPECT_EQ(K7::add_T<uint32_t>(3, 0), 3);
	EXPECT_EQ(K7::add_T<uint32_t>(4, 0), 4);
	EXPECT_EQ(K7::add_T<uint32_t>(5, 0), 5);
	EXPECT_EQ(K7::add_T<uint32_t>(6, 0), 6);

	const auto tmp = K7::add_T<uint32_t>((7u << 8u) ^ 7, 0);
	EXPECT_EQ(tmp, (0u << 8u) ^ 0);
	EXPECT_EQ(K7::add_T<uint32_t>((0u << 8u) ^ 0, 0), (0u << 8u) ^ 0);
	EXPECT_EQ(K7::add_T<uint32_t>((1u << 8u) ^ 1, 0), (1u << 8u) ^ 1);
	EXPECT_EQ(K7::add_T<uint32_t>((2u << 8u) ^ 2, 0), (2u << 8u) ^ 2);
	EXPECT_EQ(K7::add_T<uint32_t>((3u << 8u) ^ 3, 0), (3u << 8u) ^ 3);


	EXPECT_EQ(K7::add_T<uint32_t>(7, 1), 1);
	EXPECT_EQ(K7::add_T<uint32_t>(0, 1), 1);
	EXPECT_EQ(K7::add_T<uint32_t>(1, 1), 2);
	EXPECT_EQ(K7::add_T<uint32_t>(2, 1), 3);
	EXPECT_EQ(K7::add_T<uint32_t>(3, 1), 4);
	EXPECT_EQ(K7::add_T<uint32_t>(4, 1), 5);
	EXPECT_EQ(K7::add_T<uint32_t>(5, 1), 6);
	EXPECT_EQ(K7::add_T<uint32_t>(6, 1), 0);

	EXPECT_EQ(K7::add_T<uint32_t>((7u << 8u) ^ 7, 1), (0u << 8u) ^ 1);
	EXPECT_EQ(K7::add_T<uint32_t>((0u << 8u) ^ 0, 1), (0u << 8u) ^ 1);
	EXPECT_EQ(K7::add_T<uint32_t>((1u << 8u) ^ 1, 1), (1u << 8u) ^ 2);
	EXPECT_EQ(K7::add_T<uint32_t>((2u << 8u) ^ 2, 1), (2u << 8u) ^ 3);
	EXPECT_EQ(K7::add_T<uint32_t>((3u << 8u) ^ 3, 1), (3u << 8u) ^ 4);


	EXPECT_EQ(K7::add_T<uint64_t>(7, 0), 0);
	EXPECT_EQ(K7::add_T<uint64_t>(0, 0), 0);
	EXPECT_EQ(K7::add_T<uint64_t>(1, 0), 1);
	EXPECT_EQ(K7::add_T<uint64_t>(2, 0), 2);
	EXPECT_EQ(K7::add_T<uint64_t>(3, 0), 3);

	EXPECT_EQ(K7::add_T<uint64_t>((7u << 8u) ^ 7, 0), (0u << 8u) ^0);
	EXPECT_EQ(K7::add_T<uint64_t>((0u << 8u) ^ 0, 0), (0u << 8u) ^0);
	EXPECT_EQ(K7::add_T<uint64_t>((1u << 8u) ^ 1, 0), (1u << 8u) ^1);
	EXPECT_EQ(K7::add_T<uint64_t>((2u << 8u) ^ 2, 0), (2u << 8u) ^2);
	EXPECT_EQ(K7::add_T<uint64_t>((3u << 8u) ^ 3, 0), (3u << 8u) ^3);

	EXPECT_EQ(K7::add_T<uint64_t>(7, 1), 1);
	EXPECT_EQ(K7::add_T<uint64_t>(0, 1), 1);
	EXPECT_EQ(K7::add_T<uint64_t>(1, 1), 2);
	EXPECT_EQ(K7::add_T<uint64_t>(2, 1), 3);
	EXPECT_EQ(K7::add_T<uint64_t>(3, 1), 4);

	EXPECT_EQ(K7::add_T<uint64_t>((7u << 8u) ^ 7, 1), (0u << 8u) ^ 1);
	EXPECT_EQ(K7::add_T<uint64_t>((0u << 8u) ^ 0, 1), (0u << 8u) ^ 1);
	EXPECT_EQ(K7::add_T<uint64_t>((1u << 8u) ^ 1, 1), (1u << 8u) ^ 2);
	EXPECT_EQ(K7::add_T<uint64_t>((2u << 8u) ^ 2, 1), (2u << 8u) ^ 3);
	EXPECT_EQ(K7::add_T<uint64_t>((3u << 8u) ^ 3, 1), (3u << 8u) ^ 4);
}

TEST(F7, sub_T) {
	EXPECT_EQ(K7::sub_T<uint64_t>(4, 0), 4);
	EXPECT_EQ(K7::sub_T<uint64_t>(0, 0), 0);
	EXPECT_EQ(K7::sub_T<uint64_t>(1, 0), 1);
	EXPECT_EQ(K7::sub_T<uint64_t>(2, 0), 2);
	EXPECT_EQ(K7::sub_T<uint64_t>(3, 0), 3);

	EXPECT_EQ(K7::sub_T<uint64_t>(4, 1), 3);
	EXPECT_EQ(K7::sub_T<uint64_t>(0, 1), 6);
	EXPECT_EQ(K7::sub_T<uint64_t>(1, 1), 0);
	EXPECT_EQ(K7::sub_T<uint64_t>(2, 1), 1);
	EXPECT_EQ(K7::sub_T<uint64_t>(3, 1), 2);

	EXPECT_EQ(K7::sub_T<uint32_t>(4, 0), 4);
	EXPECT_EQ(K7::sub_T<uint32_t>(0, 0), 0);
	EXPECT_EQ(K7::sub_T<uint32_t>(1, 0), 1);
	EXPECT_EQ(K7::sub_T<uint32_t>(2, 0), 2);
	EXPECT_EQ(K7::sub_T<uint32_t>(3, 0), 3);

	EXPECT_EQ(K7::sub_T<uint32_t>(4, 1), 3);
	EXPECT_EQ(K7::sub_T<uint32_t>(0, 1), 6);
	EXPECT_EQ(K7::sub_T<uint32_t>(1, 1), 0);
	EXPECT_EQ(K7::sub_T<uint32_t>(2, 1), 1);
	EXPECT_EQ(K7::sub_T<uint32_t>(3, 1), 2);

	EXPECT_EQ(K7::sub_T<uint32_t>(4, 2), 2);
	EXPECT_EQ(K7::sub_T<uint32_t>(0, 2), 5);
	EXPECT_EQ(K7::sub_T<uint32_t>(1, 2), 6);
	EXPECT_EQ(K7::sub_T<uint32_t>(2, 2), 0);
	EXPECT_EQ(K7::sub_T<uint32_t>(3, 2), 1);

	EXPECT_EQ(K7::sub_T<uint8_t>(4, 0), 4);
	EXPECT_EQ(K7::sub_T<uint8_t>(0, 0), 0);
	EXPECT_EQ(K7::sub_T<uint8_t>(1, 0), 1);
	EXPECT_EQ(K7::sub_T<uint8_t>(2, 0), 2);
	EXPECT_EQ(K7::sub_T<uint8_t>(3, 0), 3);

	EXPECT_EQ(K7::sub_T<uint8_t>(4, 1), 3);
	EXPECT_EQ(K7::sub_T<uint8_t>(0, 1), 6);
	EXPECT_EQ(K7::sub_T<uint8_t>(1, 1), 0);
	EXPECT_EQ(K7::sub_T<uint8_t>(2, 1), 1);
	EXPECT_EQ(K7::sub_T<uint8_t>(3, 1), 2);

	EXPECT_EQ(K7::sub_T<uint8_t>(4, 2), 2);
	EXPECT_EQ(K7::sub_T<uint8_t>(0, 2), 5);
	EXPECT_EQ(K7::sub_T<uint8_t>(1, 2), 6);
	EXPECT_EQ(K7::sub_T<uint8_t>(2, 2), 0);
	EXPECT_EQ(K7::sub_T<uint8_t>(3, 2), 1);


	EXPECT_EQ(K7::sub_T<uint32_t>((7u << 8u) ^ 7, 0), (0u << 8u) ^0);
	EXPECT_EQ(K7::sub_T<uint32_t>((0u << 8u) ^ 0, 0), (0u << 8u) ^0);
	EXPECT_EQ(K7::sub_T<uint32_t>((1u << 8u) ^ 1, 0), (1u << 8u) ^1);
	EXPECT_EQ(K7::sub_T<uint32_t>((2u << 8u) ^ 2, 0), (2u << 8u) ^2);
	EXPECT_EQ(K7::sub_T<uint32_t>((3u << 8u) ^ 3, 0), (3u << 8u) ^3);

	EXPECT_EQ(K7::sub_T<uint32_t>((7u << 8u) ^ 7, (1u << 8u) ^ 0), (6u << 8u) ^ 0);
	EXPECT_EQ(K7::sub_T<uint32_t>((0u << 8u) ^ 0, (1u << 8u) ^ 6), (6u << 8u) ^ 1);
	EXPECT_EQ(K7::sub_T<uint32_t>((1u << 8u) ^ 1, (1u << 8u) ^ 0), (0u << 8u) ^ 1);
	EXPECT_EQ(K7::sub_T<uint32_t>((2u << 8u) ^ 2, (1u << 8u) ^ 0), (1u << 8u) ^ 2);
	EXPECT_EQ(K7::sub_T<uint32_t>((3u << 8u) ^ 3, (1u << 8u) ^ 0), (2u << 8u) ^ 3);
}

TEST(F7, mul_T) {
	EXPECT_EQ(K7::mul_T<uint32_t>(4, 0), 0);
	EXPECT_EQ(K7::mul_T<uint32_t>(0, 0), 0);
	EXPECT_EQ(K7::mul_T<uint32_t>(1, 0), 0);
	EXPECT_EQ(K7::mul_T<uint32_t>(2, 0), 0);
	EXPECT_EQ(K7::mul_T<uint32_t>(3, 0), 0);

	EXPECT_EQ(K7::mul_T<uint32_t>(4, 1), 4);
	EXPECT_EQ(K7::mul_T<uint32_t>(0, 1), 0);
	EXPECT_EQ(K7::mul_T<uint32_t>(1, 1), 1);
	EXPECT_EQ(K7::mul_T<uint32_t>(2, 1), 2);
	EXPECT_EQ(K7::mul_T<uint32_t>(3, 1), 3);

	EXPECT_EQ(K7::mul_T<uint32_t>(4, 2), 1);
	EXPECT_EQ(K7::mul_T<uint32_t>(0, 2), 0);
	EXPECT_EQ(K7::mul_T<uint32_t>(1, 2), 2);
	EXPECT_EQ(K7::mul_T<uint32_t>(2, 2), 4);
	EXPECT_EQ(K7::mul_T<uint32_t>(3, 2), 6);

	EXPECT_EQ(K7::mul_T<uint32_t>(4, 3), 5);
	EXPECT_EQ(K7::mul_T<uint32_t>(0, 3), 0);
	EXPECT_EQ(K7::mul_T<uint32_t>(1, 3), 3);
	EXPECT_EQ(K7::mul_T<uint32_t>(2, 3), 6);
	EXPECT_EQ(K7::mul_T<uint32_t>(3, 3), 2);
}

TEST(F7, add256_T) {
	const uint8x32_t t0 = uint8x32_t::set1(0);
	const uint8x32_t t1 = uint8x32_t::set1(1);
	const uint8x32_t t2 = uint8x32_t::set1(2);
	const uint8x32_t t3 = uint8x32_t::set1(3);
	const uint8x32_t t4 = uint8x32_t::set1(4);
	const uint8x32_t t5 = uint8x32_t::set1(5);
	const uint8x32_t t6 = uint8x32_t::set1(6);

	EXPECT_EQ(K7::add256_T(t0, t0), t0);
	EXPECT_EQ(K7::add256_T(t0, t1), t1);
	EXPECT_EQ(K7::add256_T(t0, t2), t2);
	EXPECT_EQ(K7::add256_T(t0, t3), t3);
	EXPECT_EQ(K7::add256_T(t1, t2), t3);
	EXPECT_EQ(K7::add256_T(t1, t3), t4);
	EXPECT_EQ(K7::add256_T(t2, t1), t3);
	EXPECT_EQ(K7::add256_T(t3, t1), t4);
	EXPECT_EQ(K7::add256_T(t1, t1), t2);
	EXPECT_EQ(K7::add256_T(t2, t2), t4);
	EXPECT_EQ(K7::add256_T(t3, t3), t6);
	EXPECT_EQ(K7::add256_T(t3, t1), t4);
	EXPECT_EQ(K7::add256_T(t3, t2), t5);
	EXPECT_EQ(K7::add256_T(t2, t3), t5);

	EXPECT_EQ(K7::add256_T(t3, t4), t0);
	EXPECT_EQ(K7::add256_T(t4, t5), t2);
}

TEST(F7, sub256_T) {
	const uint8x32_t t0 = uint8x32_t::set1(0);
	const uint8x32_t t1 = uint8x32_t::set1(1);
	const uint8x32_t t2 = uint8x32_t::set1(2);
	const uint8x32_t t3 = uint8x32_t::set1(3);
	const uint8x32_t t4 = uint8x32_t::set1(4);
	const uint8x32_t t5 = uint8x32_t::set1(5);
	const uint8x32_t t6 = uint8x32_t::set1(6);

	EXPECT_EQ(K7::sub256_T(t0, t0), t0);
	EXPECT_EQ(K7::sub256_T(t0, t1), t6);
	EXPECT_EQ(K7::sub256_T(t0, t2), t5);
	EXPECT_EQ(K7::sub256_T(t0, t3), t4);
	EXPECT_EQ(K7::sub256_T(t1, t2), t6);
	EXPECT_EQ(K7::sub256_T(t1, t3), t5);
	EXPECT_EQ(K7::sub256_T(t2, t1), t1);
	EXPECT_EQ(K7::sub256_T(t3, t1), t2);
	EXPECT_EQ(K7::sub256_T(t1, t1), t0);
	EXPECT_EQ(K7::sub256_T(t2, t2), t0);
	EXPECT_EQ(K7::sub256_T(t3, t3), t0);
	EXPECT_EQ(K7::sub256_T(t3, t2), t1);
	EXPECT_EQ(K7::sub256_T(t2, t3), t6);
}

TEST(F7, mul256_T) {
	const uint8x32_t t0 = uint8x32_t::set1(0);
	const uint8x32_t t1 = uint8x32_t::set1(1);
	const uint8x32_t t2 = uint8x32_t::set1(2);
	const uint8x32_t t3 = uint8x32_t::set1(3);
	const uint8x32_t t4 = uint8x32_t::set1(4);
	const uint8x32_t t5 = uint8x32_t::set1(5);
	const uint8x32_t t6 = uint8x32_t::set1(6);

	EXPECT_EQ(K7::mul256_T(t0, t0), t0);
	EXPECT_EQ(K7::mul256_T(t0, t1), t0);
	EXPECT_EQ(K7::mul256_T(t0, t2), t0);
	EXPECT_EQ(K7::mul256_T(t0, t3), t0);
	EXPECT_EQ(K7::mul256_T(t1, t2), t2);
	EXPECT_EQ(K7::mul256_T(t1, t3), t3);
	EXPECT_EQ(K7::mul256_T(t1, t4), t4);
	EXPECT_EQ(K7::mul256_T(t1, t5), t5);
	EXPECT_EQ(K7::mul256_T(t1, t6), t6);
	EXPECT_EQ(K7::mul256_T(t2, t1), t2);
	EXPECT_EQ(K7::mul256_T(t3, t1), t3);
	EXPECT_EQ(K7::mul256_T(t1, t1), t1);
	EXPECT_EQ(K7::mul256_T(t2, t2), t4);
	EXPECT_EQ(K7::mul256_T(t3, t3), t2);
	EXPECT_EQ(K7::mul256_T(t3, t2), t6);
	EXPECT_EQ(K7::mul256_T(t2, t3), t6);

	EXPECT_EQ(K7::mul256_T(t4, t4), t2);
	EXPECT_EQ(K7::mul256_T(t5, t5), t4);
	EXPECT_EQ(K7::mul256_T(t6, t6), t1);
}



TEST(F5, mod_T) {
	const auto tmp = K5::mod_T<uint32_t>( 0);
	EXPECT_EQ(tmp, 0);
	EXPECT_EQ(K5::mod_T<uint32_t>( 1), 1);
	EXPECT_EQ(K5::mod_T<uint32_t>( 2), 2);
	EXPECT_EQ(K5::mod_T<uint32_t>( 3), 3);
	EXPECT_EQ(K5::mod_T<uint32_t>( 4), 4);
	EXPECT_EQ(K5::mod_T<uint32_t>( 5), 0);
	EXPECT_EQ(K5::mod_T<uint32_t>( 6), 1);
	EXPECT_EQ(K5::mod_T<uint32_t>( 7), 2);
	EXPECT_EQ(K5::mod_T<uint32_t>( 8), 3);
	EXPECT_EQ(K5::mod_T<uint32_t>( 9), 4);
	EXPECT_EQ(K5::mod_T<uint32_t>(10), 0);
	EXPECT_EQ(K5::mod_T<uint32_t>(11), 1);

	/// mult edge cases
	EXPECT_EQ(K5::mod_T<uint32_t>(25), 0);
	EXPECT_EQ(K5::mod_T<uint32_t>(26), 1);
	EXPECT_EQ(K5::mod_T<uint32_t>(24), 4);
}

TEST(F5, add_T) {
	EXPECT_EQ(K5::add_T<uint32_t>(0, 0), 0);
	EXPECT_EQ(K5::add_T<uint32_t>(1, 0), 1);
	EXPECT_EQ(K5::add_T<uint32_t>(2, 0), 2);
	EXPECT_EQ(K5::add_T<uint32_t>(3, 0), 3);
	EXPECT_EQ(K5::add_T<uint32_t>(4, 0), 4);
	EXPECT_EQ(K5::add_T<uint32_t>(5, 0), 0);
	EXPECT_EQ(K5::add_T<uint32_t>(6, 0), 1);
	EXPECT_EQ(K5::add_T<uint32_t>(7, 0), 2);
	EXPECT_EQ(K5::add_T<uint32_t>(8, 0), 3);
	EXPECT_EQ(K5::add_T<uint32_t>(9, 0), 4);
	EXPECT_EQ(K5::add_T<uint32_t>(10, 0), 0);

	EXPECT_EQ(K5::add_T<uint32_t>((0u << 8u) ^ 0, 0), (0u << 8u) ^ 0);
	EXPECT_EQ(K5::add_T<uint32_t>((1u << 8u) ^ 1, 0), (1u << 8u) ^ 1);
	EXPECT_EQ(K5::add_T<uint32_t>((2u << 8u) ^ 2, 0), (2u << 8u) ^ 2);
	EXPECT_EQ(K5::add_T<uint32_t>((3u << 8u) ^ 3, 0), (3u << 8u) ^ 3);
	EXPECT_EQ(K5::add_T<uint32_t>((4u << 8u) ^ 4, 0), (4u << 8u) ^ 4);
	EXPECT_EQ(K5::add_T<uint32_t>((5u << 8u) ^ 5, 0), (0u << 8u) ^ 0);


	EXPECT_EQ(K5::add_T<uint32_t>(0, 1), 1);
	EXPECT_EQ(K5::add_T<uint32_t>(1, 1), 2);
	EXPECT_EQ(K5::add_T<uint32_t>(2, 1), 3);
	EXPECT_EQ(K5::add_T<uint32_t>(3, 1), 4);
	EXPECT_EQ(K5::add_T<uint32_t>(4, 1), 0);
	EXPECT_EQ(K5::add_T<uint32_t>(5, 1), 1);
	EXPECT_EQ(K5::add_T<uint32_t>(6, 1), 2);
	EXPECT_EQ(K5::add_T<uint32_t>(7, 1), 3);
	EXPECT_EQ(K5::add_T<uint32_t>(8, 1), 4);
	EXPECT_EQ(K5::add_T<uint32_t>(9, 1), 0);

	EXPECT_EQ(K5::add_T<uint32_t>((0u << 8u) ^ 0, 1), (0u << 8u) ^ 1);
	EXPECT_EQ(K5::add_T<uint32_t>((1u << 8u) ^ 1, 1), (1u << 8u) ^ 2);
	EXPECT_EQ(K5::add_T<uint32_t>((2u << 8u) ^ 2, 1), (2u << 8u) ^ 3);
	EXPECT_EQ(K5::add_T<uint32_t>((3u << 8u) ^ 3, 1), (3u << 8u) ^ 4);
	EXPECT_EQ(K5::add_T<uint32_t>((4u << 8u) ^ 4, 1), (4u << 8u) ^ 0);
	EXPECT_EQ(K5::add_T<uint32_t>((5u << 8u) ^ 5, 1), (0u << 8u) ^ 1);
	EXPECT_EQ(K5::add_T<uint32_t>((6u << 8u) ^ 6, 1), (1u << 8u) ^ 2);
	EXPECT_EQ(K5::add_T<uint32_t>((7u << 8u) ^ 7, 1), (2u << 8u) ^ 3);

	EXPECT_EQ(K5::add_T<uint64_t>(0, 0), 0);
	EXPECT_EQ(K5::add_T<uint64_t>(1, 0), 1);
	EXPECT_EQ(K5::add_T<uint64_t>(2, 0), 2);
	EXPECT_EQ(K5::add_T<uint64_t>(3, 0), 3);
	EXPECT_EQ(K5::add_T<uint64_t>(4, 0), 4);
	EXPECT_EQ(K5::add_T<uint64_t>(5, 0), 0);
	EXPECT_EQ(K5::add_T<uint64_t>(6, 0), 1);
	EXPECT_EQ(K5::add_T<uint64_t>(7, 0), 2);
	EXPECT_EQ(K5::add_T<uint64_t>(8, 0), 3);
	EXPECT_EQ(K5::add_T<uint64_t>(9, 0), 4);
	EXPECT_EQ(K5::add_T<uint64_t>(10, 0), 0);

	EXPECT_EQ(K5::add_T<uint64_t>((0u << 8u) ^ 0, 0), (0u << 8u) ^ 0);
	EXPECT_EQ(K5::add_T<uint64_t>((1u << 8u) ^ 1, 0), (1u << 8u) ^ 1);
	EXPECT_EQ(K5::add_T<uint64_t>((2u << 8u) ^ 2, 0), (2u << 8u) ^ 2);
	EXPECT_EQ(K5::add_T<uint64_t>((3u << 8u) ^ 3, 0), (3u << 8u) ^ 3);
	EXPECT_EQ(K5::add_T<uint64_t>((4u << 8u) ^ 4, 0), (4u << 8u) ^ 4);
	EXPECT_EQ(K5::add_T<uint64_t>((5u << 8u) ^ 5, 0), (0u << 8u) ^ 0);
	EXPECT_EQ(K5::add_T<uint64_t>((6u << 8u) ^ 6, 0), (1u << 8u) ^ 1);
	EXPECT_EQ(K5::add_T<uint64_t>((7u << 8u) ^ 7, 0), (2u << 8u) ^ 2);
	EXPECT_EQ(K5::add_T<uint64_t>((8u << 8u) ^ 8, 0), (3u << 8u) ^ 3);
	EXPECT_EQ(K5::add_T<uint64_t>((9u << 8u) ^ 9, 0), (4u << 8u) ^ 4);

	EXPECT_EQ(K5::add_T<uint64_t>(0, 1), 1);
	EXPECT_EQ(K5::add_T<uint64_t>(1, 1), 2);
	EXPECT_EQ(K5::add_T<uint64_t>(2, 1), 3);
	EXPECT_EQ(K5::add_T<uint64_t>(3, 1), 4);
	EXPECT_EQ(K5::add_T<uint64_t>(4, 1), 0);
	EXPECT_EQ(K5::add_T<uint64_t>(5, 1), 1);
	EXPECT_EQ(K5::add_T<uint64_t>(6, 1), 2);
	EXPECT_EQ(K5::add_T<uint64_t>(7, 1), 3);
	EXPECT_EQ(K5::add_T<uint64_t>(8, 1), 4);
	EXPECT_EQ(K5::add_T<uint64_t>(9, 1), 0);
	EXPECT_EQ(K5::add_T<uint64_t>(10, 1), 1);
}

TEST(F5, sub_T) {
	EXPECT_EQ(K5::sub_T<uint64_t>(0, 0), 0);
	EXPECT_EQ(K5::sub_T<uint64_t>(1, 0), 1);
	EXPECT_EQ(K5::sub_T<uint64_t>(2, 0), 2);
	EXPECT_EQ(K5::sub_T<uint64_t>(3, 0), 3);
	EXPECT_EQ(K5::sub_T<uint64_t>(4, 0), 4);
	EXPECT_EQ(K5::sub_T<uint64_t>(5, 0), 0);

	EXPECT_EQ(K5::sub_T<uint64_t>(0, 1), 4);
	EXPECT_EQ(K5::sub_T<uint64_t>(1, 1), 0);
	EXPECT_EQ(K5::sub_T<uint64_t>(2, 1), 1);
	EXPECT_EQ(K5::sub_T<uint64_t>(3, 1), 2);
	EXPECT_EQ(K5::sub_T<uint64_t>(4, 1), 3);
	EXPECT_EQ(K5::sub_T<uint64_t>(5, 1), 4);

	EXPECT_EQ(K5::sub_T<uint32_t>(0, 0), 0);
	EXPECT_EQ(K5::sub_T<uint32_t>(1, 0), 1);
	EXPECT_EQ(K5::sub_T<uint32_t>(2, 0), 2);
	EXPECT_EQ(K5::sub_T<uint32_t>(3, 0), 3);
	EXPECT_EQ(K5::sub_T<uint32_t>(4, 0), 4);
	EXPECT_EQ(K5::sub_T<uint32_t>(5, 0), 0);

	EXPECT_EQ(K5::sub_T<uint32_t>(0, 1), 4);
	EXPECT_EQ(K5::sub_T<uint32_t>(1, 1), 0);
	EXPECT_EQ(K5::sub_T<uint32_t>(2, 1), 1);
	EXPECT_EQ(K5::sub_T<uint32_t>(3, 1), 2);
	EXPECT_EQ(K5::sub_T<uint32_t>(4, 1), 3);
	EXPECT_EQ(K5::sub_T<uint32_t>(5, 1), 4);

	EXPECT_EQ(K5::sub_T<uint32_t>(0, 2), 3);
	EXPECT_EQ(K5::sub_T<uint32_t>(1, 2), 4);
	EXPECT_EQ(K5::sub_T<uint32_t>(2, 2), 0);
	EXPECT_EQ(K5::sub_T<uint32_t>(3, 2), 1);
	EXPECT_EQ(K5::sub_T<uint32_t>(4, 2), 2);
	EXPECT_EQ(K5::sub_T<uint32_t>(5, 2), 3);

	EXPECT_EQ(K5::sub_T<uint8_t>(0, 0), 0);
	EXPECT_EQ(K5::sub_T<uint8_t>(1, 0), 1);
	EXPECT_EQ(K5::sub_T<uint8_t>(2, 0), 2);
	EXPECT_EQ(K5::sub_T<uint8_t>(3, 0), 3);
	EXPECT_EQ(K5::sub_T<uint8_t>(4, 0), 4);
	EXPECT_EQ(K5::sub_T<uint8_t>(5, 0), 0);

	EXPECT_EQ(K5::sub_T<uint8_t>(0, 1), 4);
	EXPECT_EQ(K5::sub_T<uint8_t>(1, 1), 0);
	EXPECT_EQ(K5::sub_T<uint8_t>(2, 1), 1);
	EXPECT_EQ(K5::sub_T<uint8_t>(3, 1), 2);
	EXPECT_EQ(K5::sub_T<uint8_t>(4, 1), 3);
	EXPECT_EQ(K5::sub_T<uint8_t>(5, 1), 4);

	EXPECT_EQ(K5::sub_T<uint8_t>(0, 2), 3);
	EXPECT_EQ(K5::sub_T<uint8_t>(1, 2), 4);
	EXPECT_EQ(K5::sub_T<uint8_t>(2, 2), 0);
	EXPECT_EQ(K5::sub_T<uint8_t>(3, 2), 1);
	EXPECT_EQ(K5::sub_T<uint8_t>(4, 2), 2);
	EXPECT_EQ(K5::sub_T<uint8_t>(5, 2), 3);


	EXPECT_EQ(K5::sub_T<uint32_t>((0u << 8u) ^ 0, 0), (0u << 8u) ^ 0);
	EXPECT_EQ(K5::sub_T<uint32_t>((1u << 8u) ^ 1, 0), (1u << 8u) ^ 1);
	EXPECT_EQ(K5::sub_T<uint32_t>((2u << 8u) ^ 2, 0), (2u << 8u) ^ 2);
	EXPECT_EQ(K5::sub_T<uint32_t>((3u << 8u) ^ 3, 0), (3u << 8u) ^ 3);
	EXPECT_EQ(K5::sub_T<uint32_t>((4u << 8u) ^ 4, 0), (4u << 8u) ^ 4);
	EXPECT_EQ(K5::sub_T<uint32_t>((5u << 8u) ^ 5, 0), (0u << 8u) ^ 0);

	EXPECT_EQ(K5::sub_T<uint32_t>((0u << 8u) ^ 0, (1u << 8u) ^ 1), (4u << 8u) ^ 4);
	EXPECT_EQ(K5::sub_T<uint32_t>((1u << 8u) ^ 1, (1u << 8u) ^ 0), (0u << 8u) ^ 1);
	EXPECT_EQ(K5::sub_T<uint32_t>((2u << 8u) ^ 2, (1u << 8u) ^ 0), (1u << 8u) ^ 2);
	EXPECT_EQ(K5::sub_T<uint32_t>((3u << 8u) ^ 3, (1u << 8u) ^ 0), (2u << 8u) ^ 3);
	EXPECT_EQ(K5::sub_T<uint32_t>((4u << 8u) ^ 4, (1u << 8u) ^ 0), (3u << 8u) ^ 4);
	EXPECT_EQ(K5::sub_T<uint32_t>((5u << 8u) ^ 5, (1u << 8u) ^ 0), (4u << 8u) ^ 0);
}

TEST(F5, mul_T) {
	EXPECT_EQ(K5::mul_T<uint32_t>(0, 0), 0);
	EXPECT_EQ(K5::mul_T<uint32_t>(1, 0), 0);
	EXPECT_EQ(K5::mul_T<uint32_t>(2, 0), 0);
	EXPECT_EQ(K5::mul_T<uint32_t>(3, 0), 0);
	EXPECT_EQ(K5::mul_T<uint32_t>(4, 0), 0);
	EXPECT_EQ(K5::mul_T<uint32_t>(5, 0), 0);

	EXPECT_EQ(K5::mul_T<uint32_t>(0, 1), 0);
	EXPECT_EQ(K5::mul_T<uint32_t>(1, 1), 1);
	EXPECT_EQ(K5::mul_T<uint32_t>(2, 1), 2);
	EXPECT_EQ(K5::mul_T<uint32_t>(3, 1), 3);
	EXPECT_EQ(K5::mul_T<uint32_t>(4, 1), 4);
	EXPECT_EQ(K5::mul_T<uint32_t>(5, 1), 0);

	EXPECT_EQ(K5::mul_T<uint32_t>(0, 2), 0);
	EXPECT_EQ(K5::mul_T<uint32_t>(1, 2), 2);
	EXPECT_EQ(K5::mul_T<uint32_t>(2, 2), 4);
	EXPECT_EQ(K5::mul_T<uint32_t>(3, 2), 1);
	EXPECT_EQ(K5::mul_T<uint32_t>(4, 2), 3);
	EXPECT_EQ(K5::mul_T<uint32_t>(5, 2), 0);

	EXPECT_EQ(K5::mul_T<uint32_t>(0, 3), 0);
	EXPECT_EQ(K5::mul_T<uint32_t>(1, 3), 3);
	EXPECT_EQ(K5::mul_T<uint32_t>(2, 3), 1);
	EXPECT_EQ(K5::mul_T<uint32_t>(3, 3), 4);
	EXPECT_EQ(K5::mul_T<uint32_t>(4, 3), 2);
	EXPECT_EQ(K5::mul_T<uint32_t>(5, 3), 0);
}

TEST(F5, add256_T) {
	const uint8x32_t t0 = uint8x32_t::set1(0);
	const uint8x32_t t1 = uint8x32_t::set1(1);
	const uint8x32_t t2 = uint8x32_t::set1(2);
	const uint8x32_t t3 = uint8x32_t::set1(3);
	const uint8x32_t t4 = uint8x32_t::set1(4);
	const uint8x32_t t5 = uint8x32_t::set1(5);

	EXPECT_EQ(K5::add256_T(t0, t0), t0);
	EXPECT_EQ(K5::add256_T(t0, t1), t1);
	EXPECT_EQ(K5::add256_T(t0, t2), t2);
	EXPECT_EQ(K5::add256_T(t0, t3), t3);
	EXPECT_EQ(K5::add256_T(t1, t2), t3);
	EXPECT_EQ(K5::add256_T(t1, t3), t4);
	EXPECT_EQ(K5::add256_T(t2, t1), t3);
	EXPECT_EQ(K5::add256_T(t3, t1), t4);
	EXPECT_EQ(K5::add256_T(t1, t1), t2);
	EXPECT_EQ(K5::add256_T(t2, t2), t4);
	EXPECT_EQ(K5::add256_T(t3, t3), t1);
	EXPECT_EQ(K5::add256_T(t3, t1), t4);
	EXPECT_EQ(K5::add256_T(t3, t2), t0);
	EXPECT_EQ(K5::add256_T(t2, t3), t0);

	EXPECT_EQ(K5::add256_T(t3, t4), t2);
	EXPECT_EQ(K5::add256_T(t4, t5), t4);
}

TEST(F5, sub256_T) {
	const uint8x32_t t0 = uint8x32_t::set1(0);
	const uint8x32_t t1 = uint8x32_t::set1(1);
	const uint8x32_t t2 = uint8x32_t::set1(2);
	const uint8x32_t t3 = uint8x32_t::set1(3);
	const uint8x32_t t4 = uint8x32_t::set1(4);

	EXPECT_EQ(K5::sub256_T(t0, t0), t0);
	EXPECT_EQ(K5::sub256_T(t0, t1), t4);
	EXPECT_EQ(K5::sub256_T(t0, t2), t3);
	EXPECT_EQ(K5::sub256_T(t0, t3), t2);
	EXPECT_EQ(K5::sub256_T(t1, t2), t4);
	EXPECT_EQ(K5::sub256_T(t1, t3), t3);
	EXPECT_EQ(K5::sub256_T(t2, t1), t1);
	EXPECT_EQ(K5::sub256_T(t3, t1), t2);
	EXPECT_EQ(K5::sub256_T(t1, t1), t0);
	EXPECT_EQ(K5::sub256_T(t2, t2), t0);
	EXPECT_EQ(K5::sub256_T(t3, t3), t0);
	EXPECT_EQ(K5::sub256_T(t3, t2), t1);
	EXPECT_EQ(K5::sub256_T(t2, t3), t4);
}

TEST(F5, mul256_T) {
	const uint8x32_t t0 = uint8x32_t::set1(0);
	const uint8x32_t t1 = uint8x32_t::set1(1);
	const uint8x32_t t2 = uint8x32_t::set1(2);
	const uint8x32_t t3 = uint8x32_t::set1(3);
	const uint8x32_t t4 = uint8x32_t::set1(4);
	const uint8x32_t t5 = uint8x32_t::set1(5);

	EXPECT_EQ(K5::mul256_T(t0, t0), t0);
	EXPECT_EQ(K5::mul256_T(t0, t1), t0);
	EXPECT_EQ(K5::mul256_T(t0, t2), t0);
	EXPECT_EQ(K5::mul256_T(t0, t3), t0);
	EXPECT_EQ(K5::mul256_T(t1, t2), t2);
	EXPECT_EQ(K5::mul256_T(t1, t3), t3);
	EXPECT_EQ(K5::mul256_T(t1, t4), t4);
	EXPECT_EQ(K5::mul256_T(t1, t5), t0);
	EXPECT_EQ(K5::mul256_T(t2, t1), t2);
	EXPECT_EQ(K5::mul256_T(t3, t1), t3);
	EXPECT_EQ(K5::mul256_T(t1, t1), t1);
	EXPECT_EQ(K5::mul256_T(t2, t2), t4);
	EXPECT_EQ(K5::mul256_T(t3, t3), t4);
	EXPECT_EQ(K5::mul256_T(t3, t2), t1);
	EXPECT_EQ(K5::mul256_T(t2, t3), t1);
	EXPECT_EQ(K5::mul256_T(t4, t4), t1);
	EXPECT_EQ(K5::mul256_T(t5, t5), t0);
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
