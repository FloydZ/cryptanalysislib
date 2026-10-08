#include <cstdint>
#include <gtest/gtest.h>
#include <iostream>

#include "random.h"
#include "compression/compression.h"

using ::testing::EmptyTestEventListener;
using ::testing::InitGoogleTest;
using ::testing::Test;
using ::testing::TestEventListeners;
using ::testing::TestInfo;
using ::testing::TestPartResult;
using ::testing::UnitTest;

constexpr size_t limit = 1u << 12;

TEST(leb128, uint32_t_increasing) {
	using T = uint32_t;
	auto *in1 = (T 		 *)malloc(limit*sizeof(T));
	auto* out = (uint8_t *)malloc(limit*sizeof(T));
	uint8_t *pou = out;
	uint8_t *p2 = pou;
	for (size_t i = 0; i < limit; ++i) { in1[i] = i; }

	for (uint32_t i = 0; i < limit; i++) {
		const uint32_t n = leb128_encode<T>(out, in1[i]);
		out += n;
	}

	for (uint32_t i = 0; i < limit; i++) {
		const T t = leb128_decode<T>(&pou);
		EXPECT_EQ(in1[i], t);
	}

	free(in1); free(p2);
}

TEST(leb128, uint64_t_increasing) {
	using T = uint64_t;

	auto *in1 = (T 		 *)malloc(limit*sizeof(T));
	auto* out = (uint8_t *)malloc(limit*sizeof(T));
	uint8_t *pou = out;
	uint8_t *p2 = pou;
	for (size_t i = 0; i < limit; ++i) { in1[i] = 2*i; }

	for (uint32_t i = 0; i < limit; i++) {
		const uint32_t n = leb128_encode<T>(out, in1[i]);
		out += n;
	}

	for (uint32_t i = 0; i < limit; i++) {
		const T t = leb128_decode<T>(&pou);
		EXPECT_EQ(in1[i], t);
	}

	free(in1); free(p2);
}

TEST(leb128, signed_round_trip) {
	const int32_t in[] = {0, 1, -1, 127, -128, 300, -300,
	                      std::numeric_limits<int32_t>::min(),
	                      std::numeric_limits<int32_t>::max()};
	for (const int32_t v : in) {
		uint8_t buf[8] = {0};
		const size_t n = leb128_encode<int32_t>(buf, v);
		EXPECT_LE(n, 5u);

		uint8_t *p = buf;
		EXPECT_EQ(v, leb128_decode<int32_t>(&p));
		EXPECT_EQ(n, size_t(p - buf));
	}

	const int8_t w = -5;
	uint8_t buf[2] = {0};
	EXPECT_EQ(2u, leb128_encode<int8_t>(buf, w));
	uint8_t *p = buf;
	EXPECT_EQ(w, leb128_decode<int8_t>(&p));
}

TEST(leb128, skip) {
	// values with 1 to 5 bytes each
	constexpr size_t N = 200;
	uint32_t in[N];
	for (size_t i = 0; i < N; i++) { in[i] = uint32_t(1ull << (i % 32)) + uint32_t(i); }
	uint8_t buf[N * 5 + 8] = {0};
	uint8_t *p = buf;
	for (size_t i = 0; i < N; i++) { p += leb128_encode<uint32_t>(p, in[i]); }

	for (size_t k = 0; k < N; k++) {
		uint8_t *q = (uint8_t *) leb128_skip(buf, k);
		EXPECT_EQ(leb128_decode<uint32_t>(&q), in[k]);
	}
}

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
