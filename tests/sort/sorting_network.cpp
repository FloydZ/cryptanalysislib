#include <cstdint>
#include <gtest/gtest.h>


using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

#include "helper.h"
#include "random.h"
#include "sort/sorting_network/common.h"

template<typename T>
T *gen_data(const size_t size) {
	T *data = (T *) malloc(sizeof(T) * size);
	ASSERT(data);

	for (size_t i = 0; i < size; ++i) {
		data[i] = fastrandombytes_uint64();
	}

	return data;
}


TEST(SortingNetwork, staticSort) {
	constexpr size_t size = 6;
	using T = uint32_t;
	T *data = gen_data<T>(size);
	StaticSort<size> static_sort;
	static_sort(data);

	for (size_t i = 0; i < size - 1; ++i) {
		ASSERT_LE(data[i], data[i + 1]);
	}

	free(data);
}


TEST(SortingNetwork, constexpra) {
	constexpr size_t size = 10;
	using T = uint32_t;
	std::array<T, size> data = {9, 7, 8, 6, 5, 4, 2, 3, 1, 0};
	StaticSort<size> static_sort;
	static_sort(data);

	for (size_t i = 0; i < size - 1; ++i) {
		ASSERT_LE(data[i], data[i + 1]);
	}
}

TEST(SortingNetwork, timsort_constexpr) {
	constexpr size_t size = 10;
	using T = uint32_t;
	std::array<T, size> data = {9, 7, 8, 6, 5, 4, 2, 3, 1, 0};
	constexpr StaticSort<size> static_sort;
	static_sort(data);

	for (size_t i = 0; i < size - 1; ++i) {
		ASSERT_LE(data[i], data[i + 1]);
	}
}

#ifdef USE_AVX2

TEST(SortingNetwork, unt64x8_t) {
	__m256i z1 = _mm256_setr_epi64x(0, 1, 2, 3);
	__m256i z2 = _mm256_setr_epi64x(4, 5, 6, 7);
	const __m256i y1 = z1;
	const __m256i y2 = z2;
	sortingnetwork_sort_i64x8(z1, z2);
	__m256i c = _mm256_cmpeq_epi64(y1, z1);
	int mask = _mm256_movemask_ps((__m256) c);
	EXPECT_EQ(mask, (1u << 8u) - 1u);

	c = _mm256_cmpeq_epi64(y2, z2);
	mask = _mm256_movemask_ps((__m256) c);
	EXPECT_EQ(mask, (1u << 8u) - 1u);
}

// SRC: https://drops.dagstuhl.de/opus/volltexte/2021/13775/pdf/LIPIcs-SEA-2021-3.pdf
TEST(SortingNetwork, uint32x8_t) {
	__m256i z = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

	__m256i a = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	__m256i b = sortingnetwork_sort_u32x8(a);
	__m256i c = _mm256_cmpeq_epi32(a, b);
	int mask = _mm256_movemask_ps((__m256) c);
	EXPECT_EQ(mask, (1u << 8u) - 1u);

	a = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
	b = sortingnetwork_sort_u32x8(a);
	c = _mm256_cmpeq_epi32(b, z);
	mask = _mm256_movemask_ps((__m256) c);
	EXPECT_EQ(mask, (1u << 8u) - 1u);
}

TEST(SortingNetwork, uint32x16_t) {
	__m256i z1 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	__m256i z2 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);

	__m256i a1 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	__m256i a2 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
	sortingnetwork_sort_u32x16(a1, a2);
	__m256i c = _mm256_cmpeq_epi32(a1, z1);
	int mask = _mm256_movemask_ps((__m256) c);
	EXPECT_EQ(mask, (1u << 8u) - 1u);
	c = _mm256_cmpeq_epi32(a2, z2);
	mask = _mm256_movemask_ps((__m256) c);
	EXPECT_EQ(mask, (1u << 8u) - 1u);


	a1 = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
	a2 = _mm256_setr_epi32(15, 14, 13, 12, 11, 10, 9, 8);
	sortingnetwork_sort_u32x16(a1, a2);
	c = _mm256_cmpeq_epi32(a1, z1);
	c = _mm256_cmpeq_epi32(a2, z2);
	mask = _mm256_movemask_ps((__m256) c);
	EXPECT_EQ(mask, (1u << 8u) - 1u);
}

TEST(SortingNetwork, uint8x16_t) {
	uint8_t d_in[16], d_out[16];
	for (uint32_t i = 0; i < 2; ++i) {
		((uint64_t *)d_in)[i] = fastrandombytes_uint64();
	}

	const __m128i insr = _mm_load_si128((__m128i *) d_in);
	const __m128i outr = sortingnetwork_sort_u8x16(insr);
	_mm_store_si128((__m128i *) d_out, outr);
	for (uint32_t i = 0; i < 15; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}

TEST(SortingNetwork, uint8x32_t) {
	const uint8_t datas1[32] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
	const uint8_t datas2[32] = {31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
	uint8_t datas3[32];
	__m128i ins1 = _mm_load_si128((__m128i *)datas2 + 0);
	__m128i ins2 = _mm_load_si128((__m128i *)datas2 + 1);

	sortingnetwork_sort_u8x32(&ins1, &ins2);
	_mm_store_si128((__m128i *)datas3 + 0, ins1);
	_mm_store_si128((__m128i *)datas3 + 1, ins2);
	for (uint32_t i = 0; i < 32; i++) {
		EXPECT_EQ(datas3[i], datas1[i]);
	}
}

TEST(SortingNetwork, int32x128_t) {
	uint32_t data[128];
	for (uint32_t i = 0; i < 128; ++i) {
		data[i] = fastrandombytes_uint64() % (1u << 31);
	}

	sortingnetwork_sort_i32x128((__m256i *)data);

	for (uint32_t i = 0; i < 127; ++i) {
		EXPECT_LE(data[i], data[i + 1])	;
	}
}

TEST(SortingNetwork, uint32x128_t) {
	uint32_t data[128];
	for (uint32_t i = 0; i < 128; ++i) {
		data[i] = fastrandombytes_uint64();
	}

	sortingnetwork_sort_u32x128((__m256i *)data);

	for (uint32_t i = 0; i < 127; ++i) {
		EXPECT_LE(data[i], data[i + 1])	;
	}
}
#endif

#ifdef USE_AVX512F
TEST(SortingNetwork, uint64x16_t) {
	uint64_t d_in[16], d_out[16];
	for (uint32_t i = 0; i < 16; ++i) {
		d_in[i] = fastrandombytes_uint64();
	}

	__m512i a = _mm512_loadu_si512((__m512i *)(d_in + 0));
	__m512i b = _mm512_loadu_si512((__m512i *)(d_in + 8));
	sortingnetwork_sort_u64x16(a, b);
	_mm512_storeu_si512((__m512i *)(d_out + 0), a);
	_mm512_storeu_si512((__m512i *)(d_out + 8), b);
	for (uint32_t i = 0; i < 15; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
