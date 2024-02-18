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
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
