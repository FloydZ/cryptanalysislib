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
	__m256i b = a;
	b = sortingnetwork_sort_u32x8(b);
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
	sortingnetwork_sort_u32x16(a2, a1);
	__m256i c = _mm256_cmpeq_epi32(a2, z1);
	int mask = _mm256_movemask_ps((__m256) c);
	EXPECT_EQ(mask, (1u << 8u) - 1u);
	c = _mm256_cmpeq_epi32(a1, z2);
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

TEST(SortingNetwork, f32x16_t) {
	__m256 z1 = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
	__m256 z2 = _mm256_setr_ps(8, 9, 10, 11, 12, 13, 14, 15);
	__m256 a1 = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
	__m256 a2 = _mm256_setr_ps(8, 9, 10, 11, 12, 13, 14, 15);
	sortingnetwork_sort_f32x16(a2, a1);

	__m256 c = _mm256_cmp_ps(a2, z1, 0);
	int mask = _mm256_movemask_ps(c);
	EXPECT_EQ(mask, (1u << 8u) - 1u);
	c = _mm256_cmp_ps(a1, z2, 0);
	mask = _mm256_movemask_ps((__m256) c);
	EXPECT_EQ(mask, (1u << 8u) - 1u);
}

template<typename T>
constexpr bool check_correctness(const T *data, const uint32_t n) {
	for (uint32_t i = 0; i < (n-1u); ++i) {
		if (data[i] > data[i + 1]) {
			return false;
		}
	}

	return true;
}

TEST(SortingNetwork, f32xX_t) {
	constexpr size_t size = 16;
	__m256 data[size] = {0};
	float *d = (float *)data;
	for (size_t i = 0; i < size * 8; ++i) {
		d[i] = static_cast <float> ((float )fastrandombytes_uint64()) / static_cast <float> ((uint64_t)-1ull);
	}

	sortingnetwork_sort_f32x8(data[0]);
	ASSERT_EQ(check_correctness((float *)data, 8), true);
	sortingnetwork_sort_f32x16(data[0], data[1]);
	ASSERT_EQ(check_correctness((float *)data, 16), true);
	sortingnetwork_sort_f32x24(data[0], data[1], data[2]);
	ASSERT_EQ(check_correctness((float *)data, 24), true);
	sortingnetwork_sort_f32x32(data[0], data[1], data[2], data[3]);
	ASSERT_EQ(check_correctness((float *)data, 32), true);
	sortingnetwork_sort_f32x40(data[0], data[1], data[2], data[3], data[4]);
	ASSERT_EQ(check_correctness((float *)data, 40), true);
	sortingnetwork_sort_f32x48(data[0], data[1], data[2], data[3], data[4], data[5]);
	ASSERT_EQ(check_correctness((float *)data, 48), true);
	sortingnetwork_sort_f32x56(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
	ASSERT_EQ(check_correctness((float *)data, 56), true);
	sortingnetwork_sort_f32x64(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
	ASSERT_EQ(check_correctness((float *)data, 64), true);
	sortingnetwork_sort_f32x72(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
	ASSERT_EQ(check_correctness((float *)data, 72), true);
	sortingnetwork_sort_f32x80(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
	ASSERT_EQ(check_correctness((float *) data, 80), true);
	sortingnetwork_sort_f32x88(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10]);
	ASSERT_EQ(check_correctness((float *) data, 88), true);
	sortingnetwork_sort_f32x96(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11]);
	ASSERT_EQ(check_correctness((float *) data, 96), true);
	sortingnetwork_sort_f32x104(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12]);
	ASSERT_EQ(check_correctness((float *) data, 104), true);
	sortingnetwork_sort_f32x112(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13]);
	ASSERT_EQ(check_correctness((float *) data, 112), true);
	sortingnetwork_sort_f32x120(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14]);
	ASSERT_EQ(check_correctness((float *) data, 120), true);
	sortingnetwork_sort_f32x128(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]);
	ASSERT_EQ(check_correctness((float *) data, 128), true);
}

TEST(SortingNetwork, u32xX_t) {
	constexpr size_t size = 16;
	__m256i data[size] = {0};
	auto *d = (uint32_t *) data;
	for (size_t i = 0; i < size * 8; ++i) {
		d[i] = fastrandombytes_uint64();
	}

	data[0] = sortingnetwork_sort_u32x8(data[0]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 8), true);
	sortingnetwork_sort_u32x16(data[0], data[1]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 16), true);
	sortingnetwork_sort_u32x24(data[0], data[1], data[2]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 24), true);
	sortingnetwork_sort_u32x32(data[0], data[1], data[2], data[3]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 32), true);
	sortingnetwork_sort_u32x40(data[0], data[1], data[2], data[3], data[4]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 40), true);
	sortingnetwork_sort_u32x48(data[0], data[1], data[2], data[3], data[4], data[5]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 48), true);
	sortingnetwork_sort_u32x56(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 56), true);
	sortingnetwork_sort_u32x64(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 64), true);
	sortingnetwork_sort_u32x72(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 72), true);
	sortingnetwork_sort_u32x80(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 80), true);
	sortingnetwork_sort_u32x88(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 88), true);
	sortingnetwork_sort_u32x96(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 96), true);
	sortingnetwork_sort_u32x104(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 104), true);
	sortingnetwork_sort_u32x112(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 112), true);
	sortingnetwork_sort_u32x120(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 120), true);
	sortingnetwork_sort_u32x128(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]);
	ASSERT_EQ(check_correctness((uint32_t *) data, 128), true);
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

	sortingnetwork_sort_u32x128_2((__m256i *)data);

	for (uint32_t i = 0; i < 127; ++i) {
		EXPECT_LE(data[i], data[i + 1])	;
	}
}
#endif

// TODO not working
// #ifdef USE_AVX512F
// TEST(SortingNetwork, uint64x16_t) {
// 	uint64_t d_in[16], d_out[16];
// 	for (uint32_t i = 0; i < 16; ++i) {
// 		d_in[i] = fastrandombytes_uint64();
// 	}
//
// 	__m512i a = _mm512_loadu_si512((__m512i *)(d_in + 0));
// 	__m512i b = _mm512_loadu_si512((__m512i *)(d_in + 8));
// 	sortingnetwork_sort_u64x16(a, b);
// 	_mm512_storeu_si512((__m512i *)(d_out + 0), a);
// 	_mm512_storeu_si512((__m512i *)(d_out + 8), b);
// 	for (uint32_t i = 0; i < 15; ++i) {
// 		EXPECT_LE(d_out[i], d_out[i+1]);
// 	}
// }
// #endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
