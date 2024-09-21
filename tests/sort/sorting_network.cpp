#include <cstdint>
#include <gtest/gtest.h>


using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

#include "helper.h"
#include "random.h"
#include "sort/sorting_network/common.h"

/// generate rng data
template<typename T>
T *gen_data(const size_t size) {
	T *data = (T *) malloc(sizeof(T) * size);
	ASSERT(data);

	for (size_t i = 0; i < size; ++i) {
		data[i] = rng();
	}

	return data;
}

/// check if `data` is sorted
template<typename T, const bool descending=true>
constexpr bool check_correctness(const T *data, const uint32_t n) {
	if (n == 1) {
		return true;
	}

	if constexpr (descending) {
		for (uint32_t i = 0; i < (n-1u); ++i) {
			if (data[i] >= data[i + 1]) {
				return false;
			}
		}
	} else {
		for (uint32_t i = 0; i < (n-1u); ++i) {
			if (data[i] <= data[i + 1]) {
				return false;
			}
		}
	}

	return true;
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

TEST(SortingNetwork, int64x8_t) {
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

TEST(SortingNetwork, djb_int32x16_t) {
	__m256i z1 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	__m256i z2 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);

	__m256i a1 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	__m256i a2 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
	sortingnetwork_djbsort_i32x16(a2, a1);
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
	c &= _mm256_cmpeq_epi32(a2, z2);
	mask = _mm256_movemask_ps((__m256) c);
	EXPECT_EQ(mask, (1u << 8u) - 1u);

	int32_t d_in[16], d_out[16];
	for (uint32_t i = 0; i < 16; ++i) {
		d_in[i] = rng();
	}

	__m256i t1 = _mm256_loadu_si256((__m256i *)(d_in + 0));
	__m256i t2 = _mm256_loadu_si256((__m256i *)(d_in + 8));
	sortingnetwork_djbsort_i32x16(t1, t2);
	_mm256_storeu_si256((__m256i *)(d_out + 0), t1);
	_mm256_storeu_si256((__m256i *)(d_out + 8), t2);
	for (uint32_t i = 0; i < 15; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}


TEST(SortingNetwork, djb_int32x32_t) {
	int32_t d_in[32], d_out[32];
	for (uint32_t i = 0; i < 32; ++i) {
		d_in[i] = rng();
	}

	__m256i t1 = _mm256_loadu_si256((__m256i *)(d_in +  0));
	__m256i t2 = _mm256_loadu_si256((__m256i *)(d_in +  8));
	__m256i t3 = _mm256_loadu_si256((__m256i *)(d_in + 16));
	__m256i t4 = _mm256_loadu_si256((__m256i *)(d_in + 24));
	sortingnetwork_djbsort_i32x32(t1, t2, t3, t4);
	_mm256_storeu_si256((__m256i *)(d_out +  0), t1);
	_mm256_storeu_si256((__m256i *)(d_out +  8), t2);
	_mm256_storeu_si256((__m256i *)(d_out + 16), t3);
	_mm256_storeu_si256((__m256i *)(d_out + 24), t4);
	for (uint32_t i = 0; i < 31; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}

TEST(SortingNetwork, uint8x16_t) {
	uint8_t d_in[16], d_out[16];
	for (uint32_t i = 0; i < 2; ++i) {
		((uint64_t *)d_in)[i] = rng();
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

TEST(SortingNetwork, f32xX_t) {
	constexpr size_t size = 16;
	__m256 data[size] = {0};
	float *d = (float *)data;
	for (size_t i = 0; i < size * 8; ++i) {
		d[i] = static_cast <float> ((float ) rng()) / static_cast <float> ((uint64_t)-1ull);
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
		d[i] = rng();
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

TEST(SortingNetwork, small_f32xX_t) {
	constexpr size_t size = 16;
	__m256i data[size] = {0};
	auto *d = (float *) data;
	for (size_t i = 0; i < size * 8; ++i) {
		d[i] = static_cast <float> ((float ) rng()) / static_cast <float> ((uint64_t)-1ull);
	}

	for (uint32_t i = 1; i < 8*size; i++) {
		const bool b =sortingnetwork_small_f32(d, i);
		ASSERT_EQ(b, true);
		const bool k = check_correctness<float>(d, i);
		ASSERT_EQ(k, true);
	}
}

TEST(SortingNetwork, int32x128_t) {
	uint32_t data[128] __attribute__((aligned(64)));
	for (uint32_t i = 0; i < 128; ++i) {
		data[i] = rng() % (1u << 31);
	}

	sortingnetwork_sort_i32x128((__m256i *)data);

	for (uint32_t i = 0; i < 127; ++i) {
		EXPECT_LE(data[i], data[i + 1])	;
	}
}

TEST(SortingNetwork, uint32x128_t) {
	uint32_t data[128] __attribute__((aligned(64)));
	for (uint32_t i = 0; i < 128; ++i) {
		data[i] = rng();
	}

	sortingnetwork_sort_u32x128_2((__m256i *)data);

	for (uint32_t i = 0; i < 127; ++i) {
		EXPECT_LE(data[i], data[i + 1])	;
	}
}
#endif

#ifdef USE_AVX512F
TEST(SortingNetwork, avx512_uint64x16_t) {
	uint64_t d_in[16], d_out[16];
	for (uint32_t i = 0; i < 16; ++i) {
		d_in[i] = rng();
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

TEST(SortingNetwork, avx512_int32x16_t) {
	int32_t d_in[16], d_out[16];
	for (uint32_t i = 0; i < 16; ++i) {
		d_in[i] = rng();
	}

	__m512i a = _mm512_loadu_si512((__m512i *)(d_in + 0));
	avx512_sortingnetwork_sort_i32x16(a);
	_mm512_storeu_si512((__m512i *)(d_out + 0), a);
	for (uint32_t i = 0; i < 15; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}

TEST(SortingNetwork, avx512_uint32x16_t) {
	uint32_t d_in[16], d_out[16];
	for (uint32_t i = 0; i < 16; ++i) {
		d_in[i] = rng();
	}

	__m512i a = _mm512_loadu_si512((__m512i *)(d_in + 0));
	avx512_sortingnetwork_sort_u32x16(a);
	_mm512_storeu_si512((__m512i *)(d_out + 0), a);
	for (uint32_t i = 0; i < 15; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}

TEST(SortingNetwork, avx512_f32x16_t) {
	float d_in[16], d_out[16];
	for (size_t i = 0; i < 16; ++i) {
		d_in[i] = static_cast <float> ((float )rng()) / static_cast <float> ((uint64_t)-1ull);
	}

	__m512 a = _mm512_loadu_si512((__m512i *)(d_in + 0));
	avx512_sortingnetwork_sort_f32x16(a);
	_mm512_storeu_si512((__m512i *)(d_out + 0), a);
	for (uint32_t i = 0; i < 15; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}

//TEST(SortingNetwork, avx512_f64x8_t) {
//	double d_in[8], d_out[8];
//	for (size_t i = 0; i < 8; ++i) {
//		d_in[i] = 8-i;
//	}
//
//	__m512d a = _mm512_loadu_pd((__m512d *)(d_in + 0));
//	sortingnetwork_sort_f64x8(a);
//	_mm512_storeu_pd((__m512d *)(d_out + 0), a);
//	for (uint32_t i = 0; i < 8; ++i) {
//		EXPECT_LE(d_out[i], d_out[i+1]);
//	}
//}


TEST(SortingNetwork, avx512_uint32x32_t) {
	uint32_t d_in[32], d_out[32];
	for (uint32_t i = 0; i < 32; ++i) {
		d_in[i] = rng();
	}
	__m512i a = _mm512_loadu_si512((__m512i *)(d_in +  0));
	__m512i b = _mm512_loadu_si512((__m512i *)(d_in + 16));
	avx512_sortingnetwork_sort_u32x32(a, b);
	_mm512_storeu_si512((__m512i *)(d_out +  0), a);
	_mm512_storeu_si512((__m512i *)(d_out + 16), b);
	for (uint32_t i = 0; i < 31; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}

TEST(SortingNetwork, avx512_uint32x48_t) {
	constexpr size_t s = 48;
	uint32_t d_in[s], d_out[s];
	for (uint32_t i = 0; i < s; ++i) {
		d_in[i] = rng();
	}
	__m512i a = _mm512_loadu_si512((__m512i *)(d_in +  0));
	__m512i b = _mm512_loadu_si512((__m512i *)(d_in + 16));
	__m512i c = _mm512_loadu_si512((__m512i *)(d_in + 32));
	avx512_sortingnetwork_sort_u32x48(a, b, c);
	_mm512_storeu_si512((__m512i *)(d_out +  0), a);
	_mm512_storeu_si512((__m512i *)(d_out + 16), b);
	_mm512_storeu_si512((__m512i *)(d_out + 32), c);
	for (uint32_t i = 0; i < s-1; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}

TEST(SortingNetwork, avx512_uint32x64_t) {
	constexpr size_t s = 64;
	uint32_t d_in[s], d_out[s];
	for (uint32_t i = 0; i < s; ++i) {
		d_in[i] = rng();
	}
	__m512i a = _mm512_loadu_si512((__m512i *)(d_in +  0));
	__m512i b = _mm512_loadu_si512((__m512i *)(d_in + 16));
	__m512i c = _mm512_loadu_si512((__m512i *)(d_in + 32));
	__m512i d = _mm512_loadu_si512((__m512i *)(d_in + 48));
	avx512_sortingnetwork_sort_u32x64(a, b, c, d);
	_mm512_storeu_si512((__m512i *)(d_out +  0), a);
	_mm512_storeu_si512((__m512i *)(d_out + 16), b);
	_mm512_storeu_si512((__m512i *)(d_out + 32), c);
	_mm512_storeu_si512((__m512i *)(d_out + 48), d);
	for (uint32_t i = 0; i < s-1; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}

TEST(SortingNetwork, avx512_uint32x80_t) {
	constexpr size_t s = 80;
	uint32_t d_in[s], d_out[s];
	for (uint32_t i = 0; i < s; ++i) {
		d_in[i] = rng();
	}
	__m512i a = _mm512_loadu_si512((__m512i *)(d_in +  0));
	__m512i b = _mm512_loadu_si512((__m512i *)(d_in + 16));
	__m512i c = _mm512_loadu_si512((__m512i *)(d_in + 32));
	__m512i d = _mm512_loadu_si512((__m512i *)(d_in + 48));
	__m512i e = _mm512_loadu_si512((__m512i *)(d_in + 64));
	avx512_sortingnetwork_sort_u32x80(a, b, c, d, e);
	_mm512_storeu_si512((__m512i *)(d_out +  0), a);
	_mm512_storeu_si512((__m512i *)(d_out + 16), b);
	_mm512_storeu_si512((__m512i *)(d_out + 32), c);
	_mm512_storeu_si512((__m512i *)(d_out + 48), d);
	_mm512_storeu_si512((__m512i *)(d_out + 64), e);
	for (uint32_t i = 0; i < s-1; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}

TEST(SortingNetwork, avx512_uint32x96_t) {
	constexpr size_t s = 96;
	uint32_t d_in[s], d_out[s];
	for (uint32_t i = 0; i < s; ++i) {
		d_in[i] = rng();
	}
	__m512i a = _mm512_loadu_si512((__m512i *)(d_in +  0));
	__m512i b = _mm512_loadu_si512((__m512i *)(d_in + 16));
	__m512i c = _mm512_loadu_si512((__m512i *)(d_in + 32));
	__m512i d = _mm512_loadu_si512((__m512i *)(d_in + 48));
	__m512i e = _mm512_loadu_si512((__m512i *)(d_in + 64));
	__m512i f = _mm512_loadu_si512((__m512i *)(d_in + 80));
	avx512_sortingnetwork_sort_u32x96(a, b, c, d, e, f);
	_mm512_storeu_si512((__m512i *)(d_out +  0), a);
	_mm512_storeu_si512((__m512i *)(d_out + 16), b);
	_mm512_storeu_si512((__m512i *)(d_out + 32), c);
	_mm512_storeu_si512((__m512i *)(d_out + 48), d);
	_mm512_storeu_si512((__m512i *)(d_out + 64), e);
	_mm512_storeu_si512((__m512i *)(d_out + 80), f);
	for (uint32_t i = 0; i < s-1; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}

TEST(SortingNetwork, avx512_uint32_small) {
	constexpr size_t s = 96;
	alignas(64) uint32_t d_in[s];

	for (uint32_t j = 16; j <= s; j+=16) {
		for (uint32_t i = 0; i < s; ++i) {
			d_in[i] = rng();
		}

		bool r = avx512_sortingnetwork_small_uint32_t(d_in, s);
		EXPECT_TRUE(r);
		for (uint32_t i = 0; i < j - 1; ++i) {
			EXPECT_LE(d_in[i], d_in[i+1]);
		}
	}

	for (uint32_t j = 16; j <= s; j+=1) {
		for (uint32_t i = 0; i < s; ++i) {
			d_in[i] = rng();
		}

		bool r = avx512_sortingnetwork_small_uint32_t(d_in, s);
		EXPECT_TRUE(r);
		for (uint32_t i = 0; i < j - 1; ++i) {
			EXPECT_LE(d_in[i], d_in[i+1]);
		}
	}
}

TEST(SortingNetwork, avx512_int32_small) {
	constexpr size_t s = 96;
	alignas(64) int32_t d_in[s];

	for (uint32_t j = 16; j <= s; j+=16) {
		for (uint32_t i = 0; i < s; ++i) {
			d_in[i] = rng();
		}

		bool r = avx512_sortingnetwork_small_int32_t(d_in, s);
		EXPECT_TRUE(r);
		for (uint32_t i = 0; i < j - 1; ++i) {
			EXPECT_LE(d_in[i], d_in[i+1]);
		}
	}

	for (uint32_t j = 16; j <= s; j+=1) {
		for (uint32_t i = 0; i < s; ++i) {
			d_in[i] = rng();
		}

		bool r = avx512_sortingnetwork_small_int32_t(d_in, s);
		EXPECT_TRUE(r);
		for (uint32_t i = 0; i < j - 1; ++i) {
			EXPECT_LE(d_in[i], d_in[i+1]);
		}
	}
}

TEST(SortingNetwork, avx512_float_small) {
	constexpr size_t s = 96;
	alignas(64) int32_t d_in[s];

	for (uint32_t j = 16; j <= s; j+=16) {
		for (uint32_t i = 0; i < s; ++i) {
			d_in[i] = static_cast <float> ((float )rng()) / static_cast <float> ((uint64_t)-1ull);
		}

		bool r = avx512_sortingnetwork_small_int32_t(d_in, s);
		EXPECT_TRUE(r);
		for (uint32_t i = 0; i < j- 1; ++i) {
			EXPECT_LE(d_in[i], d_in[i+1]);
		}
	}

	for (uint32_t j = 16; j <= s; j+=1) {
		for (uint32_t i = 0; i < s; ++i) {
			d_in[i] = static_cast <float> ((float )rng()) / static_cast <float> ((uint64_t)-1ull);
		}

		bool r = avx512_sortingnetwork_small_int32_t(d_in, s);
		EXPECT_TRUE(r);
		for (uint32_t i = 0; i < j- 1; ++i) {
			EXPECT_LE(d_in[i], d_in[i+1]);
		}
	}
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
