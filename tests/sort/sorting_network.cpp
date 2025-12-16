#include <cstdint>
#include <cstdlib>
#include <gtest/gtest.h>


using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

#include "helper.h"
#include "random.h"
#include "sort/sorting_network/common.h"

using namespace cryptanalysislib;

/// generate rng data
template<typename T>
T *gen_data(const size_t size) {
	T *data = (T *) malloc(sizeof(T) * size);
	assert(data);

	for (size_t i = 0; i < size; ++i) {
		data[i] = cryptanalysislib::rng();
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
		EXPECT_LE(data[i], data[i + 1]);
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
		EXPECT_LE(data[i], data[i + 1]);
	}
}

TEST(SortingNetwork, timsort_constexpr) {
	constexpr size_t size = 10;
	using T = uint32_t;
	std::array<T, size> data = {9, 7, 8, 6, 5, 4, 2, 3, 1, 0};
	constexpr StaticSort<size> static_sort;
	static_sort(data);

	for (size_t i = 0; i < size - 1; ++i) {
		EXPECT_LE(data[i], data[i + 1]);
	}
}


#ifdef USE_AVX2
TEST(SortingNetwork, uint16x16_t) {
	uint16_t d_in[16], d_out[16];
	for (uint32_t i = 0; i < 16; ++i) {
		d_in[i] = rng(); //i + (i&1)*i*(1u<<8);
	}

	const __m256i in  = _mm256_loadu_si256((const __m256i *) d_in);
	const __m256i out = sortingnetwork_sort_u16x16(in);
	_mm256_storeu_si256((__m256i *)d_out, out);
	for (uint32_t i = 0; i < 15; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}
TEST(SortingNetwork, kv_uint16x16_t) {
	uint16_t k_in[16] __attribute__((aligned(32))), k_out[16] __attribute__((aligned(32)));
	uint16_t v_in[16] __attribute__((aligned(32))), v_out[16] __attribute__((aligned(32)));
	for (uint32_t i = 0; i < 16; ++i) {
		k_in[i] = rng(); //i + (i&1)*i*(1u<<8);
		v_in[i] = rng();
	}

	memcpy(k_out, k_in, 32);
	memcpy(v_out, v_in, 32);
	sortingnetwork_kvsort_u16x16((__m256i *)k_out, (__m256i *)v_out);
	for (uint32_t i = 0; i < 15; ++i) {
		EXPECT_LE(k_out[i], k_out[i+1]);
		uint32_t j = 0;
		for (; j < 15; j++) {
			if (k_out[i] == k_in[j]) { break; }
		}
		EXPECT_EQ(v_out[i], v_in[j]);
	}
}
TEST(SortingNetwork, uint16x32_t) {
	uint16_t d_in[32], d_out[32];
	for (uint32_t i = 0; i < 32; ++i) {
		d_in[i] = rand() & 0xFF;
	}
	__m256i in0 = _mm256_loadu_si256((const __m256i *)(d_in+ 0));
	__m256i in1 = _mm256_loadu_si256((const __m256i *)(d_in+16));
	sortingnetwork_sort_u16x32(in0, in1);
	_mm256_storeu_si256((__m256i *)(d_out + 0), in0);
	_mm256_storeu_si256((__m256i *)(d_out +16), in1);
	for (uint32_t i = 0; i < 31; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}
TEST(SortingNetwork, uint16x64_t) {
	uint16_t d_in[64], d_out[64];
	for (uint32_t i = 0; i < 64; ++i) {
		d_in[i] = rand() & 0xFF;
	}
	__m256i in0 = _mm256_loadu_si256((const __m256i *)(d_in+ 0));
	__m256i in1 = _mm256_loadu_si256((const __m256i *)(d_in+16));
	__m256i in2 = _mm256_loadu_si256((const __m256i *)(d_in+32));
	__m256i in3 = _mm256_loadu_si256((const __m256i *)(d_in+48));
	sortingnetwork_sort_u16x64(in0, in1, in2, in3);
	_mm256_storeu_si256((__m256i *)(d_out + 0), in0);
	_mm256_storeu_si256((__m256i *)(d_out +16), in1);
	_mm256_storeu_si256((__m256i *)(d_out +32), in2);
	_mm256_storeu_si256((__m256i *)(d_out +48), in3);
	for (uint32_t i = 0; i < 63; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}
TEST(SortingNetwork, uint16x128_t) {
	uint16_t d_in[128], d_out[128];
	for (uint32_t i = 0; i < 128; ++i) {
		d_in[i] = rand() & 0xFF;
	}
	__m256i in0 = _mm256_loadu_si256((const __m256i *)(d_in +  0));
	__m256i in1 = _mm256_loadu_si256((const __m256i *)(d_in + 16));
	__m256i in2 = _mm256_loadu_si256((const __m256i *)(d_in + 32));
	__m256i in3 = _mm256_loadu_si256((const __m256i *)(d_in + 48));
	__m256i in4 = _mm256_loadu_si256((const __m256i *)(d_in + 64));
	__m256i in5 = _mm256_loadu_si256((const __m256i *)(d_in + 80));
	__m256i in6 = _mm256_loadu_si256((const __m256i *)(d_in + 96));
	__m256i in7 = _mm256_loadu_si256((const __m256i *)(d_in +112));
	sortingnetwork_sort_u16x128(in0, in1, in2, in3, in4, in5, in6, in7);
	_mm256_storeu_si256((__m256i *)(d_out +   0), in0);
	_mm256_storeu_si256((__m256i *)(d_out +  16), in1);
	_mm256_storeu_si256((__m256i *)(d_out +  32), in2);
	_mm256_storeu_si256((__m256i *)(d_out +  48), in3);
	_mm256_storeu_si256((__m256i *)(d_out +  64), in4);
	_mm256_storeu_si256((__m256i *)(d_out +  80), in5);
	_mm256_storeu_si256((__m256i *)(d_out +  96), in6);
	_mm256_storeu_si256((__m256i *)(d_out + 112), in7);
	for (uint32_t i = 0; i < 127; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}

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

TEST(SortingNetwork, uint8x32_t_) {
	// const uint8_t datas1[32] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
	uint8_t datas2[32]; //{31,30,29,28,27,26,25,24,23,22,21,20,19,18,17,16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0};
	uint8_t datas3[32];
	for (uint32_t i = 0; i < 32; ++i) {
		datas2[i] = rng();
	}
	const __m256i ins1 = _mm256_loadu_si256((__m256i *)datas2 + 0);
	const __m256i ins2 = sortingnetwork_sort_u8x32_(ins1);
	_mm256_storeu_si256(reinterpret_cast<__m256i_u *>(datas3), ins2);
	for (uint32_t i = 1; i < 32; i++) {
		EXPECT_LE(datas3[i-1], datas3[i]);
	}
}

TEST(SortingNetwork, uint8x64_t) {
	uint8_t datas2[64];
	uint8_t datas3[64];
	for (uint32_t i = 0; i < 64; ++i) {
		datas2[i] = rng();
	}
	 __m256i i1 = _mm256_loadu_si256((const __m256i *)(datas2 +  0));
	 __m256i i2 = _mm256_loadu_si256((const __m256i *)(datas2 + 32));

	sortingnetwork_sort_u8x64(i1, i2);
	_mm256_storeu_si256((__m256i_u *)(datas3 +  0), i1);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 32), i2);
	for (uint32_t i = 1; i < 64; i++) {
		EXPECT_LE(datas3[i-1], datas3[i]);
	}
}

TEST(SortingNetwork, uint8x96_t) {
	uint8_t datas2[96];
	uint8_t datas3[96];
	for (uint32_t i = 0; i < 96; ++i) {
		datas2[i] = rng();
	}
	 __m256i i1 = _mm256_loadu_si256((const __m256i *)(datas2 +  0));
	 __m256i i2 = _mm256_loadu_si256((const __m256i *)(datas2 + 32));
	 __m256i i3 = _mm256_loadu_si256((const __m256i *)(datas2 + 64));

	sortingnetwork_sort_u8x96(i1, i2, i3);
	_mm256_storeu_si256((__m256i_u *)(datas3 +  0), i1);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 32), i2);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 64), i3);
	for (uint32_t i = 1; i < 96; i++) {
		EXPECT_LE(datas3[i-1], datas3[i]);
	}
}

TEST(SortingNetwork, uint8x128_t) {
	uint8_t datas2[128] __attribute__((aligned(64)));
	uint8_t datas3[128] __attribute__((aligned(64)));
	for (uint32_t i = 0; i < 128; ++i) {
		datas2[i] = rng();
	}
	 __m256i i1 = _mm256_loadu_si256((const __m256i *)(datas2 +  0));
	 __m256i i2 = _mm256_loadu_si256((const __m256i *)(datas2 + 32));
	 __m256i i3 = _mm256_loadu_si256((const __m256i *)(datas2 + 64));
	 __m256i i4 = _mm256_loadu_si256((const __m256i *)(datas2 + 96));
	sortingnetwork_sort_u8x128(i1, i2, i3, i4);
	_mm256_storeu_si256((__m256i_u *)(datas3 +  0), i1);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 32), i2);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 64), i3);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 96), i4);
	for (uint32_t i = 1; i < 128; i++) {
		EXPECT_LE(datas3[i-1], datas3[i]);
	}
}

//TEST(SortingNetwork, uint8x224_t) {
//	// TODO not finished
//	uint8_t datas2[224] __attribute__((aligned(64)));
//	uint8_t datas3[224] __attribute__((aligned(64)));
//	uint8_t *datas4 = datas3 + 128;
//	for (uint32_t i = 0; i < 224; ++i) {
//		datas2[i] =rng();
//	}
//	 __m256i i1 = _mm256_loadu_si256((const __m256i *)(datas2 +   0));
//	 __m256i i2 = _mm256_loadu_si256((const __m256i *)(datas2 +  32));
//	 __m256i i3 = _mm256_loadu_si256((const __m256i *)(datas2 +  64));
//	 __m256i i4 = _mm256_loadu_si256((const __m256i *)(datas2 +  96));
//	 __m256i i5 = _mm256_loadu_si256((const __m256i *)(datas2 + 128));
//	 __m256i i6 = _mm256_loadu_si256((const __m256i *)(datas2 + 160));
//	 __m256i i7 = _mm256_loadu_si256((const __m256i *)(datas2 + 192));
//	sortingnetwork_sort_u8x224(i1, i2, i3, i4, i5, i6, i7);
//	_mm256_storeu_si256((__m256i_u *)(datas3 +   0), i1);
//	_mm256_storeu_si256((__m256i_u *)(datas3 +  32), i2);
//	_mm256_storeu_si256((__m256i_u *)(datas3 +  64), i3);
//	_mm256_storeu_si256((__m256i_u *)(datas3 +  96), i4);
//	_mm256_storeu_si256((__m256i_u *)(datas3 + 128), i5);
//	_mm256_storeu_si256((__m256i_u *)(datas3 + 160), i6);
//	_mm256_storeu_si256((__m256i_u *)(datas3 + 192), i7);
//	for (uint32_t i = 1; i < 224; i++) {
//		if (datas3[i-1] > datas3[i]) {
//			std::cout << i << std::endl;
//		}
//		EXPECT_LE(datas3[i-1], datas3[i]);
//	}
//}

TEST(SortingNetwork, uint8x256_t) {
	uint8_t datas2[256] __attribute__((aligned(64)));
	uint8_t datas3[256] __attribute__((aligned(64)));
	for (uint32_t i = 0; i < 256; ++i) {
		datas2[i] = rng();
	}
	 __m256i i1 = _mm256_loadu_si256((const __m256i *)(datas2 +   0));
	 __m256i i2 = _mm256_loadu_si256((const __m256i *)(datas2 +  32));
	 __m256i i3 = _mm256_loadu_si256((const __m256i *)(datas2 +  64));
	 __m256i i4 = _mm256_loadu_si256((const __m256i *)(datas2 +  96));
	 __m256i i5 = _mm256_loadu_si256((const __m256i *)(datas2 + 128));
	 __m256i i6 = _mm256_loadu_si256((const __m256i *)(datas2 + 160));
	 __m256i i7 = _mm256_loadu_si256((const __m256i *)(datas2 + 192));
	 __m256i i8 = _mm256_loadu_si256((const __m256i *)(datas2 + 224));
	sortingnetwork_sort_u8x256(i1, i2, i3, i4, i5, i6, i7, i8);
	_mm256_storeu_si256((__m256i_u *)(datas3 +   0), i1);
	_mm256_storeu_si256((__m256i_u *)(datas3 +  32), i2);
	_mm256_storeu_si256((__m256i_u *)(datas3 +  64), i3);
	_mm256_storeu_si256((__m256i_u *)(datas3 +  96), i4);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 128), i5);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 160), i6);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 192), i7);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 224), i8);
	for (uint32_t i = 1; i < 256; i++) {
		EXPECT_LE(datas3[i-1], datas3[i]);
	}
}

TEST(SortingNetwork, uint8x288_t) {
	uint8_t datas2[288] __attribute__((aligned(64)));
	uint8_t datas3[288] __attribute__((aligned(64)));
	rng_seed();
	for (unsigned char & i : datas2) {
		i = rng();
	}
	 __m256i i1 = _mm256_loadu_si256((const __m256i *)(datas2 +   0));
	 __m256i i2 = _mm256_loadu_si256((const __m256i *)(datas2 +  32));
	 __m256i i3 = _mm256_loadu_si256((const __m256i *)(datas2 +  64));
	 __m256i i4 = _mm256_loadu_si256((const __m256i *)(datas2 +  96));
	 __m256i i5 = _mm256_loadu_si256((const __m256i *)(datas2 + 128));
	 __m256i i6 = _mm256_loadu_si256((const __m256i *)(datas2 + 160));
	 __m256i i7 = _mm256_loadu_si256((const __m256i *)(datas2 + 192));
	 __m256i i8 = _mm256_loadu_si256((const __m256i *)(datas2 + 224));
	 __m256i i9 = _mm256_loadu_si256((const __m256i *)(datas2 + 256));
	sortingnetwork_sort_u8x288(i1, i2, i3, i4, i5, i6, i7, i8, i9);
	_mm256_storeu_si256((__m256i_u *)(datas3 +   0), i1);
	_mm256_storeu_si256((__m256i_u *)(datas3 +  32), i2);
	_mm256_storeu_si256((__m256i_u *)(datas3 +  64), i3);
	_mm256_storeu_si256((__m256i_u *)(datas3 +  96), i4);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 128), i5);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 160), i6);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 192), i7);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 224), i8);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 256), i9);
	for (uint32_t i = 1; i < 288; i++) {
		EXPECT_LE(datas3[i-1], datas3[i]);
	}
}

TEST(SortingNetwork, uint8x512_t) {
	uint8_t datas2[512] __attribute__((aligned(64)));
	uint8_t datas3[512] __attribute__((aligned(64)));
	for (uint32_t i = 0; i < 512; ++i) {
		datas2[i] = rng();
	}
	 __m256i  i1 = _mm256_loadu_si256((const __m256i *)(datas2 +   0));
	 __m256i  i2 = _mm256_loadu_si256((const __m256i *)(datas2 +  32));
	 __m256i  i3 = _mm256_loadu_si256((const __m256i *)(datas2 +  64));
	 __m256i  i4 = _mm256_loadu_si256((const __m256i *)(datas2 +  96));
	 __m256i  i5 = _mm256_loadu_si256((const __m256i *)(datas2 + 128));
	 __m256i  i6 = _mm256_loadu_si256((const __m256i *)(datas2 + 160));
	 __m256i  i7 = _mm256_loadu_si256((const __m256i *)(datas2 + 192));
	 __m256i  i8 = _mm256_loadu_si256((const __m256i *)(datas2 + 224));
	 __m256i  i9 = _mm256_loadu_si256((const __m256i *)(datas2 + 256));
	 __m256i i10 = _mm256_loadu_si256((const __m256i *)(datas2 + 288));
	 __m256i i11 = _mm256_loadu_si256((const __m256i *)(datas2 + 320));
	 __m256i i12 = _mm256_loadu_si256((const __m256i *)(datas2 + 352));
	 __m256i i13 = _mm256_loadu_si256((const __m256i *)(datas2 + 384));
	 __m256i i14 = _mm256_loadu_si256((const __m256i *)(datas2 + 416));
	 __m256i i15 = _mm256_loadu_si256((const __m256i *)(datas2 + 448));
	 __m256i i16 = _mm256_loadu_si256((const __m256i *)(datas2 + 480));
	sortingnetwork_sort_u8x512(i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15, i16);
	_mm256_storeu_si256((__m256i_u *)(datas3 +   0), i1);
	_mm256_storeu_si256((__m256i_u *)(datas3 +  32), i2);
	_mm256_storeu_si256((__m256i_u *)(datas3 +  64), i3);
	_mm256_storeu_si256((__m256i_u *)(datas3 +  96), i4);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 128), i5);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 160), i6);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 192), i7);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 224), i8);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 256), i9);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 288), i10);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 320), i11);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 352), i12);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 384), i13);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 416), i14);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 448), i15);
	_mm256_storeu_si256((__m256i_u *)(datas3 + 480), i16);
	for (uint32_t i = 1; i < 512; i++) {
		EXPECT_LE(datas3[i-1], datas3[i]);
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
	EXPECT_EQ(check_correctness((float *)data, 8), true);
	sortingnetwork_sort_f32x16(data[0], data[1]);
	EXPECT_EQ(check_correctness((float *)data, 16), true);
	sortingnetwork_sort_f32x24(data[0], data[1], data[2]);
	EXPECT_EQ(check_correctness((float *)data, 24), true);
	sortingnetwork_sort_f32x32(data[0], data[1], data[2], data[3]);
	EXPECT_EQ(check_correctness((float *)data, 32), true);
	sortingnetwork_sort_f32x40(data[0], data[1], data[2], data[3], data[4]);
	EXPECT_EQ(check_correctness((float *)data, 40), true);
	sortingnetwork_sort_f32x48(data[0], data[1], data[2], data[3], data[4], data[5]);
	EXPECT_EQ(check_correctness((float *)data, 48), true);
	sortingnetwork_sort_f32x56(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
	EXPECT_EQ(check_correctness((float *)data, 56), true);
	sortingnetwork_sort_f32x64(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
	EXPECT_EQ(check_correctness((float *)data, 64), true);
	sortingnetwork_sort_f32x72(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
	EXPECT_EQ(check_correctness((float *)data, 72), true);
	sortingnetwork_sort_f32x80(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
	EXPECT_EQ(check_correctness((float *) data, 80), true);
	sortingnetwork_sort_f32x88(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10]);
	EXPECT_EQ(check_correctness((float *) data, 88), true);
	sortingnetwork_sort_f32x96(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11]);
	EXPECT_EQ(check_correctness((float *) data, 96), true);
	sortingnetwork_sort_f32x104(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12]);
	EXPECT_EQ(check_correctness((float *) data, 104), true);
	sortingnetwork_sort_f32x112(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13]);
	EXPECT_EQ(check_correctness((float *) data, 112), true);
	sortingnetwork_sort_f32x120(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14]);
	EXPECT_EQ(check_correctness((float *) data, 120), true);
	sortingnetwork_sort_f32x128(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]);
	EXPECT_EQ(check_correctness((float *) data, 128), true);
}

TEST(SortingNetwork, u32xX_t) {
	constexpr size_t size = 16;
	__m256i data[size] = {0};
	auto *d = (uint32_t *) data;
	for (size_t i = 0; i < size * 8; ++i) {
		d[i] = rng();
	}

	data[0] = sortingnetwork_sort_u32x8(data[0]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 8), true);
	sortingnetwork_sort_u32x16(data[0], data[1]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 16), true);
	sortingnetwork_sort_u32x24(data[0], data[1], data[2]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 24), true);
	sortingnetwork_sort_u32x32(data[0], data[1], data[2], data[3]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 32), true);
	sortingnetwork_sort_u32x40(data[0], data[1], data[2], data[3], data[4]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 40), true);
	sortingnetwork_sort_u32x48(data[0], data[1], data[2], data[3], data[4], data[5]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 48), true);
	sortingnetwork_sort_u32x56(data[0], data[1], data[2], data[3], data[4], data[5], data[6]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 56), true);
	sortingnetwork_sort_u32x64(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 64), true);
	sortingnetwork_sort_u32x72(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 72), true);
	sortingnetwork_sort_u32x80(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 80), true);
	sortingnetwork_sort_u32x88(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 88), true);
	sortingnetwork_sort_u32x96(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 96), true);
	sortingnetwork_sort_u32x104(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 104), true);
	sortingnetwork_sort_u32x112(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 112), true);
	sortingnetwork_sort_u32x120(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 120), true);
	sortingnetwork_sort_u32x128(data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9], data[10], data[11], data[12], data[13], data[14], data[15]);
	EXPECT_EQ(check_correctness((uint32_t *) data, 128), true);
}

TEST(SortingNetwork, small_f32xX_t) {
	constexpr size_t size = 16;
	__m256 data[size];
	auto *d = reinterpret_cast<float *>(data);
	for (size_t i = 0; i < size * 8; ++i) {
		d[i] = static_cast <float> ((float ) rng()) / static_cast <float> ((uint64_t)-1ull);
	}

	for (uint32_t i = 1; i < 8*size; i++) {
		const bool b = sortingnetwork_small_f32(d, i);
		EXPECT_EQ(b, true);
		const bool k = check_correctness<float>(d, i);
		EXPECT_EQ(k, true);
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

	__m512 a = (__m512)_mm512_loadu_si512((__m512i *)(d_in + 0));
	avx512_sortingnetwork_sort_f32x16(a);
	_mm512_storeu_si512((__m512i *)(d_out + 0), (__m512i)a);
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

TEST(SortingNetwork, avx512_uint16x32_t) {
	uint16_t d_in[32], d_out[32];
	for (uint32_t i = 0; i < 32; ++i) {
		d_in[i] = rng();
	}

	const __m512i in  = _mm512_loadu_si512((const __m512i *) d_in);
	const __m512i out = sortingnetwork_sort_u16x32_v2(in);
	_mm512_storeu_si512((__m512i *)d_out, out);
	for (uint32_t i = 0; i < 31; ++i) {
		EXPECT_LE(d_out[i], d_out[i+1]);
	}
}

TEST(SortingNetwork, avx512_kv_uint16x32_t) {
	uint16_t k_in[32] __attribute__((aligned(32))), k_out[32] __attribute__((aligned(32)));
	uint16_t v_in[32] __attribute__((aligned(32))), v_out[32] __attribute__((aligned(32)));
	for (uint32_t i = 0; i < 32; ++i) {
		k_in[i] = rng();
		v_in[i] = i;
	}

	memcpy(k_out, k_in, 64);
	memcpy(v_out, v_in, 64);
	sortingnetwork_kvsort_u16x32((__m512i *)k_out, (__m512i *)v_out);
	for (uint32_t i = 0; i < 31; ++i) {
		EXPECT_LE(k_out[i], k_out[i+1]);
		uint32_t j = 0;
		for (; j < 31; j++) {
			if (k_out[i] == k_in[j]) { break; }
		}
		EXPECT_EQ(v_out[i], v_in[j]);
	}
}

#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
