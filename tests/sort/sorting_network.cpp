#include <gtest/gtest.h>
#include <cstdint>


using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

#include "random.h"
#include "helper.h"
#include "sort/sorting_network/common.h"

template<typename T>
T *gen_data(const size_t size) {
	T *data = (T*) malloc(sizeof(T) * size);
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

	for (size_t i = 0; i < size-1; ++i) {
		ASSERT_LE(data[i], data[i+1]);
	}

	free(data);
}



#ifdef USE_AVX2
#include "sort/sorting_network/avx2.h"

// SRC: https://drops.dagstuhl.de/opus/volltexte/2021/13775/pdf/LIPIcs-SEA-2021-3.pdf
TEST(SortingNetwork, Ints8) {
	//print_U256i(a);
	//print_U256i(b);
	__m256i z = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);

	__m256i a = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
	__m256i b = sortingnetwork_sort_8_32(a);
	__m256i c = _mm256_cmpeq_epi32(a, b);
	int mask = _mm256_movemask_ps((__m256) c);
	EXPECT_EQ(mask, (1u << 8u) - 1u);

	a = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
	b = sortingnetwork_sort_8_32(a);
	c = _mm256_cmpeq_epi32(b, z);
	mask = _mm256_movemask_ps((__m256) c);
	EXPECT_EQ(mask, (1u << 8u) - 1u);
}
#endif

int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
