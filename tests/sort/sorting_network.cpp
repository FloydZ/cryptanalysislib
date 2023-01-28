#include <gtest/gtest.h>
#include <cstdint>
#include <immintrin.h>

#include "sort/sorting_network.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

union U256i {
    __m256i v;
    uint32_t a[8];
};

void print_U256i(const __m256i v){
    const U256i u = { v };

    for (const auto i : u.a) {
        printf("%d ", i);
	}

	printf("\n");
}



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


int main(int argc, char **argv) {
    InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
