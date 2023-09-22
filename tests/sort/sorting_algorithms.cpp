#include <gtest/gtest.h>
#include <cstdint>

#include "random.h"
#include "helper.h"
#include "sort/sort.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

constexpr size_t listsize = 1024;

template<typename T>
T* generate_list(const size_t len) {
	T* array = (T *)malloc(len*sizeof(T));
	for (size_t i = 0; i < len; ++i) {
		array[i] = fastrandombytes_uint64();
	}
	ASSERT(array);
	return array;
}

TEST(CrumSort, Ints8) {
	uint8_t *array8 = generate_list<uint8_t>(listsize);
	crumsort<uint8_t >(array8, listsize, [](const uint8_t *a, const uint8_t *b){
		return (*a) - (*b);
	});

    for (size_t i = 0; i < listsize-1; ++i) {
        ASSERT_LE(array8[i], array8[i+1]);
    }
	free(array8);
}

TEST(QuadSort, Ints8) {
    uint8_t *array8 = generate_list<uint8_t>(listsize);
	quadsort<uint8_t>(array8, listsize, [](uint8_t *a, uint8_t *b){
		return *a < *b;
	});

	for (size_t i = 0; i < listsize-1; ++i) {
		ASSERT_GE(array8[i], array8[i+1]);
	}

	free(array8);
}

TEST(RobinHoodSort, Ints8) {
    uint8_t *array8 = generate_list<uint8_t>(listsize);
	rhmergesort<uint8_t>(array8, listsize);

    for (size_t i = 0; i < listsize-1; ++i) {
        ASSERT_LE(array8[i], array8[i+1]);
    }

    free(array8);
}

TEST(SKASort, Ints8) {
    uint8_t *array8 = generate_list<uint8_t>(listsize);
    ska_sort(array8, array8 + listsize, [](const uint8_t in){ return in;});

    for (size_t i = 0; i < listsize-1; ++i) {
        ASSERT_LE(array8[i], array8[i+1]);
    }

    free(array8);
}

TEST(VergeSort, Ints8) {
    uint8_t *array8 = generate_list<uint8_t>(listsize);
    vergesort::vergesort(array8, array8 + listsize, [](const uint8_t in1, const uint8_t in2){
		return in1 < in2;
	});

    for (size_t i = 0; i < listsize-1; ++i) {
        ASSERT_LE(array8[i], array8[i+1]);
    }

    free(array8);
}

TEST(VVSort, Ints32) {
    uint32_t *array8 = generate_list<uint32_t>(listsize);
    vv_radix_sort(array8, listsize);

    for (size_t i = 0; i < listsize-1; ++i) {
        ASSERT_LE(array8[i], array8[i+1]);
    }

    free(array8);
}

#ifdef USE_AVX2
TEST(DJBSORT, Ints32) {
	int32_t *array8 = generate_list<int32_t>(listsize);
	int32_sort(array8, listsize);

	for (size_t i = 0; i < listsize-1; ++i) {
		ASSERT_LE(array8[i], array8[i+1]);
	}
	free(array8);
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}




