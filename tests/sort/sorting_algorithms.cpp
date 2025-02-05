#include <gtest/gtest.h>
#include <cstdint>

#include "random.h"
#include "helper.h"
#include "sort/sort.h"

using ::testing::InitGoogleTest;
using ::testing::Test;
using namespace std;

constexpr size_t listsize = 128;

template<typename T>
T* generate_list(const size_t len) {
	T* array = (T *)malloc(len*sizeof(T));
	for (size_t i = 0; i < len; ++i) {
		array[i] = rng();
	}
	assert(array);
	return array;
}


template <typename T>
class TestSort : public testing::Test {};

TYPED_TEST_SUITE_P(TestSort);
TYPED_TEST_P(TestSort, CountingSort) {
	TypeParam *array = generate_list<TypeParam>(listsize);
	counting_sort_u8(array, listsize);
	for (size_t i = 0; i < listsize-1; ++i) {
		EXPECT_LE(array[i], array[i+1]);
	}

	free(array);
}

TYPED_TEST_P(TestSort, HeapSort) {
	TypeParam *array = generate_list<TypeParam>(listsize);
	heap_sort(array, listsize);
	for (size_t i = 0; i < listsize-1; ++i) {
		EXPECT_LE(array[i], array[i+1]);
	}

	free(array);
}

TYPED_TEST_P(TestSort, MergeSort) {
	TypeParam *array = generate_list<TypeParam>(listsize);
	merge_sort(array, listsize);
	for (size_t i = 0; i < listsize-1; ++i) {
		EXPECT_LE(array[i], array[i+1]);
	}

	free(array);
}

TYPED_TEST_P(TestSort, RobinHoodSort) {
    TypeParam *array8 = generate_list<TypeParam>(listsize);
	rhmergesort<uint8_t>(array8, listsize);
    for (size_t i = 0; i < listsize-1; ++i) {
        EXPECT_LE(array8[i], array8[i+1]);
    }

    free(array8);
}

TYPED_TEST_P(TestSort, SKASort) {
    TypeParam *array8 = generate_list<TypeParam>(listsize);
    ska_sort(array8, array8 + listsize, [](const TypeParam in){ return in;});

    for (size_t i = 0; i < listsize-1; ++i) {
        EXPECT_LE(array8[i], array8[i+1]);
    }

    free(array8);
}

TYPED_TEST_P(TestSort, VergeSort) {
    TypeParam *array8 = generate_list<TypeParam>(listsize);
    vergesort::vergesort(array8, array8 + listsize,
                         [](const TypeParam in1, const TypeParam in2){
		return in1 < in2;
	});

    for (size_t i = 0; i < listsize-1; ++i) {
        EXPECT_LE(array8[i], array8[i+1]);
    }

    free(array8);
}

TYPED_TEST_P(TestSort, VVSort) {
    TypeParam *array8 = generate_list<TypeParam>(listsize);
    vv_radix_sort(array8, listsize);

    for (size_t i = 0; i < listsize-1; ++i) {
        EXPECT_LE(array8[i], array8[i+1]);
    }

    free(array8);
}

TYPED_TEST_P(TestSort, MultipleSKASort) {
	for (uint32_t t = 0; t < (1u << 19); t++) {
        TypeParam *array8 = generate_list<TypeParam>(listsize);
        ska_sort(array8, array8 + listsize, [](const TypeParam in){ return in;});

        for (size_t i = 0; i < listsize-1; ++i) {
            EXPECT_LE(array8[i], array8[i+1]);
        }

        free(array8);
	}
}

REGISTER_TYPED_TEST_SUITE_P(TestSort, CountingSort, RobinHoodSort, SKASort, VergeSort, VVSort, MultipleSKASort);
using MyTypes = ::testing::Types<uint8_t, uint16_t, uint32_t, uint64_t>;
INSTANTIATE_TYPED_TEST_SUITE_P(My, TestSort, MyTypes);

#ifdef USE_AVX2
TEST(DJBSORT, Ints32) {
	int32_t *array8 = generate_list<int32_t>(listsize);
	int32_sort(array8, listsize);

	for (size_t i = 0; i < listsize-1; ++i) {
		EXPECT_LE(array8[i], array8[i+1]);
	}
	free(array8);
}
#endif

int main(int argc, char **argv) {
	InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}




