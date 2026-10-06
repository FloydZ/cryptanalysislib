#include <gtest/gtest.h>
#include <cstdint>

#include "random.h"
#include "helper.h"
#include "sort/sort.h"
#include "sort/quicksort.h"
#include "sort/radixsort.h"

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
// TYPED_TEST_P(TestSort, CountingSort) {
// 	auto *array = generate_list<TypeParam>(listsize);
// 	counting_sort_u8(static_cast<uint8_t *>(array), listsize);
// 	for (size_t i = 0; i < listsize-1; ++i) {
// 		EXPECT_LE(array[i], array[i+1]);
// 	}
//
// 	free(array);
// }

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
	rhmergesort<TypeParam>(array8, listsize);
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

// sorts x[0..n) with insertion sort (reference)
template<typename T>
static void insertion_sort(T *x, const size_t n) {
	for (size_t i = 1; i < n; i++) {
		const T v = x[i];
		size_t j = i;
		for (; j > 0 && v < x[j-1]; j--) { x[j] = x[j-1]; }
		x[j] = v;
	}
}

// selection, quick, merge (2- and 4-way), heap and radix sort on many sizes,
// with few distinct values (many duplicates) and with full-range values
TYPED_TEST_P(TestSort, ClassicSorts) {
	for (const size_t n : {0u, 1u, 2u, 7u, 8u, 9u, 15u, 16u, 17u, 100u, 1000u}) {
		for (int rep = 0; rep < 8; rep++) {
			std::vector<TypeParam> in(n), ref(n);
			for (size_t i = 0; i < n; i++) {
				in[i] = (rep & 1) ? TypeParam(rng() % 5) : TypeParam(rng());
			}
			for (size_t i = 0; i < n; i++) { ref[i] = in[i]; }
			insertion_sort(ref.data(), n);

			for (int algo = 0; algo < 6; algo++) {
				std::vector<TypeParam> v(in);
				switch (algo) {
					case 0: selection_sort(v.data(), n); break;
					case 1: quick_sort(v.data(), n); break;
					case 2: merge_sort(v.data(), n); break;
					case 3: merge_sort4(v.data(), n); break;
					case 4: radix_sort(v.data(), n); break;
					default: heap_sort(v.data(), n); break;
				}
				EXPECT_TRUE(v == ref) << "algo=" << algo << " n=" << n;
			}

			std::vector<TypeParam> v(in);
			heap_sort_descending(v.data(), n);
			for (size_t i = 0; i < n; i++) {
				EXPECT_EQ(v[i], ref[n - 1 - i]);
			}
		}
	}
}

REGISTER_TYPED_TEST_SUITE_P(TestSort, /*CountingSort,*/ HeapSort, MergeSort, ClassicSorts, RobinHoodSort, SKASort, VergeSort, VVSort, MultipleSKASort);
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




