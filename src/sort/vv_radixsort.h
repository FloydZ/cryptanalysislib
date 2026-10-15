#ifndef CRYPTANALYSISLIB_VV_RADIXSORT_H
#define CRYPTANALYSISLIB_VV_RADIXSORT_H

#include <cstdint>
#include <cstddef>
#include <cstdlib>

/// LSD radix sort, taken from Valentin Vasseur
/// \tparam T
/// \tparam use_idx
/// \param array
/// \param idx
/// \param aux
/// \param aux2
/// \param len /
template<typename T, bool use_idx>
void vv_radix_sort(T *array, size_t *idx, T *aux, size_t *aux2, size_t len) {
    constexpr uint32_t BITS = sizeof(T)*8;
    constexpr uint32_t RADIX = 8;
    constexpr uint32_t BUCKETS = (1L << RADIX);

    auto DIGIT = [](const T A, const T B){
        return (((A) >> (BITS - ((B) + 1) * RADIX)) & (BUCKETS - 1));
    };

    for (size_t w = BITS / RADIX; w-- > 0;) {
        size_t count[BUCKETS + 1] = {0};

        for (size_t i = 0; i < len; ++i)
            ++count[DIGIT(array[i], w) + 1];

        for (size_t j = 1; j < BUCKETS - 1; ++j)
            count[j + 1] += count[j];

        for (size_t i = 0; i < len; ++i) {
            size_t cnt = count[DIGIT(array[i], w)];
            aux[cnt] = array[i];
            if constexpr (use_idx) {
                aux2[cnt] = idx[i];
            }
            ++count[DIGIT(array[i], w)];
        }

        for (size_t i = 0; i < len; ++i) {
            array[i] = aux[i];
            if constexpr (use_idx) {
                idx[i] = aux2[i];
            }
        }
    }
}

/// straight forward radix sort.
/// \tparam use_idx if set to true, additionally an const_array will used to restore the original sorting. Currently unusable
template<typename T, bool use_idx=false>
void vv_radix_sort(T *L, const size_t len) {
    if (len < 2) {
        return;
    }

    // NOTE: the buffers are allocated for every call. They used to be
    // `static`, but were freed at the end of every call (use after free on
    // the second call) and never grown for larger inputs.
    T *aux1 = (T *) malloc(sizeof(T) * len);
    size_t *aux2 = nullptr;
    size_t *idx = nullptr;
    if constexpr (use_idx) {
        aux2 = (size_t *) malloc(sizeof(size_t) * len);
        idx = (size_t *) malloc(sizeof(size_t) * len);
        for (size_t i = 0; i < len; ++i) {
            idx[i] = i;
        }
    }

	vv_radix_sort<T, use_idx>(L, idx, aux1, aux2, len);
	free(aux1);

	if constexpr (use_idx) {
        free(aux2);
        free(idx);
	}
}
#endif//CRYPTANALYSISLIB_VV_RADIXSORT_H
