#ifndef CRYPTANALYSISLIB_VV_RADIXSORT_H
#define CRYPTANALYSISLIB_VV_RADIXSORT_H

#include <cstdint>
#include <cstddef>

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
    static T *aux1 = (T *) malloc(sizeof(T) * 8 * len / 8);
    static size_t *aux2;
    static size_t *idx;
    if constexpr (use_idx) {
        aux2 = (size_t *) malloc(sizeof(size_t) * len);
        idx = (size_t *) malloc(sizeof(size_t) * len);
    }
    static uint64_t old_len = 0;

    if (old_len > len) {
        aux1 = (T *) realloc(aux1, sizeof(T) * len / 8);
        if constexpr (use_idx) {
            aux2 = (size_t *) realloc(aux2, sizeof(size_t) * len);
            idx = (size_t *) realloc(idx, sizeof(size_t) * len);
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
