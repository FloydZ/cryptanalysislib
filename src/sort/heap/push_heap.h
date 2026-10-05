#pragma once 
#include <cstdlib>
#include <algorithm>
#include <stdint.h>

// TODO: tests and benchmarks

// main source: https://github.com/WojciechMula/toys/blob/master/simd-heap/push_heap/push_heap_avx512.h
template<typename T,
         typename Compare>
constexpr void push_heap_scalar(T* start,
                                T* end,
                                Compare cmp) {
    T* array         = start;
    ssize_t index       = (end - start) - 1;
    ssize_t parent_idx  = (index - 1) / 2;

    const T new_value = array[index];
    while (parent_idx >= 0) {
        if (cmp(array[parent_idx], new_value)) {
            array[index]      = array[parent_idx];
            array[parent_idx] = new_value;

            index      = parent_idx;
            parent_idx = (index - 1) / 2;
        } else {
            break;
        }
    }
}

#ifdef USE_AVX2
#include "immintrin.h"

constexpr void push_heap_avx2(int32_t* start, size_t size) {
    if (size <= 255) {
        std::push_heap(start, start + size);
        return;
    }

    __m256i avx2_sort_values[8] = {
         _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
         _mm256_setr_epi32(1, 0, 2, 3, 4, 5, 6, 7),
         _mm256_setr_epi32(1, 2, 0, 3, 4, 5, 6, 7),
         _mm256_setr_epi32(1, 2, 3, 0, 4, 5, 6, 7),
         _mm256_setr_epi32(1, 2, 3, 4, 0, 5, 6, 7),
         _mm256_setr_epi32(1, 2, 3, 4, 5, 0, 6, 7),
         _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 0, 7),
         _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0),
    };

    size_t index = size - 1;
    while (index >= 256) {
        // 1. Construct indices from the current element to parent nodes 7 levels up
        ssize_t parent = 0;

        uint32_t tmp[8];
        tmp[0] = index;
        for (int i=1; i < 8; i++) {
            parent = (index - 1)/2;
            tmp[i] = parent;
            index = parent;
        }

        const __m256i indices = _mm256_load_si256((const __m256i*)tmp);

        // 2. Load values from the selected path
        const __m256i values = _mm256_i32gather_epi32((const int*)start, indices, sizeof(uint32_t));

        // 3. Broadcast 0th element from the vector
        const __m256i last_value = _mm256_permutevar8x32_epi32(values, _mm256_setzero_si256());

        // 3. Check if the heap property is violated.
        const __m256i mask = _mm256_cmpgt_epi32(values, last_value);

        const uint8_t any_parent_less = _mm256_movemask_ps(_mm256_castsi256_ps(mask));
        if (any_parent_less == 0) {
            const __m256i sorted = _mm256_permutevar8x32_epi32(values, avx2_sort_values[7]);
            _mm256_i32scatter_epi32(start, indices, sorted, sizeof(uint32_t));
            index = tmp[7];
            continue;
        }

        // 4. Elements on the path be should sorted, we need to locate where to insert a new value.
        const int new_index = __builtin_ctz(any_parent_less) - 1;
        if (new_index > 0) {
            const __m256i sorted = _mm256_permutevar8x32_epi32(values, avx2_sort_values[new_index]);
            _mm256_i32scatter_epi32(start, indices, sorted, sizeof(uint32_t));
        }
    }

    std::push_heap(start, start + index + 1);
}
#endif

#ifdef USE_AVX512 


void push_heap_avx512(int32_t* start, size_t size) {
    ssize_t index = size - 1;
    __m512i avx512_sort_values[16] = {
        avx512_sort_values[0]  = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 
        avx512_sort_values[1]  = _mm512_setr_epi32(1, 0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 
        avx512_sort_values[2]  = _mm512_setr_epi32(1, 2, 0, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 
        avx512_sort_values[3]  = _mm512_setr_epi32(1, 2, 3, 0, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 
        avx512_sort_values[4]  = _mm512_setr_epi32(1, 2, 3, 4, 0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 
        avx512_sort_values[5]  = _mm512_setr_epi32(1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), 
        avx512_sort_values[6]  = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 0, 7, 8, 9, 10, 11, 12, 13, 14, 15), 
        avx512_sort_values[7]  = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 0, 8, 9, 10, 11, 12, 13, 14, 15), 
        avx512_sort_values[8]  = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 0, 9, 10, 11, 12, 13, 14, 15), 
        avx512_sort_values[9]  = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 10, 11, 12, 13, 14, 15), 
        avx512_sort_values[10] = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 11, 12, 13, 14, 15), 
        avx512_sort_values[11] = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0, 12, 13, 14, 15), 
        avx512_sort_values[12] = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 0, 13, 14, 15), 
        avx512_sort_values[13] = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 0, 14, 15), 
        avx512_sort_values[14] = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 0, 15), 
        avx512_sort_values[15] = _mm512_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 0), 

    }
    if ((index >= 32767) && (index < 65535)) { // we're handling insertions only at level #15

        // 1. Build indices values
        uint32_t tmp[16];
        tmp[0]  = index;
        tmp[15] = 0;
        for (int i=1; i < 15; i++) {
            const ssize_t parent = (index - 1)/2;
            tmp[i] = parent;
            index = parent;
        }

        const __m512i indices = _mm512_load_si512(tmp);

        // 2. Load values from path between the new element and the root.
        const __m512i values = _mm512_i32gather_epi32(indices, start, sizeof(uint32_t));

        // 3. Check if the heap propery is violated.
        const __m512i new_value = _mm512_permutexvar_epi32(_mm512_setzero_si512(), values);
        const __mmask16 any_parent_less = _mm512_cmpgt_epu32_mask(values, new_value);

        // 4. Elements on the path be should sorted, we need to locate where to insert the new value.
        const int new_index = __builtin_ctz(any_parent_less) - 1;
        if (new_index > 0) {
            const __m512i sorted = _mm512_permutexvar_epi32(avx512_sort_values[new_index], values);
            _mm512_i32scatter_epi32(start, indices, sorted, sizeof(uint32_t));
        }
    } else {
        std::push_heap(start, start + size);
        abort();
    }
}
#endif
