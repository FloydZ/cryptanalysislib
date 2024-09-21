#ifndef CRYPTANALYSISLIB_ALGORITHM_ARGMIN_H
#define CRYPTANALYSISLIB_ALGORITHM_ARGMIN_H

#include <concepts>
#include <cstdlib>
#include <cstdint>
#include <type_traits>

// TODO only < needed
template<typename T>
#if __cplusplus > 201709L
    requires std::totally_ordered<T>
#endif
constexpr static inline size_t argmin(const T *a,
                                      const size_t n) noexcept {
    size_t k = 0;
    for (size_t i = 0; i < n; i++) {
        if (a[i] < a[k]) {
            k = i;
        }
    }
    
    return k;
}

#ifdef USE_AVX2 
#include <immintrin.h>
// TODO write config if pointers are aligned
// TODO write selection function in cryptanalysis namespace

// source : https://en.algorithmica.org/hpc/algorithms/argmin/
constexpr static inline size_t argmin(uint32_t *a,
                                      const size_t n) {
    // indices on the current iteration
    __m256i cur = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
    // the current minimum for each slice
    __m256i min = _mm256_set1_epi32(-1u);
    // its index (argmin) for each slice
    __m256i idx = _mm256_setzero_si256();

    for (size_t i = 0; i < n; i += 8) {
        // load a new SIMD block
        __m256i x = _mm256_load_si256((__m256i*) &a[i]);
        // find the slices where the minimum is updated
        __m256i mask = _mm256_cmpgt_epi32(min, x);
        // update the indices
        idx = _mm256_blendv_epi8(idx, cur, mask);
        // update the minimum (can also similarly use a "blend" here, but min is faster)
        min = _mm256_min_epi32(x, min);
        // update the current indices
        const __m256i eight = _mm256_set1_epi32(8);
        cur = _mm256_add_epi32(cur, eight);       // 
        // can also use a "blend" here, but min is faster
    }

    // find the argmin in the "min" __m256iister and return its real index
    uint32_t min_arr[8], idx_arr[8];
    
    _mm256_storeu_si256((__m256i*) min_arr, min);
    _mm256_storeu_si256((__m256i*) idx_arr, idx);

    size_t k = 0, m = min_arr[0];
    for (uint32_t i = 1; i < 8; i++) {
        if (min_arr[i] < m) {
            m = min_arr[k = i];
        }
    }

    return idx_arr[k];
}
#endif
#endif
