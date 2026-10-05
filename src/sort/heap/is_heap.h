#pragma once
/// source: https://github.com/WojciechMula/toys/blob/master/simd-heap/is_heap

#include <iterator>
#include <algorithm>


// TODO
// - tests and benchmarks for all functions



template<typename ForwardIterator,
         typename Compare = std::less<>>
constexpr bool is_heap_fwd(ForwardIterator start,
                           ForwardIterator end,
                           Compare cmp) {
    if (start == end) {
        return true;
    }

    auto parent  = start;
    auto current = std::next(start);
    while (current != end) {
        if (cmp(*parent, *current))
            return false;

        current = std::next(current);
        if (current == end)
            break;

        if (cmp(*parent, *current))
            return false;

        parent = std::next(parent);
        current = std::next(current);
    }

    return true;
}


template <typename RandomIterator, typename Compare = std::less<>>
bool is_heap_rnd(RandomIterator start, RandomIterator end, Compare cmp) {
    const size_t count = end - start;
    if (count <= 1)
        return true;

    size_t parent_idx = 0;
    size_t child_idx = 1;
    for (size_t i=0; i < count/2; i++) {
        const auto parent = start[parent_idx];
        if (cmp(parent, start[child_idx]) or cmp(parent, start[child_idx + 1]))
            return false;

        parent_idx += 1;
        child_idx += 2;
    }

    if (count % 2 == 1) {
        const size_t i = count - 1;
        const auto parent = start[(i - 1)/2];
        const auto child  = start[i];

        return not cmp(parent, child);
    }
    else
        return true;
}

#ifdef USE_AVX2
#include <immintrin.h>

constexpr bool is_heap_sse_epi32(const int32_t* begin,
                                 const int32_t* end) {
    const ssize_t k = 16/4; // words in a vector

    if (end - begin < 2 * k) {
        return std::is_heap(begin, end);
    }

    const int32_t* parent = begin;
    const int32_t* current = begin + 1;

    while (end - current >= 2 * k) {
        // 1. load parents
        // tmp = [p3|p2|p1|p0]
        const __m128i tmp = _mm_loadu_si128((const __m128i*)parent);
        // p0  = [p1|p1|p0|p0]
        const __m128i p0  = _mm_unpacklo_epi32(tmp, tmp);
        // p1  = [p3|p3|p2|p2]
        const __m128i p1  = _mm_unpackhi_epi32(tmp, tmp);

        // 2. load children
        const __m128i children0 = _mm_loadu_si128((const __m128i*)(current + 0));
        const __m128i children1 = _mm_loadu_si128((const __m128i*)(current + k));

        // 3. compare parents with their children
        const __m128i lt0 = _mm_cmplt_epi32(p0, children0);
        const __m128i lt1 = _mm_cmplt_epi32(p1, children1);
        const __m128i t0 = _mm_or_si128(lt0, lt1);

        if (_mm_movemask_epi8(t0))
            return false;

        if (current + 2*k > end)
            break;

        parent  += k;
        current += 2*k;
    }

    for (ssize_t i = current - begin; i < end - begin; i++) {
        if (begin[(i - 1) / 2] < begin[i]) {
            return false;
        }
    }

    return true;
}

constexpr bool is_heap_avx2_epi32(const int32_t* begin,
                                  const int32_t* end) {
    constexpr ssize_t k = 32/4; // words in a vector

    if (end - begin < 2 * k) {
        return std::is_heap(begin, end);
    }

    const int32_t* parent = begin;
    const int32_t* current = begin + 1;

    const __m256i lo = _mm256_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3);
    const __m256i hi = _mm256_setr_epi32(4, 4, 5, 5, 6, 6, 7, 7);

    while (end - current >= 2 * k) {
        // 1. load parents
        // tmp = [p3|p2|p1|p0]
        const __m256i tmp = _mm256_loadu_si256((const __m256i*)parent);
        // p0  = [p1|p1|p0|p0]
        const __m256i p0  = _mm256_permutevar8x32_epi32(tmp, lo);
        // p1  = [p3|p3|p2|p2]                               
        const __m256i p1  = _mm256_permutevar8x32_epi32(tmp, hi);

        // 2. load children
        const __m256i children0 = _mm256_loadu_si256((const __m256i*)(current + 0));
        const __m256i children1 = _mm256_loadu_si256((const __m256i*)(current + k));

        // 3. compare parents with their children
        const __m256i lt0 = _mm256_cmpgt_epi32(children0, p0);
        const __m256i lt1 = _mm256_cmpgt_epi32(children1, p1);
        const __m256i t0 = _mm256_or_si256(lt0, lt1);

        if (_mm256_movemask_epi8(t0))
            return false;

        if (current + 2*k > end)
            break;

        parent  += k;
        current += 2*k;
    }

    for (ssize_t i = current - begin; i < end - begin; i++) {
        if (begin[(i - 1) / 2] < begin[i]) {
            return false;
        }
    }

    return true;
}
#endif

#ifdef USE_AVX512

bool is_heap_avx512_epi32(const int32_t* begin, const int32_t* end) {
    const ssize_t k = 64/4; // words in a vector

    if (end - begin < 2 * k) {
        return std::is_heap(begin, end);
    }

    const int32_t* parent = begin;
    const int32_t* current = begin + 1;

    const __m512i lo = _mm512_setr_epi32(0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7);
    const __m512i hi = _mm512_setr_epi32(8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15);

    while (end - current >= 2 * k) {
        // 1. load parents
        // tmp = [p3|p2|p1|p0]
        const __m512i tmp = _mm512_loadu_si512((const __m512i*)parent);
        // p0  = [p1|p1|p0|p0]
        const __m512i p0  = _mm512_permutexvar_epi32(lo, tmp);
        // p1  = [p3|p3|p2|p2]                               
        const __m512i p1  = _mm512_permutexvar_epi32(hi, tmp);

        // 2. load children
        const __m512i children0 = _mm512_loadu_si512((const __m512i*)(current + 0));
        const __m512i children1 = _mm512_loadu_si512((const __m512i*)(current + k));

        // 3. compare parents with their children
        const __mmask16 lt0 = _mm512_cmpgt_epi32_mask(children0, p0);
        const __mmask16 lt1 = _mm512_cmpgt_epi32_mask(children1, p1);

        if (!_kortestz_mask16_u8(lt0, lt1)) {
            return false;
        }

        if (current + 2*k > end)
            break;

        parent  += k;
        current += 2*k;
    }

    for (ssize_t i = current - begin; i < end - begin; i++) {
        if (begin[(i - 1) / 2] < begin[i]) {
            return false;
        }
    }

    return true;
}
#endif
