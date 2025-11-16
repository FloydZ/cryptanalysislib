#pragma once 

#include <iterator>
#include <cstdint>

#ifdef USE_AVX512F
#include <immintrin.h>
#include "simd/simd.h"
#endif

// TODO simd version 
// TODO parrallel version

namespace cryptanalysislib {
    struct AlgorithmSetIntersectionConfig {
    public:
        const size_t aligned_instructions = false;
    	const uint32_t min_size_simd = 32;
    	const uint32_t min_size_per_thread = 16384;
    };
    constexpr static AlgorithmSetIntersectionConfig algorithmSetIntersectionConfig{};

    
    namespace internal {
        /// Source: https://ashvardanian.com/posts/simd-set-intersections-sve2-avx512/
        template <typename T>
        size_t galloping_intersection_size(T const *a, 
                                           T const *b, 
                                           const size_t n, 
                                           const size_t m) noexcept {
            size_t i = 0, j = 0, count = 0;    
            while (i < n && j < m) {
                // If elements are equal, increment both indices and count
                if (a[i] == b[j]) count++, i++, j++;
        
                // If a[i] is smaller, gallop through b to find its match
                else if (a[i] < b[j]) j = std::lower_bound(b + j, b + m, a[i]) - b;
        
                // If a[i] is larger, gallop through a to find its match
                else i = std::lower_bound(a + i, a + n, b[j]) - a;
            }
            return count;
        }

#ifdef USE_AVX512F 

        // source: https://ashvardanian.com/posts/simd-set-intersections-sve2-avx512/
        void simsimd_intersect_u16_ice(
            uint16_t const *a, uint16_t const *b,
            const size_t a_length, const size_t b_length,
            uint64_t* results) {
        
            uint16_t const* const a_end = a + a_length;
            uint16_t const* const b_end = b + b_length;
            size_t c = 0;
            union vec_t {
                __m512i zmm;
                uint16_t u16[32];
                uint8_t u8[64];
            } a_vec, b_vec;

            while (a + 32 < a_end && b + 32 < b_end) {
                a_vec.zmm = _mm512_loadu_si512((__m512i const*)a);
                b_vec.zmm = _mm512_loadu_si512((__m512i const*)b);

                // Intersecting registers with `_mm512_2intersect_epi16_mask` involves a lot of shuffling
                // and comparisons, so we want to avoid it if the slices don't overlap at all
                uint16_t a_min;
                uint16_t a_max = a_vec.u16[31];
                uint16_t b_min = b_vec.u16[0];
                uint16_t b_max = b_vec.u16[31];

                // If the slices don't overlap, advance the appropriate pointer
                while (a_max < b_min && a + 64 < a_end) {
                    a += 32;
                    a_vec.zmm = _mm512_loadu_si512((__m512i const*)a);
                    a_max = a_vec.u16[31];
                }
                a_min = a_vec.u16[0];
                while (b_max < a_min && b + 64 < b_end) {
                    b += 32;
                    b_vec.zmm = _mm512_loadu_si512((__m512i const*)b);
                    b_max = b_vec.u16[31];
                }
                b_min = b_vec.u16[0];

                // Now we are likely to have some overlap, so we can intersect the registers
                __mmask32 a_matches = _mm512_2intersect_epi16_mask(a_vec.zmm, b_vec.zmm);
                c += _mm_popcnt_u32(a_matches); // The `_popcnt32` symbol isn't recognized by MSVC

                // Determine the number of entries to skip in each array, by comparing
                // every element in the vector with the last (largest) element in the other array
                __m512i a_last_broadcasted = _mm512_set1_epi16(*(short const*)&a_max);
                __m512i b_last_broadcasted = _mm512_set1_epi16(*(short const*)&b_max);
                __mmask32 a_step_mask = _mm512_cmple_epu16_mask(a_vec.zmm, b_last_broadcasted);
                __mmask32 b_step_mask = _mm512_cmple_epu16_mask(b_vec.zmm, a_last_broadcasted);
                a += 32 - _lzcnt_u32((uint32_t)a_step_mask);
                b += 32 - _lzcnt_u32((uint32_t)b_step_mask);
            }

            // Handle the tail:
            // TODO simsimd_intersect_u16_serial(a, b, a_end - a, b_end - b, results);
            *results += c; // And merge it with the main body result
        }
#endif
    };


    /// Computes the intersection of two sorted ranges
    ///
    /// \tparam InputIt1 type of the first input iterator
    /// \tparam InputIt2 type of the second input iterator
    /// \tparam OutputIt type of the output iterator
    /// \tparam Compare type of the comparison function
    /// 
    /// NOTE: Input ranges must be sorted according to the same ordering criterion
    template<class InputIt1, 
             class InputIt2, 
             class OutputIt, 
             class Compare>
#if __cplusplus > 201709L
    	requires std::forward_iterator<InputIt1> && 
                 std::forward_iterator<InputIt2> && 
                 std::forward_iterator<OutputIt>
#endif
    /// Constructs a sorted range consisting of elements that are found in both sorted input ranges
    ///
    /// \param first1[in]: iterator to the beginning of the first range
    /// \param last1[in]: iterator to the end of the first range
    /// \param first2[in]: iterator to the beginning of the second range
    /// \param last2[in]: iterator to the end of the second range
    /// \param d_first[out]: iterator to the beginning of the destination range
    /// \param comp[in]: comparison function object
    /// \return iterator to the end of the constructed range
    constexpr
    OutputIt set_intersection(InputIt1 first1, 
                              InputIt1 last1,
                              InputIt2 first2,
                              InputIt2 last2, 
                              OutputIt d_first, 
                              Compare comp) noexcept {
        while (first1 != last1 && first2 != last2) {
            if (comp(*first1, *first2)) {
                ++first1;
            } else {
                if (!comp(*first2, *first1)) {
                    // *first1 and *first2 are equivalent.
                    *d_first++ = *first1++; 
                }
                ++first2;
            }
        }
        return d_first;
    }
    
    /// Computes the intersection of two sorted ranges using the less operator
    ///
    /// \tparam InputIt1 type of the first input iterator
    /// \tparam InputIt2 type of the second input iterator
    /// \tparam OutputIt type of the output iterator
    /// 
    /// NOTE: Input ranges must be sorted in ascending order
    template<class InputIt1, 
             class InputIt2, 
             class OutputIt>
#if __cplusplus > 201709L
    	requires std::forward_iterator<InputIt1> && 
                 std::forward_iterator<InputIt2> && 
                 std::forward_iterator<OutputIt>
#endif
    /// Constructs a sorted range consisting of elements that are found in both sorted input ranges
    ///
    /// \param first1[in]: iterator to the beginning of the first range
    /// \param last1[in]: iterator to the end of the first range
    /// \param first2[in]: iterator to the beginning of the second range
    /// \param last2[in]: iterator to the end of the second range
    /// \param d_first[out]: iterator to the beginning of the destination range
    /// \return iterator to the end of the constructed range
    constexpr
    OutputIt set_intersection(InputIt1 first1,
                              InputIt1 last1,
                              InputIt2 first2,
                              InputIt2 last2, 
                              OutputIt d_first) {
        return set_intersection(first1, last1, first2, last2, d_first, std::less{});
    }

}; // end namespace
