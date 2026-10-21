#pragma once

#include <iterator>
#include <cstdint>
#include <future>
#include <vector>

#ifdef USE_AVX512F
#include <immintrin.h>
#include "algorithm/countl.h"
#endif
#include "simd/simd.h"
#include "algorithm/algorithm.h"
#include "algorithm/bits/popcount.h"
#include "algorithm/bounds.h"
#include "algorithm/copy.h"
#include "thread/thread.h"

namespace cryptanalysislib {
    struct AlgorithmSetIntersectionConfig {
    public:
        const size_t aligned_instructions = false;
    	const uint32_t min_size_simd = 32;
    	const uint32_t min_size_per_thread = 16384;
    };
    constexpr static AlgorithmSetIntersectionConfig algorithmSetIntersectionConfig{};

    
    namespace internal {
        /// \return first position `p` in [lo, hi) with !(x[p] < v), hi if none
        template <typename T>
        constexpr size_t intersection_lower_pos(T const *x,
                                                size_t lo,
                                                size_t hi,
                                                const T v) noexcept {
            while (lo < hi) {
                const size_t mid = lo + (hi - lo) / 2;
                if (x[mid] < v) { lo = mid + 1; } else { hi = mid; }
            }
            return lo;
        }

        /// Source: https://ashvardanian.com/posts/simd-set-intersections-sve2-avx512/
        /// NOTE: the smaller element is skipped: before, `a[i] < b[j]`
        ///     searched `a[i]` in `b` (which returns `j` again), so the loop
        ///     never terminated as soon as both elements differed.
        template <typename T>
        size_t galloping_intersection_size(T const *a, 
                                           T const *b, 
                                           const size_t n, 
                                           const size_t m) noexcept {
            size_t i = 0, j = 0, count = 0;    
            while (i < n && j < m) {
                // If elements are equal, increment both indices and count
                if (a[i] == b[j]) count++, i++, j++;
        
                // If a[i] is smaller, skip all elements of a smaller than b[j]
                else if (a[i] < b[j]) i = intersection_lower_pos(a, i, n, b[j]);
        
                // If b[j] is smaller, skip all elements of b smaller than a[i]
                else j = intersection_lower_pos(b, j, m, a[i]);
            }
            return count;
        }

        /// number of common elements of the sorted, duplicate free arrays
        /// `a` and `b` (two pointer merge)
        static inline size_t intersect_u16_serial(uint16_t const *a, uint16_t const *b,
                                                  const size_t n, const size_t m) noexcept {
            size_t i = 0, j = 0, c = 0;
            while ((i < n) && (j < m)) {
                if (a[i] < b[j]) { ++i; }
                else if (b[j] < a[i]) { ++j; }
                else { ++c; ++i; ++j; }
            }
            return c;
        }

        /// SIMD kernel of `set_intersection_simd`: a block of `a` is compared
        /// with every element of a block of `b`, the matches are compress
        /// stored. Afterwards the block(s) with the smaller last element are
        /// skipped.
        /// \return the number of elements written to `out`
        template<typename T,
                 const AlgorithmSetIntersectionConfig &config = algorithmSetIntersectionConfig>
#if __cplusplus > 201709L
            requires std::unsigned_integral<T>
#endif
        size_t set_intersection_uXX_simd(const T *a, const size_t n,
                                         const T *b, const size_t m,
                                         T *out) noexcept {
            using S = SIMDSelector<T>;
            constexpr uint32_t limbs = S::LIMBS;
            using M = MaskType<limbs>;

            size_t i = 0, j = 0, c = 0;
            while (((i + limbs) <= n) && ((j + limbs) <= m)) {
                const S va = S::template load<config.aligned_instructions != 0>(a + i);
                M mask = 0;
                for (uint32_t k = 0; k < limbs; ++k) {
                    mask |= M(va == S::set1(b[j + k]));
                }

                if (mask) {
                    if constexpr (requires(T *p, const S x, const Mask<limbs> &k) { S::compress(p, x, k); }) {
                        S::compress(out + c, va, Mask<limbs>(mask));
                        c += popcount::popcount(mask);
                    } else {
                        // NOTE: the generic SIMD types have no `compress`
                        for (uint32_t k = 0; k < limbs; ++k) {
                            if ((mask >> k) & 1u) { out[c++] = a[i + k]; }
                        }
                    }
                }

                const T amax = a[i + limbs - 1u], bmax = b[j + limbs - 1u];
                if (amax <= bmax) { i += limbs; }
                if (bmax <= amax) { j += limbs; }
            }

            // tail management
            while ((i < n) && (j < m)) {
                if (a[i] < b[j]) { ++i; }
                else if (b[j] < a[i]) { ++j; }
                else { out[c++] = a[i]; ++i; ++j; }
            }
            return c;
        }

#if defined(USE_AVX512F) && defined(USE_AVX512BW)
        /// emulates the (non existing) 16 bit `vp2intersect`: bit i of the
        /// result is set iff a[i] equals any lane of `b`. `b` is compared in
        /// all 32 lane rotations.
        static inline __mmask32 mm512_2intersect_epi16_mask(const __m512i a,
                                                            const __m512i b) noexcept {
            __m512i idx = _mm512_set_epi16(31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16,
                                           15, 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0);
            const __m512i one = _mm512_set1_epi16(1), mask = _mm512_set1_epi16(31);
            __mmask32 ret = 0;
            for (uint32_t r = 0; r < 32; ++r) {
                ret |= _mm512_cmpeq_epi16_mask(a, _mm512_permutexvar_epi16(idx, b));
                idx = _mm512_and_si512(_mm512_add_epi16(idx, one), mask);
            }
            return ret;
        }

        // source: https://ashvardanian.com/posts/simd-set-intersections-sve2-avx512/
        /// adds the number of common elements of the sorted, duplicate free
        /// arrays `a` and `b` to `*results`
        inline void simsimd_intersect_u16_ice(
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

            // NOTE: compare remaining lengths, `a + 32` is UB if fewer than 32 remain
            while ((a_end - a) > 32 && (b_end - b) > 32) {
                a_vec.zmm = _mm512_loadu_si512((__m512i const*)a);
                b_vec.zmm = _mm512_loadu_si512((__m512i const*)b);

                // Intersecting registers with `_mm512_2intersect_epi16_mask` involves a lot of shuffling
                // and comparisons, so we want to avoid it if the slices don't overlap at all
                uint16_t a_min;
                uint16_t a_max = a_vec.u16[31];
                uint16_t b_min = b_vec.u16[0];
                uint16_t b_max = b_vec.u16[31];

                // If the slices don't overlap, advance the appropriate pointer
                while (a_max < b_min && (a_end - a) > 64) {
                    a += 32;
                    a_vec.zmm = _mm512_loadu_si512((__m512i const*)a);
                    a_max = a_vec.u16[31];
                }
                a_min = a_vec.u16[0];
                while (b_max < a_min && (b_end - b) > 64) {
                    b += 32;
                    b_vec.zmm = _mm512_loadu_si512((__m512i const*)b);
                    b_max = b_vec.u16[31];
                }
                b_min = b_vec.u16[0];

                // Now we are likely to have some overlap, so we can intersect the registers
                __mmask32 a_matches = mm512_2intersect_epi16_mask(a_vec.zmm, b_vec.zmm);
                c += cryptanalysislib::popcount::popcount((uint32_t)a_matches);

                // Determine the number of entries to skip in each array, by comparing
                // every element in the vector with the last (largest) element in the other array
                __m512i a_last_broadcasted = _mm512_set1_epi16(*(short const*)&a_max);
                __m512i b_last_broadcasted = _mm512_set1_epi16(*(short const*)&b_max);
                __mmask32 a_step_mask = _mm512_cmple_epu16_mask(a_vec.zmm, b_last_broadcasted);
                __mmask32 b_step_mask = _mm512_cmple_epu16_mask(b_vec.zmm, a_last_broadcasted);
                a += 32 - cryptanalysislib::countl_zero((uint32_t)a_step_mask);
                b += 32 - cryptanalysislib::countl_zero((uint32_t)b_step_mask);
            }

            // Handle the tail:
            c += intersect_u16_serial(a, b, size_t(a_end - a), size_t(b_end - b));
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

    /// Computes the intersection of two strictly increasing (sorted and
    /// duplicate free) arrays of unsigned integers with SIMD instructions
    /// NOTE: for multisets the result differs from `set_intersection`.
    /// \param a[in]: first array
    /// \param n[in]: number of elements in `a`
    /// \param b[in]: second array
    /// \param m[in]: number of elements in `b`
    /// \param out[out]: the common elements (increasing), must have space for min(n, m) elements
    /// \return number of common elements
    template<typename T,
             const AlgorithmSetIntersectionConfig &config = algorithmSetIntersectionConfig>
#if __cplusplus > 201709L
        requires std::unsigned_integral<T>
#endif
    size_t set_intersection_simd(const T *a, const size_t n,
                                 const T *b, const size_t m,
                                 T *out) noexcept {
        return internal::set_intersection_uXX_simd<T, config>(a, n, b, m, out);
    }

    /// Computes the intersection of two sorted ranges (parallel version)
    /// The first range is split into chunks (without splitting a run of
    /// equivalent elements), the corresponding part of the second range is
    /// found by binary search. Every chunk is intersected into its own
    /// buffer, which are then copied to `d_first`.
    /// NOTE: the result is the same as of the sequential version (also for multisets)
    /// \param policy[in]: execution policy
    /// \param first1[in]: iterator to the beginning of the first range
    /// \param last1[in]: iterator to the end of the first range
    /// \param first2[in]: iterator to the beginning of the second range
    /// \param last2[in]: iterator to the end of the second range
    /// \param d_first[out]: iterator to the beginning of the destination range
    /// \param comp[in]: comparison function object
    /// \return iterator to the end of the constructed range
    template<class ExecPolicy,
             class RandIt1,
             class RandIt2,
             class OutIt,
             class Compare,
             const AlgorithmSetIntersectionConfig &config = algorithmSetIntersectionConfig>
#if __cplusplus > 201709L
        requires ExecutionPolicy<ExecPolicy> &&
                 std::random_access_iterator<RandIt1> &&
                 std::random_access_iterator<RandIt2> &&
                 std::random_access_iterator<OutIt>
#endif
    OutIt set_intersection(ExecPolicy &&policy,
                           RandIt1 first1,
                           RandIt1 last1,
                           RandIt2 first2,
                           RandIt2 last2,
                           OutIt d_first,
                           Compare comp) noexcept {
        using T = typename std::iterator_traits<RandIt1>::value_type;
        const size_t n = static_cast<size_t>(std::distance(first1, last1));
        const uint32_t nthreads = should_par(policy, config, n);
        if (is_seq<ExecPolicy>(policy) || (nthreads < 2)) {
            return cryptanalysislib::set_intersection(first1, last1, first2, last2, d_first, comp);
        }

        // chunk boundaries in both ranges: a chunk of the first range ends at
        // the end of a run of equivalent elements
        std::vector<size_t> b1(nthreads + 1u), b2(nthreads + 1u);
        b1[0] = 0; b2[0] = 0;
        for (uint32_t k = 1; k < nthreads; ++k) {
            size_t p = (n * k) / nthreads;
            if (p < b1[k - 1]) { p = b1[k - 1]; }
            if (p > 0) {
                p = static_cast<size_t>(cryptanalysislib::upper_bound(first1 + p, last1, *(first1 + (p - 1)), comp) - first1);
            }
            b1[k] = p;
            b2[k] = (p == n) ? static_cast<size_t>(std::distance(first2, last2))
                             : static_cast<size_t>(internal::bounds_lower_bound(first2, last2, *(first1 + p), comp) - first2);
        }
        b1[nthreads] = n;
        b2[nthreads] = static_cast<size_t>(std::distance(first2, last2));

        auto &task_pool = *policy.pool();
        std::vector<std::vector<T>> tmp(nthreads);
        std::vector<std::future<void>> futures;
        for (uint32_t k = 0; k < nthreads; ++k) {
            futures.emplace_back(task_pool.submit([&, k]() {
                tmp[k].resize(std::min(b1[k + 1] - b1[k], b2[k + 1] - b2[k]));
                const auto e = cryptanalysislib::set_intersection(first1 + b1[k], first1 + b1[k + 1],
                                                                  first2 + b2[k], first2 + b2[k + 1],
                                                                  tmp[k].begin(), comp);
                tmp[k].resize(static_cast<size_t>(e - tmp[k].begin()));
            }));
        }
        for (auto &f : futures) { f.wait(); }
        futures.clear();

        // copy the chunks to their offsets
        size_t off = 0;
        for (uint32_t k = 0; k < nthreads; ++k) {
            const size_t o = off;
            off += tmp[k].size();
            futures.emplace_back(task_pool.submit([&, k, o]() {
                cryptanalysislib::copy(tmp[k].begin(), tmp[k].end(), d_first + o);
            }));
        }
        for (auto &f : futures) { f.wait(); }
        return d_first + off;
    }

    /// Computes the intersection of two sorted ranges using the less operator (parallel version)
    template<class ExecPolicy,
             class RandIt1,
             class RandIt2,
             class OutIt,
             const AlgorithmSetIntersectionConfig &config = algorithmSetIntersectionConfig>
#if __cplusplus > 201709L
        requires ExecutionPolicy<ExecPolicy> &&
                 std::random_access_iterator<RandIt1> &&
                 std::random_access_iterator<RandIt2> &&
                 std::random_access_iterator<OutIt>
#endif
    OutIt set_intersection(ExecPolicy &&policy,
                           RandIt1 first1,
                           RandIt1 last1,
                           RandIt2 first2,
                           RandIt2 last2,
                           OutIt d_first) noexcept {
        return cryptanalysislib::set_intersection<ExecPolicy, RandIt1, RandIt2, OutIt, std::less<>, config>
            (std::forward<ExecPolicy>(policy), first1, last1, first2, last2, d_first, std::less<>{});
    }

}; // end namespace
