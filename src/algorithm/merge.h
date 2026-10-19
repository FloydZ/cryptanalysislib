#pragma once 

#include <algorithm>
#include <cstdint>
#include <functional>
#include <future>
#include <iterator>
#include <type_traits>
#include <vector>

#include "algorithm/algorithm.h"
#include "algorithm/bounds.h"
#include "algorithm/copy.h"
#include "algorithm/rotate.h"
#include "algorithm/swap.h"
#include "thread/thread.h"

#ifdef USE_AVX2
#include "sort/sorting_network/common.h"
#endif

namespace cryptanalysislib {
    /// Configuration for merge algorithms
    struct AlgorithmMergeConfig : public AlgorithmConfig {
        constexpr static size_t min_size_per_thread = 1u << 16u;
    };
    constexpr static AlgorithmMergeConfig algorithmMergeConfig;

    namespace internal {
        /// merge path / co-rank: the number of elements of `a` among the first
        /// `d` elements of the stable merge of `a` and `b` (on ties the
        /// elements of `a` come first).
        template<class RandIt1, class RandIt2, class Compare>
        constexpr size_t merge_co_rank(const size_t d,
                                       RandIt1 a, const size_t n1,
                                       RandIt2 b, const size_t n2,
                                       Compare comp) noexcept {
            size_t lo = d > n2 ? d - n2 : 0;
            size_t hi = d < n1 ? d : n1;
            while (lo < hi) {
                const size_t i = lo + (hi - lo) / 2;
                const size_t j = d - i;
                // a[i] would be emitted before b[j-1]: take more of `a`
                if ((j > 0) && !comp(b[j - 1], a[i])) {
                    lo = i + 1;
                } else {
                    hi = i;
                }
            }
            return lo;
        }
    } // end namespace internal

#ifdef USE_AVX2
    namespace internal {
        /// merges two sorted vectors of 8 uint32_t: afterwards `lo` holds the
        /// 8 smallest, `hi` the 8 largest elements, both sorted.
        /// `a` and reversed `b` form a bitonic sequence: min/max splits it into
        /// two bitonic halves (all of `lo` <= all of `hi`), which the bitonic
        /// merge network `sortingnetwork_aftermerge_u32x8` sorts.
        static inline void merge_u32x8x2(__m256i &lo, __m256i &hi,
                                         const __m256i a, const __m256i b) noexcept {
            const __m256i rev = _mm256_setr_epi32(7, 6, 5, 4, 3, 2, 1, 0);
            const __m256i br = _mm256_permutevar8x32_epi32(b, rev);
            lo = _mm256_min_epu32(a, br);
            hi = _mm256_max_epu32(a, br);
            sortingnetwork_aftermerge_u32x8(lo);
            sortingnetwork_aftermerge_u32x8(hi);
        }

        /// merges the sorted arrays `a` (n1) and `b` (n2) into `out` (n1+n2)
        /// NOTE: Inoue et al. style: the 8 largest of the last merge step stay
        ///     in a register and are merged with the next block of the input
        ///     with the smaller head. As soon as that input has less than 8
        ///     elements left, the rest is merged scalar.
        static inline void merge_u32_avx2(const uint32_t *a, const size_t n1,
                                          const uint32_t *b, const size_t n2,
                                          uint32_t *out) noexcept {
            constexpr size_t t = 8;
            size_t i = 0, j = 0, k = 0;
            alignas(32) uint32_t rest[t];
            size_t nrest = 0;

            if ((n1 >= t) && (n2 >= t)) {
                __m256i va = _mm256_loadu_si256((const __m256i *)a);
                __m256i vb = _mm256_loadu_si256((const __m256i *)b);
                i = t; j = t;
                while (true) {
                    __m256i lo, hi;
                    merge_u32x8x2(lo, hi, va, vb);
                    _mm256_storeu_si256((__m256i *)(out + k), lo);
                    k += t;
                    vb = hi;

                    // the next block must come from the input with the smaller
                    // head, otherwise elements could be emitted out of order
                    const bool a_left = i < n1, b_left = j < n2;
                    const bool a_next = a_left && (!b_left || a[i] <= b[j]);
                    if (a_next && (i + t <= n1)) {
                        va = _mm256_loadu_si256((const __m256i *)(a + i)); i += t;
                    } else if (!a_next && b_left && (j + t <= n2)) {
                        va = _mm256_loadu_si256((const __m256i *)(b + j)); j += t;
                    } else {
                        break;
                    }
                }

                _mm256_store_si256((__m256i *)rest, vb);
                nrest = t;
            }

            // scalar 3-way merge of `rest`, a[i..n1), b[j..n2)
            size_t r = 0;
            while ((r < nrest) || (i < n1) || (j < n2)) {
                uint32_t best = UINT32_MAX;
                int src = -1;
                if (r < nrest)                     { best = rest[r]; src = 0; }
                if ((i < n1) && (src < 0 || a[i] < best)) { best = a[i]; src = 1; }
                if ((j < n2) && (src < 0 || b[j] < best)) { best = b[j]; src = 2; }
                out[k++] = best;
                if (src == 0) { r++; } else if (src == 1) { i++; } else { j++; }
            }
        }
    } // end namespace internal
#endif
    
    /// Merges the sorted ranges [first1, last1) and [first2, last2) into the
    /// sorted range starting at `d_first` (like `std::merge`, stable: on
    /// equal elements those of the first range come first)
    /// \tparam InputIt1 Input iterator type of the first range
    /// \tparam InputIt2 Input iterator type of the second range
    /// \tparam OutputIt Output iterator type
    /// \param first1/last1[in]: first sorted range
    /// \param first2/last2[in]: second sorted range
    /// \param d_first[out]: beginning of the output, must not overlap the inputs
    /// \return Iterator past the last written element
    /// NOTE: contiguous `uint32_t` ranges use the AVX2 merge (if available)
    template<class InputIt1,
             class InputIt2,
             class OutputIt>
    OutputIt merge(InputIt1 first1,
                   InputIt1 last1,
                   InputIt2 first2,
                   InputIt2 last2,
                   OutputIt d_first) {
#ifdef USE_AVX2
        // SIMD version for contiguous uint32_t ranges
        if constexpr (std::contiguous_iterator<InputIt1> &&
                      std::contiguous_iterator<InputIt2> &&
                      std::contiguous_iterator<OutputIt> &&
                      std::is_same_v<std::iter_value_t<InputIt1>, uint32_t> &&
                      std::is_same_v<std::iter_value_t<InputIt2>, uint32_t> &&
                      std::is_same_v<std::iter_value_t<OutputIt>, uint32_t>) {
            const size_t n1 = static_cast<size_t>(last1 - first1);
            const size_t n2 = static_cast<size_t>(last2 - first2);
            internal::merge_u32_avx2(std::to_address(first1), n1,
                                     std::to_address(first2), n2,
                                     std::to_address(d_first));
            return d_first + (n1 + n2);
        }
#endif
        for (; first1 != last1; ++d_first) {
            if (first2 == last2)
                return std::copy(first1, last1, d_first);
     
            if (*first2 < *first1) {
                *d_first = *first2;
                ++first2;
            } else {
                *d_first = *first1;
                ++first1;
            }
        }

        return std::copy(first2, last2, d_first);
    }
   
    /// Merges two ranges sorted w.r.t. `comp` (like `std::merge`, stable)
    /// \tparam Compare less-than comparison, `comp(a, b)` true iff a < b
    /// \param first1/last1[in]: first sorted range
    /// \param first2/last2[in]: second sorted range
    /// \param d_first[out]: beginning of the output, must not overlap the inputs
    /// \param comp[in]: comparison function
    /// \return Iterator past the last written element
    template<class InputIt1,
             class InputIt2,
             class OutputIt,
             class Compare>
    OutputIt merge(InputIt1 first1,
                   InputIt1 last1,
                   InputIt2 first2,
                   InputIt2 last2,
                   OutputIt d_first,
                   Compare comp) {
        for (; first1 != last1; ++d_first) {
            if (first2 == last2)
                return std::copy(first1, last1, d_first);
     
            if (comp(*first2, *first1)) {
                *d_first = *first2;
                ++first2;
            } else {
                *d_first = *first1;
                ++first1;
            }
        }

        return std::copy(first2, last2, d_first);
    }

    /// Merges two sorted ranges (parallel version). Every thread computes an
    /// equally large part of the output: its start in both inputs is found by
    /// a binary search (merge path / co-rank), so the result equals the
    /// sequential (stable) merge.
    /// \param policy[in]: Execution policy specifying parallelization strategy
    /// \param first1/last1[in]: first sorted range
    /// \param first2/last2[in]: second sorted range
    /// \param d_first[out]: beginning of the output, n1+n2 elements
    /// \param comp[in]: less-than comparison
    /// \return Iterator to the end of the output range
    template<class ExecPolicy,
             class RandIt1,
             class RandIt2,
             class RandIt3,
             class Compare,
             const AlgorithmMergeConfig &config = algorithmMergeConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt1> &&
             std::random_access_iterator<RandIt2> &&
             std::random_access_iterator<RandIt3>
#endif
    RandIt3 merge(ExecPolicy &&policy,
                  RandIt1 first1,
                  RandIt1 last1,
                  RandIt2 first2,
                  RandIt2 last2,
                  RandIt3 d_first,
                  Compare comp) noexcept {
        // `<`: the sequential overload without comparator (SIMD for uint32_t)
        constexpr bool is_less = std::is_same_v<Compare, std::less<>>;
        const size_t n1 = static_cast<size_t>(std::distance(first1, last1));
        const size_t n2 = static_cast<size_t>(std::distance(first2, last2));
        const size_t n = n1 + n2;
        const uint32_t nthreads = should_par(policy, config, n);
        if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
            if constexpr (is_less) {
                return cryptanalysislib::merge(first1, last1, first2, last2, d_first);
            } else {
                return cryptanalysislib::merge(first1, last1, first2, last2, d_first, comp);
            }
        }

        auto &task_pool = *policy.pool();
        std::vector<std::future<void>> futures;
        for (uint32_t t = 0; t < nthreads; ++t) {
            const size_t d0 = (n * t) / nthreads;
            const size_t d1 = (n * (t + 1)) / nthreads;
            futures.emplace_back(task_pool.submit([=]() {
                const size_t i0 = internal::merge_co_rank(d0, first1, n1, first2, n2, comp);
                const size_t i1 = internal::merge_co_rank(d1, first1, n1, first2, n2, comp);
                const size_t j0 = d0 - i0, j1 = d1 - i1;
                if constexpr (is_less) {
                    cryptanalysislib::merge(first1 + i0, first1 + i1,
                                            first2 + j0, first2 + j1,
                                            d_first + d0);
                } else {
                    cryptanalysislib::merge(first1 + i0, first1 + i1,
                                            first2 + j0, first2 + j1,
                                            d_first + d0, comp);
                }
            }));
        }

        for (auto &f : futures) { f.wait(); }
        return d_first + n;
    }

    /// parallel merge with `<`
    template<class ExecPolicy,
             class RandIt1,
             class RandIt2,
             class RandIt3,
             const AlgorithmMergeConfig &config = algorithmMergeConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt1> &&
             std::random_access_iterator<RandIt2> &&
             std::random_access_iterator<RandIt3>
#endif
    RandIt3 merge(ExecPolicy &&policy,
                  RandIt1 first1,
                  RandIt1 last1,
                  RandIt2 first2,
                  RandIt2 last2,
                  RandIt3 d_first) noexcept {
        return cryptanalysislib::merge(std::forward<ExecPolicy>(policy),
                                       first1, last1, first2, last2, d_first, std::less<>());
    }

    namespace internal {
        /// merges the sorted [first, middle) (len1 elements) and [middle,
        /// last) (len2 elements) in place without extra memory: the larger
        /// part is split in the middle, the matching split point of the other
        /// part is found by a binary search, the two inner pieces are swapped
        /// by a rotation, and both halves are merged recursively.
        /// O(n log n) moves and comparisons.
        template<class BidirIt, class Compare>
        constexpr void merge_without_buffer(BidirIt first,
                                            BidirIt middle,
                                            BidirIt last,
                                            const std::iter_difference_t<BidirIt> len1,
                                            const std::iter_difference_t<BidirIt> len2,
                                            Compare &comp) noexcept {
            if ((len1 == 0) || (len2 == 0)) {
                return;
            }

            if ((len1 + len2) == 2) {
                if (comp(*middle, *first)) {
                    cryptanalysislib::iter_swap(first, middle);
                }
                return;
            }

            BidirIt first_cut = first, second_cut = middle;
            std::iter_difference_t<BidirIt> len11, len22;
            if (len1 > len2) {
                len11 = len1 / 2;
                std::advance(first_cut, len11);
                second_cut = internal::bounds_lower_bound(middle, last, *first_cut, comp);
                len22 = std::distance(middle, second_cut);
            } else {
                len22 = len2 / 2;
                std::advance(second_cut, len22);
                first_cut = internal::bounds_upper_bound(first, middle, *second_cut, comp);
                len11 = std::distance(first, first_cut);
            }

            const BidirIt new_middle = cryptanalysislib::rotate(first_cut, middle, second_cut);
            merge_without_buffer(first, first_cut, new_middle, len11, len22, comp);
            merge_without_buffer(new_middle, second_cut, last, len1 - len11, len2 - len22, comp);
        }
    } // end namespace internal

    /// Merges the consecutive sorted ranges [first, middle) and [middle, last)
    /// into the sorted range [first, last) (like `std::inplace_merge`,
    /// stable). Needs no extra memory, O(n log n).
    /// \param first[in]: beginning of the first sorted range
    /// \param middle[in]: end of the first, beginning of the second sorted range
    /// \param last[in]: end of the second sorted range
    /// \param comp[in]: less-than comparison
    template<class BidirIt,
             class Compare>
#if __cplusplus > 201709L
    requires std::bidirectional_iterator<BidirIt>
#endif
    constexpr void inplace_merge(BidirIt first,
                                 BidirIt middle,
                                 BidirIt last,
                                 Compare comp) noexcept {
        internal::merge_without_buffer(first, middle, last,
                                       std::distance(first, middle),
                                       std::distance(middle, last), comp);
    }

    /// same as above, with `<`
    template<class BidirIt>
#if __cplusplus > 201709L
    requires std::bidirectional_iterator<BidirIt>
#endif
    constexpr void inplace_merge(BidirIt first,
                                 BidirIt middle,
                                 BidirIt last) noexcept {
        cryptanalysislib::inplace_merge(first, middle, last, std::less<>());
    }

    /// Merges the consecutive sorted ranges [first, middle) and [middle, last)
    /// into the sorted range [first, last) (parallel version of
    /// `inplace_merge`, stable): the parallel `merge` into a buffer, which
    /// is copied back in parallel.
    /// NOTE: needs n elements of extra memory, and the elements are copied:
    /// 	`T` must be default constructible and copyable
    /// \param policy[in]: Execution policy specifying parallelization strategy
    /// \param first[in]: beginning of the first sorted range
    /// \param middle[in]: end of the first, beginning of the second sorted range
    /// \param last[in]: end of the second sorted range
    /// \param comp[in]: less-than comparison
    template<class ExecPolicy,
             class RandIt,
             class Compare,
             const AlgorithmMergeConfig &config = algorithmMergeConfig>
#if __cplusplus > 201709L
    requires ExecutionPolicy<ExecPolicy> &&
             std::random_access_iterator<RandIt>
#endif
    void inplace_merge(ExecPolicy &&policy,
                       RandIt first,
                       RandIt middle,
                       RandIt last,
                       Compare comp) noexcept {
        using T = std::iter_value_t<RandIt>;
        const size_t n = static_cast<size_t>(std::distance(first, last));
        const uint32_t nthreads = should_par(policy, config, n);
        if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
            cryptanalysislib::inplace_merge(first, middle, last, comp);
            return;
        }

        std::vector<T> buf(n);
        cryptanalysislib::merge(policy, first, middle, middle, last, buf.begin(), comp);
        cryptanalysislib::copy(policy, buf.begin(), buf.end(), first);
    }

    /// same as above, with `<`
    template<class ExecPolicy,
             class RandIt,
             const AlgorithmMergeConfig &config = algorithmMergeConfig>
#if __cplusplus > 201709L
    requires ExecutionPolicy<ExecPolicy> &&
             std::random_access_iterator<RandIt>
#endif
    void inplace_merge(ExecPolicy &&policy,
                       RandIt first,
                       RandIt middle,
                       RandIt last) noexcept {
        cryptanalysislib::inplace_merge
            <ExecPolicy, RandIt, std::less<>, config>
            (std::forward<ExecPolicy>(policy), first, middle, last, std::less<>());
    }
}; // end namespace cryptanalysislib
