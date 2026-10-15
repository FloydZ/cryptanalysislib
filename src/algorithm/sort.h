#pragma once 

#include <algorithm>
#include <cstddef>
#include <functional>
#include <future>
#include <iterator>
#include <vector>

#include "algorithm/algorithm.h"
#include "algorithm/copy.h"
#include "algorithm/heap.h"
#include "algorithm/merge.h"
#include "algorithm/partition.h"
#include "algorithm/swap.h"
#include "sort/ska_sort.h"
#include "sort/timsort.h"

namespace cryptanalysislib {
    template<class ForwardIt, class Compare>
    constexpr ForwardIt is_sorted_until(ForwardIt first, ForwardIt last, Compare comp);

    /// Finds the end of the longest sorted prefix of [first, last)
    /// (like `std::is_sorted_until`)
    /// \param first[in]: Iterator to the beginning of the range
    /// \param last[in]: Iterator to the end of the range
    /// \return first iterator `it` with `*it < *(it - 1)`, or last
    template<class ForwardIt>
    constexpr //< since C++20
    ForwardIt is_sorted_until(ForwardIt first,
                              ForwardIt last) {
        return cryptanalysislib::is_sorted_until(first, last, std::less<>());
    }

    /// Finds the end of the longest prefix of [first, last) sorted w.r.t.
    /// `comp` (like `std::is_sorted_until`)
    /// \param first[in]: Iterator to the beginning of the range
    /// \param last[in]: Iterator to the end of the range
    /// \param comp[in]: less-than comparison
    /// \return first iterator `it` with `comp(*it, *(it - 1))`, or last
    template<class ForwardIt,
             class Compare>
    constexpr
    ForwardIt is_sorted_until(ForwardIt first,
                              ForwardIt last,
                              Compare comp) {
        if (first != last) {
            ForwardIt next = first;
            while (++next != last) {
                if (comp(*next, *first)) {
                    return next;
                }
                first = next;
            }
        }

        return last;
    }
    
    /// Checks whether [first, last) is sorted in non-descending order
    /// (like `std::is_sorted`)
    /// \param first[in]: Iterator to the beginning of the range
    /// \param last[in]: Iterator to the end of the range
    /// \return true if sorted (also if the range is empty)
    template<class ForwardIt>
    bool is_sorted(ForwardIt first,
                   ForwardIt last) {
        return cryptanalysislib::is_sorted_until(first, last) == last;
    }
   
    /// Checks whether [first, last) is sorted w.r.t. `comp` (like `std::is_sorted`)
    /// \param first[in]: Iterator to the beginning of the range
    /// \param last[in]: Iterator to the end of the range
    /// \param comp[in]: less-than comparison
    /// \return true if sorted (also if the range is empty)
    template<class ForwardIt, class Compare>
    bool is_sorted(ForwardIt first,
                   ForwardIt last,
                   Compare comp) {
        return cryptanalysislib::is_sorted_until(first, last, comp) == last;
    }

    /// Sorts [first, last) by the key `extract_key(*it)` (radix sort, `ska_sort`).
    /// NOTE: `extract_key` takes a single element, the overload with a
    ///     two argument comparator is in `sort/sort.h`.
    /// \param first[in]: Iterator to the beginning of the range
    /// \param last[in]: Iterator to the end of the range
    /// \param extract_key[in]: returns the sort key (integer, float, ...) of an element
    template<class RandIt,
             class ExtractKey>
        requires std::random_access_iterator<RandIt> &&
                 std::invocable<ExtractKey &, std::iter_reference_t<RandIt>>
    constexpr inline void sort(RandIt first,
                               RandIt last,
                               ExtractKey extract_key) noexcept {
        ::ska_sort(first, last, extract_key);
    }




template<typename RandomIt, typename Compare>
constexpr void partial_sort(RandomIt first, RandomIt middle, RandomIt last, Compare comp);

/// Rearranges [first, last) such that [first, middle) contains the
/// `middle - first` smallest elements in ascending order (like
/// `std::partial_sort`); the order of the rest is unspecified.
/// NOTE: before, this forwarded to `std::partial_sort`
template<typename RandomIt>
constexpr //< since C++20
void partial_sort(RandomIt first, RandomIt middle, RandomIt last)
{
    typedef typename std::iterator_traits<RandomIt>::value_type VT;
    cryptanalysislib::partial_sort(first, middle, last, std::less<VT>());
}
namespace impl
{
    template<typename RandomIt, typename Compare>
    constexpr //< since C++20
    void sift_down(RandomIt first, RandomIt last, const Compare& comp)
    {
        // sift down element at “first”
        const auto length = static_cast<std::size_t>(last - first);
        std::size_t current = 0;
        std::size_t next = 2;
        while (next < length)
        {
            if (comp(*(first + next), *(first + (next - 1))))
                --next;
            if (!comp(*(first + current), *(first + next)))
                return;
            cryptanalysislib::iter_swap(first + current, first + next);
            current = next;
            next = 2 * current + 2;
        }
        --next;
        if (next < length && comp(*(first + current), *(first + next)))
            cryptanalysislib::iter_swap(first + current, first + next);
    }
 
    template<typename RandomIt, typename Compare>
    constexpr //< since C++20
    void heap_select(RandomIt first, RandomIt middle, RandomIt last, const Compare& comp)
    {
        std::make_heap(first, middle, comp);
        for (auto i = middle; i != last; ++i)
        {
            if (comp(*i, *first))
            {
                cryptanalysislib::iter_swap(first, i);
                impl::sift_down(first, middle, comp);
            }
        }
    }
} // namespace impl
 
/// Rearranges [first, last) such that [first, middle) contains the
/// `middle - first` smallest elements w.r.t. `comp`, sorted (like
/// `std::partial_sort`): heap select + heap sort, O(n log k).
template<typename RandomIt, typename Compare>
constexpr //< since C++20
void partial_sort(RandomIt first, RandomIt middle, RandomIt last, Compare comp)
{
    impl::heap_select(first, middle, last, comp);
    std::sort_heap(first, middle, comp);
}

/// Sorts [first, last) w.r.t. `comp` and keeps the order of equivalent
/// elements (like `std::stable_sort`), via `gfx::timsort`.
/// NOTE: the `requires` clause makes it more constrained than
/// 	`std::stable_sort`, so unqualified calls which see both are not ambiguous.
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param comp[in]: less-than comparison
template<class RandIt,
         class Compare>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
void stable_sort(RandIt first,
                 RandIt last,
                 Compare comp) noexcept {
    gfx::timsort(first, last, comp);
}

/// same as above, with `<`
template<class RandIt>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
void stable_sort(RandIt first,
                 RandIt last) noexcept {
    cryptanalysislib::stable_sort(first, last, std::less<>());
}

/// Copies the `min(last - first, d_last - d_first)` smallest elements of
/// [first, last) w.r.t. `comp`, sorted, to [d_first, d_last) (like
/// `std::partial_sort_copy`): a max-heap of the smallest elements seen so
/// far, O(n log k).
/// \param first[in]: Iterator to the beginning of the input range
/// \param last[in]: Iterator to the end of the input range
/// \param d_first[out]: Iterator to the beginning of the output range
/// \param d_last[out]: Iterator to the end of the output range
/// \param comp[in]: less-than comparison
/// \return iterator past the last written element
template<class InputIt,
         class RandIt,
         class Compare>
#if __cplusplus > 201709L
    requires std::input_iterator<InputIt> &&
             std::random_access_iterator<RandIt>
#endif
constexpr RandIt partial_sort_copy(InputIt first,
                                   InputIt last,
                                   RandIt d_first,
                                   RandIt d_last,
                                   Compare comp) noexcept {
    if (d_first == d_last) {
        return d_last;
    }

    RandIt r = d_first;
    for (; (first != last) && (r != d_last); ++first, ++r) {
        *r = *first;
    }

    cryptanalysislib::make_heap(d_first, r, comp);
    const auto len = r - d_first;
    for (; first != last; ++first) {
        // smaller than the largest kept element: replace it
        if (comp(*first, *d_first)) {
            internal::heap_sift_down(d_first, len, 0, std::iter_value_t<RandIt>(*first), comp);
        }
    }

    cryptanalysislib::sort_heap(d_first, r, comp);
    return r;
}

/// same as above, with `<`
template<class InputIt,
         class RandIt>
#if __cplusplus > 201709L
    requires std::input_iterator<InputIt> &&
             std::random_access_iterator<RandIt>
#endif
constexpr RandIt partial_sort_copy(InputIt first,
                                   InputIt last,
                                   RandIt d_first,
                                   RandIt d_last) noexcept {
    return cryptanalysislib::partial_sort_copy(first, last, d_first, d_last, std::less<>());
}

namespace impl {
    /// sorts the (small) range [first, last) by insertion
    template<typename RandomIt, typename Compare>
    constexpr void insertion_sort(RandomIt first, RandomIt last, Compare &comp) {
        if (first == last) {
            return;
        }

        for (RandomIt i = first + 1; i != last; ++i) {
            std::iter_value_t<RandomIt> v = std::move(*i);
            RandomIt j = i;
            for (; (j != first) && comp(v, *(j - 1)); --j) {
                *j = std::move(*(j - 1));
            }
            *j = std::move(v);
        }
    }

    /// moves the median of `*a`, `*b`, `*c` to `*result`
    template<typename RandomIt, typename Compare>
    constexpr void move_median_to_first(RandomIt result, RandomIt a, RandomIt b, RandomIt c, Compare &comp) {
        if (comp(*a, *b)) {
            if (comp(*b, *c)) {
                cryptanalysislib::iter_swap(result, b);
            } else if (comp(*a, *c)) {
                cryptanalysislib::iter_swap(result, c);
            } else {
                cryptanalysislib::iter_swap(result, a);
            }
        } else if (comp(*a, *c)) {
            cryptanalysislib::iter_swap(result, a);
        } else if (comp(*b, *c)) {
            cryptanalysislib::iter_swap(result, c);
        } else {
            cryptanalysislib::iter_swap(result, b);
        }
    }

    /// Hoare partition of [first, last) around `*pivot`, which is outside
    /// the range. Unguarded: both scans stop at the median of three elements
    /// placed by `move_median_to_first`.
    /// \return first element of the right part (all elements >= pivot)
    template<typename RandomIt, typename Compare>
    constexpr RandomIt unguarded_partition(RandomIt first, RandomIt last, RandomIt pivot, Compare &comp) {
        while (true) {
            while (comp(*first, *pivot)) {
                ++first;
            }
            --last;
            while (comp(*pivot, *last)) {
                --last;
            }
            if (!(first < last)) {
                return first;
            }
            cryptanalysislib::iter_swap(first, last);
            ++first;
        }
    }
} // namespace impl

/// Rearranges [first, last) such that `*nth` is the element which would be
/// there if the range was sorted w.r.t. `comp`, no element before `nth` is
/// greater and no element after it is smaller (like `std::nth_element`).
/// Introselect: quickselect with median of three pivots, after
/// 2*log2(n) partitions heap select, i.e. O(n) on average, O(n log n) worst
/// case.
/// \param first[in]: Iterator to the beginning of the range
/// \param nth[in]: position of the element to select
/// \param last[in]: Iterator to the end of the range
/// \param comp[in]: less-than comparison
template<class RandIt,
         class Compare>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
constexpr void nth_element(RandIt first,
                           RandIt nth,
                           RandIt last,
                           Compare comp) noexcept {
    if ((first == last) || (nth == last)) {
        return;
    }

    uint32_t depth = 0;
    for (auto n = last - first; n > 1; n >>= 1) {
        depth += 2;
    }

    while ((last - first) > 16) {
        if (depth == 0) {
            // [first, nth] holds the nth + 1 smallest elements, `*first` the largest
            impl::heap_select(first, nth + 1, last, comp);
            cryptanalysislib::iter_swap(first, nth);
            return;
        }
        depth -= 1;

        impl::move_median_to_first(first, first + 1, first + (last - first) / 2, last - 1, comp);
        const RandIt cut = impl::unguarded_partition(first + 1, last, first, comp);
        if (cut <= nth) {
            first = cut;
        } else {
            last = cut;
        }
    }

    impl::insertion_sort(first, last, comp);
}

/// same as above, with `<`
template<class RandIt>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
constexpr void nth_element(RandIt first,
                           RandIt nth,
                           RandIt last) noexcept {
    cryptanalysislib::nth_element(first, nth, last, std::less<>());
}

/// Configuration for the parallel `stable_sort`, `partial_sort_copy` and
/// `nth_element`
struct AlgorithmSortConfig : public AlgorithmConfig {
    constexpr static size_t min_size_per_thread = 1u << 14u;
};
constexpr static AlgorithmSortConfig algorithmSortConfig;

/// Sorts [first, last) w.r.t. `comp` and keeps the order of equivalent
/// elements (parallel version of `stable_sort`): every thread stable sorts
/// one chunk, then pairs of sorted runs are merged in rounds, alternating
/// between [first, last) and a buffer. In every round each pair is split
/// into parts proportional to its size by the merge path (co-rank), so all
/// threads merge in every round, not only in the first ones.
/// NOTE: needs n elements of extra memory, and the elements are copied: `T`
/// 	must be default constructible and copyable
/// \param policy[in]: Execution policy specifying parallelization strategy
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param comp[in]: less-than comparison
template<class ExecPolicy,
         class RandIt,
         class Compare,
         const AlgorithmSortConfig &config = algorithmSortConfig>
#if __cplusplus > 201709L
    requires ExecutionPolicy<ExecPolicy> &&
             std::random_access_iterator<RandIt>
#endif
void stable_sort(ExecPolicy &&policy,
                 RandIt first,
                 RandIt last,
                 Compare comp) noexcept {
    using T = std::iter_value_t<RandIt>;
    const size_t n = static_cast<size_t>(std::distance(first, last));
    const uint32_t nthreads = should_par(policy, config, n);
    if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
        cryptanalysislib::stable_sort(first, last, comp);
        return;
    }

    auto &task_pool = *policy.pool();
    std::vector<std::future<void>> futures;
    std::vector<size_t> bound(nthreads + 1);
    for (uint32_t k = 0; k <= nthreads; ++k) {
        bound[k] = (n * k) / nthreads;
    }
    for (uint32_t k = 0; k < nthreads; ++k) {
        futures.emplace_back(task_pool.submit([&comp, b = first + bound[k], e = first + bound[k + 1]]() {
            cryptanalysislib::stable_sort(b, e, comp);
        }));
    }
    internal::get_futures(futures);
    futures.clear();

    // merges the parts [d0, d1) of the merge of src[a, b) and src[b, c)
    // into dst[a + d0, a + d1)
    const auto merge_part = [&comp](auto src, auto dst,
                                    const size_t a, const size_t b, const size_t c,
                                    const size_t d0, const size_t d1) {
        const size_t n1 = b - a, n2 = c - b;
        const size_t i0 = internal::merge_co_rank(d0, src + a, n1, src + b, n2, comp);
        const size_t i1 = internal::merge_co_rank(d1, src + a, n1, src + b, n2, comp);
        if constexpr (std::is_same_v<Compare, std::less<>>) {
            cryptanalysislib::merge(src + a + i0, src + a + i1,
                                    src + b + (d0 - i0), src + b + (d1 - i1), dst + a + d0);
        } else {
            cryptanalysislib::merge(src + a + i0, src + a + i1,
                                    src + b + (d0 - i0), src + b + (d1 - i1), dst + a + d0, comp);
        }
    };

    std::vector<T> buf(n);
    bool in_buf = false;
    while (bound.size() > 2) {
        std::vector<size_t> next{0};
        const size_t runs = bound.size() - 1;
        for (size_t r = 0; r < runs; r += 2) {
            const size_t a = bound[r], b = bound[r + 1];
            const size_t c = (r + 1 < runs) ? bound[r + 2] : b;
            const size_t m = c - a;
            const size_t parts = std::max<size_t>(1u, (m * nthreads + n / 2) / n);
            for (size_t t = 0; t < parts; ++t) {
                const size_t d0 = (m * t) / parts, d1 = (m * (t + 1)) / parts;
                futures.emplace_back(task_pool.submit([=, &merge_part, &buf]() {
                    if (in_buf) {
                        merge_part(buf.begin(), first, a, b, c, d0, d1);
                    } else {
                        merge_part(first, buf.begin(), a, b, c, d0, d1);
                    }
                }));
            }
            next.push_back(c);
        }
        internal::get_futures(futures);
        futures.clear();
        bound = std::move(next);
        in_buf = !in_buf;
    }

    if (in_buf) {
        cryptanalysislib::copy(policy, buf.begin(), buf.end(), first);
    }
}

/// same as above, with `<`
template<class ExecPolicy,
         class RandIt,
         const AlgorithmSortConfig &config = algorithmSortConfig>
#if __cplusplus > 201709L
    requires ExecutionPolicy<ExecPolicy> &&
             std::random_access_iterator<RandIt>
#endif
void stable_sort(ExecPolicy &&policy,
                 RandIt first,
                 RandIt last) noexcept {
    cryptanalysislib::stable_sort
        <ExecPolicy, RandIt, std::less<>, config>
        (std::forward<ExecPolicy>(policy), first, last, std::less<>());
}

/// Copies the `min(last - first, d_last - d_first)` smallest elements of
/// [first, last) w.r.t. `comp`, sorted, to [d_first, d_last) (parallel
/// version of `partial_sort_copy`): every thread selects the smallest
/// elements of its chunk (sequential `partial_sort_copy`) into a buffer,
/// the candidates are sorted by the parallel `stable_sort` and the
/// smallest ones copied to the output.
/// NOTE: needs up to n + min(n, k * nthreads) elements of extra memory:
/// 	`T` must be default constructible and copyable
/// \param policy[in]: Execution policy specifying parallelization strategy
/// \param first[in]: Iterator to the beginning of the input range
/// \param last[in]: Iterator to the end of the input range
/// \param d_first[out]: Iterator to the beginning of the output range
/// \param d_last[out]: Iterator to the end of the output range
/// \param comp[in]: less-than comparison
/// \return iterator past the last written element
template<class ExecPolicy,
         class RandIt1,
         class RandIt2,
         class Compare,
         const AlgorithmSortConfig &config = algorithmSortConfig>
#if __cplusplus > 201709L
    requires ExecutionPolicy<ExecPolicy> &&
             std::random_access_iterator<RandIt1> &&
             std::random_access_iterator<RandIt2>
#endif
RandIt2 partial_sort_copy(ExecPolicy &&policy,
                          RandIt1 first,
                          RandIt1 last,
                          RandIt2 d_first,
                          RandIt2 d_last,
                          Compare comp) noexcept {
    using T = std::iter_value_t<RandIt1>;
    const size_t n = static_cast<size_t>(std::distance(first, last));
    const size_t k = static_cast<size_t>(std::distance(d_first, d_last));
    const uint32_t nthreads = should_par(policy, config, n);
    if (is_seq<ExecPolicy>(policy) || (nthreads < 2) || (k == 0)) {
        return cryptanalysislib::partial_sort_copy(first, last, d_first, d_last, comp);
    }

    // offsets of the candidates of every chunk in `cand`
    std::vector<size_t> off(nthreads + 1, 0);
    for (uint32_t t = 0; t < nthreads; ++t) {
        const size_t lo = (n * t) / nthreads, hi = (n * (t + 1)) / nthreads;
        off[t + 1] = off[t] + std::min(k, hi - lo);
    }

    std::vector<T> cand(off[nthreads]);
    auto &task_pool = *policy.pool();
    std::vector<std::future<void>> futures;
    for (uint32_t t = 0; t < nthreads; ++t) {
        const size_t lo = (n * t) / nthreads, hi = (n * (t + 1)) / nthreads;
        futures.emplace_back(task_pool.submit([&comp, &cand, &off, first, lo, hi, t]() {
            cryptanalysislib::partial_sort_copy(first + lo, first + hi,
                                                cand.begin() + off[t], cand.begin() + off[t + 1], comp);
        }));
    }
    internal::get_futures(futures);

    cryptanalysislib::stable_sort(policy, cand.begin(), cand.end(), comp);
    const size_t r = std::min(k, n);
    return cryptanalysislib::copy(policy, cand.begin(), cand.begin() + r, d_first);
}

/// same as above, with `<`
template<class ExecPolicy,
         class RandIt1,
         class RandIt2,
         const AlgorithmSortConfig &config = algorithmSortConfig>
#if __cplusplus > 201709L
    requires ExecutionPolicy<ExecPolicy> &&
             std::random_access_iterator<RandIt1> &&
             std::random_access_iterator<RandIt2>
#endif
RandIt2 partial_sort_copy(ExecPolicy &&policy,
                          RandIt1 first,
                          RandIt1 last,
                          RandIt2 d_first,
                          RandIt2 d_last) noexcept {
    return cryptanalysislib::partial_sort_copy
        <ExecPolicy, RandIt1, RandIt2, std::less<>, config>
        (std::forward<ExecPolicy>(policy), first, last, d_first, d_last, std::less<>());
}

/// Rearranges [first, last) such that `*nth` is the element which would be
/// there if the range was sorted w.r.t. `comp`, no element before `nth` is
/// greater and no element after it is smaller (parallel version of
/// `nth_element`). Quickselect with the parallel `stable_partition`: the
/// range is split into the elements less than, equal to and greater than a
/// median of three pivot, and only the part containing `nth` is processed
/// further. Once the range is too small for several threads, or after
/// 2*log2(n) rounds, the sequential `nth_element` finishes.
/// NOTE: the pivot is copied, the partitions copy the elements into a
/// 	buffer: `T` must be default constructible and copyable
/// \param policy[in]: Execution policy specifying parallelization strategy
/// \param first[in]: Iterator to the beginning of the range
/// \param nth[in]: position of the element to select
/// \param last[in]: Iterator to the end of the range
/// \param comp[in]: less-than comparison
template<class ExecPolicy,
         class RandIt,
         class Compare,
         const AlgorithmSortConfig &config = algorithmSortConfig>
#if __cplusplus > 201709L
    requires ExecutionPolicy<ExecPolicy> &&
             std::random_access_iterator<RandIt>
#endif
void nth_element(ExecPolicy &&policy,
                 RandIt first,
                 RandIt nth,
                 RandIt last,
                 Compare comp) noexcept {
    if ((first == last) || (nth == last)) {
        return;
    }

    uint32_t depth = 0;
    for (auto n = last - first; n > 1; n >>= 1) {
        depth += 2;
    }

    while (!is_seq<ExecPolicy>(policy) &&
           (should_par(policy, config, static_cast<size_t>(last - first)) >= 2) &&
           (depth > 0)) {
        depth -= 1;

        const auto size = last - first;
        std::iter_value_t<RandIt> pivot = first[0];
        const auto &x = first[size / 2], &y = first[size - 1];
        if (comp(pivot, x)) {
            if (comp(x, y)) {
                pivot = x;
            } else if (comp(pivot, y)) {
                pivot = y;
            }
        } else if (comp(y, pivot)) {
            pivot = comp(x, y) ? y : x;
        }

        const RandIt cut = cryptanalysislib::stable_partition(policy, first, last,
            [&comp, &pivot](const auto &e) { return comp(e, pivot); });
        if (nth < cut) {
            last = cut;
            continue;
        }

        // [cut, last): no element is less than the pivot
        const RandIt cut2 = cryptanalysislib::stable_partition(policy, cut, last,
            [&comp, &pivot](const auto &e) { return !comp(pivot, e); });
        if (nth < cut2) {
            // `*nth` is equivalent to the pivot
            return;
        }
        first = cut2;
    }

    cryptanalysislib::nth_element(first, nth, last, comp);
}

/// same as above, with `<`
template<class ExecPolicy,
         class RandIt,
         const AlgorithmSortConfig &config = algorithmSortConfig>
#if __cplusplus > 201709L
    requires ExecutionPolicy<ExecPolicy> &&
             std::random_access_iterator<RandIt>
#endif
void nth_element(ExecPolicy &&policy,
                 RandIt first,
                 RandIt nth,
                 RandIt last) noexcept {
    cryptanalysislib::nth_element
        <ExecPolicy, RandIt, std::less<>, config>
        (std::forward<ExecPolicy>(policy), first, nth, last, std::less<>());
}

};
