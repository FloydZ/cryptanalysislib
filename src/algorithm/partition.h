#pragma once 

#include <algorithm>
#include <iterator>
#include <utility>

#include <future>
#include <vector>

#include "algorithm/rotate.h"
#include "algorithm/algorithm.h"
#include "algorithm/copy.h"
#include "algorithm/find.h"
#include "thread/thread.h"


/// TODO simd versions

namespace cryptanalysislib {

/// Checks whether all elements satisfying `p` come before all others
/// (like `std::is_partitioned`)
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param p[in]: unary predicate
/// \return true if [first, last) is partitioned by `p` (also if empty)
template<class InputIt,
         class UnaryPred>
constexpr 
bool is_partitioned(InputIt first,
                    InputIt last,
                    UnaryPred p) {
    for (; first != last; ++first)
        if (!p(*first))
            break;
    for (; first != last; ++first)
        if (p(*first))
            return false;
    return true;
}

/// Finds the end of the first group of a range partitioned by `p` via binary
/// search (like `std::partition_point`)
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param p[in]: unary predicate, the range must be partitioned by it
/// \return iterator to the first element for which `p` is false (or last)
template<class ForwardIt,
         class UnaryPred>
constexpr
ForwardIt partition_point(ForwardIt first,
                          ForwardIt last,
                          UnaryPred p) {
    for (auto length = std::distance(first, last); 0 < length; )
    {
        auto half = length / 2;
        auto middle = std::next(first, half);
        if (p(*middle))
        {
            first = std::next(middle);
            length -= (half + 1);
        }
        else
            length = half;
    }
 
    return first;
}

/// Reorders [first, last) such that all elements for which `p` is true come
/// first (like `std::partition`). NOT stable, see `stable_partition`.
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param p[in]: unary predicate
/// \return iterator to the first element of the second group
template<class ForwardIt,
         class UnaryPred>
constexpr
ForwardIt partition(ForwardIt first,
                    ForwardIt last, 
                    UnaryPred p) {
    first = std::find_if_not(first, last, p);
    if (first == last)
        return first;
 
    for (auto i = std::next(first); i != last; ++i)
        if (p(*i))
        {
            std::iter_swap(i, first);
            ++first;
        }
 
    return first;
}

/// Copies the elements for which `p` is true to `d_first_true` and the others
/// to `d_first_false`, keeping their order (like `std::partition_copy`)
/// \param first[in]: Iterator to the beginning of the input range
/// \param last[in]: Iterator to the end of the input range
/// \param d_first_true[out]: output for the elements satisfying `p`
/// \param d_first_false[out]: output for the other elements
/// \param p[in]: unary predicate
/// \return pair of iterators past the last written element of both outputs
template<class InputIt,
         class OutputIt1,
         class OutputIt2,
         class UnaryPred>
constexpr
std::pair<OutputIt1, OutputIt2>
    partition_copy(InputIt first,
                   InputIt last,
                   OutputIt1 d_first_true,
                   OutputIt2 d_first_false,
                   UnaryPred p) {
    for (; first != last; ++first) {
        if (p(*first)) {
            *d_first_true = *first;
            ++d_first_true;
        } else {
            *d_first_false = *first;
            ++d_first_false;
        }
    }
 
    return std::pair<OutputIt1, OutputIt2>(d_first_true, d_first_false);
}

} // end namespace cryptanalysislib

namespace cryptanalysislib::internal {
    /// stable partition of the `len` elements starting at `first`
    /// divide and conquer: partition both halves, then rotate the
    /// `false` part of the left half behind the `true` part of the right half.
    /// \return iterator to the first element for which `p` is false
    template<class ForwardIt,
             class UnaryPred>
    constexpr ForwardIt stable_partition_rec(ForwardIt first,
                                             const size_t len,
                                             UnaryPred &p) {
        if (len == 1) {
            ForwardIt next = first;
            ++next;
            return p(*first) ? next : first;
        }

        const size_t half = len / 2;
        ForwardIt middle = first;
        for (size_t i = 0; i < half; ++i) { ++middle; }

        ForwardIt left = stable_partition_rec(first, half, p);
        ForwardIt right = stable_partition_rec(middle, len - half, p);
        // [left, middle) are false, [middle, right) are true
        return cryptanalysislib::rotate(left, middle, right);
    }
} // end namespace cryptanalysislib::internal

namespace cryptanalysislib {

/// Reorders [first, last) such that all elements for which `p` is true come
/// first, keeping the relative order within both groups.
/// NOTE: in place; `p` is called exactly once per element; O(n log n) moves.
/// \return iterator to the first element of the second group
template<class ForwardIt,
         class UnaryPred>
constexpr
ForwardIt stable_partition(ForwardIt first,
                           ForwardIt last, 
                           UnaryPred p) {
    // skip the prefix, which is already in place
    while ((first != last) && p(*first)) {
        ++first;
    }

    if (first == last) {
        return first;
    }

    // `*first` is known to be false: partition the rest and move `*first`
    // behind its `true` elements
    ForwardIt next = first;
    ++next;

    size_t len = 0;
    for (ForwardIt it = next; it != last; ++it) {
        len += 1;
    }

    const ForwardIt r = (len == 0) ? next : cryptanalysislib::internal::stable_partition_rec(next, len, p);
    return cryptanalysislib::rotate(first, next, r);
}
} // end namespace cryptanalysislib

namespace cryptanalysislib {
    /// Configuration for the parallel partition algorithms
    struct AlgorithmPartitionConfig : public AlgorithmConfig {
        constexpr static size_t min_size_per_thread = 1u << 15u;
    };
    constexpr static AlgorithmPartitionConfig algorithmPartitionConfig;

    namespace internal {
        /// pass 1 of the parallel stable partition: counts the `true` elements
        /// of each of the `nthreads` chunks of the `n` elements at `first`
        template<class ExecPolicy,
                 class RandIt,
                 class UnaryPred>
        std::vector<size_t> partition_count_parallel(ExecPolicy &&policy,
                                                     RandIt first,
                                                     const size_t n,
                                                     UnaryPred &p,
                                                     const uint32_t nthreads) noexcept {
            auto &task_pool = *policy.pool();
            std::vector<size_t> ntrue(nthreads, 0);
            std::vector<std::future<void>> futures;
            for (uint32_t k = 0; k < nthreads; ++k) {
                const size_t lo = (n * k) / nthreads, hi = (n * (k + 1)) / nthreads;
                futures.emplace_back(task_pool.submit([&ntrue, first, lo, hi, k, &p]() {
                    size_t c = 0;
                    for (size_t i = lo; i < hi; ++i) { c += p(first[i]) ? 1u : 0u; }
                    ntrue[k] = c;
                }));
            }
            for (auto &f : futures) { f.wait(); }
            return ntrue;
        }

        /// pass 2 of the parallel stable partition: copies every chunk to its
        /// prefix sum offsets in both outputs (`ntrue` from pass 1)
        /// \return (#true, #false)
        template<class ExecPolicy,
                 class RandIt,
                 class OutIt1,
                 class OutIt2,
                 class UnaryPred>
        std::pair<size_t, size_t> partition_scatter_parallel(ExecPolicy &&policy,
                                                             RandIt first,
                                                             const size_t n,
                                                             OutIt1 d_first_true,
                                                             OutIt2 d_first_false,
                                                             UnaryPred &p,
                                                             const std::vector<size_t> &ntrue) noexcept {
            auto &task_pool = *policy.pool();
            const uint32_t nthreads = static_cast<uint32_t>(ntrue.size());
            size_t tsum = 0;
            std::vector<std::future<void>> futures;
            for (uint32_t k = 0; k < nthreads; ++k) {
                const size_t lo = (n * k) / nthreads, hi = (n * (k + 1)) / nthreads;
                const size_t toff = tsum, foff = lo - tsum;
                tsum += ntrue[k];
                futures.emplace_back(task_pool.submit([=, &p]() {
                    cryptanalysislib::partition_copy(first + lo, first + hi,
                                                     d_first_true + toff,
                                                     d_first_false + foff, p);
                }));
            }
            for (auto &f : futures) { f.wait(); }
            return {tsum, n - tsum};
        }
    } // end namespace internal

    /// Checks whether all elements satisfying `p` come before all others
    /// (parallel version): the first element not satisfying `p` is searched
    /// in parallel, then in parallel whether any later element satisfies `p`.
    template<class ExecPolicy,
             class RandIt,
             class UnaryPred,
             const AlgorithmPartitionConfig &config = algorithmPartitionConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
    bool is_partitioned(ExecPolicy &&policy,
                        RandIt first,
                        RandIt last,
                        UnaryPred p) noexcept {
        const size_t n = static_cast<size_t>(std::distance(first, last));
        if (is_seq<ExecPolicy>(policy) || should_par(policy, config, n) == 0) {
            return cryptanalysislib::is_partitioned(first, last, p);
        }

        const RandIt f = cryptanalysislib::find_if_not(policy, first, last, p);
        return cryptanalysislib::find_if(policy, f, last, p) == last;
    }

    /// Stable partition copy (parallel version), see `partition_copy`
    /// NOTE: `p` is evaluated twice per element (so it must be pure).
    /// \return pair of iterators past the last written element of both outputs
    template<class ExecPolicy,
             class RandIt,
             class OutIt1,
             class OutIt2,
             class UnaryPred,
             const AlgorithmPartitionConfig &config = algorithmPartitionConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt> &&
             std::random_access_iterator<OutIt1> &&
             std::random_access_iterator<OutIt2>
#endif
    std::pair<OutIt1, OutIt2> partition_copy(ExecPolicy &&policy,
                                             RandIt first,
                                             RandIt last,
                                             OutIt1 d_first_true,
                                             OutIt2 d_first_false,
                                             UnaryPred p) noexcept {
        const size_t n = static_cast<size_t>(std::distance(first, last));
        const uint32_t nthreads = should_par(policy, config, n);
        if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
            return cryptanalysislib::partition_copy(first, last, d_first_true, d_first_false, p);
        }

        const auto ntrue = internal::partition_count_parallel(policy, first, n, p, nthreads);
        const auto c = internal::partition_scatter_parallel(policy, first, n, d_first_true, d_first_false, p, ntrue);
        return {d_first_true + c.first, d_first_false + c.second};
    }

    /// Stable partition (parallel version): the `true` elements are counted
    /// in parallel, then both groups are copied in parallel into one buffer
    /// (n elements of extra memory), which is copied back in parallel.
    /// NOTE: `p` is evaluated twice per element (so it must be pure).
    /// NOTE: the elements are copied, not moved: `T` must be copyable
    /// \return iterator to the first element of the second group
    template<class ExecPolicy,
             class RandIt,
             class UnaryPred,
             const AlgorithmPartitionConfig &config = algorithmPartitionConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
    RandIt stable_partition(ExecPolicy &&policy,
                            RandIt first,
                            RandIt last,
                            UnaryPred p) noexcept {
        using T = typename std::iterator_traits<RandIt>::value_type;
        const size_t n = static_cast<size_t>(std::distance(first, last));
        const uint32_t nthreads = should_par(policy, config, n);
        if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
            return cryptanalysislib::stable_partition(first, last, p);
        }

        const auto ntrue = internal::partition_count_parallel(policy, first, n, p, nthreads);
        size_t tsum = 0;
        for (const size_t c : ntrue) { tsum += c; }

        std::vector<T> tmp(n);
        internal::partition_scatter_parallel(policy, first, n, tmp.begin(), tmp.begin() + tsum, p, ntrue);
        cryptanalysislib::copy(policy, tmp.begin(), tmp.end(), first);
        return first + tsum;
    }
} // end namespace cryptanalysislib
