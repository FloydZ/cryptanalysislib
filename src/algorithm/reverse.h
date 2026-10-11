#pragma once 
#include <future>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "algorithm/algorithm.h"
#include "algorithm/swap.h"
#include "thread/thread.h"

/// NOTE: no explicit SIMD version: the swap loop below is auto-vectorized;
/// 	an explicit `TxN_t` version (block loads from both ends, lane reverse)
/// 	measured equal to it (2^16..2^24 uint32_t, AVX2).

namespace cryptanalysislib {
    /// Configuration for the parallel reverse
    struct AlgorithmReverseConfig : public AlgorithmConfig {
        constexpr static size_t min_size_per_thread = 1u << 16u;
    };
    constexpr static AlgorithmReverseConfig algorithmReverseConfig;

/// Reverses the order of the elements in [first, last) (like `std::reverse`)
/// NOTE: call it qualified (`cryptanalysislib::reverse`): an unqualified call
/// 	on `std::` iterators also finds `std::reverse` via ADL (ambiguous).
/// \tparam BidirIt bidirectional iterator type
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
template<class BidirIt>
constexpr
void reverse(BidirIt first,
             BidirIt last) noexcept {
    using iter_cat = typename std::iterator_traits<BidirIt>::iterator_category;
 
    // Tag dispatch, e.g. calling reverse_impl(first, last, iter_cat()),
    // can be used in C++14 and earlier modes.
    if constexpr (std::is_base_of_v<std::random_access_iterator_tag, iter_cat>) {
        if (first == last) {
            return;
        }
 
        for (--last; first < last; (void)++first, --last) {
            cryptanalysislib::iter_swap(first, last);
        }
    } else {
        while (first != last && first != --last) {
            cryptanalysislib::iter_swap(first++, last);
        }
    }
}

/// Copies [first, last) in reverse order to `d_first` (like `std::reverse_copy`)
/// \tparam BidirIt bidirectional iterator type of the input
/// \tparam OutputIt output iterator type
/// \param first[in]: Iterator to the beginning of the input range
/// \param last[in]: Iterator to the end of the input range
/// \param d_first[out]: beginning of the output, must not overlap the input
/// \return Iterator past the last written element
template<class BidirIt,
         class OutputIt>
constexpr 
OutputIt reverse_copy(BidirIt first,
                      BidirIt last,
                      OutputIt d_first) {
    for (; first != last; ++d_first) {
        *d_first = *(--last);
    }
    return d_first;
}

/// Reverses [first, last) (parallel version): the first half is split
/// between the threads, every thread swaps its part with the mirrored part
/// of the second half.
/// \param policy[in]: Execution policy specifying parallelization strategy
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
template<class ExecPolicy,
         class RandIt,
         const AlgorithmReverseConfig &config = algorithmReverseConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
void reverse(ExecPolicy &&policy,
             RandIt first,
             RandIt last) noexcept {
    const size_t n = static_cast<size_t>(std::distance(first, last));
    const uint32_t nthreads = should_par(policy, config, n);
    if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
        cryptanalysislib::reverse(first, last);
        return;
    }

    const size_t half = n / 2;
    auto &task_pool = *policy.pool();
    std::vector<std::future<void>> futures;
    for (uint32_t k = 0; k < nthreads; ++k) {
        const size_t lo = (half * k) / nthreads, hi = (half * (k + 1)) / nthreads;
        futures.emplace_back(task_pool.submit([first, n, lo, hi]() {
            for (size_t i = lo; i < hi; ++i) {
                cryptanalysislib::iter_swap(first + i, first + (n - 1 - i));
            }
        }));
    }
    for (auto &f : futures) { f.wait(); }
}
} // end namespace cryptanalysislib
