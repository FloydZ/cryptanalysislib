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
#include "algorithm/remove.h"
#include "algorithm/bits/popcount.h"
#include "thread/thread.h"
#include "memory/memcpy.h"

#ifdef USE_AVX512F
#include "simd/simd.h"
#endif

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

namespace internal {
    /// SIMD `partition_copy` of `n` uint32_t on the predicate `x < pivot`
    /// \return (#true, #false)
    static inline std::pair<size_t, size_t> partition_copy_less_u32(const uint32_t *in,
                                                                    const size_t n,
                                                                    uint32_t *d_first_true,
                                                                    uint32_t *d_first_false,
                                                                    const uint32_t pivot) noexcept {
        size_t i = 0, t = 0, f = 0;
#if defined(USE_AVX512F)
        // the compress stores only write the selected lanes
        const uint32x16_t p = uint32x16_t::set1(pivot);
        for (; (i + 16) <= n; i += 16) {
            const uint32x16_t d = uint32x16_t::unaligned_load(in + i);
            const uint16_t mt = uint32x16_t::lt(d, p);
            const uint16_t mf = (uint16_t)~mt;
            uint32x16_t::compress(d_first_true + t, d, Mask<16>(mt));
            uint32x16_t::compress(d_first_false + f, d, Mask<16>(mf));
            const uint32_t c = cryptanalysislib::popcount::popcount((uint32_t)mt);
            t += c; f += 16u - c;
        }
#elif defined(USE_AVX2)
        // NOTE: the lanes are moved to the front of a register, which is
        // 	stored into a buffer, as the outputs may not have space for 8
        // 	elements. AVX2 has no unsigned compare: the sign bit is flipped.
        const __m256i sign = _mm256_set1_epi32((int)0x80000000u);
        const __m256i p = _mm256_xor_si256(_mm256_set1_epi32((int)pivot), sign);
        alignas(32) uint32_t buf[8];
        for (; (i + 8) <= n; i += 8) {
            const __m256i d = _mm256_loadu_si256((const __m256i *)(in + i));
            const __m256i lt = _mm256_cmpgt_epi32(p, _mm256_xor_si256(d, sign));
            const uint32_t mt = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(lt));
            const uint32_t mf = (~mt) & 0xffu;
            const uint32_t c = cryptanalysislib::popcount::popcount(mt);

            const __m256i pt = _mm256_loadu_si256((const __m256i *)remove_u32_lut[mt].data());
            _mm256_store_si256((__m256i *)buf, _mm256_permutevar8x32_epi32(d, pt));
            cryptanalysislib::memcpy(d_first_true + t, buf, c);

            const __m256i pf = _mm256_loadu_si256((const __m256i *)remove_u32_lut[mf].data());
            _mm256_store_si256((__m256i *)buf, _mm256_permutevar8x32_epi32(d, pf));
            cryptanalysislib::memcpy(d_first_false + f, buf, 8u - c);
            t += c; f += 8u - c;
        }
#endif
        for (; i < n; ++i) {
            if (in[i] < pivot) {
                d_first_true[t++] = in[i];
            } else {
                d_first_false[f++] = in[i];
            }
        }
        return {t, f};
    }
} // end namespace internal

/// `partition_copy` on the predicate `x < pivot`: copies the elements smaller
/// than `pivot` to `d_first_true` and the others to `d_first_false`, keeping
/// their order. For contiguous ranges of uint32_t the comparisons and
/// copies are done with SIMD (AVX2, AVX-512).
/// \param first[in]: Iterator to the beginning of the input range
/// \param last[in]: Iterator to the end of the input range
/// \param d_first_true[out]: output for the elements < pivot
/// \param d_first_false[out]: output for the other elements
/// \param pivot[in]:
/// \return pair of iterators past the last written element of both outputs
template<class InputIt,
         class OutputIt1,
         class OutputIt2,
         class T = typename std::iterator_traits<InputIt>::value_type>
std::pair<OutputIt1, OutputIt2>
    partition_copy_less(InputIt first,
                        InputIt last,
                        OutputIt1 d_first_true,
                        OutputIt2 d_first_false,
                        const T &pivot) noexcept {
    if constexpr (std::contiguous_iterator<InputIt> &&
                  std::contiguous_iterator<OutputIt1> &&
                  std::contiguous_iterator<OutputIt2> &&
                  std::is_same_v<std::iter_value_t<InputIt>, uint32_t> &&
                  std::is_same_v<std::iter_value_t<OutputIt1>, uint32_t> &&
                  std::is_same_v<std::iter_value_t<OutputIt2>, uint32_t>) {
        const auto r = internal::partition_copy_less_u32(std::to_address(first),
                                                         static_cast<size_t>(last - first),
                                                         std::to_address(d_first_true),
                                                         std::to_address(d_first_false),
                                                         uint32_t(pivot));
        return {d_first_true + r.first, d_first_false + r.second};
    }

    return cryptanalysislib::partition_copy(first, last, d_first_true, d_first_false,
        [&pivot](const auto &x) { return x < pivot; });
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
