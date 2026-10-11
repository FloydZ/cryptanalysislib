#pragma once

#include <array>
#include <cstdint>
#include <future>
#include <iterator>
#include <type_traits>
#include <vector>

#include "find.h"
#include "algorithm/algorithm.h"
#include "algorithm/bits/popcount.h"
#include "thread/thread.h"

#ifdef USE_AVX2
#include <immintrin.h>
#endif


namespace cryptanalysislib {
#ifdef USE_AVX2
    namespace internal {
        /// for every 8 bit keep-mask: the `permutevar8x32` indices which move
        /// the kept lanes (in order) to the front
        constexpr std::array<std::array<uint32_t, 8>, 256> remove_u32_lut = []() {
            std::array<std::array<uint32_t, 8>, 256> t{};
            for (uint32_t m = 0; m < 256; ++m) {
                uint32_t k = 0;
                for (uint32_t i = 0; i < 8; ++i) {
                    if ((m >> i) & 1u) { t[m][k++] = i; }
                }
                for (; k < 8; ++k) { t[m][k] = 0; }
            }
            return t;
        }();

        /// stable inplace `remove` of `value` from `n` uint32_t
        /// NOTE: the write position never passes the read position, and a store
        ///     only overwrites the block which is already loaded, so inplace is safe
        /// \return new number of elements
        static inline size_t remove_u32_avx2(uint32_t *data,
                                             const size_t n,
                                             const uint32_t value) noexcept {
            const __m256i v = _mm256_set1_epi32((int)value);
            size_t i = 0, w = 0;
            for (; (i + 8) <= n; i += 8) {
                const __m256i d = _mm256_loadu_si256((const __m256i *)(data + i));
                const uint32_t eq = (uint32_t)_mm256_movemask_ps(_mm256_castsi256_ps(_mm256_cmpeq_epi32(d, v)));
                const uint32_t keep = (~eq) & 0xffu;
                const __m256i perm = _mm256_loadu_si256((const __m256i *)remove_u32_lut[keep].data());
                _mm256_storeu_si256((__m256i *)(data + w), _mm256_permutevar8x32_epi32(d, perm));
                w += cryptanalysislib::popcount::popcount(keep);
            }

            for (; i < n; ++i) {
                if (data[i] != value) { data[w++] = data[i]; }
            }
            return w;
        }
    } // end namespace internal
#endif

    template<class ForwardIt, 
             class T = typename std::iterator_traits<ForwardIt>::value_type>
    ForwardIt remove(ForwardIt first,
                     ForwardIt last,
                     const T& value) {
#ifdef USE_AVX2
        if constexpr (std::contiguous_iterator<ForwardIt> &&
                      std::is_same_v<std::iter_value_t<ForwardIt>, uint32_t> &&
                      std::is_convertible_v<T, uint32_t>) {
            const size_t n = static_cast<size_t>(last - first);
            return first + internal::remove_u32_avx2(std::to_address(first), n, uint32_t(value));
        }
#endif
        first = cryptanalysislib::find(first, last, value);
        if (first != last)
            for (ForwardIt i = first; ++i != last;)
                if (!(*i == value))
                    *first++ = std::move(*i);
        return first;
    }
    
    template<class ForwardIt,
             class UnaryPred>
    ForwardIt remove_if(ForwardIt first,
                        ForwardIt last,
                        UnaryPred p) {
        first = cryptanalysislib::find_if(first, last, p);
        if (first != last)
            for (ForwardIt i = first; ++i != last;)
                if (!p(*i))
                    *first++ = std::move(*i);
        return first;
    }
    
    template<class InputIt,
             class OutputIt,
             class T = typename std::iterator_traits<InputIt>::value_type>
    constexpr OutputIt remove_copy(InputIt first,
                                   InputIt last,
                                   OutputIt d_first,
                                   const T& value) {
        for (; first != last; ++first)
            if (!(*first == value))
                *d_first++ = *first;
        return d_first;
    }
    
    template<class InputIt,
             class OutputIt,
             class UnaryPred>
    constexpr OutputIt remove_copy_if(InputIt first,
                                      InputIt last,
                                      OutputIt d_first,
                                      UnaryPred p) {
        for (; first != last; ++first) {
            if (!p(*first)) {
                *d_first++ = *first;
            }
        }
        return d_first;
    }

    /// Configuration for the parallel remove algorithms
    /// NOTE: measured (gcc, 25% removed): the SIMD `remove` of uint32_t is
    ///     memory bound, its parallel version only pays off for large inputs
    ///     (2^20: 85 µs both, 2^22: 330 → 214 µs), hence `min_size_simd_parallel`.
    ///     A 20-round hash predicate in `remove_if`: 2^20: 10.7 → 1.5 ms.
    struct AlgorithmRemoveConfig : public AlgorithmConfig {
        constexpr static size_t min_size_per_thread = 1u << 15u;
        constexpr static size_t min_size_simd_parallel = 1u << 21u;
    };
    constexpr static AlgorithmRemoveConfig algorithmRemoveConfig;

    namespace internal {
        /// stable parallel remove: every chunk is compacted inplace in
        /// parallel by `chunk_remove(chunk_first, chunk_last)` (which returns
        /// the new chunk end), then the compacted chunks are moved together,
        /// in order.
        /// NOTE: the moves are sequential: moving the chunks in parallel is
        ///     not safe, as a destination can overlap the source of an earlier
        ///     chunk. Every move goes to the left, so a forward loop is safe.
        template<class ExecPolicy,
                 class RandIt,
                 class ChunkRemove>
        RandIt remove_parallel(ExecPolicy &&policy,
                               RandIt first,
                               const size_t n,
                               ChunkRemove chunk_remove,
                               const uint32_t nthreads) noexcept {
            auto &task_pool = *policy.pool();
            std::vector<std::future<RandIt>> futures;
            for (uint32_t k = 0; k < nthreads; ++k) {
                const size_t lo = (n * k) / nthreads, hi = (n * (k + 1)) / nthreads;
                futures.emplace_back(task_pool.submit([first, lo, hi, &chunk_remove]() {
                    return chunk_remove(first + lo, first + hi);
                }));
            }

            RandIt out = futures[0].get();
            for (uint32_t k = 1; k < nthreads; ++k) {
                RandIt src = first + (n * k) / nthreads;
                const RandIt end = futures[k].get();
                if (out == src) {
                    // nothing removed so far: no self move assignment
                    out = end;
                    continue;
                }
                for (; src != end; ++src, ++out) {
                    *out = std::move(*src);
                }
            }
            return out;
        }
    } // end namespace internal

    /// stable `remove_if` (parallel version), see `internal::remove_parallel`
    /// \return iterator to the new end of the range
    template<class ExecPolicy,
             class RandIt,
             class UnaryPred,
             const AlgorithmRemoveConfig &config = algorithmRemoveConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
    RandIt remove_if(ExecPolicy &&policy,
                     RandIt first,
                     RandIt last,
                     UnaryPred p) noexcept {
        const size_t n = static_cast<size_t>(std::distance(first, last));
        const uint32_t nthreads = should_par(policy, config, n);
        if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
            return cryptanalysislib::remove_if(first, last, p);
        }

        return internal::remove_parallel(policy, first, n,
            [&p](RandIt f, RandIt l) { return cryptanalysislib::remove_if(f, l, p); },
            nthreads);
    }

    /// stable `remove` (parallel version), see `internal::remove_parallel`.
    /// Every chunk uses the sequential `remove` (SIMD for uint32_t).
    /// \return iterator to the new end of the range
    template<class ExecPolicy,
             class RandIt,
             class T = typename std::iterator_traits<RandIt>::value_type,
             const AlgorithmRemoveConfig &config = algorithmRemoveConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
    RandIt remove(ExecPolicy &&policy,
                  RandIt first,
                  RandIt last,
                  const T &value) noexcept {
        const size_t n = static_cast<size_t>(std::distance(first, last));
#ifdef USE_AVX2
        if constexpr (std::contiguous_iterator<RandIt> &&
                      std::is_same_v<std::iter_value_t<RandIt>, uint32_t>) {
            if (n < config.min_size_simd_parallel) {
                return cryptanalysislib::remove(first, last, value);
            }
        }
#endif
        const uint32_t nthreads = should_par(policy, config, n);
        if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
            return cryptanalysislib::remove(first, last, value);
        }

        return internal::remove_parallel(policy, first, n,
            [&value](RandIt f, RandIt l) { return cryptanalysislib::remove(f, l, value); },
            nthreads);
    }
}; // end namespace cryptanalysislib
