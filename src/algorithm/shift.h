#pragma once

#include <future>
#include <iterator>
#include <vector>

#include "algorithm/algorithm.h"
#include "thread/thread.h"


namespace cryptanalysislib {
    template<class ForwardIt>
#if __cplusplus > 201709L
    requires std::forward_iterator<ForwardIt>
#endif
    ForwardIt shift_left(ForwardIt first,
                         ForwardIt last,
                         typename std::iterator_traits<ForwardIt>::difference_type n);

    template<class ForwardIt>
#if __cplusplus > 201709L
    requires std::bidirectional_iterator<ForwardIt>
#endif
    ForwardIt shift_right(ForwardIt first,
                          ForwardIt last,
                          typename std::iterator_traits<ForwardIt>::difference_type n);

    /// Shifts the elements in the range [first, last) by n positions
    /// Elements are moved forward (towards first) if n is negative,
    /// or backward (towards last) if n is positive
    /// Elements moved before first or beyond last are discarded
    /// 
    /// \tparam ForwardIt Forward iterator type
    /// \param first Iterator to the first element in the range
    /// \param last Iterator one past the last element in the range
    /// \param n Number of positions to shift (positive shifts right, negative shifts left)
    /// \return Iterator to the new position of the first element that was not discarded
    template<class ForwardIt>
#if __cplusplus > 201709L
    requires std::bidirectional_iterator<ForwardIt>
#endif
    ForwardIt shift(ForwardIt first,
                    ForwardIt last, 
                    typename std::iterator_traits<ForwardIt>::difference_type n) {
        // positive: towards `last` (= shift_right), negative: towards `first` (= shift_left)
        if (n > 0) {
            return cryptanalysislib::shift_right(first, last, n);
        }

        if (n < 0) {
            cryptanalysislib::shift_left(first, last, -n);
        }

        return first;
    }
    
    /// Shifts elements left by n positions
    /// Elements are moved backward (towards first)
    /// 
    /// \tparam ForwardIt Forward iterator type
    /// \param first Iterator to the first element in the range
    /// \param last Iterator one past the last element in the range
    /// \param n Number of positions to shift left
    /// \return Iterator to the new end of the range (last - n)
    template<class ForwardIt>
#if __cplusplus > 201709L
    requires std::forward_iterator<ForwardIt>
#endif
    ForwardIt shift_left(ForwardIt first,
                         ForwardIt last,
                         typename std::iterator_traits<ForwardIt>::difference_type n) {
        if (n == 0) {
            return last;
        }
        
        if (n >= std::distance(first, last)) {
            // All elements would be shifted out of range
            return first;
        }
        
        // Move elements from [first+n, last) to [first, last-n), front to back
        ForwardIt src = first;
        for (typename std::iterator_traits<ForwardIt>::difference_type i = 0; i < n; ++i) {
            ++src;
        }
        for (; src != last; ++src, ++first) {
            *first = std::move(*src);
        }
        return first;
    }
    
    /// Shifts elements right by n positions
    /// Elements are moved forward (towards last)
    /// 
    /// \tparam ForwardIt Forward iterator type
    /// \param first Iterator to the first element in the range
    /// \param last Iterator one past the last element in the range
    /// \param n Number of positions to shift right
    /// \return Iterator to the new beginning of the range (first + n)
    template<class ForwardIt>
#if __cplusplus > 201709L
    requires std::bidirectional_iterator<ForwardIt>
#endif
    ForwardIt shift_right(ForwardIt first,
                          ForwardIt last,
                          typename std::iterator_traits<ForwardIt>::difference_type n) {
        using diff_t = typename std::iterator_traits<ForwardIt>::difference_type;
        if (n <= 0) {
            return first;
        }

        diff_t size = 0;
        for (ForwardIt it = first; it != last; ++it) {
            size += 1;
        }

        if (n >= size) {
            // All elements would be shifted out of range
            return last;
        }

        // Move elements from [first, last-n) to [first+n, last), back to front
        ForwardIt src = first;
        for (diff_t i = 0; i < size - n; ++i) {
            ++src;
        }

        ForwardIt dst = last;
        while (src != first) {
            --src;
            --dst;
            *dst = static_cast<typename std::iterator_traits<ForwardIt>::value_type &&>(*src);
        }

        // `dst` is now first + n
        return dst;
    }
    
    /// Configuration for the parallel shift algorithms
    /// NOTE: shifting is memory bound: measured (gcc, uint32_t, n = 1000):
    ///     2^20 elements: 51 µs sequential, 53 µs parallel; 2^24: 3.4 ms vs
    ///     2.1 ms. Hence at least 2^20 elements per thread.
    struct AlgorithmShiftConfig : public AlgorithmConfig {
        const size_t min_size_per_thread = 1u << 20u;
    };
    constexpr static AlgorithmShiftConfig algorithmShiftConfig;

    namespace internal {
        /// parallel shift of the `m` elements at `src` to `dst` (`dst = src - n`
        /// for a left shift, `dst = src + n` for a right shift). Every thread
        /// moves one block. The blocks overlap only at their borders: the
        /// first `n` source elements of block k+1 (left shift: the first `n`
        /// destination elements of block k+1) are overwritten by its
        /// neighbour, so they are saved first (phase 1), then all blocks are
        /// moved (phase 2). Needs `n <= m / nthreads`.
        template<const bool left,
                 class ExecPolicy,
                 class RandIt>
        void shift_parallel(ExecPolicy &&policy,
                            RandIt first,
                            const size_t m,
                            const size_t n,
                            const uint32_t nthreads) noexcept {
            using T = typename std::iterator_traits<RandIt>::value_type;
            auto &task_pool = *policy.pool();
            auto lo = [m, nthreads](const uint32_t k) { return (m * k) / nthreads; };

            // phase 1: the `n` elements at the start of every block k >= 1
            // (positions relative to the source start for a right shift and
            // to the destination start for a left shift)
            std::vector<std::vector<T>> saved(nthreads);
            {
                std::vector<std::future<void>> futures;
                for (uint32_t k = 1; k < nthreads; ++k) {
                    futures.emplace_back(task_pool.submit([&, k]() {
                        saved[k].resize(n);
                        RandIt s = first + lo(k);
                        for (size_t i = 0; i < n; ++i) { saved[k][i] = std::move(s[i]); }
                    }));
                }
                for (auto &f : futures) { f.wait(); }
            }

            // phase 2: move every block
            std::vector<std::future<void>> futures;
            for (uint32_t k = 0; k < nthreads; ++k) {
                futures.emplace_back(task_pool.submit([&, k]() {
                    const size_t b = lo(k), e = lo(k + 1);
                    if constexpr (left) {
                        // dst[i] = src[i + n] for i in [b, e), front to back;
                        // the sources in [e, e + n) are the saved elements of
                        // block k + 1 (the last block reads unshifted data)
                        const size_t direct = (k + 1 < nthreads) ? e - n : e;
                        for (size_t i = b; i < direct; ++i) {
                            first[i] = std::move(first[i + n]);
                        }
                        for (size_t i = direct; i < e; ++i) {
                            first[i] = std::move(saved[k + 1][i - direct]);
                        }
                    } else {
                        // dst[i + n] = src[i] for i in [b, e), back to front;
                        // the sources in [b, b + n) of block k >= 1 were saved
                        const size_t direct = (k > 0) ? b + n : b;
                        for (size_t i = e; i-- > direct;) {
                            first[i + n] = std::move(first[i]);
                        }
                        for (size_t i = direct; i-- > b;) {
                            first[i + n] = std::move(saved[k][i - b]);
                        }
                    }
                }));
            }
            for (auto &f : futures) { f.wait(); }
        }
    } // end namespace internal

    /// Shifts elements left by n positions (parallel version), see
    /// `internal::shift_parallel`
    /// Elements are moved backward (towards first)
    ///
    /// \tparam ExecPolicy Execution policy type
    /// \tparam RandIt random access iterator type
    /// \param policy Execution policy
    /// \param first Iterator to the first element in the range
    /// \param last Iterator one past the last element in the range
    /// \param n Number of positions to shift left
    /// \return Iterator to the new end of the range (last - n)
    template<class ExecPolicy,
             class RandIt,
             const AlgorithmShiftConfig &config = algorithmShiftConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
    RandIt shift_left(ExecPolicy&& policy,
                      RandIt first,
                      RandIt last,
                      typename std::iterator_traits<RandIt>::difference_type n) noexcept {
        const auto len = std::distance(first, last);
        if (n <= 0) {
            return last;
        }
        if (n >= len) {
            // All elements would be shifted out of range
            return first;
        }

        const size_t m = static_cast<size_t>(len - n);
        uint32_t nthreads = should_par(policy, config, m);
        // every block must contain at least `n` elements
        if (nthreads > m / static_cast<size_t>(n)) { nthreads = static_cast<uint32_t>(m / static_cast<size_t>(n)); }
        if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
            return cryptanalysislib::shift_left(first, last, n);
        }

        internal::shift_parallel<true>(policy, first, m, static_cast<size_t>(n), nthreads);
        return first + m;
    }
    
    /// Shifts elements right by n positions (parallel version), see
    /// `internal::shift_parallel`
    /// Elements are moved forward (towards last)
    ///
    /// \tparam ExecPolicy Execution policy type
    /// \tparam RandIt random access iterator type
    /// \param policy Execution policy
    /// \param first Iterator to the first element in the range
    /// \param last Iterator one past the last element in the range
    /// \param n Number of positions to shift right
    /// \return Iterator to the new beginning of the range (first + n)
    template<class ExecPolicy,
             class RandIt,
             const AlgorithmShiftConfig &config = algorithmShiftConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
    RandIt shift_right(ExecPolicy&& policy,
                       RandIt first,
                       RandIt last,
                       typename std::iterator_traits<RandIt>::difference_type n) noexcept {
        const auto len = std::distance(first, last);
        if (n <= 0) {
            return first;
        }
        if (n >= len) {
            // All elements would be shifted out of range
            return last;
        }

        const size_t m = static_cast<size_t>(len - n);
        uint32_t nthreads = should_par(policy, config, m);
        if (nthreads > m / static_cast<size_t>(n)) { nthreads = static_cast<uint32_t>(m / static_cast<size_t>(n)); }
        if (is_seq<ExecPolicy>(policy) || nthreads < 2) {
            return cryptanalysislib::shift_right(first, last, n);
        }

        internal::shift_parallel<false>(policy, first, m, static_cast<size_t>(n), nthreads);
        return first + n;
    }
}; // end namespace cryptanalysislib
