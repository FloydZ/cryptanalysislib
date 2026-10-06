#pragma once

#include <utility>
#include <iterator>

#include "simd/simd.h"
#include "algorithm/algorithm.h"
#include "algorithm/bits/ffs.h"

namespace cryptanalysislib {

	/// Configuration for mismatch algorithms with threading and alignment settings
	struct AlgorithmMismatchConfig /* : public AlgorithmConfig */ {
		// NOTE multithreaded find is extremly slow
		const size_t min_size_per_thread = 1048576u;
		const bool aligned_instructions = false;
	};
    constexpr static AlgorithmMismatchConfig algorithmMismatchConfig;

    namespace internal {
        
        // TODO test 
        /// SIMD-optimized mismatch finding for integral arrays
        /// \tparam T Integral type for array elements
        /// \tparam config Algorithm configuration (default: algorithmMismatchConfig)
        /// \param data1[in]: First array of integers
        /// \param data2[in]: Second array of integers
        /// \param n[in]: Length of both arrays
        /// \return Index of first mismatch, or n if arrays are equal
        template<typename T,
                 const AlgorithmMismatchConfig &config = algorithmMismatchConfig>
            requires std::is_integral_v<T>
        [[nodiscard]] constexpr static inline size_t mismatch_simd_uXX(const T *data1,
                                                                        const T *data2,
                                                                        const size_t n) noexcept {
            using S = SIMDSelector<T>;
            
            constexpr size_t t = S::LIMBS;
            static_assert(t <= 64);
            // one bit per lane
            constexpr uint64_t full = (t == 64) ? uint64_t(-1ull) : ((1ull << t) - 1ull);
            size_t i = 0;
            
            // SIMD loop - process S::LIMBS elements at a time
            for (; (i + t) <= n; i += t) {
                const auto d1 = S::template load<config.aligned_instructions>(data1 + i);
                const auto d2 = S::template load<config.aligned_instructions>(data2 + i);
                const uint64_t cmp = uint64_t(d1 == d2) & full;
                
                // If comparison result is not all ones (i.e., there's a mismatch)
                if (cmp != full) [[unlikely]] {
                    // Find the first mismatch within this SIMD chunk
                    const uint64_t mismatch_mask = (~cmp) & full;
                    return i + ffs<uint64_t>(mismatch_mask) - 1u;
                }
            }
            
            // Handle remaining elements (tail management)
            for (; i < n; i++) {
                if (data1[i] != data2[i]) {
                    return i;
                }
            }
            
            return n; // Arrays are equal
        }
        /// Parallel search for the first index k < n with !p(first1[k], first2[k]).
        /// Every chunk works on absolute offsets into both ranges.
        /// \return k, or n if the first n elements match
        template<class ExecPolicy,
                 class InputIt1,
                 class InputIt2,
                 class BinaryPred>
        size_t mismatch_parallel(ExecPolicy &&policy,
                                 InputIt1 first1,
                                 const size_t n,
                                 InputIt2 first2,
                                 BinaryPred p,
                                 const uint32_t nthreads) noexcept {
            auto chunk = [first1, first2, n, p](InputIt1 b, InputIt1 e) noexcept -> size_t {
                const size_t off = static_cast<size_t>(b - first1);
                InputIt2 b2 = first2 + off;
                for (size_t k = off; b != e; ++b, ++b2, ++k) {
                    if (!p(*b, *b2)) {
                        return k;
                    }
                }
                return n;
            };

            auto futures = internal::parallel_chunk_for_1(policy, first1, first1 + n,
                                                          chunk, (size_t *)nullptr, 1, nthreads);
            // wait for every chunk and keep the smallest mismatch index
            size_t ret = n;
            for (auto &f : futures) {
                const size_t k = f.get();
                if (k < ret) {
                    ret = k;
                }
            }
            return ret;
        }
    } // end namespace internal

    /// Finds first mismatch between two ranges using default equality (sequential version)
    /// \tparam InputIt1 Forward iterator type for first range
    /// \tparam InputIt2 Forward iterator type for second range
    /// \tparam config Algorithm configuration (default: algorithmMismatchConfig)
    /// \param first1[in]: Iterator to the beginning of the first range
    /// \param last1[in]: Iterator to the end of the first range
    /// \param first2[in]: Iterator to the beginning of the second range
    /// \return Pair of iterators pointing to the first mismatch
    template<class InputIt1, 
             class InputIt2,
             const AlgorithmMismatchConfig config=algorithmMismatchConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt1> &&
                 std::forward_iterator<InputIt2>
#endif
    constexpr std::pair<InputIt1, InputIt2>
    mismatch(InputIt1 first1, 
             InputIt1 last1, 
             InputIt2 first2) noexcept {
        while (first1 != last1 && *first1 == *first2) {
            ++first1, ++first2;
        }
        return std::make_pair(first1, first2);
    }

    /// Finds first mismatch between two ranges using default equality (parallel version)
    /// \tparam ExecPolicy Execution policy type for parallel execution
    /// \tparam InputIt1 Random access iterator type for first range
    /// \tparam InputIt2 Random access iterator type for second range
    /// \tparam config Algorithm configuration (default: algorithmMismatchConfig)
    /// \param policy[in]: Execution policy specifying parallelization strategy
    /// \param first1[in]: Iterator to the beginning of the first range
    /// \param last1[in]: Iterator to the end of the first range
    /// \param first2[in]: Iterator to the beginning of the second range
    /// \return Pair of iterators pointing to the first mismatch
    template<class ExecPolicy,
             class InputIt1, 
             class InputIt2,
             const AlgorithmMismatchConfig config=algorithmMismatchConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<InputIt1> &&
                 std::random_access_iterator<InputIt2>
#endif
    constexpr std::pair<InputIt1, InputIt2>
    mismatch(ExecPolicy &&policy,
             InputIt1 first1, 
             InputIt1 last1, 
             InputIt2 first2) noexcept {
		using diff_t = typename std::iterator_traits<InputIt1>::difference_type;
		const diff_t size = std::distance(first1, last1);
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::mismatch
                <InputIt1, InputIt2, config>
                (first1, last1, first2);
		}

        const size_t k = internal::mismatch_parallel(policy, first1, static_cast<size_t>(size),
                                                     first2, [](const auto &a, const auto &b) { return a == b; }, nthreads);
        return {first1 + k, first2 + k};
    }
    
    /// Finds first mismatch between two ranges using custom predicate (sequential version)
    /// \tparam InputIt1 Forward iterator type for first range
    /// \tparam InputIt2 Forward iterator type for second range
    /// \tparam BinaryPred Binary predicate type for element comparison
    /// \tparam config Algorithm configuration (default: algorithmMismatchConfig)
    /// \param first1[in]: Iterator to the beginning of the first range
    /// \param last1[in]: Iterator to the end of the first range
    /// \param first2[in]: Iterator to the beginning of the second range
    /// \param p[in]: Binary predicate for element comparison
    /// \return Pair of iterators pointing to the first mismatch
    template<class InputIt1,
             class InputIt2, 
             class BinaryPred,
             const AlgorithmMismatchConfig config=algorithmMismatchConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt1> &&
                 std::forward_iterator<InputIt2> &&
    			 std::regular_invocable<BinaryPred,
									const typename InputIt1::value_type&,
                                    const typename InputIt2::value_type&>
#endif
    constexpr std::pair<InputIt1, InputIt2>
    mismatch(InputIt1 first1,
             InputIt1 last1,
             InputIt2 first2, 
             BinaryPred p) noexcept {
        while (first1 != last1 && p(*first1, *first2)) {
            ++first1, ++first2;
        }

        return std::make_pair(first1, first2);
    }

    /// Finds first mismatch between two ranges using custom predicate (parallel version)
    /// \tparam ExecPolicy Execution policy type for parallel execution
    /// \tparam InputIt1 Random access iterator type for first range
    /// \tparam InputIt2 Random access iterator type for second range
    /// \tparam BinaryPred Binary predicate type for element comparison
    /// \tparam config Algorithm configuration (default: algorithmMismatchConfig)
    /// \param policy[in]: Execution policy specifying parallelization strategy
    /// \param first1[in]: Iterator to the beginning of the first range
    /// \param last1[in]: Iterator to the end of the first range
    /// \param first2[in]: Iterator to the beginning of the second range
    /// \param p[in]: Binary predicate for element comparison
    /// \return Pair of iterators pointing to the first mismatch
    template<class ExecPolicy,
             class InputIt1, 
             class InputIt2,
             class BinaryPred,
             const AlgorithmMismatchConfig config=algorithmMismatchConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<InputIt1> &&
                 std::random_access_iterator<InputIt2> &&
    			 std::regular_invocable<BinaryPred,
									const typename InputIt1::value_type&,
                                    const typename InputIt2::value_type&>
#endif
    constexpr std::pair<InputIt1, InputIt2>
    mismatch(ExecPolicy &&policy,
             InputIt1 first1, 
             InputIt1 last1, 
             InputIt2 first2,
             BinaryPred p) noexcept {
		using diff_t = typename std::iterator_traits<InputIt1>::difference_type;
		const diff_t size = std::distance(first1, last1);
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::mismatch
                <InputIt1, InputIt2, BinaryPred, config>
                (first1, last1, first2, p);
		}

        const size_t k = internal::mismatch_parallel(policy, first1, static_cast<size_t>(size),
                                                     first2, p, nthreads);
        return {first1 + k, first2 + k};
    }

    /// Finds first mismatch between two full ranges using default equality (sequential version)
    /// \tparam InputIt1 Forward iterator type for first range
    /// \tparam InputIt2 Forward iterator type for second range
    /// \tparam config Algorithm configuration (default: algorithmMismatchConfig)
    /// \param first1[in]: Iterator to the beginning of the first range
    /// \param last1[in]: Iterator to the end of the first range
    /// \param first2[in]: Iterator to the beginning of the second range
    /// \param last2[in]: Iterator to the end of the second range
    /// \return Pair of iterators pointing to the first mismatch
    template<class InputIt1, 
             class InputIt2,
             const AlgorithmMismatchConfig config=algorithmMismatchConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt1> &&
                 std::forward_iterator<InputIt2>
#endif
    constexpr std::pair<InputIt1, InputIt2>
    mismatch(InputIt1 first1,
             InputIt1 last1, 
             InputIt2 first2, 
             InputIt2 last2) noexcept {
        while (first1 != last1 && first2 != last2 && *first1 == *first2) {
            ++first1, ++first2;
        }
     
        return std::make_pair(first1, first2);
    }
    
    /// Finds first mismatch between two full ranges using default equality (parallel version)
    /// \tparam ExecPolicy Execution policy type for parallel execution
    /// \tparam InputIt1 Random access iterator type for first range
    /// \tparam InputIt2 Random access iterator type for second range
    /// \tparam config Algorithm configuration (default: algorithmMismatchConfig)
    /// \param policy[in]: Execution policy specifying parallelization strategy
    /// \param first1[in]: Iterator to the beginning of the first range
    /// \param last1[in]: Iterator to the end of the first range
    /// \param first2[in]: Iterator to the beginning of the second range
    /// \param last2[in]: Iterator to the end of the second range
    /// \return Pair of iterators pointing to the first mismatch
    template<class ExecPolicy,
             class InputIt1, 
             class InputIt2,
             const AlgorithmMismatchConfig config=algorithmMismatchConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<InputIt1> &&
                 std::random_access_iterator<InputIt2>
#endif
    constexpr std::pair<InputIt1, InputIt2>
    mismatch(ExecPolicy &&policy,
             InputIt1 first1, 
             InputIt1 last1, 
             InputIt2 first2,
             InputIt2 last2) noexcept {
		using diff_t = typename std::iterator_traits<InputIt1>::difference_type;
		const diff_t size = std::distance(first1, last1);
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::mismatch
                <InputIt1, InputIt2, config>
                (first1, last1, first2, last2);
		}

        const auto n1 = last1 - first1;
        const auto n2 = last2 - first2;
        const size_t n = static_cast<size_t>(n1 < n2 ? n1 : n2);
        const size_t k = internal::mismatch_parallel(policy, first1, n,
                                                     first2, [](const auto &a, const auto &b) { return a == b; }, nthreads);
        return {first1 + k, first2 + k};
    }
    
    /// Finds first mismatch between two full ranges using custom predicate (sequential version)
    /// \tparam InputIt1 Forward iterator type for first range
    /// \tparam InputIt2 Forward iterator type for second range
    /// \tparam BinaryPred Binary predicate type for element comparison
    /// \tparam config Algorithm configuration (default: algorithmMismatchConfig)
    /// \param first1[in]: Iterator to the beginning of the first range
    /// \param last1[in]: Iterator to the end of the first range
    /// \param first2[in]: Iterator to the beginning of the second range
    /// \param last2[in]: Iterator to the end of the second range
    /// \param p[in]: Binary predicate for element comparison
    /// \return Pair of iterators pointing to the first mismatch
    template<class InputIt1, 
             class InputIt2, 
             class BinaryPred,
             const AlgorithmMismatchConfig config=algorithmMismatchConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt1> &&
                 std::forward_iterator<InputIt2> &&
    			 std::regular_invocable<BinaryPred,
									const typename InputIt1::value_type&,
                                    const typename InputIt2::value_type&>
#endif
    constexpr std::pair<InputIt1, InputIt2>
    mismatch(InputIt1 first1,
             InputIt1 last1,
             InputIt2 first2, 
             InputIt2 last2, 
             BinaryPred p)  {
        while (first1 != last1 && first2 != last2 && p(*first1, *first2)) {
            ++first1, ++first2;
        }
     
        return std::make_pair(first1, first2);
    }

    /// Finds first mismatch between two full ranges using custom predicate (parallel version)
    /// \tparam ExecPolicy Execution policy type for parallel execution
    /// \tparam InputIt1 Random access iterator type for first range
    /// \tparam InputIt2 Random access iterator type for second range
    /// \tparam BinaryPred Binary predicate type for element comparison
    /// \tparam config Algorithm configuration (default: algorithmMismatchConfig)
    /// \param policy[in]: Execution policy specifying parallelization strategy
    /// \param first1[in]: Iterator to the beginning of the first range
    /// \param last1[in]: Iterator to the end of the first range
    /// \param first2[in]: Iterator to the beginning of the second range
    /// \param last2[in]: Iterator to the end of the second range
    /// \param p[in]: Binary predicate for element comparison
    /// \return Pair of iterators pointing to the first mismatch
    template<class ExecPolicy,
             class InputIt1, 
             class InputIt2,
             class BinaryPred,
             const AlgorithmMismatchConfig config=algorithmMismatchConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<InputIt1> &&
                 std::random_access_iterator<InputIt2> &&
    			 std::regular_invocable<BinaryPred,
									const typename InputIt1::value_type&,
                                    const typename InputIt2::value_type&>
#endif
    constexpr std::pair<InputIt1, InputIt2>
    mismatch(ExecPolicy &&policy,
             InputIt1 first1, 
             InputIt1 last1, 
             InputIt2 first2,
             InputIt2 last2,
             BinaryPred p) noexcept {
		using diff_t = typename std::iterator_traits<InputIt1>::difference_type;
		const diff_t size = std::distance(first1, last1);
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::mismatch
                <InputIt1, InputIt2, BinaryPred, config>
                (first1, last1, first2, last2, p);
		}

        const auto n1 = last1 - first1;
        const auto n2 = last2 - first2;
        const size_t n = static_cast<size_t>(n1 < n2 ? n1 : n2);
        const size_t k = internal::mismatch_parallel(policy, first1, n,
                                                     first2, p, nthreads);
        return {first1 + k, first2 + k};
    }
}; // end namespace
