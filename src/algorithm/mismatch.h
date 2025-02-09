#pragma once

#include <utility>
#include <iterator>

#include "algorithm/algorithm.h"

namespace cryptanalysislib {

	struct AlgorithmMismatchConfig /* : public AlgorithmConfig */ {
		// NOTE multithreaded find is extremly slow
		const size_t min_size_per_thread = 1048576u;
		const bool aligned_instructions = false;
	};
    constexpr static AlgorithmMismatchConfig algorithmMismatchConfig;

    // TODO simd implementation

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

        auto ret = std::make_pair(last1, first2);
        auto futures = internal::parallel_chunk_for_2(
                            std::forward<ExecPolicy>(policy), 
                            first1, last1, last1,
                            cryptanalysislib::mismatch<InputIt1, InputIt2, config>,
                            ret, nthreads);
        internal::get_futures(futures);
        for (auto &future : futures) {
            if (future.first != first2) {
                return future;
            }
        }
    }
    
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

        auto ret = std::make_pair(last1, first2);
        auto futures = internal::parallel_chunk_for_2(
                            std::forward<ExecPolicy>(policy), 
                            first1, last1, last1,
                            cryptanalysislib::mismatch<InputIt1, InputIt2, BinaryPred, config>,
                            ret, nthreads, p);
        internal::get_futures(futures);
        for (auto &future : futures) {
            if (future.first != first2) {
                return future;
            }
        }
    }

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

        auto ret = std::make_pair(last1, first2);
        auto futures = internal::parallel_chunk_for_2(
                            std::forward<ExecPolicy>(policy), 
                            first1, last1, last1,
                            cryptanalysislib::mismatch<InputIt1, InputIt2, config>,
                            ret, nthreads, last2);
        internal::get_futures(futures);
        for (auto &future : futures) {
            if (future.first != first2) {
                return future;
            }
        }
    }
    
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

        auto ret = std::make_pair(last1, first2);
        auto futures = internal::parallel_chunk_for_2(
                            std::forward<ExecPolicy>(policy), 
                            first1, last1, last1,
                            cryptanalysislib::mismatch<InputIt1, InputIt2, BinaryPred, config>,
                            ret, nthreads, last2, p);
        internal::get_futures(futures);
        for (auto &future : futures) {
            if (future.first != first2) {
                return future;
            }
        }
    }
}; // end namespace
