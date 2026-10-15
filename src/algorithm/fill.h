#ifndef CRYPTANALYSISLIB_ALGORITHM_FILL_H
#define CRYPTANALYSISLIB_ALGORITHM_FILL_H

#include <functional>

#include "thread/thread.h"
#include "memory/memory.h"
#include "algorithm/algorithm.h"

namespace cryptanalysislib {

    /// Configuration for fill algorithms
    struct AlgorithmFillConfig : public AlgorithmConfig {
        const size_t min_size_per_thread = 1048576;
    };
    
    constexpr static AlgorithmFillConfig algorithmFillConfig;

    
    // NOTE: `fill`/`fill_n` use `cryptanalysislib::memset`, which is the
    // 	SIMD (TxN_t) implementation, so no separate SIMD kernels are needed
    
	/// Fills a range with a specified value (sequential version)
	/// \tparam Iterator Forward iterator type for the range
	/// \tparam config Algorithm configuration (default: algorithmFillConfig)
	/// \param first[in]: Iterator to the beginning of the range
	/// \param last[in]: Iterator to the end of the range
	/// \param value[in]: Value to fill the range with
	template <class Iterator,
			  const AlgorithmFillConfig &config=algorithmFillConfig>
#if __cplusplus > 201709L
    requires std::forward_iterator<Iterator>
#endif
	constexpr void fill(Iterator first,
	 				    Iterator last,
	 				    const typename Iterator::value_type& value) noexcept {
        using T = Iterator::value_type;
		// NOTE: `memset` writes [first, last) as an array: only for contiguous
		// 	iterators, and `first` must not be dereferenced for an empty range
		if constexpr (std::contiguous_iterator<Iterator>) {
			const size_t s = static_cast<size_t>(last - first);
			if (s == 0) {
				return;
			}
			cryptanalysislib::memset<T>(&(*first), value, s);
		} else {
			for (; first != last; ++first) {
				*first = value;
			}
		}
    }

	/// Fills n elements with a specified value (sequential version)
	/// \tparam Iterator Forward iterator type for the range
	/// \tparam Size Integral type for count
	/// \tparam config Algorithm configuration (default: algorithmFillConfig)
	/// \param first[in]: Iterator to the beginning of the range
	/// \param n[in]: Number of elements to fill
	/// \param value[in]: Value to fill the elements with
	/// \return Iterator to the end of the filled range
	template <class Iterator,
			  class Size,
			  const AlgorithmFillConfig &config=algorithmFillConfig>
#if __cplusplus > 201709L
    requires std::forward_iterator<Iterator>
#endif
    Iterator fill_n(Iterator first,
					const Size n,
				    const typename Iterator::value_type& value) {
        if (n <= 0) {
            return first;
        }
        Iterator last = internal::advanced(first, n);
        cryptanalysislib::fill(first, last, value);
        return last;
    }

	/// Fills a range with a specified value (parallel version)
	/// \tparam ExecPolicy Execution policy type for parallel execution
	/// \tparam RandIt Random access iterator type for the range
	/// \tparam config Algorithm configuration (default: algorithmFillConfig)
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: Iterator to the beginning of the range
	/// \param last[in]: Iterator to the end of the range
	/// \param value[in]: Value to fill the range with
template <class ExecPolicy,
			  class RandIt,
			  const AlgorithmFillConfig &config=algorithmFillConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
	void fill(ExecPolicy &&policy,
			  RandIt first,
			  RandIt last,
			  const typename RandIt::value_type& value) noexcept {
		const size_t size = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::fill<RandIt, config>(first, last, value);
		}

		internal::parallel_chunk_for_1_wait(std::forward<ExecPolicy>(policy),
										    first, last,
											cryptanalysislib::fill<RandIt, config>,
											(void*)nullptr,
											1,
											nthreads,
											value);
	}

	/// Fills n elements with a specified value (parallel version)
	/// \tparam ExecPolicy Execution policy type for parallel execution
	/// \tparam RandIt Random access iterator type for the range
	/// \tparam Size Integral type for count
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: Iterator to the beginning of the range
	/// \param n[in]: Number of elements to fill
	/// \param value[in]: Value to fill the elements with
	/// \return Iterator to the end of the filled range
	template <class ExecPolicy,
			  class RandIt,
			  class Size>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt>
#endif
    RandIt fill_n(ExecPolicy &&policy,
				  RandIt first,
				  const Size n,
				  const typename RandIt::value_type& value) {
        if (n <= 0) {
            return first;
        }
        RandIt last = internal::advanced(first, n);
        cryptanalysislib::fill(std::forward<ExecPolicy>(policy), first, last, value);
        return last;
    }


}; // end namespace cryptanalysislib
#endif
