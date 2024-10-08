#ifndef CRYPTANALYSISLIB_ALGORITHM_FILL_H
#define CRYPTANALYSISLIB_ALGORITHM_FILL_H

#include <functional>

#include "thread/thread.h"
#include "memory/memory.h"
#include "algorithm/algorithm.h"

namespace cryptanalysislib {

struct AlgorithmFillConfig : public AlgorithmConfig {
    constexpr static size_t min_size_per_thread = 1u<<10u;
};

constexpr static AlgorithmFillConfig algorithmFillConfig;

	template <class Iterator,
			  const AlgorithmFillConfig &config=algorithmFillConfig>
#if __cplusplus > 201709L
    requires std::bidirectional_iterator<Iterator>
#endif
	constexpr void fill(Iterator first,
	 				    Iterator last,
	 				    const typename Iterator::value_type& value) noexcept {
        using T = Iterator::value_type;
		const size_t s = static_cast<size_t>(std::distance(first, last));
		cryptanalysislib::memset<T>(&(*first), value, s);
    }

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

}; // end namespace cryptanalysislib
#endif
