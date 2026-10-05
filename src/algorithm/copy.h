#ifndef CRYPTANALYSISLIB_ALGORITHM_COPY_H
#define CRYPTANALYSISLIB_ALGORITHM_COPY_H

#include "memory/memory.h"
#include "algorithm/algorithm.h"

namespace cryptanalysislib {

/// Configuration for copy algorithms with minimum size per thread threshold
struct AlgorithmCopyConfig : public AlgorithmConfig {
    constexpr static size_t min_size_per_thread = 262144;
};

constexpr static AlgorithmCopyConfig algorithmCopyConfig;

/// Copies elements from one range to another
/// \tparam RandIt1 Random access iterator type for source range
/// \tparam RandIt2 Random access iterator type for destination range
/// \tparam config Algorithm configuration (default: algorithmCopyConfig)
/// \param first Iterator to the beginning of the source range
/// \param last Iterator to the end of the source range
/// \param dest Iterator to the beginning of the destination range
/// \return Iterator to the end of the destination range
template <class RandIt1,
          class RandIt2,
          const AlgorithmCopyConfig &config=algorithmCopyConfig>
#if __cplusplus > 201709L
    requires std::forward_iterator<RandIt1> &&
             std::forward_iterator<RandIt2>
#endif
constexpr RandIt2 copy(RandIt1 first, 
                       RandIt1 last, 
                       RandIt2 dest) noexcept {
    using T = RandIt1::value_type;
    const size_t s = static_cast<size_t>(std::distance(first, last));
    cryptanalysislib::template memcpy<T>(&(*dest), &(*first), s);
    std::advance(dest, s);
    return dest;
}

/// Copies n elements from source to destination
/// \tparam RandIt1 Random access iterator type for source range
/// \tparam Size Integral type for count
/// \tparam RandIt2 Random access iterator type for destination range
/// \param first Iterator to the beginning of the source range
/// \param n Number of elements to copy
/// \param dest Iterator to the beginning of the destination range
/// \return Iterator to the end of the destination range
template<class RandIt1,
	     class Size,
	     class RandIt2,
         const AlgorithmCopyConfig &config=algorithmCopyConfig>
#if __cplusplus > 201709L
	requires std::forward_iterator<RandIt1> &&
	         std::forward_iterator<RandIt2>
#endif
constexpr RandIt2 copy_n(RandIt1 first,
	                     const Size n,
	                     RandIt2 dest) noexcept {
	if (n <= 0) {
		return dest;
	}

	RandIt1 last = internal::advanced(first, n);
	cryptanalysislib::copy
        <RandIt1, RandIt2, config>
        (first, last, dest);
	return internal::advanced(dest, n);
}

/// Copies elements from one range to another
/// \tparam ExecPolicy Execution policy type for parallel execution
/// \tparam RandIt1 Random access iterator type for source range
/// \tparam RandIt2 Random access iterator type for destination range
/// \tparam config Algorithm configuration (default: algorithmCopyConfig)
/// \param policy Execution policy specifying parallelization strategy
/// \param first Iterator to the beginning of the source range
/// \param last Iterator to the end of the source range
/// \param dest Iterator to the beginning of the destination range
/// \return Iterator to the end of the destination range
template <class ExecPolicy,
          class RandIt1, 
          class RandIt2,
          const AlgorithmCopyConfig &config=algorithmCopyConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt1> &&
             std::random_access_iterator<RandIt2>
#endif
RandIt2 copy(ExecPolicy &&policy,
             RandIt1 first,
             RandIt1 last,
             RandIt2 dest) noexcept {

    const size_t size = static_cast<size_t>(std::distance(first, last));
    const uint32_t nthreads = should_par(policy, config, size);
    if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
        return cryptanalysislib::copy(first, last, dest);
    }

    auto futures = internal::parallel_chunk_for_2(std::forward<ExecPolicy>(policy), 
                                                  first, last, dest,
                                                  cryptanalysislib::copy<RandIt1, RandIt2, config>,
                                                  (RandIt2*)nullptr, nthreads);
    internal::get_futures(futures);
    return internal::advanced(dest, std::distance(first, last));
}

/// Copies n elements from source to destination (parallel version)
/// \tparam ExecPolicy Execution policy type for parallel execution
/// \tparam RandIt1 Random access iterator type for source range
/// \tparam Size Integral type for count
/// \tparam RandIt2 Random access iterator type for destination range
/// \tparam config Algorithm configuration (default: algorithmCopyConfig)
/// \param policy Execution policy specifying parallelization strategy
/// \param first Iterator to the beginning of the source range
/// \param n Number of elements to copy
/// \param dest Iterator to the beginning of the destination range
/// \return Iterator to the end of the destination range
template <class ExecPolicy,
          class RandIt1,
          class Size,
          class RandIt2,
          const AlgorithmCopyConfig &config=algorithmCopyConfig>
RandIt2 copy_n(ExecPolicy &&policy,
			   RandIt1 first,
			   const Size n,
			   RandIt2 dest) noexcept {
    if (n <= 0) {
        return dest;
    }

    RandIt1 last = internal::advanced(first, n);
    cryptanalysislib::copy
        <ExecPolicy, RandIt1, RandIt2, config>
        (std::forward<ExecPolicy>(policy), first, last, dest);
    return internal::advanced(dest, n);
}
#ifdef USE_CUDA

#endif

}; // end namespace cryptanalysislib
#endif
