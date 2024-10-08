#ifndef CRYPTANALYSISLIB_ALGORITHM_COPY_H
#define CRYPTANALYSISLIB_ALGORITHM_COPY_H

#include <functional>

#include "thread/thread.h"
#include "memory/memory.h"
#include "algorithm/algorithm.h"

namespace cryptanalysislib {

struct AlgorithmCopyConfig : public AlgorithmConfig {
    constexpr static size_t min_size_per_thread = 1u<<10u;
};

constexpr static AlgorithmCopyConfig algorithmCopyConfig;

/// \tparam RandIt1
/// \tparam RandIt2
/// \tparam config
/// \param first
/// \param last
/// \param dest
/// \return
template <class RandIt1,
          class RandIt2,
          const AlgorithmCopyConfig &config=algorithmCopyConfig>
#if __cplusplus > 201709L
    requires std::bidirectional_iterator<RandIt1> &&
             std::bidirectional_iterator<RandIt2>
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

/// \tparam RandIt1
/// \tparam Size
/// \tparam RandIt2
/// \param first
/// \param n
/// \param dest
/// \return
template<class RandIt1,
	     class Size,
	     class RandIt2,
         const AlgorithmCopyConfig &config=algorithmCopyConfig>
#if __cplusplus > 201709L
	requires std::bidirectional_iterator<RandIt1> &&
	         std::bidirectional_iterator<RandIt2>
#endif
constexpr RandIt2 copy_n(RandIt1 first,
	                     const Size n,
	                     RandIt2 dest) noexcept {
	if (n <= 0) {
		return dest;
	}

	RandIt1 last = internal::advanced(first, n);
	cryptanalysislib::copy<RandIt1, RandIt2, config>(first, last, dest);
	return internal::advanced(dest, n);
}

/// \tparam ExecPolicy
/// \tparam RandIt1
/// \tparam RandIt2
/// \tparam config
/// \param policy
/// \param first
/// \param last
/// \param dest
/// \return
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

/// \tparam ExecPolicy
/// \tparam RandIt1
/// \tparam Size
/// \tparam RandIt2
/// \tparam config
/// \param policy
/// \param first
/// \param n
/// \param dest
/// \return
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
    cryptanalysislib::copy<ExecPolicy, RandIt1, RandIt2, config>(std::forward<ExecPolicy>(policy), first, last, dest);
    return internal::advanced(dest, n);
}

}; // end namespace cryptanalysislib
#endif
