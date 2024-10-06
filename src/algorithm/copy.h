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

/// 
template <class ExecPolicy, 
          class RandIt1, 
          class RandIt2,
          const AlgorithmCopyConfig &config=algorithmCopyConfig>
#if __cplusplus > 201709L
    requires std::bidirectional_iterator<RandIt1> &&
             std::bidirectional_iterator<RandIt2>
#endif
RandIt2
copy(ExecPolicy &&policy, 
     RandIt1 first, 
     RandIt1 last, 
     RandIt2 dest) noexcept {

    const size_t size = static_cast<size_t>(std::distance(first, last));
    const uint32_t ntreads = should_par(policy, config, size);
    if (is_seq<ExecPolicy>(policy)) {
        return cryptanalysislib::copy(first, last, dest);
    }

    auto futures = internal::parallel_chunk_for_2(std::forward<ExecPolicy>(policy), 
                                                  first, last, dest,
                                                  std::copy<RandIt1, RandIt2>, (RandIt2*)nullptr);
    internal::get_futures(futures);
    return internal::advanced(dest, std::distance(first, last));
}
}; // end namespace cryptanalysislib
#endif
