#ifndef CRYPTANALYSISLIB_ALGORITHM_H
#define CRYPTANALYSISLIB_ALGORITHM_H

#include <cstdlib>
#include "thread/thread.h"

namespace cryptanalysislib {

///
struct AlgorithmConfig {
    constexpr static size_t alignment = 8;
};

/// extends the functionality of `is_par`/`is_seq` by deciding during runtime
/// if threads should be used (maybe the problem is to small) and if yes,
/// how many.
/// \param policy 
/// \param config derivation of `AlgorithmConfig` which implements a `min_size_per_thread` field.
/// \param size problem size: number of elements to be processed 
/// \return the number of threads that should be used. If zero it means that 
///     no threading should be used.
template <class ExecPolicy,
          class AlgorithmConfigClass>
[[nodiscard]] constexpr inline static uint32_t should_par(const ExecPolicy& policy,
                                                          const AlgorithmConfigClass &config,
                                                          const size_t size) noexcept {
    if(!policy.par_allowed()) {
        return 0;
    }

    if (size < config.min_size_per_thread) {
        return 0;
    }

    const uint32_t pnt = policy.pool()->get_num_threads();
    const uint32_t nt = std::min(
        (uint32_t)((size+config.min_size_per_thread - 1)/config.min_size_per_thread),
        pnt);
    return nt;
}

}; // end namespace cryptanalysislib
#endif
