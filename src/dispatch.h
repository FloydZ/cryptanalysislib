#ifndef CRYPTANALYSISLIB_DISPATCH_H
#define CRYPTANALYSISLIB_DISPATCH_H

#include <cstdlib>
#include <cstdint>
#include <functional>

#include "helper.h"
#include "cpucycles.h"

struct BenchmarkConfig {
    constexpr static size_t number_iterations = 1u<<10;
};
constexpr static BenchmarkConfig benchmarkConfig{};

/// @tparam config
/// @tparam F
/// @tparam Args
/// @param f
/// @param args
/// @return
template<const BenchmarkConfig &config=benchmarkConfig,
         typename F,
         typename ...Args>
__attribute__((noinline))
static size_t genereric_bench(F &&f,
                       Args &&...args) noexcept {
    size_t c = 0;
    for (size_t i = 0; i < config.number_iterations; i++) {
        c -= cpucycles();
        std::invoke(f, args...);
        c += cpucycles();
    }

    return c;
};

///
/// @tparam F
/// @tparam Args
/// @tparam config
/// @param out
/// @param f
/// @param args
/// @return
template<typename F,
         typename ...Args,
         const BenchmarkConfig &config=benchmarkConfig>
__attribute__((noinline))
static size_t genereric_dispatch(F &out,
                                 std::vector<F> &f,
                                 Args &&...args) noexcept {
    size_t mc = -1ull, min_pos = 0;
    for (size_t i = 0; i < f.size(); i++) {
        const size_t cycles = genereric_bench
                                <config>
                                (f[i], args...);
        if (cycles < mc) {
            min_pos = i;
            mc = cycles;
        }
    }

    out = f[min_pos];
    return min_pos;
};


#endif
