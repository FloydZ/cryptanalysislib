#ifndef CRYPTANALYSISLIB_DISPATCH_H
#define CRYPTANALYSISLIB_DISPATCH_H

#include <cstdlib>
#include <cstdint>
#include <functional>

#include "helper.h"
#include "cpucycles.h"


struct BenchmarkConfig {
public:
    constexpr static size_t number_iterations = 1u<<5;
};

constexpr static BenchmarkConfig benchmarkConfig{};

template<typename F,
         typename Args,
         const BenchmarkConfig &config=benchmarkConfig>
size_t inline genereric_bench(F &&f,
                              Args &&args...) noexcept {
    size_t c = 0;
    for (size_t i = 0; i < config.number_iterations; i++) {
        c -= cpucycles();
        std::invoke(f, args);
        c += cpucycles();
    }

    return c;
};


template<typename F,
         typename Args,
         const BenchmarkConfig &config=benchmarkConfig>
size_t inline genereric_dispatch(F &out, F *f,
                                 Args *args,
                                 const uint32_t n) noexcept {
    ASSERT(n > 0);
    std::vector<size_t> cycles{n};
    for (size_t i = 0; i < n; i++) {
        cycles = genereric_bench
                    <F, Args, config>
                    (f[i], args[i]);
    }
  
    size_t mc = cycles[0];
    size_t min_pos = 0;
    for (uint32_t i = 1; i < n; i++) {
        const size_t c = cycles[i];
        if (c < mc) {
            mc = c;
            min_pos = i;
        }
    }

    out = f[min_pos];
    return min_pos;
};


#endif
