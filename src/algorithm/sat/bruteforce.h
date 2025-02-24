#include <cstdint>
#include <vector>

#include "math/math.h"
#include "print/print.h"
#include "simd/simd.h"

// TODO logging of solutions
// TODO move to ./src/combination/grey.h
constexpr static inline uint32_t idxq(const uint32_t i,
                                      const uint32_t j) noexcept {
  return j * (j - 1u) / 2u + i;
}

constexpr static inline uint32_t to_gray(uint32_t i) noexcept {
  return (i ^ (i >> 1u));
}

constexpr static inline uint32_t next_gray(const uint32_t a) noexcept {
    return a ^ __builtin_ctz(a+1);
}

using namespace cryptanalysislib;

template <typename S,
          typename T,
          const uint32_t n>
requires SIMDAble<S>
void sat_bruteforce(const T *clauses,
                    const uint32_t nr_clauses) {
    assert(S::limbs == nr_clauses);

    constexpr size_t size = 1u << n;
    constexpr T m = (1ull << n) - 1ull;

    const S c = S::load(clauses);
    const S nc = S::not_(c);
    const S zero = S::set1(0);
    for (size_t i = 0; i < size; i++) {
        // TODO write python script which unrolls everything
        const T x_ = to_gray(i);
        const S x = S::set1(x_);
        const S invx = S::not_(x);

        const S t1 = c & x;
        const S t2 = nc & invx;
        const S t3 = t1 | t2;
        const T t = t3 > zero;
        if (t == m) {
            // found a solution
            print_binary(x, n);
        }
    }
}

/// \tparam T
/// \tparam n number of variables in each clause
/// \param clauses[in]: layout: 
///               31                    0
///     clause 1: [ lit_31, ..., lit_0 ]
///     clause 2: []
///       ...
///     clause n: []
/// \param nr_clauses[in]: number of clauses in 
template <typename T,
          const uint32_t n>
void sat_bruteforce(const T *clauses,
                    const uint32_t nr_clauses) {
    constexpr size_t size = 1u << n;
    constexpr T m = (1ull << n) - 1ull;
    std::vector<T> c(nr_clauses);
    std::vector<T> nc(nr_clauses);

    // copy everything into a local buffer
    for (uint32_t i = 0; i < nr_clauses; i++) {
        c[i] = clauses[i]; nc[i] = (~clauses[i]) & m;
    }

    for (size_t i = 0; i < size; i++) {
        uint32_t x = to_gray(i);
        uint32_t invx = ~x;
        uint32_t b = 0;
        for (uint32_t j = 0; j < nr_clauses; j++) {
            const uint32_t t1 = c[j] & x;
            const uint32_t t2 = nc[j] & invx;
            b += (t1 | t2) > 0;
        }
        
        if (b == nr_clauses) {
            // found a solution
            print_binary(x, n);
        }

    }
}
