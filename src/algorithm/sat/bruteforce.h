#include <cstdint>
#include <vector>

#include "math/math.h"
#include "helper.h"
#include "algorithm/countr.h"
#include "print/print.h"
#include "simd/simd.h"
#include "combination/grey.h"


template<typename T>
constexpr static inline 
T idxq(const T i,
       const T j) noexcept {
  return j * (j - 1u) / 2u + i;
}

using namespace cryptanalysislib;

/// SIMD version of the SAT bruteforce: every SIMD lane holds one clause, all
/// 2^n assignments are enumerated in Gray code order.
/// NOTE: before, this did not compile (`S::limbs`, `print_binary` of a
///     vector) and was wrong: the lane mask was compared against the
///     variable mask `m`, and `~c` was not masked to the `n` variable bits
///     (every clause looked satisfied by its high bits).
/// \tparam S SIMD type, one clause per lane
/// \tparam T limb type of `S`
/// \tparam n number of variables (clause bit i = polarity of variable i)
/// \tparam print if true, every solution is also printed
/// \param clauses[in]: exactly S::LIMBS clauses, layout as below
/// \param nr_clauses[in]: number of clauses, must be S::LIMBS
/// \return all solutions (assignments, bit i = variable i), in Gray code order
template <typename S,
          typename T,
          const uint32_t n,
          const bool print = true>
requires SIMDAble<S>
std::vector<T> sat_bruteforce(const T *clauses,
                              const uint32_t nr_clauses) {
    assert(S::LIMBS == nr_clauses);
    (void)nr_clauses;

    constexpr size_t size = 1u << n;
    constexpr T m = (1ull << n) - 1ull;
    constexpr uint64_t lanes = (S::LIMBS >= 64) ? uint64_t(-1ull) : ((1ull << S::LIMBS) - 1ull);
    std::vector<T> solutions;

    const S mm = S::set1(m);
    const S c = S::load(clauses) & mm;
    const S nc = S::not_(c) & mm;
    const S zero = S::set1(0);

    // the current assignment `x_ = to_gray(i)`, broadcasted and negated
    T x_ = 0;
    S x = zero, invx = mm;

    // NOTE: the Gray code is enumerated in blocks of `B` assignments: within
    // 	a block the flipped variables (ctz(k)) are compile time constants, so
    // 	every block is fully unrolled. Only the first flip of a block
    // 	depends on the block index.
    constexpr uint32_t lb = n < 4u ? n : 4u;
    constexpr size_t B = 1ull << lb;
    for (size_t i = 0; i < size; i += B) {
        if (i) {
            const S f = S::set1(T(1) << cryptanalysislib::countr_zero(i));
            x_ ^= T(1) << cryptanalysislib::countr_zero(i);
            x = x ^ f;
            invx = invx ^ f;
        }

        constexpr_for<size_t(0), B, size_t(1)>([&](const auto k) {
            if constexpr (k > 0) {
                constexpr T flip = T(1) << cryptanalysislib::countr_zero(size_t(k));
                const S f = S::set1(flip);
                x_ ^= flip;
                x = x ^ f;
                invx = invx ^ f;
            }

            const S t3 = (c & x) | (nc & invx);
            // one bit per lane: clause satisfied
            const uint64_t t = uint64_t(t3 > zero) & lanes;
            if (t == lanes) {
                // found a solution
                solutions.push_back(x_);
                if constexpr (print) {
                    print_binary(x_, n);
                }
            }
        });
    }
    return solutions;
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
/// \tparam print if true, every solution is also printed (as before)
/// \return all solutions (assignments, bit i = variable i), in Gray code order
template <typename T,
          const uint32_t n,
          const bool print = true>
std::vector<T> sat_bruteforce(const T *clauses,
                              const uint32_t nr_clauses) {
    constexpr size_t size = 1u << n;
    constexpr T m = (1ull << n) - 1ull;
    std::vector<T> c(nr_clauses);
    std::vector<T> nc(nr_clauses);

    std::vector<T> solutions;

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
            solutions.push_back(T(x));
            if constexpr (print) {
                print_binary(x, n);
            }
        }

    }
    return solutions;
}
