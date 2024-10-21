#ifndef CRYPTANALYSISLIB_ALGORITHM_INT2WEIGHT_H
#define CRYPTANALYSISLIB_ALGORITHM_INT2WEIGHT_H

#include <cstdint>

#include "math/math.h"
#include "algorithm/bits.h"

/// \tparam T
/// \param weights	output list
/// \param in in element, which is mapped to a bit sequence
/// \param n	bitlength of n
/// \param wt max index size
/// \param k how many indices to generate
template<typename D, typename T>
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T> &&
			 std::is_arithmetic_v<D>
#endif
constexpr void int2weights(D *weights,
                           const T in,
                           const uint32_t n,
                           const uint32_t wt,
                           const uint32_t k) noexcept {
	T a = in;
	uint32_t wn = n;
	uint32_t wk = wt;
	uint32_t set = 0;
	while (wn != 0) {
		if ((set == wt) || (set == k)) {
			break;
		} else if (wn + set == wt) {
			weights[set] = wn - 1;
			wn -= 1;
			set += 1;
		} else if (a < binom(wn - 1, wk)) {
			wn -= 1;
		} else {
			a -= binom(wn - 1, wk);
			weights[set] = wn - 1u;
			wn -= 1;
			wk -= 1;
			set += 1;
		}
	}
}

/// \tparam D
/// \tparam T
/// \param weights
/// \param in
/// \param n
/// \param wt
/// \param k
template<typename D,
		 typename T>
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T> &&
			 std::is_arithmetic_v<D>
#endif
constexpr void int2weight_bits(D *weights,
                               const T in,
                               const uint32_t n,
                               const uint32_t wt,
                               const uint32_t k) noexcept {
	T a = in;
	uint32_t wn = n;
	uint32_t wk = wt;
	uint32_t set = 0;
	*weights = 0ull;
	while (wn != 0) {
		if ((set == wt) || (set == k)) {
			break;
		} else if (wn + set == wt) {
			set_bit(weights, wn - 1, 1);
			// *weights ^= 1 << (wn - 1);
			wn -= 1;
			set += 1;
		} else if (a < binom(wn - 1, wk)) {
			wn -= 1;
		} else {
			a -= binom(wn - 1, wk);
			// *weights ^= 1ull << (wn - 1);
			set_bit(weights, wn - 1, 1);
			wn -= 1;
			wk -= 1;
			set += 1;
		}
	}
}

template<typename D,
		 typename T>
#if __cplusplus > 201709L
	requires std::is_arithmetic_v<T> &&
	         std::is_arithmetic_v<D>
#endif
void int2weights(std::vector<D> &weights,
				const T in,
				const uint32_t n,
				const uint32_t wt) {
	int2weights(weights.data(), in, n, wt);
}
#endif
