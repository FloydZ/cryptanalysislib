#ifndef CRYPTANALYSISLIB_ALGORITHM_INT2WEIGHT_H
#define CRYPTANALYSISLIB_ALGORITHM_INT2WEIGHT_H

#include <stdint.h>
#include "math/math.h"

/// \tparam T
/// \param weights	output list
/// \param in in element, which is mapped to a bit sequence
/// \param n	bitlength of n
/// \param wt max index size
/// \param k how many indices to generate
template<typename T>
	requires std::is_arithmetic_v<T>
constexpr void int2weights(uint32_t *weights,
                 const T in,
        		 const uint32_t n,
        		 const uint32_t wt,
                 const uint32_t k) noexcept {
	T a = in;
	uint32_t wn = n;
	uint32_t wk = wt;
	// uint64_t v = 0;
	uint32_t set = 0;
	while (wn != 0) {
		if ((set == wt) || (set == k)) {
			break;
		} else if (wn + set == wt) {
			// v += (1ULL << (wn - 1));
			weights[set] = wn - 1;
			wn -= 1;
			set += 1;
		} else if (a < binom(wn - 1, wk)) {
			wn -= 1;
		} else {
			a -= binom(wn - 1, wk);
			// v += (1ULL << (wn - 1));
			weights[set] = wn - 1u;
			wn -= 1;
			wk -= 1;
			set += 1;
		}
	}

	// return v;
}

template<typename T>
requires std::is_arithmetic_v<T>
void int2weights(std::vector<T> &weights,
				const T in,
				const uint32_t n,
				const uint32_t wt) {
	int2weights(weights.data(), in, n, wt);
}
#endif
