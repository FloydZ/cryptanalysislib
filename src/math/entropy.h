#ifndef CRYPTANALYSISLIB_MATH_ENTROPY_H
#define CRYPTANALYSISLIB_MATH_ENTROPY_H

#include "math/log.h"

/// Entropy function
/// \param x input
/// \return H[x] := -x*Log2[x] - (1 - x)*Log2[1 - x];
constexpr double HH(const double x) noexcept {
	if (x <= 0)
		return 0.;

	if (x >= 1.)
		return 0.;

	return -x * log2(x) - (1. - x) * log2(1. - x);
}

#endif//CRYPTANALYSISLIB_MATH_ENTROPY_H
