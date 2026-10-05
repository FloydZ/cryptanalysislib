#ifndef CRYPTANALYSISLIB_ALGORITHM_INT2WEIGHT_H
#define CRYPTANALYSISLIB_ALGORITHM_INT2WEIGHT_H

#include <cstdint>

#include "math/math.h"
#include "algorithm/bits.h"

// TODO add namespace cryptanalysislib 

/// Converts integer to weight positions using combinatorial ranking
/// \tparam D Output data type for weight positions
/// \tparam T Input integer type
/// \param weights[out]: Output array of weight positions
/// \param in[in]: Input element to be mapped to bit sequence
/// \param n[in]: Bit length of the sequence
/// \param wt[in]: Maximum weight (number of set bits)
/// \param k[in]: Maximum number of indices to generate
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

/// Converts integer to weight positions stored as bits
/// \tparam D Output data type for bit representation
/// \tparam T Input integer type
/// \param weights[out]: Output bit array where set bits indicate positions
/// \param in[in]: Input element to be mapped to bit sequence
/// \param n[in]: Bit length of the sequence
/// \param wt[in]: Maximum weight (number of set bits)
/// \param k[in]: Maximum number of indices to generate
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

/// Converts integer to weight positions using vector output
/// \tparam D Output data type for weight positions
/// \tparam T Input integer type
/// \param weights[out]: Output vector of weight positions
/// \param in[in]: Input element to be mapped to bit sequence
/// \param n[in]: Bit length of the sequence
/// \param wt[in]: Maximum weight (number of set bits)
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
