#ifndef CRYPTANALYSISLIB_PERMUTATION_H
#define CRYPTANALYSISLIB_PERMUTATION_H

#include <cstdint>
#include <cstdlib>

#include "helper.h"

class PermutationConfig {
};

constexpr static PermutationConfig permutationConfig;

// template <const PermutationConfig &config=permutationConfig>
class Permutation {
public:
	// The swap operations in LAPACK format.
	uint32_t *values;

	// The length of the swap const_array.
	uint32_t length;

	/// @param length
	Permutation(const uint32_t length) noexcept {
		this->values = (uint32_t *)malloc(sizeof(uint32_t) * length);
		ASSERT(values);
		this->length = length;
		for (uint32_t i = 0; i < length; ++i) {
			this->values[i] = i;
		}
	}

	~Permutation() {
		free(values);
	}
};

#endif//CRYPTANALYSISLIB_PERMUTATION_H
