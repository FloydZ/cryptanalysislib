#ifndef CRYPTANALYSISLIB_PERMUTATION_H
#define CRYPTANALYSISLIB_PERMUTATION_H

#include <cstdint>
#include <cstdlib>
#include <cassert>

#include "helper.h"
#include "memory/memory.h"

class PermutationConfig {
    /// TODO
};

constexpr static PermutationConfig permutationConfig;

// template <const PermutationConfig &config=permutationConfig>
class Permutation {
public:
	// The swap operations in LAPACK format.
	uint32_t *values;

	// The length of the swap const_array.
	uint32_t length;

	/// \param length[in]
	Permutation(const uint32_t length) noexcept {
		this->values = (uint32_t *)malloc(sizeof(uint32_t) * length);
		assert(values);
		this->length = length;
		for (uint32_t i = 0; i < length; ++i) {
			this->values[i] = i;
		}
	}

	// NOTE: owns `values`. Before, the implicit copies shared the buffer, so
	// 	it was freed twice.
	Permutation(const Permutation &other) noexcept : Permutation(other.length) {
		cryptanalysislib::memcpy(values, other.values, length);
	}

	Permutation(Permutation &&other) noexcept :
	    values(other.values), length(other.length) {
		other.values = nullptr;
		other.length = 0;
	}

	Permutation &operator=(const Permutation &other) noexcept {
		if (this != &other) {
			if (length != other.length) {
				free(values);
				values = (uint32_t *)malloc(sizeof(uint32_t) * other.length);
				assert(values);
				length = other.length;
			}
			cryptanalysislib::memcpy(values, other.values, length);
		}
		return *this;
	}

	Permutation &operator=(Permutation &&other) noexcept {
		if (this != &other) {
			free(values);
			values = other.values;
			length = other.length;
			other.values = nullptr;
			other.length = 0;
		}
		return *this;
	}

    /// 
	~Permutation() {
		free(values);
	}
};

#endif//CRYPTANALYSISLIB_PERMUTATION_H
