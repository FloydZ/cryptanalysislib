#ifndef CRYPTANALYSISLIB_PERMUTATION_H
#define CRYPTANALYSISLIB_PERMUTATION_H

#include <cstdint>
#include <cstdlib>

/**
* \brief Permutations.
*/
class Permutation {
public:
	/**
	* The swap operations in LAPACK format.
	*/
	uint32_t *values;

	/**
	* The length of the swap array.
	*/
	uint32_t length;

	Permutation(uint32_t length) {
		this->values = (uint32_t *)malloc(sizeof(uint32_t) * length);
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
