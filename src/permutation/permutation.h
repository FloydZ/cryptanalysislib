#ifndef CRYPTANALYSISLIB_PERMUTATION_H
#define CRYPTANALYSISLIB_PERMUTATION_H

#include <cstdint>
#include <cstdlib>

/**
* \brief Permutations.
*/
typedef struct mzp_t {
	/**
	* The swap operations in LAPACK format.
	*/
	uint32_t *values;

	/**
	* The length of the swap array.
	*/
	uint32_t length;
} mzp_t;// note that this is NOT mpz_t

mzp_t *mzp_init(uint32_t length) {
	mzp_t *P  = (mzp_t *)malloc(sizeof(mzp_t));
	P->values = (uint32_t *)malloc(sizeof(uint32_t) * length);
	P->length = length;
	for (uint32_t i = 0; i < length; ++i) { P->values[i] = i; }
	return P;
}

void mzp_free(mzp_t *P) {
	free(P->values);
	free(P);
}
#endif//CRYPTANALYSISLIB_PERMUTATION_H
