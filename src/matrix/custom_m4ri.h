#ifndef SMALLSECRETLWE_CUSTOM_MATRIX_H
#define SMALLSECRETLWE_CUSTOM_MATRIX_H

#include "helper.h"
#include "random.h"
#include "simd/simd.h"

#include <type_traits>
#include <utility>
#include <array>
#include <memory>

/// TODO: dass alles in `binary_matrix` aufgehen lassen




#define MATRIX_AVX_PADDING(len) (((len + 255) / 256) * 256)

#define MAX_K 8
#define WORD_SIZE (8 * sizeof(word))

/// private forward definition
mzd_t *_mzd_transpose(mzd_t *DST, mzd_t const *A);

uint32_t hamming_weight(mzd_t *ptr, const uint32_t row=0) {
	uint32_t ret = 0;

	// NOTE: don't change the type
	for (int i = 0; i < ptr->ncols; ++i) {
		ret += mzd_read_bit(ptr, (int)row, i);
	}

	return ret;
}
uint32_t hamming_weight_column(mzd_t *ptr, const uint32_t col) {
	uint32_t ret = 0;

	// NOTE: don't change the type
	for (int i = 0; i < ptr->nrows; ++i) {
		ret += mzd_read_bit(ptr, i, (int)col);
	}

	return ret;
}








/// custom function to append in_matrix to out_matrix.
/// needed by bjmm.h
/// \param out_matrix input/output
/// \param in_matrix input
/// \param start_r start row; offset from the in_matrix
/// \param start_c start column offset from the out_matrix
void mzd_append(mzd_t *__restrict__ out_matrix,
                const mzd_t *__restrict__ const in_matrix,
                const uint32_t start_r,
                const uint32_t start_c) noexcept {
	assert((uint32_t)out_matrix->nrows > start_r);
	assert((uint32_t)out_matrix->ncols > start_c);
	assert(out_matrix->nrows - (int)start_r == in_matrix->nrows);

	// NOTE: don't change the type
	for (uint32_t row = 0; row < in_matrix->nrows; ++row) {
		for (uint32_t col = 0; col < in_matrix->ncols; ++col) {
			auto bit = mzd_read_bit(in_matrix, row, col);
			mzd_write_bit(out_matrix, row + (int)start_r, col + (int)start_c, bit);
		}
	}
}





// The same as `mzd_concat` but with the oddity that a new matrix with avx padding is generated if necessary.
// This is sometimes called augment
mzd_t *matrix_concat(mzd_t *C, mzd_t const *A, mzd_t const *B) noexcept {
	if (A->nrows != B->nrows) { m4ri_die("mzd_concat: Bad arguments to concat!\n"); }

	if (C == NULL) {
		C = mzd_init(A->nrows, MATRIX_AVX_PADDING(A->ncols + B->ncols));
		// This is the important point, we `resize` the ouput matrix to its normal expected size.
		C->ncols = A->ncols + B->ncols;
	} else if (C->nrows != A->nrows || C->ncols < (A->ncols + B->ncols)) {
		m4ri_die("mzd_concat: C has wrong dimension!\n");
	}

	for (rci_t i = 0; i < A->nrows; ++i) {
		word *dst_truerow = C->rows[i];
		word *src_truerow = A->rows[i];
		for (wi_t j = 0; j < A->width; ++j) { dst_truerow[j] = src_truerow[j]; }
	}

	for (rci_t i = 0; i < B->nrows; ++i) {
		for (rci_t j = 0; j < B->ncols; ++j) {
			mzd_write_bit(C, i, j + A->ncols, mzd_read_bit(B, i, j));
		}
	}

	__M4RI_DD_MZD(C);
	return C;
}

// generates the DOOM shits
// in must be the syndrom in column form
// writes the shifts directly into the out matrix.
mzd_t *matrix_down_shift_into_matrix(mzd_t *out, const mzd_t *in, const uint32_t col, const uint32_t i) noexcept {
	if ((in->ncols != 1) || (in->nrows != out->nrows)) m4ri_die("matrix_shift: Bad argument!\n");

	if (out == nullptr) {
		out = mzd_init(in->nrows, in->nrows);
	}

	for (uint32_t j = 0; j < in->nrows; ++j) {
		mzd_write_bit(out, (j + i) % in->nrows, col, mzd_read_bit(in, j, 0));
	}

	return out;
}

// generates the DOOM shits
// in must be the syndrom in column form
// writes the shifts directly into the out matrix.
mzd_t *matrix_up_shift_into_matrix(mzd_t *out, const mzd_t *in, const uint32_t col, const uint32_t i) noexcept {
	if (in->ncols != 1) m4ri_die("matrix_shift: Bad argument!\n");

	if (out == nullptr) {
		out = mzd_init(in->nrows, in->nrows);
	}

	for (uint32_t j = 0; j < in->nrows; ++j) {
		mzd_write_bit(out, j, col, mzd_read_bit(in, (j + i) % in->nrows, 0));
	}

	return out;
}

// generates the DOOM shits
// in must be the syndrom in column form
mzd_t *matrix_down_shift(mzd_t *out, const mzd_t *in, uint32_t i) noexcept {
	if (in->ncols != 1) m4ri_die("matrix_shift: Bad argument!\n");

	if (out == nullptr) {
		out = mzd_init(in->nrows, in->ncols);
	}

	for (uint32_t j = 0; j < in->nrows; ++j) {
		mzd_write_bit(out, (j + i) % in->nrows, 0, mzd_read_bit(in, j, 0));
	}

	return out;
}

// generates the DOOM shits
// in must be the syndrom in column form
mzd_t *matrix_up_shift(mzd_t *out, const mzd_t *in, uint32_t i) noexcept {
	if (in->ncols != 1) m4ri_die("matrix_shift: Bad argument!\n");

	if (out == nullptr) {
		out = mzd_init(in->nrows, in->ncols);
	}

	for (uint32_t j = 0; j < in->nrows; ++j) {
		mzd_write_bit(out, j, 0, mzd_read_bit(in, (j + i) % in->nrows, 0));
	}

	return out;
}





#endif //SMALLSECRETLWE_CUSTOM_MATRIX_H
