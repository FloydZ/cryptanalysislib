#ifndef SMALLSECRETLWE_CUSTOM_MATRIX_H
#define SMALLSECRETLWE_CUSTOM_MATRIX_H

#include "m4ri/m4ri.h"
#include "m4ri/mzd.h"
#include "glue_m4ri.h"

#ifdef USE_AVX2
#include <immintrin.h>
#include <emmintrin.h>
#endif

#define MATRIX_AVX_PADDING(len) (((len + 255) / 256) * 256)


typedef struct customMatrixData {
	int **rev;
	int **diff;
	// Unpadded number of columns
	int real_nr_cols;

	int working_nr_cols;

	uint64_t *lookup_table;
} customMatrixData;

#define MAX_K 7
#define WORD_SIZE (8 * sizeof(word))

/// custom function to append in_matrix to out_matrix.
/// \param out_matrix
/// \param in_matrix
/// \param start_r
/// \param start_c
void mzd_append(mzd_t *out_matrix, const mzd_t *const in_matrix, const int start_r, const int start_c) {
	ASSERT(out_matrix->nrows > start_r);
	ASSERT(out_matrix->ncols > start_c);
	ASSERT(out_matrix->nrows - start_r == in_matrix->nrows);
	for (int row = 0; row < in_matrix->nrows; ++row) {
		for (int col = 0; col < in_matrix->ncols; ++col) {
			auto bit = mzd_read_bit(in_matrix, row, col);
			mzd_write_bit(out_matrix, row + start_r, col + start_c, bit);
		}
	}
}


mzd_t *_mzd_transpose(mzd_t *DST, mzd_t const *A);

mzd_t *matrix_transpose(mzd_t *DST, mzd_t const *A) {
	if (DST == NULL) {
		DST = mzd_init(A->ncols, A->nrows);
	} else if (__M4RI_UNLIKELY(DST->nrows < A->ncols || DST->ncols < A->nrows)) {
		m4ri_die("mzd_transpose: Wrong size for return matrix.\n");
	} else {
		/** it seems this is taken care of in the subroutines, re-enable if running into problems **/
		// mzd_set_ui(DST,0);
	}

	if (A->nrows == 0 || A->ncols == 0) return mzd_copy(DST, A);

	if (__M4RI_LIKELY(!mzd_is_windowed(DST) && !mzd_is_windowed(A)))
		return _mzd_transpose(DST, A);
	int A_windowed = mzd_is_windowed(A);
	if (A_windowed) A = mzd_copy(NULL, A);
	if (__M4RI_LIKELY(!mzd_is_windowed(DST)))
		_mzd_transpose(DST, A);
	else {
		mzd_t *D = mzd_init(DST->nrows, DST->ncols);
		_mzd_transpose(D, A);
		mzd_copy(DST, D);
		mzd_free(D);
	}
	if (A_windowed) mzd_free((mzd_t *)A);
	return DST;
}

// The same as `mzd_concat` but with the oddity that a new matrix with avx padding is generated if necessary.
// This is sometimes called augment
mzd_t *matrix_concat(mzd_t *C, mzd_t const *A, mzd_t const *B) {
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
mzd_t *matrix_down_shift_into_matrix(mzd_t *out, const mzd_t *in, const uint32_t col, const uint32_t i) {
	if ((in->ncols != 1) || (in->nrows != out->nrows)) m4ri_die("matrix_shift: Bad argument!\n");

	if (out == nullptr) {
		out = mzd_init(in->nrows, in->nrows);
	}

	for (int j = 0; j < in->nrows; ++j) {
		mzd_write_bit(out, (j + i) % in->nrows, col, mzd_read_bit(in, j, 0));
	}

	return out;
}

// generates the DOOM shits
// in must be the syndrom in column form
// writes the shifts directly into the out matrix.
mzd_t *matrix_up_shift_into_matrix(mzd_t *out, const mzd_t *in, const uint32_t col, const uint32_t i) {
	if (in->ncols != 1) m4ri_die("matrix_shift: Bad argument!\n");

	if (out == nullptr) {
		out = mzd_init(in->nrows, in->nrows);
	}

	for (int j = 0; j < in->nrows; ++j) {
		mzd_write_bit(out, j, col, mzd_read_bit(in, (j + i) % in->nrows, 0));
	}

	return out;
}

// generates the DOOM shits
// in must be the syndrom in column form
mzd_t *matrix_down_shift(mzd_t *out, const mzd_t *in, uint32_t i) {
	if (in->ncols != 1) m4ri_die("matrix_shift: Bad argument!\n");

	if (out == nullptr) {
		out = mzd_init(in->nrows, in->ncols);
	}

	for (int j = 0; j < in->nrows; ++j) {
		mzd_write_bit(out, (j + i) % in->nrows, 0, mzd_read_bit(in, j, 0));
	}

	return out;
}

// generates the DOOM shits
// in must be the syndrom in column form
mzd_t *matrix_up_shift(mzd_t *out, const mzd_t *in, uint32_t i) {
	if (in->ncols != 1) m4ri_die("matrix_shift: Bad argument!\n");

	if (out == nullptr) {
		out = mzd_init(in->nrows, in->ncols);
	}

	for (int j = 0; j < in->nrows; ++j) {
		mzd_write_bit(out, j, 0, mzd_read_bit(in, (j + i) % in->nrows, 0));
	}

	return out;
}

// create A with `mzp_t *A = mzp_init(length)
// input permutation `A` should have initilaised with: `for(int i = 0; i < A->length; i++) A->value[i] = i` or something.
void matrix_create_random_permutation(mzd_t *A, mzp_t *P) {
	mzd_t * AT= mzd_init(A->ncols,A->nrows);
	mzd_transpose(AT, A);
	// dont permute the last column since it is the syndrome
	for (uint32_t i = 0; i < uint32_t(P->length-1); ++i) {
		word pos = fastrandombytes_uint64() % (P->length - i);

		ASSERT(i+pos < uint64_t(P->length));

		auto tmp = P->values[i];
		P->values[i] = P->values[i+pos];
		P->values[pos+i] = tmp;

		mzd_row_swap(AT, i, i+pos);
	}
	mzd_transpose(A, AT);
	mzd_free(AT);
}

// optimized version to which you have additionally pass the transposed, So its not created/freed every time
void matrix_create_random_permutation(mzd_t *A, mzd_t *AT, mzp_t *P) {
	matrix_transpose(AT, A);
	// dont permute the last column since it is the syndrome
	for (uint32_t i = 0; i < uint32_t(P->length-1); ++i) {
		word pos = fastrandombytes_uint64() % (P->length - i);

		ASSERT(i+pos < uint32_t(P->length));

		auto tmp = P->values[i];
		P->values[i] = P->values[i+pos];
		P->values[pos+i] = tmp;

		mzd_row_swap(AT, i, i+pos);
	}
	matrix_transpose(A, AT);
}


void xor_avx1_new(uint8_t *x, uint8_t *y, uint8_t *z, unsigned n) {
	for (unsigned i = 0; i < n; i += 1) {
#ifdef USE_AVX2
#ifdef USE_AVX2_SPECIAL_ALIGNMENT
		__m256i x_avx = _mm256_load_si256((__m256i *)x + i);
		__m256i y_avx = _mm256_load_si256((__m256i *)y + i);
		__m256i z_avx = _mm256_xor_si256(x_avx, y_avx);
		_mm256_store_si256((__m256i *)z + i, z_avx);
#else
		__m256 x_avx = _mm256_loadu_ps((float*)x + 8*i);
		__m256 y_avx = _mm256_loadu_ps((float*)y + 8*i);
		__m256 z_avx = _mm256_xor_ps(x_avx,y_avx);
		_mm256_storeu_ps((float*)z + 8*i, z_avx);
#endif
#else
		((uint64_t *)z)[4*i] = ((uint64_t *)x)[4*i] ^ ((uint64_t *)y)[4*i];
		((uint64_t *)z)[4*i+1] = ((uint64_t *)x)[4*i+1] ^ ((uint64_t *)y)[4*i+1];
		((uint64_t *)z)[4*i+2] = ((uint64_t *)x)[4*i+2] ^ ((uint64_t *)y)[4*i+2];
		((uint64_t *)z)[4*i+3] = ((uint64_t *)x)[4*i+3] ^ ((uint64_t *)y)[4*i+3];
#endif
	}
}


void matrix_swap_rows_new(mzd_t *M, size_t i, size_t j) {
	word *tmp = M->rows[i];
	M->rows[i] = M->rows[j];
	M->rows[j] = tmp;
}


size_t matrix_gauss_submatrix(mzd_t *M, const size_t r, size_t c, size_t rows,
                              size_t cols, size_t k) {
	size_t start_row = r;
	size_t j;
	size_t cols_padded_ymm = MATRIX_AVX_PADDING(cols) / 256;
	for (j = c; j < c + k; ++j) {
		int found = 0;
		for (size_t i = start_row; i < rows; ++i) {
			for (size_t l = 0; l < j - c; ++l) {
				if ((M->rows[i][(c + l) / WORD_SIZE] >> ((c + l) % WORD_SIZE)) & 1) {
					xor_avx1_new((uint8_t *) M->rows[r + l], (uint8_t *) M->rows[i], (uint8_t *) M->rows[i],
					             cols_padded_ymm);
				}
			}

			if ((M->rows[i][j / WORD_SIZE] >> (j % WORD_SIZE)) & 1) {
				matrix_swap_rows_new(M, i, start_row);
				for (size_t l = r; l < start_row; ++l) {
					if ((M->rows[l][j / WORD_SIZE] >> (j % WORD_SIZE)) & 1) {
						xor_avx1_new((uint8_t *) M->rows[start_row], (uint8_t *) M->rows[l], (uint8_t *) M->rows[l],
						             cols_padded_ymm);
					}
				}
				++start_row;
				found = 1;
				break;
			}
		}

		if (found == 0) {
			break;
		}
	}
	return j - c;
}


void matrix_make_table(mzd_t *M, size_t r, size_t cols, size_t k, uint64_t *T, int **diff) {
	size_t cols_padded = MATRIX_AVX_PADDING(cols);
	size_t cols_padded_word = cols_padded / 64;
	size_t cols_padded_ymm = cols_padded / 256;

	for (size_t i = 0; i < cols_padded_word; ++i) {
		T[i] = 0L;
	}

	for (size_t i = 0; i + 1 < 1UL << k; ++i) {
		xor_avx1_new((uint8_t *)M->rows[r + diff[k][i]], (uint8_t *)T,
		             (uint8_t *)(T + cols_padded_word), cols_padded_ymm);
		T += cols_padded_word;
	}
}


word matrix_read_bits(mzd_t *M, size_t x, size_t y, size_t n) {
	int spot = y % WORD_SIZE;
	word block = y / WORD_SIZE;
	int spill = spot + n - WORD_SIZE;
	word temp = (spill <= 0) ? M->rows[x][block] << -spill
	                         : (M->rows[x][block + 1] << (WORD_SIZE - spill)) |
	                           (M->rows[x][block] >> spill);
	return temp >> (WORD_SIZE- n);
}


void matrix_process_rows(mzd_t *M, size_t rstart, size_t cstart, size_t rstop,
                         size_t k, size_t cols, uint64_t *T, int **rev) {
	size_t cols_padded = MATRIX_AVX_PADDING(cols);
	size_t cols_padded_ymm = cols_padded / 256;
	size_t cols_padded_word = cols_padded / 64;

	for (size_t r = rstart; r < rstop; ++r) {
		size_t x0 = rev[k][matrix_read_bits(M, r, cstart, k)];
		if (x0) {
			xor_avx1_new((uint8_t *) (T + x0 * cols_padded_word), (uint8_t *) M->rows[r],
			             (uint8_t *) M->rows[r], cols_padded_ymm);
		}
	}
}

///
/// \param M 			matrix
/// \param k 			m4ri k
/// \param rstop 		row stop
/// \param matrix_data 	helper data
/// \param cstart 		column start.
/// \return
size_t matrix_echelonize_partial(mzd_t *M, size_t k, size_t rstop, customMatrixData *matrix_data, size_t cstart) {
	int **rev = matrix_data->rev;
	int **diff = matrix_data->diff;
	uint64_t  *xor_rows = matrix_data->lookup_table;

	size_t rows = M->nrows;
	size_t cols = matrix_data->working_nr_cols;

	size_t kk = k;

	size_t r = 0;
	size_t c = cstart;

	while (c < rstop) {
		if (c + kk > rstop) {
			kk = rstop - c;
		}

		size_t kbar = matrix_gauss_submatrix(M, r, c, rows, cols, kk);
		if (kk != kbar) break;

		if (kbar > 0) {
			matrix_make_table(M, r, cols, kbar, xor_rows, diff);
			matrix_process_rows(M, r + kbar, c, rows, kbar, cols, xor_rows, rev);
			matrix_process_rows(M, 0, c, r, kbar, cols, xor_rows, rev);
		}

		r += kbar;
		c += kbar;
	}

	return r;
}

// additionally to the m4ri algorithm a fix is applied if m4ri fails
/// \param M
/// \param k
/// \param rstop
/// \param matrix_data
/// \param cstart
/// \param fix_col 		how many columns must be solved at the end of this function. Must be != 0
/// \param look_ahead 	how many the algorithm must start ahead of `fix_col` to search for a pivo element. Can be zero
/// \param permutation
/// \return
size_t matrix_echelonize_partial_plusfix(mzd_t *M, size_t k, size_t rstop,
										 customMatrixData *matrix_data, size_t cstart,
										 size_t fix_col, size_t look_ahead,
                                         mzp_t *permutation) {
	size_t rang = matrix_echelonize_partial(M, k, rstop, matrix_data, cstart);
	for (size_t b = rang; b < rstop; ++b) {
		bool found = false;
		// find a column where in the last row is a one
		for (size_t i = fix_col+look_ahead; i < size_t(M->ncols); ++i) {
			if (mzd_read_bit(M, b, i)) {
				found = true;
				if (i == b)
					break;

				std::swap(permutation->values[i-look_ahead], permutation->values[b]);
				mzd_col_swap(M, b, i);
				break;
			}
		}

		// if something found, fix this row by adding it to each row where a 1 one.
		if (found) {
			for (size_t i = 0; i < b; ++i) {
				if (mzd_read_bit(M, i, b)) {
					mzd_row_xor(M, i, b);
				}
			}
		} else {
			// if we were not able to fix the gaussian elimination, we return the original rang which was solved
			return rang;
		}
	}

	// return the full rang, if we archived it.
	return fix_col;
}

size_t matrix_echelonize_partial(mzd_t *M, size_t k, size_t rstop, customMatrixData *matrix_data) {
	return matrix_echelonize_partial(M, k, rstop, matrix_data, 0);
}

static int gray_new(int i, int k) {
	int lastbit = 0;
	int res = 0;
	for (int j = k; j-- > 0;) {
		int bit = i & (1 << j);
		res |= (lastbit >> 1) ^ bit;
		lastbit = bit;
	}
	return res;
}


void matrix_alloc_gray_code(int ***rev, int ***diff) {
	*rev = (int **) malloc((MAX_K + 1) * sizeof(int *));
	*diff = (int **) malloc((MAX_K + 1) * sizeof(int *));

	for (size_t k = 0; k <= MAX_K; ++k) {
		(*rev)[k] = (int *) malloc((1 << k) * sizeof(int));
		(*diff)[k] = (int *) malloc((1 << k) * sizeof(int));
	}
}


void matrix_free_gray_code(int **rev, int **diff) {
	for (size_t k = 0; k <= MAX_K; ++k) {
		free(rev[k]);
		free(diff[k]);
	}
	free(rev);
	free(diff);
}


void matrix_build_gray_code(int **rev, int **diff) {
	for (size_t k = 0; k <= MAX_K; ++k) {
		for (size_t i = 0; i < 1UL << k; ++i) {
			rev[k][gray_new(i, k)] = i;
		}

		for (size_t i = k + 1; i-- > 0;) {
			for (size_t j = 1; j < (1UL << i) + 1; ++j) {
				diff[k][j * (1 << (k - i)) - 1] = k - i;
			}
		}
	}
}

// special copy, which can be useful if you working with avx alignment
mzd_t *matrix_copy(mzd_t *N, mzd_t const *P) {
	if (N == P) return N;

	if (N == NULL) {
		N = mzd_init(P->nrows, P->ncols);
	}
	word *p_truerow, *n_truerow;
	wi_t const wide = P->width - 1;
	word mask_end   = P->high_bitmask;
	for (rci_t i = 0; i < P->nrows; ++i) {
		p_truerow = P->rows[i];
		n_truerow = N->rows[i];
		for (wi_t j = 0; j < wide; ++j) n_truerow[j] = p_truerow[j];
		n_truerow[wide] = (n_truerow[wide] & ~mask_end) | (p_truerow[wide] & mask_end);
	}
	__M4RI_DD_MZD(N);
	return N;
}

customMatrixData* init_matrix_data(int nr_columns){
	customMatrixData *matrix_data = (customMatrixData *) malloc(sizeof(customMatrixData));

	matrix_data->real_nr_cols = nr_columns;
	matrix_data->working_nr_cols = nr_columns;  // this can be overwritten

	matrix_alloc_gray_code(&matrix_data->rev, &matrix_data->diff);
	matrix_build_gray_code(matrix_data->rev, matrix_data->diff);

	matrix_data->lookup_table = (uint64_t *) aligned_alloc(4096, (MATRIX_AVX_PADDING(nr_columns) / 8) * (1<<MAX_K));
	return matrix_data;
}

void free_matrix_data(customMatrixData* matrix_data){
	matrix_free_gray_code(matrix_data->rev, matrix_data->diff);
	free(matrix_data->lookup_table);
	free(matrix_data);
}

mzd_t *matrix_init(rci_t r, rci_t c) {
	auto padding = MATRIX_AVX_PADDING(c);
	return mzd_init(r, padding);
}

/// ein versuch die matrix so zu splitten, dass die H matrix 256 alignent ist.
mzd_t *matrix_init_split(const mzd_t *A, const mzd_t *s, const uint32_t nkl, const uint32_t c) {
	ASSERT(s->nrows==1);

	auto padding = MATRIX_AVX_PADDING(A->ncols + c);
	mzd_t *r = mzd_init(A->nrows, padding);

	for (uint32_t row = 0; row < uint32_t(A->nrows); row++) {
		for (uint col = 0; col < nkl; ++col) {
			mzd_write_bit(r, row, col, mzd_read_bit(A, row, col));
		}

		for (int col = nkl; col < A->ncols; ++col) {
			mzd_write_bit(r, row, col+c, mzd_read_bit(A, row, col));
		}

		mzd_write_bit(r, row, A->ncols+c, mzd_read_bit(s, 0, row));

	}
	return r;
}

// the same as above.
static void print_matrix(const char *s, const mzd_t *A, int start_row=-1, int end_row=-1){
	const int ss = start_row == -1 ? 0 : start_row;
	const int ee = end_row == -1 ? A->nrows : end_row;

	printf("%s\n", s);
	for (int i = ss; i < ee; ++i) {
		mzd_fprint_row(stdout, A, i);
	}
	printf("\n");
}

static void print_matrix(const char *s, const mzd_t *A, int start_row, int end_row, int start_col, int end_col){
	const int sstart_row = start_row == -1 ? 0 : start_row;
	const int eend_row = end_row == -1 ? A->nrows : end_row;

	const int sstart_col = start_col == -1 ? 0 : start_col;
	const int eend_col = end_col == -1 ? A->ncols : end_col;

	mzd_t *B = mzd_init(eend_row-sstart_row, eend_col-sstart_col);
	mzd_submatrix(B, A, sstart_row, sstart_col, eend_row, eend_col);

	printf("%s\n", s);
	for (int i = 0; i < eend_row-sstart_row; ++i) {
		mzd_fprint_row(stdout, B, i);
	}
	printf("\n");
	mzd_free(B);
}

#endif //SMALLSECRETLWE_CUSTOM_MATRIX_H
