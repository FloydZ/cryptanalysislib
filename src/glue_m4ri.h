#ifndef SMALLSECRETLWE_GLUE_M4RI_H
#define SMALLSECRETLWE_GLUE_M4RI_H

#include "m4ri/m4ri.h"
#include "m4ri/misc.h"
#include "random.h"

#include "helper.h"



int mzd_copy_row_internal(word *out, const word *in, const int start, const int end){
	ASSERT(end > start && start >= 0);
	for (int i = start; i < end; ++i) {
		BIT b = __M4RI_GET_BIT(in[i/m4ri_radix], i%m4ri_radix);
		__M4RI_WRITE_BIT(out[(i-start)/m4ri_radix], (i-start)%m4ri_radix, b);
	}

	return end-start;
}


void mzd_clear(mzd_t *A) {
	for (int i = 0; i < A->nrows; ++i) {
		for (int j = 0; j < A->width; ++j) {
			A->rows[i][j] = 0;
		}
	}
}
void mzd_row_and(mzd_t *out,  const mzd_t *in1, const mzd_t *in2, const rci_t i) {
	ASSERT(out->nrows > i && out->ncols == in1->ncols && out->ncols == in2->ncols);
	for (int l = 0; l < out->width; ++l) {
		out->rows[i][l] = in1->rows[i][l] & in2->rows[i][l];
	}
}

void mzd_row_or(mzd_t *out, const rci_t i, const mzd_t *in, const rci_t j) {
	ASSERT(out->nrows > i && in->nrows > j && out->ncols == in->ncols);
	for (int l = 0; l < out->width; ++l) {
		out->rows[i][l] |= in->rows[j][l];
	}
}


// xor row i and j together ans save the result in i. But only xor between scol and ecol.
template<const uint32_t scol, const uint32_t ecol>
void mzd_row_xor(mzd_t *A, const rci_t i, const rci_t j) {
	ASSERT(A->nrows > i && A->nrows > j);

	constexpr uint32_t start_limb = scol/64;
	constexpr uint32_t end_limb = (ecol+63)/64;

	constexpr uint32_t start_offset = scol%64;
	constexpr uint32_t end_offset = ecol%64;

	constexpr uint64_t low_bitmask = start_offset == 0 ? -1 : __M4RI_RIGHT_BITMASK(64-start_offset);
	constexpr uint64_t high_bitmask = __M4RI_LEFT_BITMASK(end_offset);

	if constexpr (start_limb == end_limb){
		constexpr uint64_t mask = low_bitmask & high_bitmask;
		A->rows[i][start_limb] ^= A->rows[j][start_limb] & mask;
		return;
	}


	A->rows[i][start_limb] ^= A->rows[j][start_limb] & low_bitmask;
	LOOP_UNROLL();
	for (uint32_t k = start_limb + 1; k < end_limb-1; ++k) {
		A->rows[i][k] ^= A->rows[j][k];

	}
	A->rows[i][end_limb] ^= A->rows[j][end_limb] & high_bitmask;
}

// tested. Only work on the first limbs of a row. Usefull if a matrix was initialized with AVX alignment.
template<const uint32_t ecol>
void mzd_row_xor(mzd_t *A, const rci_t i, const rci_t j) {
	ASSERT(A->nrows > i && A->nrows > j);
	constexpr uint64_t high_bitmask = __M4RI_LEFT_BITMASK(ecol%64);
	constexpr uint32_t limbs = ((ecol+63) / 64)-1;
	for (uint32_t l = 0; l < limbs; ++l) {
		A->rows[i][l] ^= A->rows[j][l];
	}
	A->rows[i][limbs] ^= A->rows[j][limbs] & high_bitmask;

}

template<const uint32_t ecol>
void mzd_row_xor_unroll(mzd_t *out, const rci_t i, const rci_t j) {
	constexpr uint32_t limbs = (ecol+63) / 64;
	ASSERT(out->nrows > i && out->nrows > j);

    #pragma unroll
	for (uint32_t l = 0; l < limbs; ++l) {
		out->rows[i][l] ^= out->rows[j][l];
	}
}




void mzd_row_xor(mzd_t *out, const rci_t i, const rci_t j, const rci_t k) {
	assert(out->nrows > i && out->nrows > j && out->nrows > k);
	for (uint32_t l = 0; l < uint32_t(out->width); ++l) {
		out->rows[i][l] ^= out->rows[j][l] ^ out->rows[k][l];
	}
}

void mzd_row_xor(mzd_t *out, const rci_t i, const mzd_t *in, const rci_t j) {
	// ASSERT(out->nrows > i && in->nrows > j && out->ncols == in->ncols);
	ASSERT(out->nrows > i && in->nrows > j);
	for (uint32_t l = 0; l < uint32_t(out->width); ++l) {
		out->rows[i][l] ^= in->rows[j][l];
	}
}

void mzd_row_xor(mzd_t *out, const rci_t i, const mzd_t *in1, const rci_t j1, const mzd_t *in2, const rci_t j2) {
	ASSERT(out->nrows > i && in1->nrows > j1 && in2->nrows > j2 && out->ncols == in1->ncols);
	for (uint32_t l = 0; l < uint32_t(out->width); ++l) {
		out->rows[i][l] = in1->rows[j1][l] ^ in2->rows[j2][l];
	}
}

uint32_t mzd_row_xor_weight(mzd_t *out, const rci_t i, const uint64_t *in) {
	uint32_t weight = 0;
	for (uint32_t l = 0; l < uint32_t(out->width); ++l) {
		weight += __builtin_popcountll(out->rows[i][l] ^ in[l]);
	}
	return weight;
}

uint32_t mzd_row_xor_weight(mzd_t *out, const rci_t i, const mzd_t *in, const rci_t j) {
	ASSERT(out->nrows > i && in->nrows > j && out->ncols == in->ncols);
	uint32_t weight = 0;
	for (uint32_t l = 0; l < uint32_t(out->width); ++l) {
		weight += __builtin_popcountll(out->rows[i][l] ^ in->rows[j][l]);
	}
	return weight;
}

uint32_t mzd_row_and_weight(mzd_t *out, const rci_t i, const mzd_t *in, const rci_t j) {
	ASSERT(out->nrows > i && in->nrows > j && out->ncols == in->ncols);
	uint32_t weight = 0;
	for (uint32_t l = 0; l < uint32_t(out->width); ++l) {
		weight += __builtin_popcountll(out->rows[i][l] & in->rows[j][l]);
	}
	return weight;
}

uint32_t mzd_row_and_xor_weight(const mzd_t *in1, const mzd_t *in2, const rci_t i) {
	ASSERT(in1->nrows > i && in2->nrows > i && in1->ncols == in2->ncols);
	uint32_t weight = 0;
	for (uint32_t l = 0; l < uint32_t(in1->width); ++l) {
		weight += __builtin_popcountll((in1->rows[i][l] ^ in1->rows[i+1][l]) & in2->rows[i][l]);
	}
	return weight;
}

void mzd_row_and_not(mzd_t *out, const rci_t i, const mzd_t *in, const rci_t j) {
	ASSERT(out->nrows > i && in->nrows > j && out->ncols == in->ncols);
	for (int l = 0; l < out->width; ++l) {
		out->rows[i][l] &= ~in->rows[j][l];
	}
}

void mzd_set_row(mzd_t *out, const rci_t i) {
	ASSERT(out->nrows > i);
	for (int l = 0; l < out->width-1; ++l) {
		out->rows[i][l] = uint64_t(-1);
	}

	out->rows[i][out->width-1] = uint64_t(-1) & out->high_bitmask;
}


// calculates the hamming weight of the sum of two rows within a given range of columns without save the sum of the two rows.
template<const uint32_t start_col, const uint32_t end_col>
__INLINE__ unsigned int hamming_weight_sum_row(const mzd_t *A, const uint32_t row1, const uint32_t row2) {
	constexpr uint32_t start_limb = start_col/64;
	constexpr uint32_t end_limb = end_col/64;

	constexpr uint32_t start_offset = start_col%64;
	constexpr uint32_t end_offset = end_col%64;

	constexpr uint64_t low_bitmask = start_offset == 0 ? -1 : __M4RI_RIGHT_BITMASK(64-start_offset);
	constexpr uint64_t high_bitmask = __M4RI_LEFT_BITMASK(end_offset);

	if constexpr (start_limb == end_limb){
		constexpr uint64_t mask = low_bitmask & high_bitmask;
		return __builtin_popcountll((A->rows[row1][start_limb] ^ A->rows[row2][start_limb]) & mask);
	}

	unsigned int weight = __builtin_popcountll((A->rows[row1][start_limb] ^ A->rows[row2][start_limb]) & low_bitmask);

	LOOP_UNROLL();
	for (uint32_t i = start_limb+1; i < end_limb; ++i) {
		weight += __builtin_popcountll(A->rows[row1][i] ^ A->rows[row2][i]);
	}

	return weight + __builtin_popcountll((A->rows[row1][end_limb] ^ A->rows[row2][end_limb])& high_bitmask);
}

__INLINE__ unsigned int hamming_weight_sum_row(const mzd_t *A, const uint32_t row1, const uint32_t row2, const uint32_t start_col, const uint32_t end_col) {
	const uint32_t start_limb = start_col/64;
	const uint32_t end_limb = end_col/64;

	const uint32_t start_offset = start_col%64;
	const uint32_t end_offset = end_col%64;

	const uint64_t low_bitmask = start_offset == 0 ? uint64_t(-1) : __M4RI_RIGHT_BITMASK(64-start_offset);
	const uint64_t high_bitmask = __M4RI_LEFT_BITMASK(end_offset);

	if (start_limb == end_limb){
		const uint64_t mask = low_bitmask & high_bitmask;
		return __builtin_popcountll((A->rows[row1][start_limb] ^ A->rows[row2][start_limb]) & mask);
	}

	unsigned int weight = __builtin_popcountll((A->rows[row1][start_limb] ^ A->rows[row2][start_limb]) & low_bitmask);

	LOOP_UNROLL();
	for (uint32_t i = start_limb+1; i < end_limb; ++i) {
		weight += __builtin_popcountll(A->rows[row1][i] ^ A->rows[row2][i]);
	}

	return weight + __builtin_popcountll((A->rows[row1][end_limb] ^ A->rows[row2][end_limb])& high_bitmask);
}

template<const uint32_t start_col, const uint32_t end_col>
__INLINE__ unsigned int hamming_weight_sum_row(const mzd_t *A, const uint32_t row1, const uint32_t row2, const uint32_t row3) {
	constexpr uint32_t start_limb = start_col/64;
	constexpr uint32_t end_limb = end_col/64;

	constexpr uint32_t start_offset = start_col%64;
	constexpr uint32_t end_offset = end_col%64;

	constexpr uint64_t low_bitmask = start_offset == 0 ? -1 : __M4RI_RIGHT_BITMASK(64-start_offset);
	constexpr uint64_t high_bitmask = __M4RI_LEFT_BITMASK(end_offset);

	if constexpr (start_limb == end_limb){
		constexpr uint64_t mask = low_bitmask & high_bitmask;
		return __builtin_popcountll((A->rows[row1][start_limb] ^
		                             A->rows[row2][start_limb] ^
		                             A->rows[row3][start_limb]) & mask);
	}

	unsigned int weight = __builtin_popcountll((A->rows[row1][start_limb] ^
	                                            A->rows[row2][start_limb] ^
	                                            A->rows[row3][start_limb]) & low_bitmask);

	LOOP_UNROLL();
	for (uint32_t i = start_limb+1; i < end_limb; ++i) {
		weight += __builtin_popcountll(A->rows[row1][i] ^ A->rows[row2][i] ^ A->rows[row3][i]);
	}

	return weight + __builtin_popcountll(  (A->rows[row1][end_limb] ^
	                                        A->rows[row2][end_limb] ^
	                                        A->rows[row3][end_limb]) & high_bitmask);
}

__INLINE__ unsigned int hamming_weight_sum_row(const mzd_t *A, const uint32_t row1, const uint32_t row2, const uint32_t row3, const uint32_t start_col, const uint32_t end_col) {
	const uint32_t start_limb = start_col/64;
	const uint32_t end_limb = end_col/64;

	const uint32_t start_offset = start_col%64;
	const uint32_t end_offset = end_col%64;

	const uint64_t low_bitmask = start_offset == 0 ? uint64_t(-1) : __M4RI_RIGHT_BITMASK(64-start_offset);
	const uint64_t high_bitmask = __M4RI_LEFT_BITMASK(end_offset);

	if (start_limb == end_limb){
		const uint64_t mask = low_bitmask & high_bitmask;
		return __builtin_popcountll((A->rows[row1][start_limb] ^
		                             A->rows[row2][start_limb] ^
		                             A->rows[row3][start_limb]) & mask);
	}

	unsigned int weight = __builtin_popcountll((A->rows[row1][start_limb] ^
	                                            A->rows[row2][start_limb] ^
	                                            A->rows[row3][start_limb]) & low_bitmask);

	LOOP_UNROLL();
	for (uint32_t i = start_limb+1; i < end_limb; ++i) {
		weight += __builtin_popcountll(A->rows[row1][i] ^ A->rows[row2][i] ^ A->rows[row3][i]);
	}

	return weight + __builtin_popcountll(  (A->rows[row1][end_limb] ^
	                                        A->rows[row2][end_limb] ^
	                                        A->rows[row3][end_limb]) & high_bitmask);
}

template<const uint32_t srow, const uint32_t erow>
unsigned int hamming_weight_column(const mzd_t *A, const uint32_t col) {
	ASSERT(A->ncols > col);
	unsigned int ret = 0;
	for (uint32_t i = srow; i < erow; ++i) { ret += mzd_read_bit(A, i, col); }
	return ret;
}

/// calculates the hamming weight of the column `col`
/// \param A
/// \return
uint32_t hamming_weight_column(const mzd_t *A, const uint32_t col) {
	// Allow to calc the weight of the syndrome?
	ASSERT(uint32_t(A->ncols) >= col);

	uint32_t ret = 0;
	for (uint32_t i = 0; i < uint32_t(A->nrows); ++i) { ret += mzd_read_bit(A, i, col); }
	return ret;
}



__FORCEINLINE__ uint32_t hamming_weight_row(const mzd_t *A, const uint32_t row) {
	uint32_t weight = 0u;

	LOOP_UNROLL();
	for (uint32_t i = 0; i < uint32_t(A->width-1); ++i) {
		weight += __builtin_popcountll(A->rows[row][i]);
	}

	return weight + __builtin_popcountll(A->rows[row][A->width-1] & A->high_bitmask);
}

template<const uint32_t start_col, const uint32_t end_col>
__FORCEINLINE__ uint32_t hamming_weight_row(const mzd_t *A, const uint32_t row){
	constexpr uint32_t start_limb = start_col/64;
	constexpr uint32_t end_limb = end_col/64;

	constexpr uint32_t start_offset = start_col%64;
	constexpr uint32_t end_offset = end_col%64;

	constexpr uint64_t low_bitmask = start_offset == 0 ? -1 : __M4RI_RIGHT_BITMASK(64-start_offset);
	constexpr uint64_t high_bitmask = __M4RI_LEFT_BITMASK(end_offset);

	if constexpr (start_limb == end_limb){
		constexpr uint64_t mask = low_bitmask & high_bitmask;
		return __builtin_popcountll(A->rows[row][start_limb] & mask);
	}

	uint32_t weight = __builtin_popcountll(A->rows[row][start_limb] & low_bitmask);

	LOOP_UNROLL();
	for (uint32_t i = start_limb+1; i < end_limb-1; ++i) {
		weight += __builtin_popcountll(A->rows[row][i]);
	}

	return weight + __builtin_popcountll(A->rows[row][end_limb] & high_bitmask);
}

__INLINE__ unsigned int hamming_weight_row(const mzd_t *A, const uint32_t row, const uint32_t start_col){
	const uint32_t start_limb = start_col/64;
	const uint32_t hh = start_col%64;
	const uint64_t low_bitmask = hh == 0 ? uint64_t(-1) : __M4RI_RIGHT_BITMASK(64-hh);

	unsigned int weight = __builtin_popcountll(A->rows[row][start_limb] & low_bitmask);

	LOOP_UNROLL();
	for (uint32_t i = start_limb+1; i < uint32_t(A->width-1); ++i) {
		weight += __builtin_popcountll(A->rows[row][i]);
	}

	return weight + __builtin_popcountll(A->rows[row][A->width-1] & A->high_bitmask);
}

inline uint32_t hamming_weight_row(const mzd_t *A, const uint32_t row, const uint32_t start_col, const uint32_t end_col){
	const uint32_t start_limb = start_col/64;
	const uint32_t end_limb = end_col/64;

	const uint32_t start_offset = start_col%64;
	const uint32_t end_offset = end_col%64;

	const uint64_t low_bitmask = start_offset == 0 ? uint64_t(-1) : __M4RI_RIGHT_BITMASK(64-start_offset);
	const uint64_t high_bitmask = __M4RI_LEFT_BITMASK(end_offset);

	if (start_limb == end_limb){
		const uint64_t mask = low_bitmask & high_bitmask;
		return __builtin_popcountll(A->rows[row][start_limb] & mask);
	}

	uint32_t weight = __builtin_popcountll(A->rows[row][start_limb] & low_bitmask);

	LOOP_UNROLL();
	for (uint32_t i = start_limb+1; i < uint32_t(A->width-1); ++i) {
		weight += __builtin_popcountll(A->rows[row][i]);
	}

	return weight + __builtin_popcountll(A->rows[row][A->width-1] & A->high_bitmask);
}

inline uint32_t hamming_weight(const mzd_t *A) {
	ASSERT(A->nrows == 1);
	uint32_t weight = 0u;

	LOOP_UNROLL();
	for (uint32_t i = 0; i < uint32_t(A->width-1); ++i) {
		weight += __builtin_popcountll(A->rows[0][i]);
	}

	return weight + __builtin_popcountll(A->rows[0][A->width-1] & A->high_bitmask);
}

template <const uint32_t limbs>
inline uint32_t hamming_weight(const mzd_t *A) {
	ASSERT(A->nrows == 1);

	uint32_t weight = 0u;

	LOOP_UNROLL();
	for (uint32_t i = 0; i < uint32_t(A->width-1); ++i) {
		weight += __builtin_popcountll(A->rows[0][i]);
	}

	return weight+__builtin_popcountll(A->rows[0][A->width-1] & A->high_bitmask);
}


template<class Label>
int m4ri_cmp_row(const word *in1, const Label &in2, const int len) {
	for (int i = 0; i < len; ++i) {
		if(in1[i] < in2[i])
			return -1;
		if(in1[i] > in2[i])
			return 1;
	}
	return 0;
}


// this function can compare everything.
//  - row vector vs row vector
//  - column vector vs row vector
//  full matrix comparison not implemented.
// return 1/-1 if unequal. 0 if equal
int m4ri_cmp_row(const mzd_t *in1, const mzd_t *in2) {
	if ((in1->nrows == 1) && (in2->nrows == 1)) {                           // Both row vectors
		if (in1->width != in2->width)
			return 1;
		return m4ri_cmp_row(in1->rows[0], in2->rows[0], in1->width);
	} else if ((in1->ncols == 1) && (in2->nrows == 1)) {                    // column x row
		if (in1->nrows != in2->ncols)
			return 1;
		for (int i = 0; i < in1->nrows; ++i) {
			if (in1->rows[i][0] != word(mzd_read_bit(in2, 0, i)))
				return 1;
		}
	} else {
		std::cout << "not implemented\n";
	}

	return 0;
}

inline void SWAP_BITS(word *row, const int pos1, const int pos2) {
	BIT p1 = __M4RI_GET_BIT(row[pos1/m4ri_radix], pos1 % m4ri_radix);
	BIT p2 = __M4RI_GET_BIT(row[pos2/m4ri_radix], pos2 % m4ri_radix);
	__M4RI_WRITE_BIT(row[pos1/ m4ri_radix], pos1 % m4ri_radix, p2);
	__M4RI_WRITE_BIT(row[pos2/ m4ri_radix], pos2 % m4ri_radix, p1);
}


template <const uint32_t limbs>
inline int m4ri_cmp_row_testing(const mzd_t *in1, const mzd_t *in2) {
	ASSERT(in1->nrows == in2->nrows && in1->nrows == 1 && in1->ncols == in2->ncols);

	LOOP_UNROLL();
	for (uint32_t i = 0; i < limbs; ++i) {
		if (in1->rows[0][i] != in2->rows[0][i])
			return 1;
	}

	return 0;
}

// copies: out[0,..., end-start] = in[start,...,end]
// make sure space is allocated.
int m4ri_copy_row(word *out, const word *in, const int start, const int end){
	ASSERT(end > start && start >= 0);
	for (int i = start; i < end; ++i) {
		BIT b = __M4RI_GET_BIT(in[i/m4ri_radix], i%m4ri_radix);
		__M4RI_WRITE_BIT(out[(i-start)/m4ri_radix], (i-start)%m4ri_radix, b);
	}

	return end-start;
}

// Adds `row1` and `row2` of matrix `in` and writes the result to `out_row` in matrix `out`.
// Also back permutes the out_row
void mzd_add_row(mzd_t *out, const mzd_t *in, const std::vector<uint8_t> &perm, const uint32_t out_row,  const uint32_t row1, const uint32_t row2) {
	ASSERT(out->ncols == in->ncols && row1 < uint32_t(in->nrows) && row2 < uint32_t(in->nrows) && out_row < uint32_t(out->nrows));

	for (int i = 0; i < in->width-1; ++i) {
		out->rows[out_row][i] = in->rows[row1][i] ^ in->rows[row2][i];
	}
	const unsigned int j = in->width-1;
	out->rows[out_row][j] = (in->rows[row1][j] ^ in->rows[row2][j]) & out->high_bitmask;

	//
	for (uint64_t i = 0; i < perm.size(); ++i) {
		SWAP_BITS(out->rows[out_row], perm[i], i);
	}
}

void mzd_add_row_simple(mzd_t *out, const uint32_t out_row,  const mzd_t *in, const uint32_t row1) {
	ASSERT(out->ncols == in->ncols && row1 < uint32_t(in->nrows) && out_row < uint32_t(out->nrows));

	for (int i = 0; i < in->width-1; ++i) {
		out->rows[out_row][i] ^= in->rows[row1][i];
	}
	const unsigned int j = in->width-1;
	out->rows[out_row][j] ^= (in->rows[row1][j] & out->high_bitmask);
}

void copy_submatrix(mzd_t *out_matrix, const mzd_t *const in_matrix, const int start_r, const int start_c, const int end_r, const int end_c){
	ASSERT(start_r >= 0 && start_r < end_r && end_r <= in_matrix->nrows);
	ASSERT(start_c >= 0 && start_c < end_c && end_c <= in_matrix->ncols);
	ASSERT(out_matrix->nrows == end_r - start_r);
	ASSERT(out_matrix->ncols == end_c - start_c);

	for(int i = start_r; i < end_r; i++){
		m4ri_copy_row(out_matrix->rows[i-start_r], in_matrix->rows[i], start_c, end_c);
	}
}

/// generates a a full rank matrix.
/// \param A
/// \return	the rank of the matrix
int m4ri_random_full_rank(mzd_t *A) {
	ASSERT(A->nrows <= A->ncols);

	mzd_t *tmp = mzd_init(A->nrows, A->ncols);
	int full = 1;

	while (true){
		mzd_randomize(A);
		mzd_copy(tmp, A);

		if(mzd_echelonize(tmp, full) == A->nrows)
			break;
	}

	mzd_free(tmp);
	return A->nrows;
}

// its important to observe that to create a random permutation on a row,
// one is only allowed to swap a bit in round i at position i with a bit at position i < j < len;
/// \param row 				__must__ not be NULL pointer
/// \param len 				number of bits in `row`
/// \param weight_per_row
void matrix_generate_random_row_weighted(word *row, const uint64_t len, const uint64_t weight_per_row) {
	ASSERT(weight_per_row <= len && len > 0);

	uint32_t limbs = weight_per_row / 64;
	row[limbs] = (1u << (weight_per_row % 64)) - 1u;

	for (uint32_t i = 0; i < limbs; ++i) {
		row[i] = ~(row[i] & 0u);
	}

	for (uint32_t i = 0; i < len-1; ++i) {
		word pos = fastrandombytes_uint64() % (len - i);
		SWAP_BITS(row, i, i + pos);
	}
}

void matrix_generate_random_weighted(mzd_t *A, const uint64_t weight_per_row) {
	ASSERT(weight_per_row <= uint64_t(A->ncols) && A->ncols > 0 && A->nrows > 0);

	for (int i = 0; i < A->nrows; ++i) {
		matrix_generate_random_row_weighted(A->rows[i], A->ncols, weight_per_row);
	}
}

///
/// \param generator
/// \param parity_check
void parity_check_matrix_to_generator_matrix(mzd_t *generator, const mzd_t *parity_check) {
	const uint32_t n = parity_check->ncols;
	const uint32_t k = n - parity_check->nrows;

	ASSERT(k == (uint32_t)generator->nrows);
	ASSERT(n == (uint32_t)generator->ncols);

	// force a identiy matrix on the first n-k coordinates of the parity check matrix
	for (uint32_t i = 0; i < n-k; i++) {
		for (uint32_t j = 0; j < n-k; j++) {
			if (i == j) {
				ASSERT(mzd_read_bit(parity_check, i, j) == 1);
				continue;
			}

			ASSERT(mzd_read_bit(parity_check, i, j) == 0);
		}
	}

	// read parity check matrix column wise
	for (uint32_t j = 0; j < k; j++) {
		// for all columns
		for (uint32_t i = 0; i < n - k; i++) {
			// for all rows
			const uint32_t bit = mzd_read_bit(parity_check, i, n-k+j);
			mzd_write_bit(generator, j, i, bit);
		}
	}

	// write the identiy matrix in the last k coordinates of the generator matrix
	for (uint32_t i = 0; i < k; i++) {
		for (uint32_t j = 0; j < k; j++) {
			if (i == j) {
				mzd_write_bit(generator, i, n - k + j, 1);
			}

			mzd_write_bit(generator, i, n - k + j, 0);
		}
	}
}

///
/// \param parity_check
/// \param generator
void generator_matrix_to_parity_check_matrix(mzd_t *parity_check, const mzd_t *generator) {
	const uint32_t n = generator->ncols;
	const uint32_t k = generator->nrows;

	ASSERT(n-k  == (uint32_t)parity_check->nrows);
	ASSERT(n    == (uint32_t)parity_check->ncols);

	// force a identy matrix in the last k coordinates of the genertor matrix
	for (uint32_t i = 0; i < k; i++) {
		for (uint32_t j = 0; j < k; j++) {
			if (i == j) {
				ASSERT(mzd_read_bit(generator, i, j + n - k) == 1);
				continue;
			}

			ASSERT(mzd_read_bit(generator, i, j + n - k) == 0);
		}
	}



	// read generator matrix column wise
	for (uint32_t j = 0; j < n-k; j++) {
		// for all columns
		for (uint32_t i = 0; i < k; i++) {
			// for all rows
			const uint32_t bit = mzd_read_bit(generator, (int)i, n-k+j);
			mzd_write_bit(parity_check, (int)j, (int)i, bit);
		}
	}

	// write the identiy matrix in the first n-k coordinates of the parity check matrix
	for (uint32_t i = 0; i < n-k; i++) {
		for (uint32_t j = 0; j < n-k; j++) {
			if (i == j) {
				mzd_write_bit(parity_check, (int)i, (int)j, 1);
			}

			mzd_write_bit(parity_check, (int)i, (int)j, 0);
		}
	}
}
#endif //SMALLSECRETLWE_GLUE_M4RI_H
