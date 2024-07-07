#ifndef CRYPTANALYSISLIB_FQMATRIX_H
#define CRYPTANALYSISLIB_FQMATRIX_H

#ifndef CRYPTANALYSISLIB_MATRIX_H
#error "dont use include <list/enumeration/fq_matrix.h>, use #include <list/enumeration/marix.h> instead"
#endif

#include "matrix/matrix.h"

/// matrix implementation which is row major
/// \tparam T base type something like `uint32_t` or `uint64_t`
/// \tparam nrows number of rows
/// \tparam ncols number of columns
template<typename T,
         const uint32_t nrows,
         const uint32_t ncols,
         const uint32_t q,
         const bool packed = true>
class FqMatrix : public FqMatrix_Meta<T, nrows, ncols, q, packed> {
public:
};

/// matrix implementation which is row major
/// NOTE special case for q=3
/// \tparam T base type something like `uint32_t` or `uint64_t`
/// \tparam nrows number of rows
/// \tparam ncols number of columns
template<typename T,
         const uint32_t nrows,
         const uint32_t ncols,
         const bool packed>
class FqMatrix<T, nrows, ncols, 3, packed> : public FqMatrix_Meta<T, nrows, ncols, 3, packed> {
public:
	/// this is just defined, because Im lazy
	static constexpr uint32_t q = 3;

	/// needed type definitions
	using typename FqMatrix_Meta<T, nrows, ncols, q, packed>::RowType;
	using typename FqMatrix_Meta<T, nrows, ncols, q, packed>::DataType;

	/// needed vars
	using FqMatrix_Meta<T, nrows, ncols, q, packed>::__data;
	using FqMatrix_Meta<T, nrows, ncols, q, packed>::ROWS;
	using FqMatrix_Meta<T, nrows, ncols, q, packed>::COLS;

	/// needed functions
	using FqMatrix_Meta<T, nrows, ncols, q, packed>::get;
	using FqMatrix_Meta<T, nrows, ncols, q, packed>::set;
	using FqMatrix_Meta<T, nrows, ncols, q, packed>::swap_rows;
	using FqMatrix_Meta<T, nrows, ncols, q, packed>::swap_cols;
	using FqMatrix_Meta<T, nrows, ncols, q, packed>::clear;
	using FqMatrix_Meta<T, nrows, ncols, q, packed>::transpose;
	using FqMatrix_Meta<T, nrows, ncols, q, packed>::column_popcnt;
	using FqMatrix_Meta<T, nrows, ncols, q, packed>::row_popcnt;
};


#endif//CRYPTANALYSISLIB_FQMATRIX_H
