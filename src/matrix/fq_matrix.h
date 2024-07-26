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
         const bool packed = true,
         typename R=void>
class FqMatrix : public FqMatrix_Meta<T, nrows, ncols, q, packed, R> {
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
class FqMatrix<T, nrows, ncols, 3, packed, void> : public FqMatrix_Meta<T, nrows, ncols, 3, packed, void> {
public:
	/// this is just defined, because Im lazy
	static constexpr uint32_t q = 3;

	using R = void;

	/// needed type definitions
	using typename FqMatrix_Meta<T, nrows, ncols, q, packed, R>::RowType;
	using typename FqMatrix_Meta<T, nrows, ncols, q, packed, R>::DataType;

	/// needed vars
	using FqMatrix_Meta<T, nrows, ncols, q, packed, R>::__data;
	using FqMatrix_Meta<T, nrows, ncols, q, packed, R>::ROWS;
	using FqMatrix_Meta<T, nrows, ncols, q, packed, R>::COLS;

	/// needed functions
	using FqMatrix_Meta<T, nrows, ncols, q, packed, R>::get;
	using FqMatrix_Meta<T, nrows, ncols, q, packed, R>::set;
	using FqMatrix_Meta<T, nrows, ncols, q, packed, R>::swap_rows;
	using FqMatrix_Meta<T, nrows, ncols, q, packed, R>::swap_cols;
	using FqMatrix_Meta<T, nrows, ncols, q, packed, R>::clear;
	using FqMatrix_Meta<T, nrows, ncols, q, packed, R>::transpose;
	using FqMatrix_Meta<T, nrows, ncols, q, packed, R>::column_popcnt;
	using FqMatrix_Meta<T, nrows, ncols, q, packed, R>::row_popcnt;
};


#endif//CRYPTANALYSISLIB_FQMATRIX_H
