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
         const uint64_t q,
         const bool packed = true,
         typename R=void>
#if __cplusplus > 201709L
    // TODO
#endif
class FqMatrix : public FqMatrixMeta<T, nrows, ncols, q, packed, R> {
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
#if __cplusplus > 201709L
    // TODO
#endif
class FqMatrix<T, nrows, ncols, 3, packed, void> : public FqMatrixMeta<T, nrows, ncols, 3, packed, void> {
public:
	/// this is just defined, because Im lazy
	static constexpr uint32_t q = 3;

	using R = void;

	using S = FqMatrixMeta<T, nrows, ncols, q, packed, R>;
	/// needed type definitions
	using typename S::RowType;
	using typename S::DataType;

	/// needed vars
	using S::__data;
	using S::ROWS;
	using S::COLS;

	/// needed functions
	using S::get;
	using S::set;
	using S::swap;
	using S::swap_rows;
	using S::swap_cols;
	using S::clear;
	using S::transpose;
	using S::column_popcnt;
	using S::row_popcnt;
	using S::mul;
	using S::add;
	using S::sub;
	using S::sub_transpose;
	using S::sub_matrix;
	using S::gaus;
	using S::fix_gaus;
	using S::m4ri;
	using S::info;
};


#endif//CRYPTANALYSISLIB_FQMATRIX_H
