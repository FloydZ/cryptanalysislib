#ifndef CRYPTANALYSISLIB_FQMATRIX_H
#define CRYPTANALYSISLIB_FQMATRIX_H

#include "matrix/matrix.h"

/// matrix implementation which is row major
/// \tparam T base type something like `uint32_t` or `uint64_t`
/// \tparam nrows number of rows
/// \tparam ncols number of columns
template<typename T,
         const uint32_t nrows,
         const uint32_t ncols,
         const uint32_t q,
         const bool packed=false>
class FqMatrix: public FqMatrix_Meta<T, nrows, ncols, q, packed>{
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
class FqMatrix<T, nrows, ncols, 3, packed> : public  FqMatrix_Meta<T, nrows, ncols, 3, packed> {
public:
	/// this is just defined, because Im lazy
	static constexpr uint32_t q = 3;

	/// needed type definitions
	using typename FqMatrix_Meta<T, nrows,ncols, q, packed>::RowType;
	using FqMatrix_Meta<T, nrows,ncols, q, packed>::DataType;

	/// needed vars
	using FqMatrix_Meta<T, nrows,ncols, q, packed>::__data;
	using FqMatrix_Meta<T, nrows,ncols, q, packed>::ROWS;
	using FqMatrix_Meta<T, nrows,ncols, q, packed>::COLS;

	/// needed functions
	using FqMatrix_Meta<T, nrows,ncols, q, packed>::get;
	using FqMatrix_Meta<T, nrows,ncols, q, packed>::set;
	using FqMatrix_Meta<T, nrows,ncols, q, packed>::swap_rows;
	using FqMatrix_Meta<T, nrows,ncols, q, packed>::swap_cols;
	using FqMatrix_Meta<T, nrows,ncols, q, packed>::clear;

	// TODO reactivate
	/////
	///// \param stop row to stop with the elimination
	///// \return number of rows systemize
	//constexpr uint32_t gaus(const uint32_t stop = -1) noexcept {
	//	const std::size_t n = nrows, m = ncols -1;
	//	RowType tmp;
	//	uint32_t row = 0;
	//	for(uint32_t col = 0; (col < m) && (row < n) && (row < stop); col++) {
	//		int sel = -1;
	//		// get pivot element
	//		for (uint32_t i = row; i < n; i++) {
	//			if (get(i, col) == 1) {
	//				sel = i;
	//				break;
	//			}

	//			if (get(i, col) == 2) {
	//				sel = i;
	//				// times two the whole row
	//				for (uint32_t j = 0; j <= m; ++j) {
	//					set((2*get(i, j))%3, i, j);
	//				}
	//				break;
	//			}
	//		}

	//		if (sel == -1)
	//			return row;

	//		swap_rows(sel, row);

	//		for(uint32_t i = 0; i < n; i++) {
	//			if (i != row) {
	//				if (get(i, col) == 0) {
	//					continue;
	//				}

	//				uint32_t c = get(i, col);
	//				if (c == 0) {
	//					continue;
	//				} else if (c == 1) {
	//					RowType::sub(__data[i], __data[i], __data[row]);
	//				} else {
	//					/// TODO test
	//					RowType::add(__data[i], __data[i], __data[row]);
	//					/// TODO add should be also goot
	//					//RowType::times2_mod3(tmp, __data[row]);
	//					//RowType::sub(__data[i], __data[i], tmp);
	//				}
	//			}
	//		}

	//		row++;
	//	}

	//	return row;
	//}

	///// \param permutation	currently column permutation of the input matrix. Its needed because we might rearrange
	/////						columns to further execute the gaussian elimination
	///// \param rang			current rang of the matrix.
	///// \param fix_col		up to which rang should the matrix be solved?
	///// \param look_ahead   how many coordinated is the algorithm allowed to look ahead.
	///// \return the new rang of the matrix.
	//uint32_t fix_gaus(uint32_t *permutation,
	//                  const uint32_t rang,
	//                  const uint32_t fix_col,
	//                  const uint32_t lookahead) noexcept {
	//	for (uint32_t b = rang; b < fix_col; ++b) {
	//		bool found = false;
	//		// find a column in which a one is found
	//		for (uint32_t i = b+1 ; i < lookahead; ++i) {
	//			if (get(b, i) == 1u) {
	//				found = true;

	//				std::swap(permutation[i], permutation[b]);
	//				swap_cols(i, b);
	//			}
	//		}
	//		if (found) {
	//			// fix the column
	//			for (uint32_t i = 0; i < b; ++i) {
	//				uint32_t a = get(i, b);
	//				if (a > 0) {
	//					RowType::add(__data[i], __data[i], __data[b]);
	//				}
	//				if (a == 2) {
	//					RowType::add(__data[i], __data[i], __data[b]);
	//				}
	//			}
	//		} else {
	//			// Sorry nothing found:
	//			return rang;
	//		}
	//	}

	//	return ROWS;
	//}
};


#endif//CRYPTANALYSISLIB_FQMATRIX_H
