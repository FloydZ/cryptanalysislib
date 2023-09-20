#ifndef CRYPTANALYSISLIB_BINARYMATRIX_H
#define CRYPTANALYSISLIB_BINARYMATRIX_H

#include "matrix/fq_matrix.h"
#include "matrix/custom_m4ri.h"

/// matrix implementation which wrapped mzd
/// \tparam T MUST be uint64_t
/// \tparam nrows number of rows
/// \tparam ncols number of columns
template<typename T, const uint32_t nrows, const uint32_t ncols>
class FqMatrix<T, nrows, ncols, 2>: private FqMatrix_Meta<T, nrows, ncols, 2> {
public:
	/// that's only because I'm lazy
	static constexpr uint32_t q = 2;

	/// needed typedefs
	using RowType = T*;
	using DataType = uint8_t;
	mzd_t *__data;
	customMatrixData *__matrix_data;
	const uint32_t m4ri_k = matrix_opt_k(nrows, ncols);

	/// needed functions
	using FqMatrix_Meta<T, nrows,ncols, q>::set;
	using FqMatrix_Meta<T, nrows,ncols, q>::identity;
	using FqMatrix_Meta<T, nrows,ncols, q>::is_equal;
	using FqMatrix_Meta<T, nrows,ncols, q>::weight_column;
	using FqMatrix_Meta<T, nrows,ncols, q>::swap;

	/// simple constructor
	constexpr FqMatrix() noexcept {
		static_assert(sizeof(T) == 8);
		__data = matrix_init(nrows, ncols);
		__matrix_data = init_matrix_data(ncols);
	}

	/// constructor for M4RI
	constexpr FqMatrix(mzd_t *in) noexcept {
		static_assert(sizeof(T) == 8);
		__data = in;
	}

	/// simple deconstructor
	constexpr ~FqMatrix() noexcept {
		mzd_free(__data);
		free_matrix_data(__matrix_data);
	}

	/// copy constructor
	constexpr FqMatrix(const FqMatrix &A) noexcept {
		__data = matrix_init(nrows, ncols);
		matrix_copy(__data, A.__data);
		__matrix_data = init_matrix_data(ncols);
	}

	/// constructor reading from string
	constexpr FqMatrix(const char* data, const uint32_t cols=ncols) noexcept {
		__data = mzd_from_str(nrows, cols, data);
		__matrix_data = init_matrix_data(ncols);
	}

	/// copy operator
	constexpr void copy(const FqMatrix &A) noexcept {
		matrix_copy(__data, A.__data);
	}

	/// \param data data to set
	/// \param i row
	/// \param j colum
	constexpr void set(const uint32_t data, const uint32_t i, const uint32_t j) noexcept {
		mzd_write_bit(__data, i, j, data);
	}

	/// gets the i-th row and j-th column
	/// \param i row
	/// \param j colum
	/// \return entry in this place
	[[nodiscard]] constexpr DataType get(const uint32_t i, const uint32_t j) const noexcept {
		ASSERT(i < nrows && j <= ncols);
		return mzd_read_bit(__data, i, j);
	}

	/// \param i row number (zero indexed)
	/// \return a const ref to a row
	[[nodiscard]] constexpr const RowType& get(const uint32_t i) const noexcept {
		ASSERT(i < nrows);
		return __data->rows[i];
	}


	/// \return the number of `T` each row is made of
	[[nodiscard]] constexpr uint32_t limbs_per_row() const noexcept {
		return __data->width;
	}

	/// clears the matrix
	/// \return
	constexpr void clear() noexcept {
		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < limbs_per_row(); ++j) {
				__data->rows[i][j] = 0;
			}
		}
	}

	/// clears the matrix
	/// \return
	constexpr void zero() noexcept {
		clear();
	}

	/// generates a fully random matrix
	constexpr void random() noexcept {
		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < limbs_per_row() - 1u; ++j) {
				__data->rows[i][j] = fastrandombytes_uint64();
			}

			__data->rows[i][limbs_per_row() - 1u] = fastrandombytes_uint64() & __data->high_bitmask;
		}
	}

	constexpr void fill(const uint32_t data) noexcept {
		ASSERT(data < 2);
		if (data == 0) {
			clear();
			return;
		}

		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < limbs_per_row() - 1u; ++j) {
				__data->rows[i][j] = T(-1ull);
			}

			__data->rows[i][limbs_per_row() - 1u] = +(-1ull) & __data->high_bitmask;
		}
	}

	/// simple additions
	/// \param out output
	/// \param in1 input
	/// \param in2 input
	constexpr static void add(FqMatrix&out,
							  const FqMatrix &in1,
							  const FqMatrix &in2) noexcept {
		mzd_add(out.__data, in1.__data, in2.__data);
	}

	/// simple subtraction
	/// \param out output
	/// \param in1 input
	/// \param in2 input
	constexpr static void sub(FqMatrix&out,
							  const FqMatrix &in1,
							  const FqMatrix &in2) noexcept {
		mzd_add(out.__data, in1.__data, in2.__data);
	}

	constexpr static FqMatrix augment(const FqMatrix &in1, const FqMatrix &in2) {
		/// TODO
		return in1;
	}

	/// direct transpose of the full matrix
	/// NOTE: no expansion is possible
	/// \param B output
	/// \param A input
	constexpr static void transpose(FqMatrix<T, ncols, nrows, q> &B,
									FqMatrix<T, nrows, ncols, q> &A) noexcept {
		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < ncols; ++j) {
				const auto data = A.get(i, j);
				B.set(data, j, i);
			}
		}
		// TODO matrix_transpose(B.__data, A.__data);
	}

	/// direct transpose of the full matrix
	/// NOTE: no expansion is possible
	/// \param B output
	/// \param A input
	template<typename Tprime, const uint32_t nrows_prime, const uint32_t ncols_prime, const uint32_t qprime>
	constexpr static void transpose(FqMatrix<Tprime, nrows_prime, ncols_prime, qprime> &B,
									FqMatrix<T, nrows, ncols, q> &A,
	                                const uint32_t srow,
	                                const uint32_t scol) noexcept {
		ASSERT(srow < nrows);
		ASSERT(scol < ncols);
		// checks must be transposed to
		ASSERT(scol < nrows_prime);
		ASSERT(srow < ncols_prime);
		ASSERT(ncols <= nrows_prime);
		ASSERT(nrows <= ncols_prime);

		for (uint32_t i = srow; i < nrows; ++i) {
			for (uint32_t j = scol; j < ncols; ++j) {
				const auto data = A.get(i, j);
				B.set(data, j, i);
			}
		}
	}

	/// NOTE this re-aligns the output to (0, 0)
	/// \param B output matrix
	/// \param A input matrix
	/// \param srow start row (inclusive, of A)
	/// \param scol start col (inclusive, of A)
	/// \param erow end row (exclusive, of A)
	/// \param ecol end col (exclusive, of A)
	template<typename Tprime, const uint32_t nrows_prime, const uint32_t ncols_prime, const uint32_t qprime>
	static constexpr void sub_matrix(FqMatrix<Tprime, nrows_prime, ncols_prime, qprime> &B,
									 const FqMatrix &A,
									 const uint32_t srow, const uint32_t scol,
									 const uint32_t erow, const uint32_t ecol) {
		ASSERT(srow < erow);
		ASSERT(scol < ecol);
		ASSERT(srow < nrows);
		ASSERT(scol < ncols);
		ASSERT(erow <= nrows);
		ASSERT(ecol <= ncols);
		ASSERT(erow-srow <= nrows_prime);
		ASSERT(ecol-scol <= ncols_prime);

		for (uint32_t row = srow; row < erow; ++row) {
			for (uint32_t col = scol; col < ecol; ++col) {
				const DataType data = A.get(row, col);
				B.set(data, row-srow, col-scol);
			}
		}
	}

	/// swap to columns
	/// \param i column 1
	/// \param j column 2
	constexpr void swap_cols(const uint16_t i, const uint16_t j) noexcept {
		mzd_col_swap(__data, i, j);
	}

	/// swap rows
	/// \param i first row
	/// \param j second row
	constexpr void swap_rows(const uint16_t i,
							 const uint16_t j) noexcept {
		mzd_row_swap(__data, i, j);
	}

	constexpr void permute_cols(FqMatrix<T, ncols, nrows, q> &AT,
								int *permutation,
								const uint32_t len) noexcept {
		uint64_t data[2];
		mzp_t *P = (mzp_t *)data;
		P->length = len;
		P->values = permutation;
		matrix_create_random_permutation(__data, AT.__data, P);
	}

	constexpr uint32_t gaus(const uint32_t stop=nrows) noexcept {
		return matrix_echelonize_partial(__data, m4ri_k, stop, __matrix_data, 0);
	}

	[[nodiscard]] constexpr uint32_t fix_gaus(int *__restrict__ permutation,
											  const uint32_t rang,
											  const uint32_t fix_col,
											  const uint32_t lookahead) noexcept {
		uint64_t data[2];
		mzp_t *P = (mzp_t *)data;
		P->length = ncols;
		P->values = permutation;
		return matrix_fix_gaus(__data, rang, fix_col, fix_col, lookahead, P);
	}

	constexpr void matrix_vector_mul(const FqMatrix<T, 1, ncols, q> &v) noexcept {
		/// TODO
	}

	/// prints the current matrix
	/// \param name postpend the name of the matrix
	/// \param binary print as binary
	/// \param compress_spaces if true, do not print spaces between the elements
	/// \param syndrome if true, print the last line as the syndrome
	constexpr void print(const std::string &name="",
	                     bool binary=true,
	                     bool transposed=false,
	                     bool compress_spaces=false,
	                     bool syndrome=false) const noexcept {
		mzd_print(__data);
	}

	constexpr bool binary() noexcept { return true; }
};

#endif//CRYPTANALYSISLIB_BINARYMATRIX_H
