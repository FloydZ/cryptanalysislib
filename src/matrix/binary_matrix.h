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

	/// simple deconstructor
	constexpr ~FqMatrix() noexcept {
		mzd_free(__data);
		free_matrix_data(__matrix_data);
	}

	/// copy constructor
	constexpr FqMatrix(const FqMatrix &A) noexcept {
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
	constexpr void set(const DataType data, const uint32_t i, const uint32_t j) noexcept {
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

	/// direct transpose of the full matrix
	/// \param B output
	/// \param A input
	constexpr static void transpose(FqMatrix_Meta<T, ncols, nrows, q> &B,
									FqMatrix_Meta<T, nrows, ncols, q> &A) noexcept {
		/// TODO
	}

	/// swap to columns
	/// \param i column 1
	/// \param j column 2
	constexpr void swap_cols(const uint16_t i, const uint16_t j) noexcept {
		/// TODO
	}

	/// swap rows
	/// \param i first row
	/// \param j second row
	constexpr void swap_rows(const uint16_t i,
							 const uint16_t j) noexcept {
		/// TODO
	}

	constexpr void permute_cols(FqMatrix_Meta<T, ncols, nrows, q> &AT,
								uint32_t *permutation,
								const uint32_t len) noexcept {
		/// TODO
	}

	constexpr void matrix_vector_mul(const FqMatrix_Meta<T, 1, ncols, q> &v) noexcept {
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
};

#endif//CRYPTANALYSISLIB_BINARYMATRIX_H
