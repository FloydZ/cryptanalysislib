#ifndef DECODING_MATRIX_H
#define DECODING_MATRIX_H

#include <cstdint>

#include "helper.h"
#include "random.h"
#include "container/fq_packed_vector.h"
#include "container/fq_vector.h"

/// TODO comments an alle functionen
/// matrix implementation which is row major
/// \tparam T base type something like `uint32_t` or `uint64_t`
/// \tparam nrows number of rows
/// \tparam ncols number of columns
/// \tparam q base field size
template<typename T, const uint32_t nrows, const uint32_t ncols, const uint32_t q, const bool packed=true>
class FqMatrix_Meta {
public:
	static constexpr uint32_t ROWS = nrows;
	static constexpr uint32_t COLS = ncols;

	// Types
	using RowType = typename std::conditional<packed,
	                                          kAryContainer_T<T, ncols, q>,
	                                          kAryPackedContainer_T<T, ncols, q>
	                                          >::type;
	using DataType = typename RowType::DataType;
	using InternalRowType = RowType;

	// Variables
	std::array<RowType, nrows> __data;

	/// copies the input matrix
	/// \param A input matrix
	constexpr void copy(const FqMatrix_Meta &A) {
		// __data.copy(A.__data, sizeof __data);
		memcpy(__data.data(), A.__data.data(), nrows*sizeof(RowType));
	}

	/// sets all entries in a matrix
	/// \param data value to set all cells to
	constexpr void set(DataType data) noexcept {
		for (uint32_t i = 0; i < ROWS; ++i) {
			for (uint32_t j = 0; j < COLS; ++j) {
				__data[i].set(data, j);
			}
		}
	}

	/// sets an entry in a matrix
	/// \param data value to set the cell to
	/// \param i row
	/// \param j column
	constexpr void set(DataType data, const uint32_t i, const uint32_t j) noexcept {
		ASSERT(i < ROWS && j <= COLS);
		ASSERT(data < q);
		__data[i].set(data, j);
	}

	/// gets the i-th row and j-th column
	/// \param i row
	/// \param j colum
	/// \return entry in this place
	[[nodiscard]] constexpr DataType get(const uint32_t i, const uint32_t j) const noexcept {
		ASSERT(i < nrows && j <= ncols);
		return __data[i][j];
	}

	/// \return the number of `T` each row is made of
	[[nodiscard]] constexpr uint32_t limbs_per_row() const noexcept {
		return RowType::internal_limbs;
	}

	/// copy constructor
	/// \param A
	FqMatrix_Meta(const FqMatrix_Meta &A) noexcept {
		clear();

		for (uint32_t row = 0; row < nrows; ++row) {
			for (uint32_t col = 0; col < limbs_per_row(); ++col) {
				__data[row].data()[col] = A.__data[row].data(col);
			}
		}
	}

	/// empty constructor
	constexpr FqMatrix_Meta() noexcept {
		clear();
	}

	/// constructor: read from string. The string should be of the format:
	///  "010202120..."
	/// e.g. one big string, without any `\n\0`
	/// \param data input data
	constexpr FqMatrix_Meta(const char* data, const uint32_t cols=ncols) noexcept {
		clear();

		char input[2] = {0};

		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < cols; ++j) {
				strncpy(input, data + i * cols + j, 1);
				const int a = atoi(input);

				ASSERT(a < q);

				set(DataType(a), i, j);
			}
		}
	}

	/// TODO move this into the special F3 implementation
	/// constructor: read from string. The string should be of the format:
	///  "010202120..."
	/// e.g. one big string, without any `\n\0`
	/// \param data input data
	//constexpr FqMatrix(const char* data) noexcept {
	//	clear();

	//	for (uint32_t i = 0; i < nrows; ++i) {
	//		for (uint32_t j = 0; j < ncols; ++j) {
	//			if (data[i*ncols + j] == '0') {
	//				set(DataType(0), i, j);
	//			} else if (data[i*ncols + j] == '1') {
	//				set(DataType(1), i, j);
	//			} else if (data[i*ncols + j] == '2') {
	//				set(DataType(2), i, j);
	//			} else {
	//				ASSERT(false);
	//			}
	//		}
	//	}
	//}

	/// clears the matrix
	/// \return
	constexpr void clear() noexcept {
		for (uint32_t i = 0; i < nrows; ++i) {
			__data[i].zero();
		}
	}

	/// clears the matrix
	/// \return
	constexpr void zero() noexcept {
		clear();
	}

	/// generates a fully random matrix
	constexpr void random() noexcept {
		for (uint32_t row = 0; row < nrows; ++row) {
			__data[row].random();
		}
	}

	/// direct transpose of the full matrix
	/// \param B output
	/// \param A input
	constexpr static void transpose(FqMatrix_Meta<T, ncols, nrows, q> &B,
	                      const FqMatrix_Meta<T, nrows, ncols, q> &A) noexcept {
		for (uint32_t row = 0; row < nrows; ++row) {
			for (uint32_t col = 0; col < ncols; ++col) {
				const DataType data = A.get(row, col);
				B.set(data, col, row);
			}
		}
	}

	///
	/// \tparam RowTypeB
	/// \tparam RowTypeA
	/// \param B
	/// \param A
	//template<class RowTypeB, class  RowTypeA>
	//static void transpose2(FqMatrix<RowTypeB> &B, const FqMatrix<RowTypeA> &A) {
	//	ASSERT(B.nrows == A.ncols && B.ncols == A.nrows);

	//	constexpr DataType mask = 3;
	//	for (uint32_t row = 0; row < A.nrows; ++row) {
	//		uint32_t pos = 0;
	//		for (uint32_t rowl = 0; rowl < RowTypeA::limbs(); ++rowl) {
	//			DataType data = A.__data[row].limb(rowl);

	//			const uint32_t limit = rowl == RowTypeA::limbs() - 1 ? (A.ncols%64) : 32;
	//			for (uint32_t i = 0; i < limit; ++i) {
	//				B.set(data&mask, pos+i, row);
	//				data >>= 2;
	//			}

	//			pos += 32;
	//		}
	//	}
	//}

	/// submatrix transpose within the full matrix. Meaning the
	/// output matrix must have at least the size of the input matrix.
	/// \param B output
	/// \param A input
	/// \param srow start row (inclusive)
	/// \param scol start column (inclusive)
	template<typename Tprime, const uint32_t nrows_prime, const uint32_t ncols_prime, const uint32_t qprime>
	constexpr static void transpose(FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, qprime> &B,
	                      const FqMatrix_Meta &A,
						  const uint32_t srow, const uint32_t scol) {
		ASSERT(srow < nrows);
		ASSERT(scol < ncols);
		// checks must be transposed to
		ASSERT(scol < nrows_prime);
		ASSERT(srow < ncols_prime);
		ASSERT(ncols <= nrows_prime);
		ASSERT(nrows <= ncols_prime);

		for (uint32_t row = srow; row < nrows; ++row) {
			for (uint32_t col = scol; col < ncols; ++col) {
				const DataType data = A.get(row, col);
				B.set(data, col, row);
			}
		}
	}

	/// only transpose a sub-matrix.
	/// NOTE: in contrast to the above function, is this function realigning the
	/// the output. Meaning the output is written starting from (0,0)
	/// \param B output matrix
	/// \param A input matrix
	/// \param srow start row (inclusive, of A)
	/// \param scol start col (inclusive, of A)
	template<typename Tprime, const uint32_t nrows_prime, const uint32_t ncols_prime, const uint32_t qprime>
	constexpr static void sub_transpose(FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, qprime> &B,
	                      		const FqMatrix_Meta &A,
						  		const uint32_t srow, const uint32_t scol) noexcept {
		ASSERT(srow < nrows);
		ASSERT(scol < ncols);
		// checks must be transposed to
		ASSERT(scol < nrows_prime);
		ASSERT(srow < ncols_prime);

		for (uint32_t row = srow; row < nrows; ++row) {
			for (uint32_t col = scol; col < ncols; ++col) {
				const DataType data = A.get(row, col);
				B.set(data, col- scol, row- srow);
			}
		}
	}

	///
	/// \param B output matrix
	/// \param A input matrix
	/// \param srow start row (inclusive, of A)
	/// \param scol start col (inclusive, of A)
	/// \param erow end row (exclusive, of A)
	/// \param ecol end col (exclusive, of A)
	template<typename Tprime, const uint32_t nrows_prime, const uint32_t ncols_prime, const uint32_t qprime>
	constexpr static void sub_transpose(FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, qprime> &B,
	                      const FqMatrix_Meta &A,
						  const uint32_t srow, const uint32_t scol,
						  const uint32_t erow, const uint32_t ecol) noexcept {
		ASSERT(srow < erow);
		ASSERT(scol < ecol);
		ASSERT(srow < nrows);
		ASSERT(scol < ncols);
		ASSERT(erow <= nrows);
		ASSERT(ecol <= ncols);
		// checks must be transposed to
		ASSERT(scol < nrows_prime);
		ASSERT(srow < ncols_prime);
		ASSERT(ecol-scol <= nrows_prime);
		ASSERT(erow-srow <= ncols_prime);

		for (uint32_t row = srow; row < erow; ++row) {
			for (uint32_t col = scol; col < ecol; ++col) {
				const DataType data = A.get(row, col);
				B.set(data, col-scol, row-srow);
			}
		}
	}

	/// \param B output matrix
	/// \param A input matrix
	/// \param srow start row (inclusive, of A)
	/// \param scol start col (inclusive, of A)
	/// \param erow end row (exclusive, of A)
	/// \param ecol end col (exclusive, of A)
	template<typename Tprime, const uint32_t nrows_prime, const uint32_t ncols_prime, const uint32_t qprime>
	static constexpr void sub_matrix(FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, qprime> &B,
	                                 const FqMatrix_Meta &A,
						  			 const uint32_t srow, const uint32_t scol,
						  			 const uint32_t erow, const uint32_t ecol)
	        noexcept {
		ASSERT(srow < erow);
		ASSERT(scol < ecol);
		ASSERT(srow < nrows);
		ASSERT(scol < ncols);
		ASSERT(erow <= nrows);
		ASSERT(ecol <= ncols);
		ASSERT(srow < nrows_prime);
		ASSERT(scol < ncols_prime);

		for (uint32_t row = srow; row < erow; ++row) {
			for (uint32_t col = scol; col < ecol; ++col) {
				const DataType data = A.get(row, col);
				B.set(data, row-srow, col-scol);
			}
		}
	}

	/// simple gaussian elimination
	/// \param stop
	/// \return
	[[nodiscard]] constexpr uint32_t gaus(const uint32_t stop = -1) noexcept {
		const std::size_t  m = ncols -1;
		alignas(32) RowType tmp;
		uint32_t row = 0;
		for(uint32_t col = 0; (col < m) && (row < nrows) && (row < stop); col++) {
			int sel = -1;
			DataType inv = get(row, col);
			inv = (0u-inv+q) % q;
			// get pivot element
			for (uint32_t i = row; i < nrows; i++) {
				if (get(i, col) == 1u) {
					sel = i;
					break;
				}

				if (get(i, col) == q) {
					sel = i;
					// neg the row
					__data[i].neg();
					break;
				}
			}

			/// early exit
			if (sel == -1)
				return row;

			///move up
			swap_rows(sel, row);

			/// solve all remaining coordinates (iterate over all rows)
			for(uint32_t i = 0; i < nrows; i++) {
				if (i == row) {
					continue;
				}
				if (get(i, col) == 0u) {
					continue;
				}

				// negate
				DataType c = (q - unsigned (get(i, col))) % q;
				RowType::scalar(tmp, __data[row], c);
				RowType::add(__data[i], __data[i], tmp);
			}

			row++;
		}

		return row;
	}

	/// \param permutation	currently column permutation of the input matrix. Its needed because we might rearrange
	///						columns to further execute the gaussian elimination
	/// \param rang			current rang of the matrix.
	/// \param fix_col		up to which rang should the matrix be solved?
	/// \param look_ahead   how many coordinated is the algorithm allowed to look ahead.
	/// \return the new rang of the matrix.
	[[nodiscard]] constexpr uint32_t fix_gaus(uint32_t *permutation,
	                                          const uint32_t rang,
	                                          const uint32_t fix_col,
	                                          const uint32_t lookahead) noexcept {
		RowType tmp;
		for (uint32_t b = rang; b < fix_col; ++b) {
			bool found = false;
			// find a column in which a one is found
			for (uint32_t i = b+1 ; i < lookahead; ++i) {
				if (get(b, i) == 1u) {
					found = true;

					std::swap(permutation[i], permutation[b]);
					swap_cols(i, b);
				}
			}

			if (found) {
				// fix the column
				for (uint32_t i = 0; i < ROWS; ++i) {
					if (i == b) {
						continue;
					}

					// negate
					DataType a = (q -get(i, b)) % q;
					RowType::scalar(tmp, __data[b], a);
					RowType::add(__data[i], __data[i], tmp);
				}
			} else {
				// Sorry nothing found:
				return rang;
			}
		}

		return ROWS;
	}

	/// swap to elements within the matrix
	/// \param i1 row of the first element
	/// \param j1 column of the first element
	/// \param i2 row of the second element
	/// \param j2 column of the second element
	constexpr void swap(const uint16_t i1,
	                    const uint16_t j1,
	                    const uint16_t i2,
	                    const uint16_t j2) noexcept {
		uint32_t tmp = get(i1, j1);
		set(get(i2, j2), i1, i2);
		set(tmp, i2, j2);
	}

	/// swap to columns
	/// \param i column 1
	/// \param j column 2
	constexpr void swap_cols(const uint16_t i, const uint16_t j) noexcept {
		ASSERT(i < ncols);
		ASSERT(j < ncols);

		/// early exit
		if (i == j)
			return;

		RowType tmp;
		tmp.zero();
		for (uint32_t row = 0; row < nrows; ++row)
			tmp.set(__data[row].get(i), row);

		for (uint32_t row = 0; row < nrows; ++row)
			__data[row].set(__data[row].get(j), i);

		for (uint32_t row = 0; row < nrows; ++row)
			__data[row].set(tmp.get(row), j);
	}

	/// swap rows
	/// \param i first row
	/// \param j second row
	constexpr void swap_rows(const uint16_t i,
	                         const uint16_t j) noexcept {
		ASSERT(i < nrows && j < nrows);
		if (i == j)
			return;

		RowType tmp = __data[i];
		__data[i] = __data[j];
		__data[j] = tmp;
	}

	/// choose and apply a new random permutation
	/// \param AT transposed matrix
	/// \param permutation given permutation (is overwritten)
	/// \param len length of the permutation
	template<typename Tprime, const uint32_t nrows_prime, const uint32_t ncols_prime, const uint32_t qprime>
	constexpr void permute_cols(FqMatrix_Meta<Tprime, nrows, ncols, qprime> &AT,
	                            uint32_t *permutation,
	                            const uint32_t len) noexcept {
		ASSERT(ncols >= len);

		transpose(AT, *this);
		for (uint32_t i = 0; i < len; ++i) {
			uint32_t pos = fastrandombytes_uint64() % (len - i);
			ASSERT(i+pos < len);

			auto tmp = permutation[i];
			permutation[i] = permutation[i+pos];
			permutation[pos+i] = tmp;

			AT.swap_rows(i, i+pos);
		}

		transpose(*this, AT);
	}

	/// appending the syndrome as the last column
	/// \param syndromeT syndrome colum
	/// \param col
	template<typename Tprime, const uint32_t ncols_prime, const uint32_t qprime>
	void append_syndromeT(const FqMatrix_Meta<Tprime, 1, ncols_prime, qprime> &syndrome_T,
	                      const uint32_t col) noexcept {
		ASSERT(syndrome_T.ncols == nrows);
		for (uint32_t i = 0; i < nrows; ++i) {
			set(syndrome_T.get(0, i), i, col);
		}
	}

	/// compute the weight of a column
	/// \param col column
	/// \return hamming weight
	[[nodiscard]] constexpr inline uint32_t weight_column(const uint32_t col) const noexcept {
		ASSERT(col < ncols);
		uint32_t weight = 0;
		for (uint32_t i = 0; i < nrows; ++i) {
			weight += __data[i][col];
		}

		return weight;
	}

	/// this is the multiplication if the input vector is in row format
	/// NOTE: this function is inplace
	/// NOTE: this = this*v (result is written into the first row)
	/// NOTE: technically that's not correct as the output is a row vector
	/// 		whereas the container is a full matrix
	/// this = this*v
	/// \param v vector
	constexpr void matrix_vector_mul(const FqMatrix_Meta<T, 1, ncols, q> &v) noexcept {
		FqMatrix_Meta tmp;
		for (uint32_t i = 0; i < nrows; ++i) {
			/// uint32_t to make sure that no overflow happens
			uint32_t sum = 0;
			for (uint32_t j = 0; j < ncols; ++j) {
				uint32_t a = get(i, j);
				uint32_t b = v.get(0, j);
				uint32_t c = (a*b)%q;
				sum += c;
			}

			sum = sum % q;
			tmp.set(sum, 0, i);
		}

		this->copy(tmp);
	}

	/// this is the multiplication if the input vector is in col format
	/// NOTE: this function is inplace
	/// NOTE: this = this*v (result is written into the first colum)
	/// NOTE: technically that's not correct as the output is a col vector
	/// 		whereas the container is a full matrix
	/// \param v vector
	constexpr void matrix_col_vector_mul(const FqMatrix_Meta<T, ncols, 1, q> &v) noexcept {
		FqMatrix_Meta tmp;
		for (uint32_t i = 0; i < nrows; ++i) {
			/// uint32_t to make sure that no overflow happens
			uint32_t sum = 0;
			for (uint32_t j = 0; j < ncols; ++j) {
				uint32_t a = get(i, j);
				uint32_t b = v.get(j, 0);
				uint32_t c = (a*b)%q;
				sum += c;
			}

			sum = sum % q;
			tmp.set(sum, i, 0);
		}

		this->copy(tmp);
	}

	/// this is the multiplication if the input vector is in row format
	/// NOTE: in comparison to the other matrix vector multiplication
	/// 	this function write the result into a separate output vector
	/// NOTE: out = this*v
	/// \param out output column vector
	/// \param v input row vector
	constexpr void matrix_vector_mul(FqMatrix_Meta<T, nrows, 1, q> &out,
	                       const FqMatrix_Meta<T, 1, ncols, q> &v) noexcept {
		for (uint32_t i = 0; i < nrows; ++i) {
			uint32_t sum = 0;
			for (uint32_t j = 0; j < ncols; ++j) {
				uint32_t a = get(i, j);
				uint32_t b = v.get(0, j);
				uint32_t c = (a*b)%q;
				sum += c;
			}

			sum = sum % q;
			out.set(sum, i, 0);
		}
	}

	/// this is the multiplication if the input vector is in row format
	/// NOTE: in comparison to the other matrix vector multiplication
	/// 	this function write the result into a separate output vector
	/// NOTE: out = this*v
	/// \param out output column vector
	/// \param v input column vector
	constexpr void matrix_col_vector_mul(FqMatrix_Meta<T, nrows, 1u, q> &out,
	                           const FqMatrix_Meta<T, nrows, 1u, q> v) noexcept {
		FqMatrix_Meta<T, nrows, 1u, q> tmp;
		for (uint32_t i = 0; i < nrows; ++i) {
			uint32_t sum = 0;
			uint32_t b = v.get(i, 0);
			for (uint32_t j = 0; j < ncols; ++j) {
				uint32_t a = get(i, j);
				uint32_t c = (a*b)%q;
				sum += c;
			}

			sum = sum % q;
			tmp.set(sum, i, 0);
		}

		out.copy(tmp);
	}


	/// out = this*in
	/// \param out output matrix of size [nrows, ncols']
	/// \param v input matrix of size [ncols, ncols']
	template<const uint32_t ncols_prime>
	constexpr void matrix_matrix_mul(FqMatrix_Meta<T, nrows, ncols_prime, q> &out,
	                       			 const FqMatrix_Meta<T, ncols, ncols_prime, q> in) noexcept {
		// over all columns in `in`
		for (uint32_t i = 0; i < ncols_prime; ++i) {
			// for each row in *this
			for (uint32_t j = 0; j < nrows; ++j) {
				uint32_t sum = 0;
				// for each element in the row
				for (uint32_t k = 0; k < ncols; ++k) {
					uint32_t a = get(j, k);
					uint32_t b = in.get(k, i);
					uint32_t c = (a*b)%q;
					sum += c;
				}
				uint32_t tmp = out.get(j, i);
				tmp = (tmp + sum) % q;
				out.set(sum, j, i);
			}
		}
	}

	/// prints the current matrix
	/// \param name postpend the name of the matrix
	/// \param binary print as binary
	/// \param compress_spaces if true, do not print spaces between the elements
	/// \param syndrome if true, print the last line as the syndrome
	constexpr void print(const std::string &name="",
	                     bool binary=false,
	                     bool compress_spaces=false,
	                     bool syndrome=false) const noexcept {
		constexpr uint32_t bits = constexpr_bits_log2(q);
		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < ncols; ++j) {
				//printf("%0*d ", bits, int(get(i, j)));
				std::cout << int(get(i, j));
				if (not compress_spaces) {
					std::cout << " ";
				}

			}

			if (syndrome) {
				// Now the syndrome
				std::cout << "  " << int(get(i, ncols)) << "\n";
			} else {
				if (i != (nrows-1))
					std::cout << "\n";
			}
		}
		std::cout << " " << name << "\n";
	}
};

/// matrix implementation which is row major
/// \tparam T base type something like `uint32_t` or `uint64_t`
/// \tparam nrows number of rows
/// \tparam ncols number of columns
template<typename T, const uint32_t nrows, const uint32_t ncols, const uint32_t q>
class FqMatrix: public  FqMatrix_Meta<T, nrows, ncols, q>{
public:
};

/// matrix implementation which is row major
/// NOTE special case for q=3
/// \tparam T base type something like `uint32_t` or `uint64_t`
/// \tparam nrows number of rows
/// \tparam ncols number of columns
template<typename T, const uint32_t nrows, const uint32_t ncols>
class FqMatrix<T, nrows, ncols, 3> : public  FqMatrix_Meta<T, nrows, ncols, 3> {

	/// this is just defined, because Im lazy
	static constexpr uint32_t q = 3;

	/// needed type definitions
	using typename FqMatrix_Meta<T, nrows,ncols, q>::RowType;

	/// needed vars
	using FqMatrix_Meta<T, nrows,ncols, q>::__data;
	using FqMatrix_Meta<T, nrows,ncols, q>::ROWS;
	using FqMatrix_Meta<T, nrows,ncols, q>::COLS;

	/// needed functions
	using FqMatrix_Meta<T, nrows,ncols, q>::get;
	using FqMatrix_Meta<T, nrows,ncols, q>::set;
	using FqMatrix_Meta<T, nrows,ncols, q>::swap_rows;
	using FqMatrix_Meta<T, nrows,ncols, q>::swap_cols;
public:

	///
	/// \param stop row to stop with the elimination
	/// \return number of rows systemize
	constexpr uint32_t gaus(const uint32_t stop = -1) noexcept {
		const std::size_t n = nrows, m = ncols -1;
		alignas(32) RowType tmp;
		uint32_t row = 0;
		for(uint32_t col = 0; (col < m) && (row < n) && (row < stop); col++) {
			int sel = -1;
			// get pivot element
			for (uint32_t i = row; i < n; i++) {
				if (get(i, col) == 1) {
					sel = i;
					break;
				}

				if (get(i, col) == 2) {
					sel = i;
					// times two the whole row
					for (uint32_t j = 0; j <= m; ++j) {
						set((2*get(i, j))%3, i, j);
					}
					break;
				}
			}

			if (sel == -1)
				return row;

			swap_rows(sel, row);

			for(uint32_t i = 0; i < n; i++) {
				if (i != row) {
					if (get(i, col) == 0) {
						continue;
					}

					uint32_t c = get(i, col);
					if (c == 0) {
						continue;
					} else if (c == 1) {
						RowType::sub(__data[i], __data[i], __data[row]);
					} else {
						/// TODO add should be also goot
						RowType::times2_mod3(tmp, __data[row]);
						RowType::sub(__data[i], __data[i], tmp);
					}
				}
			}

			row++;
		}

		return row;
	}

	/// \param permutation	currently column permutation of the input matrix. Its needed because we might rearrange
	///						columns to further execute the gaussian elimination
	/// \param rang			current rang of the matrix.
	/// \param fix_col		up to which rang should the matrix be solved?
	/// \param look_ahead   how many coordinated is the algorithm allowed to look ahead.
	/// \return the new rang of the matrix.
	uint32_t fix_gaus(uint32_t *permutation,
	                  const uint32_t rang,
	                  const uint32_t fix_col,
	                  const uint32_t lookahead) noexcept {
		for (uint32_t b = rang; b < fix_col; ++b) {
			bool found = false;
			// find a column in which a one is found
			for (uint32_t i = b+1 ; i < lookahead; ++i) {
				if (get(b, i) == 1u) {
					found = true;

					std::swap(permutation[i], permutation[b]);
					swap_cols(i, b);
				}
			}
			if (found) {
				// fix the column
				for (uint32_t i = 0; i < b; ++i) {
					uint32_t a = get(i, b);
					if (a > 0) {
						RowType::add(__data[i], __data[i], __data[b]);
					}
					if (a == 2) {
						RowType::add(__data[i], __data[i], __data[b]);
					}
				}
			} else {
				// Sorry nothing found:
				return rang;
			}
		}

		return ROWS;
	}

};
#endif//DECODING_MATRIX_H
