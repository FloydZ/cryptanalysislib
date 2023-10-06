#ifndef CRYPTANALYSISLIB_MATRIX_H
#define CRYPTANALYSISLIB_MATRIX_H

#include <cstdint>

#include "helper.h"
#include "random.h"
#include "container/fq_packed_vector.h"
#include "container/fq_vector.h"


#if __cplusplus > 201709L
/// These are only needed for the matrix/matrix
/// and matrix/vector multiplication
/// \tparam LabelType
template<class LabelType>
concept LabelTypeAble = requires(LabelType c) {
	LabelType::LENGTH;

	requires requires(const uint32_t i) {
		c[i];
		c.get(i);
		c.set(i, i);
	};
};

/// \tparam ValueType
template<class ValueType>
concept ValueTypeAble = requires(ValueType c) {
	ValueType::LENGTH;

	requires requires(const uint32_t i) {
		c[i];
		c.get(i);
		c.set(i, i);
	};
};

/// The following functions need to be implemented by
/// all matrix types
template<class MatrixType>
concept MatrixAble = requires(MatrixType c) {
	typename MatrixType::DataType;

	requires requires(const uint32_t i,
	                  const typename MatrixType::DataType a) {
		c.get(i, i);
		c.get(i);
		c.set(a, i, i);
		c.set(a);
		c.copy(c);

		c.clear();
		c.zero();
		c.identity();
		c.fill(i);
		c.random();

		c.weight_column(i);

		MatrixType::augment(c, c);

		MatrixType::add(c, c, c);
		MatrixType::sub(c, c, c);
		// TODO allow for different types MatrixType::transpose(c, c);
		MatrixType::sub_transpose(c, c, i, i);
		MatrixType::sub_matrix(c, c, i, i, i, i);

		//TODO c.matrix_vector_mul(c);
		//TODO c.matrix_matrix_mul(c);

		c.gaus();
		//TODO c.fix_gaus(*i, i, i, i);
		//TODO c.m4ri();

		c.swap(i, i, i, i);
		c.swap_cols(i, i);
		c.swap_rows(i, i);
	};
};
#endif


/// matrix implementation which is row major
/// \tparam T base type something like `uint32_t` or `uint64_t`
/// \tparam nrows number of rows
/// \tparam ncols number of columns
/// \tparam q base field size
template<typename T,
		const uint32_t nrows,
		const uint32_t ncols,
		const uint32_t q,
		const bool packed=false>
class FqMatrix_Meta {
public:
	static constexpr uint32_t ROWS = nrows;
	static constexpr uint32_t COLS = ncols;

	// Types
	using RowType = typename std::conditional<packed,
			kAryPackedContainer_T<T, ncols, q>,
			kAryContainer_T<T, ncols, q>
	>::type;
	typedef typename RowType::DataType DataType;
	using InternalRowType = RowType;

	// Variables
	std::array<RowType, nrows> __data;

	constexpr FqMatrix_Meta() noexcept {
		clear();
	}

	/// copy constructor
	/// \param A
	constexpr FqMatrix_Meta(const FqMatrix_Meta &A) noexcept {
		clear();

		for (uint32_t row = 0; row < nrows; ++row) {
			for (uint32_t col = 0; col < limbs_per_row(); ++col) {
				__data[row].data()[col] = A.__data[row].data(col);
			}
		}
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

	/// copies the input matrix
	/// \param A input matrix
	constexpr void copy(const FqMatrix_Meta &A) {
		// __data.copy(A.__data, sizeof __data);
		memcpy(__data.data(), A.__data.data(), nrows*sizeof(RowType));
	}


	/// copy a smaller matrix into to big matrix
	template<typename Tprime, const uint32_t nrows_prime, const uint32_t ncols_prime, const uint32_t qprime, const bool packed_prime>
	constexpr void copy_sub(const FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, qprime, packed_prime> &A,
							const uint32_t srow, const uint32_t scol) {
		static_assert(nrows_prime <= nrows);
		static_assert(ncols_prime <= ncols);
		ASSERT(nrows_prime + srow <= nrows);
		ASSERT(ncols_prime + scol <= ncols);

		for (uint32_t i = 0; i < nrows_prime; i++) {
			for (uint32_t j = 0; j < ncols_prime; j++) {
				const DataType data = A.get(i, j);
				set(i + srow, j + scol, data);
			}	
		}
	}

	/// sets all entries in a matrix
	/// \param data value to set all cells to
	constexpr void set(DataType data) noexcept {
		for (uint32_t i = 0; i < ROWS; ++i) {
			for (uint32_t j = 0; j < COLS; ++j) {
				set(data, i, j);
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

	/// \param i row number (zero indexed)
	/// \return a const ref to a row
	[[nodiscard]] constexpr const RowType& get(const uint32_t i) const noexcept {
		ASSERT(i < nrows);
		return __data[i];
	}

	/// \param i row number (zero indexed)
	/// \return a mut ref to a row
	[[nodiscard]] constexpr RowType& get(const uint32_t i) noexcept {

		ASSERT(i < nrows);
		return __data[i];
	}

	/// \param i row number (zero indexed)
	/// \return a const ref to a row
	[[nodiscard]] constexpr const RowType& operator[](const uint32_t i) const noexcept {
		return get(i);
	}

	/// \param i row number (zero indexed)
	/// \return a mut ref to a row
	[[nodiscard]] constexpr RowType& operator[](const uint32_t i) noexcept {
		return get(i);
	}
	/// creates an identity matrix
	/// \return
	constexpr void identity(const DataType val=1) noexcept {
		clear();

		for (uint32_t i = 0; i < std::min(nrows, ncols); ++i) {
			set(val, i, i);
		}
	}

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

	/// fills the matrix with a symbol
	constexpr void fill(const T in) noexcept {
		for (uint32_t row = 0; row < nrows; ++row) {
			for (uint32_t col = 0; col < ncols; ++col) {
				set(in, row, col);
			}
		}
	}

	/// simple additions
	/// \param out output
	/// \param in1 input
	/// \param in2 input
	constexpr static void add(FqMatrix_Meta &out,
							  const FqMatrix_Meta &in1,
							  const FqMatrix_Meta &in2) noexcept {
		for (uint32_t i = 0; i < nrows; ++i) {
			RowType::add(out.__data[i], in1.get(i), in2.get(i));
		}
	}

	/// simple subtract
	/// \param out output
	/// \param in1 input
	/// \param in2 input
	constexpr static void sub(FqMatrix_Meta &out,
							  const FqMatrix_Meta &in1,
							  const FqMatrix_Meta &in2) noexcept {
		for (uint32_t i = 0; i < nrows; ++i) {
			RowType::sub(out.__data[i], in1.get(i), in2.get(i));
		}
	}

	/// simple scalar operations: A*in
	constexpr void scalar(const DataType in) noexcept {
		for (uint32_t i = 0; i < nrows; i++) {
			RowType::scalar(__data[i], __data[i], in);
		}
	}

	/// simple scalar operations: A*in
	constexpr static void scalar(FqMatrix_Meta &out,
								 const FqMatrix_Meta &in,
								 const DataType scalar) noexcept {
		for (uint32_t i = 0; i < nrows; i++) {
			RowType::scalar(out.__data[i], in.__data[i], scalar);
		}
	}

	/// direct transpose of the full matrix
	constexpr FqMatrix_Meta<T, ncols, nrows, q> transpose() const noexcept {
		FqMatrix_Meta<T, ncols, nrows, q> ret;
		ret.zero();
		for (uint32_t row = 0; row < nrows; ++row) {
			for (uint32_t col = 0; col < ncols; ++col) {
				const DataType data = get(row, col);
				ret.set(data, col, row);
			}
		}

		return ret;
	}


	/// direct transpose of the full matrix
	/// \param B output
	/// \param A input
	constexpr static void transpose(FqMatrix_Meta<T, ncols, nrows, q, packed> &B,
									FqMatrix_Meta<T, nrows, ncols, q, packed> &A) noexcept {
		for (uint32_t row = 0; row < nrows; ++row) {
			for (uint32_t col = 0; col < ncols; ++col) {
				const DataType data = A.get(row, col);
				B.set(data, col, row);
			}
		}
	}

	/// submatrix transpose within the full matrix. Meaning the
	/// output matrix must have at least the size of the input matrix.
	/// \param B output
	/// \param A input
	/// \param srow start row (inclusive)
	/// \param scol start column (inclusive)
	template<typename Tprime, const uint32_t nrows_prime, const uint32_t ncols_prime, const uint32_t qprime>
	constexpr static void transpose(FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, qprime, packed> &B,
									FqMatrix_Meta &A,
									const uint32_t srow,
									const uint32_t scol) noexcept {
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
	template<typename Tprime, const uint32_t nrows_prime, const uint32_t ncols_prime, const uint32_t qprime, const bool packedprime>
	constexpr static void sub_transpose(FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, qprime, packedprime> &B,
										const FqMatrix_Meta &A,
										const uint32_t srow,
										const uint32_t scol) noexcept {
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
	template<typename Tprime, const uint32_t nrows_prime, const uint32_t ncols_prime, const uint32_t qprime, const bool packedprime>
	constexpr static void sub_transpose(FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, qprime, packedprime> &B,
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

	/// NOTE this re-aligns the output to (0, 0)
	/// \param B output matrix
	/// \param A input matrix
	/// \param srow start row (inclusive, of A)
	/// \param scol start col (inclusive, of A)
	/// \param erow end row (exclusive, of A)
	/// \param ecol end col (exclusive, of A)
	template<typename Tprime, const uint32_t nrows_prime, const uint32_t ncols_prime, const uint32_t qprime, const bool packedprime>
	static constexpr void sub_matrix(FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, qprime, packedprime> &B,
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
		ASSERT(erow-srow <= nrows_prime);
		ASSERT(ecol-scol <= ncols_prime);

		for (uint32_t row = srow; row < erow; ++row) {
			for (uint32_t col = scol; col < ecol; ++col) {
				const DataType data = A.get(row, col);
				B.set(data, row-srow, col-scol);
			}
		}
	}

	/// simple gaussian elimination
	/// \param stop stop the elimination in the following row
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

				if (get(i, col) == (q-1u)) {
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
				DataType c = (q - get(i, col)) % q;
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
		uint32_t b = rang;
		for (; b < fix_col; ++b) {
			bool found = false;

			// find a column in which a one is found
			for (uint32_t col = b ; col < lookahead; ++col) {
				for (uint32_t row=b; row<b+1; row++) {
					if (get(row, col) == 1u) {
						found = true;

						std::swap(permutation[col], permutation[b]);
						swap_cols(col, b);
						if (row != b)
							swap_rows(b, row);
					}
				}
			}

			if (found) {
				// fix the column
				for (uint32_t i = 0; i < ROWS; ++i) {
					if (i == b) {
						continue;
					}

					if(get(i, b) == 0) {
						continue;
					}

					// negate
					DataType a = (q -get(i, b)) % q;
					RowType::scalar(tmp, __data[b], a);
					RowType::add(__data[i], __data[i], tmp);
				}
			} else {
				// Sorry nothing found:
				break;
			}
		}

#ifdef DEBUG
		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < b; ++j) {
				ASSERT(get(i, j) == (i==j));
			}
		}
#endif
		return b;
	}

	/// TODO currently not implemented
	/// \tparam r
	/// \param r_stop
	/// \return
	template<const uint32_t r>
	constexpr uint32_t m4ri(const uint32_t r_stop) noexcept {
		static_assert(r > 0);
		constexpr uint32_t qm1 = q-1;
		ASSERT(false);

		/// computes q**r
		constexpr auto compute_size = [](){
		  size_t tmp = qm1;
		  for (uint32_t i = 0; i < r - 1; i++) {
			  tmp *= qm1;
		  }

		  return tmp;
		};

		/// data container
		//static RowType buckets[compute_size()];

		/// computes the index within the precomputation table
		constexpr auto compute_index =
			[&](const size_t row_index, const size_t col_index) {
			  size_t ret = 0;

			  (void) row_index;
			  (void) col_index;
			  for (uint32_t i = 0; i < r; ++i) {

			  }

			  return ret;
		};

		/// init the buckets
		constexpr auto init = [&](const uint32_t start_row) {
		  ASSERT(start_row + r <= nrows);
		  for (uint32_t i = 0; i < qm1; ++i) {
			  //size_t offset = 0;

			  // simply copy each row
			  for (uint32_t j = 0; j < r; ++j) {
				  //buckets[offset] = get(start_row + j);
				  //offset += q;
			  }

			  for (uint32_t j = 0; j < r; ++j) {

			  }
		  }
		};

		return r_stop;
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

	/// TODO check the following things:
	/// 	- is transposing really faster?
	/// 	- one can skip the first transpose, if one applies the additional permutation also in `fix_gaus`
	/// choose and apply a new random permutation
	/// \param AT transposed matrix
	/// \param permutation given permutation (is overwritten)
	/// \param len length of the permutation
	constexpr void permute_cols(FqMatrix_Meta<T, ncols, nrows, q, packed> &AT,
								uint32_t *permutation,
								const uint32_t len) noexcept {
		ASSERT(ncols >= len);

		this->transpose(AT, *this, 0, 0);
		for (uint32_t i = 0; i < len; ++i) {
			uint32_t pos = fastrandombytes_uint64() % (len - i);
			ASSERT(i+pos < len);

			auto tmp = permutation[i];
			permutation[i] = permutation[i+pos];
			permutation[pos+i] = tmp;

			AT.swap_rows(i, i+pos);
		}

		FqMatrix_Meta<T, ncols, nrows, q, packed>::transpose(*this, AT, 0, 0);
	}

	/// appending the syndrome as the last column
	/// \param syndromeT syndrome colum
	/// \param col
	template<typename Tprime, const uint32_t qprime>
	constexpr void append_syndromeT(const FqMatrix_Meta<Tprime, nrows, 1, qprime, packed> &syndrome_T,
									const uint32_t col) noexcept {
		ASSERT(syndrome_T.ROWS == nrows);
		for (uint32_t i = 0; i < nrows; ++i) {
			auto data = syndrome_T.get(i, 0);
			__data[i].set(data, col);
		}
	}

	///
	/// \tparam Tprime
	/// \tparam ncols_prime
	/// \tparam qprime
	/// \param A
	/// \param B
	/// \return
	template<const uint32_t ncols_prime>
	constexpr static FqMatrix_Meta<T, nrows, ncols+ncols_prime, q, packed>
	augment(const FqMatrix_Meta<T, nrows, ncols, q, packed> &A,
			const FqMatrix_Meta<T, nrows, ncols_prime, q, packed> &B) noexcept {
		FqMatrix_Meta<T, nrows, ncols+ncols_prime, q, packed> ret;
		ret.clear();

		// copy the first matrix
		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < ncols; ++j) {
				const DataType data = A.get(i, j);
				ret.set(data, i, j);
			}
		}

		// copy the second matrix
		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < ncols_prime; ++j) {
				const DataType data = B.get(i, j);
				ret.set(data, i, j+ncols);
			}
		}

		return ret;
	}

	/// compute the hamming weight of a column
	/// \param col column
	/// \return hamming weight
	[[nodiscard]] constexpr inline uint32_t weight_column(const uint32_t col) const noexcept {
		ASSERT(col < ncols);
		uint32_t weight = 0;
		for (uint32_t i = 0; i < nrows; ++i) {
			weight += get(i, col) > 0;
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
	constexpr void matrix_vector_mul(const FqMatrix_Meta<T, 1, ncols, q, packed> &v) noexcept {
		FqMatrix_Meta tmp;
		for (uint32_t i = 0; i < nrows; ++i) {
			/// uint32_t to make sure that no overflow happens
			DataType sum = 0;
			for (uint32_t j = 0; j < ncols; ++j) {
				uint32_t a = get(i, j);
				uint32_t b = v.get(0, j);
				uint32_t c = (a*b)%q;
				sum += c;
				sum %= q;
			}

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
	constexpr void matrix_col_vector_mul(const FqMatrix_Meta<T, ncols, 1, q, packed> &v) noexcept {
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
	constexpr void matrix_vector_mul(FqMatrix_Meta<T, nrows, 1, q, packed> &out,
									 const FqMatrix_Meta<T, 1, ncols, q, packed> &v) const noexcept {
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
	/// \param v input row vector
	constexpr void matrix_row_vector_mul(FqMatrix_Meta<T, 1, nrows, q, packed> &out,
										 const FqMatrix_Meta<T, 1, ncols, q, packed> &v) const noexcept {
		for (uint32_t i = 0; i < nrows; ++i) {
			uint32_t sum = 0;
			for (uint32_t j = 0; j < ncols; ++j) {
				uint32_t a = get(i, j);
				uint32_t b = v.get(0, j);
				uint32_t c = (a*b)%q;
				sum += c;
			}

			sum = sum % q;
			out.set(sum, 0, i);
		}
	}

	/// special version of the row_vector multiplication
	/// in which the input/output are not matrices but
	/// vectors.
	/// \tparam LabelType must fulfill the following function
	///					get(i), set(i)
	/// \tparam ValueType must fulfill the following function
	/// 				get(i), set(i), ::LENGTH
	/// \param out
	/// \param in
	/// \return
	template<class LabelType, class ValueType>
#if __cplusplus > 201709L
	requires LabelTypeAble<LabelType> &&
			 ValueTypeAble<ValueType>
#endif
	constexpr void matrix_row_vector_mul2(LabelType &out, const ValueType &in) const noexcept {
		constexpr uint32_t IN_COLS = ValueType::LENGTH;
		constexpr uint32_t OUT_COLS = LabelType::LENGTH;
		static_assert(IN_COLS == COLS);
		static_assert(OUT_COLS == ROWS);

		for (uint32_t i = 0; i < nrows; ++i) {
			uint64_t sum = 0;
			for (uint32_t j = 0; j < ncols; ++j) {
				uint32_t a = get(i, j);
				uint32_t b = in.get(j);
				uint32_t c = (a*b)%q;
				sum += c;
				sum %= q;
			}

			sum = sum % q;
			out.set(sum, i);
		}
	}

	/// this is the multiplication if the input vector is in row format
	/// NOTE: in comparison to the other matrix vector multiplication
	/// 	this function write the result into a separate output vector
	/// NOTE: out = this*v
	/// \param out output column vector
	/// \param v input column vector
	constexpr void matrix_col_vector_mul(FqMatrix_Meta<T, nrows, 1u, q, packed> &out,
										 const FqMatrix_Meta<T, nrows, 1u, q, packed> v) noexcept {
		FqMatrix_Meta<T, nrows, 1u, q, packed> tmp;
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
	constexpr void matrix_matrix_mul(FqMatrix_Meta<T, nrows, ncols_prime, q, packed> &out,
									 const FqMatrix_Meta<T, ncols, ncols_prime, q, packed> in) noexcept {
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

	///
	/// \param A
	/// \param B
	/// \return true if equal
	constexpr static bool is_equal(const FqMatrix_Meta &A, const FqMatrix_Meta &B) {
		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < ncols; ++j) {
				if (A.get(i, j) != B.get(i, j))
					return false;
			}
		}

		return true;
	}

	/// prints the current matrix
	/// \param name postpend the name of the matrix
	/// \param binary print as binary
	/// \param compress_spaces if true, do not print spaces between the elements
	/// \param syndrome if true, print the last line as the syndrome
	constexpr void print(const std::string &name="",
						 bool binary=false,
						 bool transposed=false,
						 bool compress_spaces=false,
						 bool syndrome=false) const noexcept {
		constexpr uint32_t bits = constexpr_bits_log2(q);
		if (binary) {
			for (uint32_t j = 0; j < ncols; ++j) {
				for (uint32_t i = 0; i < nrows; ++i) {
					if (transposed) {
						print_binary(get(i, j), bits);
					} else {
						print_binary(get(j, i), bits);
					}

					if (not compress_spaces) {
						std::cout << " ";
					}
				}
			}
			return;

		}

		if (transposed) {
			for (uint32_t j = 0; j < ncols; ++j) {
				for (uint32_t i = 0; i < nrows; ++i) {
					std::cout << int(get(i, j));
					if (not compress_spaces) {
						std::cout << " ";
					}

				}
			}
		} else {
			for (uint32_t i = 0; i < nrows; ++i) {
				for (uint32_t j = 0; j < ncols; ++j) {
					std::cout << int(get(i, j));
					if (not compress_spaces) {
						std::cout << " ";
					}
				}

				if (syndrome) {
					// Now the syndrome
					std::cout << "  " << int(get(i, ncols)) << "\n";
				} else {
					if (i != (nrows - 1))
						std::cout << "\n";
				}
			}
		}
		std::cout << " " << name << "\n";
	}

	/// some simple functions
	constexpr bool binary() noexcept { return false; }

	/// these two functions exist, as there are maybe matrix implementations
	/// you want to wrap, which are not constant sized
	constexpr uint32_t rows() noexcept { return ROWS; }
	constexpr uint32_t cols() noexcept { return COLS; }

	/// \return the number of `T` each row is made of
	[[nodiscard]] constexpr uint32_t limbs_per_row() const noexcept {
		return RowType::internal_limbs;
	}

};


#endif//CRYPTANALYSISLIB_MATRIX_H
