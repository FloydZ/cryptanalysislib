#ifndef CRYPTANALYSISLIB_MATRIX_H
#define CRYPTANALYSISLIB_MATRIX_H

#include <cstdint>

#include "helper.h"
#include "container/fq_packed_vector.h"
#include "container/fq_vector.h"
#include "permutation/permutation.h"
#include "random.h"
#include "reflection/reflection.h"


#if __cplusplus > 201709L
/// These are only needed for the matrix/matrix
/// and matrix/vector multiplication
/// \tparam LabelType
template<class LabelType>
concept LabelTypeAble = requires(LabelType c) {
	LabelType::length();
	LabelType::info();

	requires requires(const uint32_t i) {
		c[i];
		c.get(i);
		c.set(i, i);
	};
};

/// \tparam ValueType
template<class ValueType>
concept ValueTypeAble = requires(ValueType c) {
	ValueType::length();
	ValueType::info();

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
	MatrixType::info();

	requires requires(const uint32_t i,
	                  const uint32_t *ii,
	                  MatrixType &m,
	                  const MatrixType &cm,
	                  Permutation &P,
	                  const typename MatrixType::DataType a) {
		c.get(i, i);
		c.get(i);
		c.set(a, i, i);
		c.set(a);
		c.copy(c);

		c.rows();
		c.cols();

		c.clear();
		c.zero();
		c.identity();
		c.fill(i);
		c.random();

		c.column_popcnt(i);
		c.row_popcnt(i);

		MatrixType::add(c, c, c);
		MatrixType::sub(c, c, c);
		MatrixType::sub_transpose(c, c, i, i);
		MatrixType::sub_matrix(c, c, i, i, i, i);

		// the problem is that c is of a different type
		// MatrixType::transpose(c, c);
		// MatrixType::mul(m, m, m);

		c.gaus();
		c.fix_gaus(P, i, i);
		c.m4ri();

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
/// \tparam packed if true the rowtype to `kAryPackedContainer`
/// \tparam R helper type to overwrite the rowtype. Overwrites packed if != void
template<typename T,
         const uint32_t _nrows,
         const uint32_t _ncols,
         const uint32_t q,
         const bool packed = false,
         typename R=void>
struct FqMatrix_Meta {
public:
	static constexpr uint32_t nrows = _nrows;
	static constexpr uint32_t ncols = _ncols;
	static constexpr uint32_t ROWS = nrows;
	static constexpr uint32_t COLS = ncols;

	// Types
	using __RowType = typename std::conditional<packed,
	                                            kAryPackedContainer_T<T, ncols, q>,
	                                            kAryContainer_T<T, ncols, q>>::type;
	// typedef __RowType RowType;
	using RowType = typename std::conditional<std::is_same_v<R, void>, __RowType, R>::type;
	typedef typename RowType::DataType DataType;
	using InternalRowType = RowType;
	// define itself
	using MatrixType = FqMatrix_Meta<T, nrows, ncols, q, packed, R>;
	using S = FqMatrix_Meta<T, nrows, ncols, q, packed, R>;

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
			for (uint32_t col = 0; col < limbs(); ++col) {
				const auto d = A.get(row, col);
				set(d, row, col);
			}
		}
	}

	/// constructor: read from string. The string should be of the format:
	///  "010202120..."
	/// e.g. one big string, without any `\n\0`
	/// \param data input data
	constexpr FqMatrix_Meta(const char *data,
	                        const uint32_t cols = ncols) noexcept {
		from_string(data, cols);
	}

	///
	/// \param data
	/// \param cols
	/// \return
	constexpr void from_string(const char *data,
	                           const uint32_t cols = ncols) noexcept {
		clear();

		/// TODO generalize over field
		char input[2] = {0};
		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < cols; ++j) {
				strncpy(input, data + i * cols + j, 1);
				const int a = atoi(input);

				ASSERT(a >= 0);
				const uint32_t aa = a;
				ASSERT(aa < q);

				set(DataType(aa), i, j);
			}
		}
	}

	/// copies the input matrix
	/// \param A input matrix
	constexpr void copy(const FqMatrix_Meta &A) {
		std::copy(A.__data.begin(), A.__data.end(), __data.begin());
	}


	/// copy a smaller matrix into to big matrix
	template<typename Tprime,
	         const uint32_t nrows_prime,
	         const uint32_t ncols_prime,
	         const uint32_t qprime,
	         const bool packed_prime>
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

	/// generates a random row with exactly weigh w
	constexpr inline void random_row_with_weight(const uint32_t row,
	                                             const uint32_t w) {
		zero_row(row);

		for (uint64_t i = 0; i < w; ++i) {
			const uint64_t data = fastrandombytes_uint64();
			set(1 + (data % (q-1)), row, i);
		}

		// now permute
		for (uint64_t i = 0; i < ncols; ++i) {
			uint64_t pos = fastrandombytes_uint64() % (ncols - i);
			auto t = get(row, i);
			set(get(row, i + pos), row, i);
			set(t, row, i + pos);
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
	constexpr inline void set(DataType data,
	                          const uint32_t i,
	                          const uint32_t j) noexcept {
		ASSERT(i < ROWS && j <= COLS);
		ASSERT((uint32_t) data < q);
		__data[i].set(data, j);
	}

	/// gets the i-th row and j-th column
	/// \param i row
	/// \param j colum
	/// \return entry in this place
	[[nodiscard]] constexpr inline DataType get(const uint32_t i,
	                                            const uint32_t j) const noexcept {
		ASSERT(i < nrows && j <= ncols);
		return __data[i][j];
	}

	/// \param i row number (zero indexed)
	/// \return a const ref to a row
	[[nodiscard]] constexpr inline const RowType &get(const uint32_t i) const noexcept {
		ASSERT(i < nrows);
		return __data[i];
	}

	/// \param i row number (zero indexed)
	/// \return a mut ref to a row
	[[nodiscard]] constexpr inline RowType &get(const uint32_t i) noexcept {

		ASSERT(i < nrows);
		return __data[i];
	}

	/// \param i row number (zero indexed)
	/// \return a const ref to a row
	[[nodiscard]] constexpr inline const RowType &operator[](const uint32_t i) const noexcept {
		return get(i);
	}

	/// \param i row number (zero indexed)
	/// \return a mut ref to a row
	[[nodiscard]] constexpr inline RowType &operator[](const uint32_t i) noexcept {
		return get(i);
	}

	/// creates an identity matrix
	constexpr void identity(const DataType val = 1) noexcept {
		clear();

		for (uint32_t i = 0; i < std::min(nrows, ncols); ++i) {
			set(val, i, i);
		}
	}

	/// clears the matrix
	constexpr void clear() noexcept {
		for (uint32_t i = 0; i < nrows; ++i) {
			__data[i].zero();
		}
	}

	/// clears the matrix
	constexpr inline void zero() noexcept {
		clear();
	}


	constexpr void zero_row(const uint32_t row) noexcept {
		ASSERT(row < nrows);
		for (uint32_t i = 0; i < ncols; ++i) {
			set(0, row, i);
		}
	}

	/// generates a fully random matrix with full rank
	constexpr void random() noexcept {
		clear();
		for (uint32_t row = 0; row < nrows; ++row) {
			__data[row].random();
		}

		for (uint32_t i = 0; i < std::min(nrows, ncols); i++) {
			const auto d = 1 + fastrandombytes_uint64(q-1u);
			set(d, i, i);
		}

		for (uint32_t row = 0; row < std::min(ncols, nrows); ++row) {
			for (uint32_t row2 = 0; row2 < std::min(nrows, ncols); ++row2) {
				if (row == row2) { continue; }
				if ((fastrandombytes_uint64() & 1u) == 1u) {
					RowType::add(__data[row2], __data[row2], __data[row]);
				} else {
					RowType::add(__data[row], __data[row2], __data[row]);
				}
			}
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

	/// simple scalar operations: A*scalar
	constexpr void scalar(const DataType &scalar) noexcept {
		for (uint32_t i = 0; i < nrows; i++) {
			RowType::scalar(__data[i], __data[i], scalar);
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
	constexpr FqMatrix_Meta<T, ncols, nrows, q, packed> transpose() const noexcept {
		FqMatrix_Meta<T, ncols, nrows, q, packed> ret;
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
	constexpr static void transpose(FqMatrix_Meta<T, ncols, nrows, q, packed, R> &B,
	                                const FqMatrix_Meta<T, nrows, ncols, q, packed, R> &A) noexcept {
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
	template<typename Tprime,
	         const uint32_t nrows_prime,
	         const uint32_t ncols_prime,
	         const uint32_t qprime>
	constexpr static void transpose(FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, qprime, packed> &B,
	                                FqMatrix_Meta &A, const uint32_t srow, const uint32_t scol) noexcept {
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
	template<typename Tprime,
	         const uint32_t nrows_prime,
	         const uint32_t ncols_prime,
	         const uint32_t qprime,
	         const bool packedprime,
	         typename Rprime>
	constexpr static void sub_transpose(FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, qprime, packedprime, Rprime> &B,
	                                    const FqMatrix_Meta &A,
	                                    const uint32_t srow,
	                                    const uint32_t scol) noexcept {
		static_assert(std::is_same_v<R, Rprime>);
		ASSERT(srow < nrows);
		ASSERT(scol < ncols);
		// checks must be transposed to
		ASSERT(scol < nrows_prime);
		ASSERT(srow < ncols_prime);

		for (uint32_t row = srow; row < nrows; ++row) {
			for (uint32_t col = scol; col < ncols; ++col) {
				const DataType data = A.get(row, col);
				B.set(data, col - scol, row - srow);
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
	template<typename Tprime,
	         const uint32_t nrows_prime,
	         const uint32_t ncols_prime,
	         const uint32_t qprime,
	         const bool packedprime,
	         typename Rprime>
	constexpr static void sub_transpose(FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, qprime, packedprime, Rprime> &B,
	                                    const FqMatrix_Meta &A,
	                                    const uint32_t srow, const uint32_t scol,
	                                    const uint32_t erow, const uint32_t ecol) noexcept {
		static_assert(std::is_same_v<R, Rprime>);
		ASSERT(srow < erow);
		ASSERT(scol < ecol);
		ASSERT(srow < nrows);
		ASSERT(scol < ncols);
		ASSERT(erow <= nrows);
		ASSERT(ecol <= ncols);
		// checks must be transposed to
		ASSERT(scol < nrows_prime);
		ASSERT(srow < ncols_prime);
		ASSERT(ecol - scol <= nrows_prime);
		ASSERT(erow - srow <= ncols_prime);

		for (uint32_t row = srow; row < erow; ++row) {
			for (uint32_t col = scol; col < ecol; ++col) {
				const DataType data = A.get(row, col);
				B.set(data, col - scol, row - srow);
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
	template<typename Tprime,
	         const uint32_t nrows_prime,
	         const uint32_t ncols_prime,
	         const uint32_t qprime,
	         const bool packedprime,
	         typename Rprime>
	static constexpr void sub_matrix(FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, qprime, packedprime, Rprime> &B,
	                                 const FqMatrix_Meta &A,
	                                 const uint32_t srow, const uint32_t scol,
	                                 const uint32_t erow, const uint32_t ecol) noexcept {
		static_assert(std::is_same_v<R, Rprime>);
		ASSERT(srow < erow);
		ASSERT(scol < ecol);
		ASSERT(srow < nrows);
		ASSERT(scol < ncols);
		ASSERT(erow <= nrows);
		ASSERT(ecol <= ncols);
		ASSERT(erow - srow <= nrows_prime);
		ASSERT(ecol - scol <= ncols_prime);

		for (uint32_t row = srow; row < erow; ++row) {
			for (uint32_t col = scol; col < ecol; ++col) {
				const DataType data = A.get(row, col);
				B.set(data, row - srow, col - scol);
			}
		}
	}

	/// simple gaussian elimination
	/// \param stop stop the elimination in the following row
	/// \return
	[[nodiscard]] constexpr uint32_t gaus(const uint32_t stop = -1) noexcept {
		const std::size_t m = ncols - 1;
		alignas(32) RowType tmp;
		uint32_t row = 0;
		for (uint32_t col = 0; (col < m) && (row < nrows) && (row < stop); col++) {
			int sel = -1;
			// get pivot element
			for (uint32_t i = row; i < nrows; i++) {
				if (get(i, col) == 1u) {
					sel = i;
					break;
				}

				if (get(i, col) == (q - 1u)) {
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
			for (uint32_t i = 0; i < nrows; i++) {
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
	[[nodiscard]] constexpr uint32_t fix_gaus(Permutation &P,
	                                          const uint32_t rang,
	                                          const uint32_t fix_col) noexcept {
		RowType tmp;
		uint32_t b = rang;
		for (; b < fix_col; ++b) {
			bool found = false;

			// find a column in which a one is found
			// -1 because of the syndrome
			for (uint32_t col = b; col < ncols - 1; ++col) {
				for (uint32_t row = b; row < b + 1; row++) {
					if (get(row, col) == 1u) {
						found = true;

						std::swap(P.values[col], P.values[b]);
						swap_cols(col, b);
						if (row != b)
							swap_rows(b, row);

						break;
					}
				}
			}

			if (found) {
				// fix the column
				for (uint32_t i = 0; i < ROWS; ++i) {
					if (i == b) {
						continue;
					}

					if (get(i, b) == 0) {
						continue;
					}

					// negate
					DataType a = (q - get(i, b)) % q;
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
				ASSERT(get(i, j) == (i == j));
			}
		}
#endif
		return b;
	}

	/// \tparam r
	/// \param r_stop
	/// \return
	template<const uint32_t r = 4>
	constexpr uint32_t m4ri(const uint32_t rstop = nrows) noexcept {
		static_assert(r > 0);

		/// computes q**r
		constexpr auto compute_size = []() {
			size_t ret = 0;
			size_t multiplier = q;
			for (uint32_t i = 0; i < r; i++) {
				ret += multiplier;
				multiplier *= q;
			}

			return ret;
		};

		/// data container
		constexpr uint32_t precompute_size = compute_size();
		RowType table[precompute_size];
		table[0].clear();
		std::array<bool, precompute_size> computed;

		/// computes the index within the pre computation table
		/// \param row_index: left start row
		/// \param col_index: left start column
		auto compute_index =
		        [&](const size_t row_index, const size_t col_index, const uint32_t kk) {
			        size_t ret = 0;
			        size_t multiplier = 1;
			        for (uint32_t i = 0; i < kk; ++i) {
				        DataType a = get(row_index, col_index + i);
				        ret += multiplier * ((q - a) % q);
				        multiplier *= q;
			        }

			        ASSERT(ret < precompute_size);
			        return ret;
		        };

		/// \param pos position with in the `Table`. computed by `compute_index`
		/// \param row top left point to start the systematization
		/// \param col top left point to start the systematization
		/// \param start_row = top left point which is already a unity matrix on a kk x kk square
		auto compute_table_entry =
		        [&](const size_t pos, const uint32_t row, const uint32_t col,
		            const uint32_t start_row, const uint32_t kk) {
			        alignas(32) RowType tmp;

			        // the fist row must be unrolled to overwrite the previous entry
			        DataType a = get(row, col);
			        a = (q - a % q);
			        RowType::scalar(table[pos], __data[start_row], a);
			        for (uint32_t i = 1; i < kk; ++i) {
				        DataType a = get(row, col + i);
				        if (a == 0) { continue; }
				        a = (q - a % q);
				        RowType::scalar(tmp, __data[start_row + i], a);
				        RowType::add(table[pos], table[pos], tmp);
			        }

			        computed[pos] = true;
		        };


		/// \param row top left point of the systematized sub-matrix
		/// \param col top left point of the systematized sub-matrix
		/// \param kk size of the systemized submatrix
		/// \param rstart
		auto process_rows =
		        [&](const uint32_t row, const uint32_t col, const uint32_t kk, const uint32_t rstart, const uint32_t rstop) {
			        ASSERT(rstart <= rstop);
			        for (uint32_t i = rstart; i < rstop; i++) {
				        size_t pos = compute_index(i, col, kk);
				        if (!computed[pos]) {
					        compute_table_entry(pos, i, col, row, kk);
				        }

				        RowType::add(__data[i], __data[i], table[pos]);
			        }
		        };

		/// computes the gaus on a kk x kk square starting from (row, col)
		/// NOTE: kk = r normally
		auto sub_gaus =
		        [&](const uint32_t row, const uint32_t col, const uint32_t kk) {
			        alignas(32) RowType tmp;
			        for (uint32_t i = row; i < row + kk; ++i) {
			        retry:

				        uint32_t sel = -1u;
				        const uint32_t current_col = col + i - row;

				        /// pivoting
				        for (uint32_t pivot_row = i; pivot_row < nrows; pivot_row++) {
					        if (get(pivot_row, current_col) == 1u) {
						        sel = pivot_row;
						        break;
					        }

					        if (get(pivot_row, current_col) == (q - 1u)) {
						        sel = pivot_row;
						        __data[pivot_row].neg();
						        break;
					        }
				        }

				        /// no pivot found
				        if (sel == -1u) { return i - row; }

				        swap_rows(i, sel);

				        /// if the pivot row is taken from outside of the kk x kk square
				        /// we need to resolve it
				        if (sel >= row + kk) {
					        for (uint32_t j = col; j < col + kk; ++j) {
						        if (j == i) { continue; }

						        const DataType a = get(i, j);
						        if (a == 0) { continue; }

						        // negate
						        DataType c = (q - a) % q;
						        RowType::scalar(tmp, __data[row + j - col], c);
						        RowType::add(__data[i], __data[i], tmp);
					        }
				        }

				        /// this is stupid: while fixing the pivot row, it can happen that
				        /// the pivot element gets zero out. We catch this case and restart.
				        if (get(i, current_col) == 0) goto retry;

				        // one final fixup
				        if (get(i, current_col) != 1) {
					        RowType::scalar(tmp, __data[current_col], (q - get(i, current_col) % q));
					        RowType::add(__data[i], __data[i], tmp);
				        }

				        /// solve the column in the kk x kk square
				        for (uint32_t j = row; j < row + kk; ++j) {
					        if (j == i) { continue; }

					        const DataType a = get(j, current_col);
					        if (a == 0) { continue; }
					        // negate
					        DataType c = (q - a) % q;
					        RowType::scalar(tmp, __data[i], c);
					        RowType::add(__data[j], __data[j], tmp);
				        }
			        }

			        return kk;
		        };

		uint32_t row = 0, col = 0, kk = r;
		while (col < rstop) {
			std::fill(computed.begin(), computed.end(), 0);
			if (col + kk > rstop) { kk = rstop - col; }
			const uint32_t kbar = sub_gaus(row, col, kk);

			if (kk != kbar) { break; }

			if (kbar > 0) {
				/// process below
				process_rows(row, col, kk, row + kbar, nrows);
				/// process above
				process_rows(row, col, kk, 0, row);
			}

			row += kbar;
			col += kbar;
		}

		return col;
	}


	/// NOTE: the input matrix must be systemized
	/// \tparam c
	/// \tparam max_row
	/// \return
	template<const uint32_t c, const uint32_t max_row>
	[[nodiscard]] constexpr uint32_t markov_gaus(Permutation &P) noexcept {
		static_assert(c > 0);
		static_assert(max_row > 0);
		static_assert(max_row <= nrows);
#ifdef DEBUG
		auto check_correctness = [this]() {
			for (uint32_t i = 0; i < nrows; ++i) {
				for (uint32_t j = 0; j < max_row; ++j) {
					if (get(i, j) != (i == j)) {
						print();
					}

					ASSERT(get(i, j) == (i == j));
				}
			}
		};
		check_correctness();
#endif
		uint32_t additional_to_solve = 0;
		RowType tmp;

		/// chose a new random permutation on only c coordinates
		std::array<uint32_t, c> perm;
		for (uint32_t i = 0; i < c; ++i) {
			perm[i] = fastrandombytes_uint64() % (ncols - 1);
		}

		/// apply the random permutation
		for (uint32_t i = 0; i < c; ++i) {
			std::swap(P.values[i], P.values[perm[i]]);
			swap_cols(i, perm[i]);

			///
			if (perm[i] < max_row) {
				swap_rows(i, perm[i]);
			}
		}

		/// fix the wrong columns
		for (uint32_t i = 0; i < c; ++i) {
			bool found = false;
			/// pivoting
			if (get(i, i) != 1u) {
				/// try to find from below
				for (uint32_t j = i + 1; j < c; ++j) {
					if (__data[j][i] == 1u) {
						swap_rows(j, i);
						found = true;
						break;
					}

					if (__data[j][i] == (q - 1u)) {
						swap_rows(j, i);
						__data[i].neg();
						found = true;
						break;
					}
				}

				if (!found) {
					/// if we are here, we failed to find a pivot element in colum i
					/// in the first rows. Now we permute in a unity column from between
					/// [c, max_row).
					/// We simply use the first free one
					const uint32_t column_to_take = max_row - 1u - additional_to_solve;
					additional_to_solve += 1u;
					std::swap(P.values[i], P.values[column_to_take]);
					perm[i] = c;
					swap_cols(i, column_to_take);
					swap_rows(i, column_to_take);

					// now we can skip the rest, as there is already a unity vector
					continue;
				}
			}

			// TODO: currently only 1 and q-1 are pivot rows
			//const DataType scal = get(i, i);
			//ASSERT(scal);
			//if (scal > 1u) {
			//	RowType::scalar(tmp, __data[i], scal-1u);
			//	tmp.print();
			//	RowType::add(__data[i], __data[i], tmp);
			//}

			ASSERT(get(i, i));
			/// first clear above
			for (uint32_t j = 0; j < nrows; ++j) {
				if (i == j) continue;
				uint32_t scal = get(j, i);
				if (scal) {
					RowType::scalar(tmp, __data[i], q - scal);
					RowType::add(__data[j], __data[j], tmp);
				}
			}
		}


		/// last but not least
		for (int32_t i = additional_to_solve; i > 0; --i) {
			bool found = false;
			/// pivoting
			for (uint32_t j = max_row - i; j < ncols; ++j) {
				if (__data[max_row - i][j] == 1u) {
					swap_cols(j, max_row - i);
					std::swap(P.values[j], P.values[max_row - i]);
					found = true;
					break;
				}

				if (__data[max_row - i][j] == (q-1)) {
					swap_cols(j, max_row - i);
					std::swap(P.values[j], P.values[max_row - i]);
					__data[max_row - i].neg();
					found = true;
					break;
				}
			}

			if (!found) {
				return max_row - i;
			}

			for (uint32_t j = 0; j < nrows; ++j) {
				if ((max_row - i) == j) continue;
				uint32_t scal = get(j, max_row - i);
				if (scal) {
					RowType::scalar(tmp, __data[max_row - i], q - scal);
					RowType::add(__data[j], __data[j], tmp);
				}
			}
		}

#ifdef DEBUG
		check_correctness();
#endif
		return max_row;
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

		std::array<DataType, nrows> col{};
		for (uint32_t row = 0; row < nrows; ++row) {
			col[row] = get(row, i);
		}

		for (uint32_t row = 0; row < nrows; ++row) {
			set(get(row, j), row, i);
		}

		for (uint32_t row = 0; row < nrows; ++row) {
			set(col[row], row, j);
		}
	}

	/// swap rows
	/// \param i first row
	/// \param j second row
	constexpr void swap_rows(const uint16_t i,
	                         const uint16_t j) noexcept {
		ASSERT(i < nrows && j < nrows);
		if (i == j) {
			return;
		}

		RowType tmp = __data[i];
		__data[i] = __data[j];
		__data[j] = tmp;
	}

	/// choose and apply a new random permutation
	/// \param AT transposed matrix
	/// \param permutation given permutation (is overwritten)
	/// \param len length of the permutation
	constexpr void permute_cols(FqMatrix_Meta<T, ncols, nrows, q, packed> &AT,
	                            Permutation &P) noexcept {
		ASSERT(ncols >= P.length);

		this->transpose(AT, *this, 0, 0);
		for (uint32_t i = 0; i < P.length; ++i) {
			uint32_t pos = fastrandombytes_uint64() % (P.length - i);
			ASSERT(i + pos < P.length);

			auto tmp = P.values[i];
			P.values[i] = P.values[i + pos];
			P.values[pos + i] = tmp;

			AT.swap_rows(i, i + pos);
		}

		FqMatrix_Meta<T, ncols, nrows, q, packed>::transpose(*this, AT, 0, 0);
	}

	/// NOTE: is slower than the implementation utilizing the
	/// \param permutation
	/// \param len
	/// \return
	constexpr void permute_cols(Permutation &P) noexcept {
		for (uint32_t i = 0; i < P.length; ++i) {
			uint32_t pos = fastrandombytes_uint64() % (P.length - i);
			ASSERT(i + pos < P.length);

			auto tmp = P.values[i];
			P.values[i] = P.values[i + pos];
			P.values[pos + i] = tmp;

			swap_cols(i, i + pos);
		}
	}

	///
	/// \tparam Tprime
	/// \tparam nrows_prime
	/// \tparam ncols_prime
	/// \tparam ncols_prime2
	/// \param ret
	/// \param in1
	/// \param in2
	/// \return
	template<typename Tprime,
	         const uint32_t nrows_prime,
	         const uint32_t ncols_prime,
	         const uint32_t ncols_prime2
	         >
	constexpr static FqMatrix_Meta<T, nrows, ncols_prime + ncols_prime2, q, packed>
	augment(FqMatrix_Meta<T, nrows, ncols_prime + ncols_prime2, q, packed> &ret,
	        const FqMatrix_Meta<Tprime, nrows_prime, ncols_prime, q, packed> &in1,
	        const FqMatrix_Meta<Tprime, nrows_prime, ncols_prime2, q, packed> &in2) noexcept {
		/// NOTE: we allow not equally sized matrices to augment,
		/// but the augmented matrix we be zero extended
		static_assert(nrows_prime <= nrows);
		ret.clear();

		for (uint32_t i = 0; i < nrows_prime; ++i) {
			for (uint32_t j = 0; j < ncols_prime; ++j) {
				const T data = in1.get(i, j);
				ret.set(data, i, j);
			}
		}

		for (uint32_t i = 0; i < nrows_prime; ++i) {
			for (uint32_t j = 0; j < ncols_prime2; ++j) {
				const T data = in2.get(i, j);
				ret.set(data, i, ncols_prime + j);
			}
		}

		return ret;
	}

	/// compute the hamming weight of a column
	/// \param col column index
	/// \return hamming weight
	[[nodiscard]] constexpr inline uint32_t column_popcnt(const uint32_t col) const noexcept {
		ASSERT(col < ncols);
		uint32_t weight = 0;
		for (uint32_t i = 0; i < nrows; ++i) {
			weight += get(i, col) > 0;
		}

		return weight;
	}

	/// compute the hamming weight of a row
	/// \param row index
	/// \return hamming weight
	[[nodiscard]] constexpr inline uint32_t row_popcnt(const uint32_t row) const noexcept {
		ASSERT(row < nrows);
		uint32_t weight = 0;
		for (uint32_t i = 0; i < ncols; ++i) {
			weight += get(row, i) > 0;
		}

		return weight;
	}

	///
	/// \param C
	/// \param A
	/// \param B
	/// \return
	constexpr static void mul(
			FqMatrix_Meta &C,
			const FqMatrix_Meta &A,
			const FqMatrix_Meta &B) noexcept {

		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < ncols; ++j) {
				uint64_t sum = 0;
				for (uint32_t k = 0; k < ncols; ++k) {
					uint32_t a = A.get(i, k);
					uint32_t b = B.get(k, j);
					uint32_t c = (a * b) % q;
					sum += c;
				}

				C.set(sum % q, i, j);
			}
		}
	}

	/// compute C = this*B
	template<const uint32_t ncols_prime>
	constexpr static void mul(
	        FqMatrix_Meta<T, nrows, ncols_prime, q, packed> &C,
	        const FqMatrix_Meta<T, nrows, ncols, q, packed> &A,
	        const FqMatrix_Meta<T, ncols, ncols_prime, q, packed> &B) noexcept {

		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < ncols_prime; ++j) {
				uint64_t sum = 0;
				for (uint32_t k = 0; k < ncols; ++k) {
					uint32_t a = A.get(i, k);
					uint32_t b = B.get(k, j);
					uint32_t c = (a * b) % q;
					sum += c;
				}

				C.set(sum % q, i, j);
			}
		}
	}

	/// allows for transposed input
	template<const uint32_t ncols_prime>
	constexpr static void mul_transposed(
	        FqMatrix_Meta<T, nrows, ncols_prime, q, packed> &C,
	        const FqMatrix_Meta<T, nrows, ncols, q, packed> &A,
	        const FqMatrix_Meta<T, ncols_prime, ncols, q, packed> &B) noexcept {
		MatrixType::template mul<ncols_prime>(C, A, B.transpose());
	}

	/// \param A input
	/// \param B input
	/// \return true if equal
	[[nodiscard]] constexpr static bool is_equal(const FqMatrix_Meta &A,
	                               const FqMatrix_Meta &B) {
		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < ncols; ++j) {
				if (A.get(i, j) != B.get(i, j)) {
					return false;
				}
			}
		}

		return true;
	}

	/// special version of the row_vector multiplication
	/// in which the input/output are not matrices but
	/// vectors.
	/// \tparam LabelType must fulfill the following function
	///					get(i), set(i)
	/// \tparam ValueType must fulfill the following function
	/// 				get(i), set(i),
	/// \param out
	/// \param in
	/// \return
	template<class LabelType, class ValueType>
#if __cplusplus > 201709L
	    requires LabelTypeAble<LabelType> &&
	             ValueTypeAble<ValueType>
#endif
	constexpr void mul(LabelType &out,
	                   const ValueType &in) const noexcept {
		using DataType = typename LabelType::DataType;
		constexpr uint32_t IN_COLS = ValueType::length();
		constexpr uint32_t OUT_COLS = LabelType::length();
		static_assert((IN_COLS == COLS)  || (IN_COLS == ROWS)) ;
		static_assert((OUT_COLS == ROWS) || (OUT_COLS == COLS));

		if constexpr ((OUT_COLS == COLS) && (IN_COLS == ROWS)) {
			// transposed multiplication
			for (uint32_t i = 0; i < OUT_COLS; ++i) {
				DataType sum = 0;
				for (uint32_t j = 0; j < IN_COLS; ++j) {
					auto a = get(j, i);
					auto b = in.get(j);
					auto c = (a * b) % q;
					sum = (sum + c) % q;
				}

				out.set(sum % q, i);
			}

			return;
		}

		if constexpr ((OUT_COLS && ROWS) && (IN_COLS == COLS)) {
			// normal multiplication
			for (uint32_t i = 0; i < nrows; ++i) {
				DataType sum = 0;
				for (uint32_t j = 0; j < ncols; ++j) {
					auto a = get(i, j);
					auto b = in.get(j);
					auto c = (a * b) % q;
					sum = (sum + c) % q;
				}
				out.set(sum % q, i);
			}

			return;
		}

		ASSERT(false);
	}

	/// prints the current matrix
	/// \param name postfix the name of the matrix
	/// \param binary print as binary
	/// \param compress_spaces if true, do not print spaces between the elements
	/// \param syndrome if true, print the last line as the syndrome
	constexpr void print(const std::string &name = "",
	                     bool binary = false,
	                     bool transposed = false,
	                     bool compress_spaces = false,
	                     bool syndrome = false) const noexcept {
		constexpr uint32_t bits = bits_log2(q);
		if (binary) {
			for (uint32_t j = 0; j < ncols; ++j) {
				//std::cout << std::setw(3) << j << ": ";
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
				//std::cout << std::setw(3) << j << ": ";
				for (uint32_t i = 0; i < nrows; ++i) {
					std::cout << int(get(i, j));
					if (not compress_spaces) {
						std::cout << " ";
					}
				}
			}
		} else {
			for (uint32_t i = 0; i < nrows; ++i) {
				//std::cout << std::setw(3) << i << ": ";
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

		if (name.length() > 0) {
			std::cout << " " << name << "\n";
		}
	}

	/// some simple functions
	[[nodiscard]] constexpr inline bool binary() const noexcept { return false; }

	/// these two functions exist, as there are maybe matrix implementations
	/// you want to wrap, which are not constant sized
	[[nodiscard]] constexpr inline uint32_t rows() const noexcept { return ROWS; }
	[[nodiscard]] constexpr inline uint32_t cols() const noexcept { return COLS; }

	/// get a full limb, instead of just a column
	/// \param row row
	/// \param limb column but instead you get a limb
	/// \return
	[[nodiscard]] constexpr inline T limb(const uint32_t row,
	                        const uint32_t limb) const noexcept {
		ASSERT(row < ROWS);
		ASSERT(limb < limbs());
		return __data[row].ptr(limb);
	}

	/// returns the number of limbs within each row
	[[nodiscard]] static constexpr inline uint32_t limbs() noexcept {
		return RowType::internal_limbs;
	}

	///
	constexpr static void info() noexcept {
		std::cout << "{ name: \"FqMatrix_Meta\""
				  << ", nrows: " << nrows
				  << ", ncols: " << ncols
				  << ", sizeof(Matrix): " << sizeof(MatrixType)
				  << ", sizeof(Row): " << sizeof(RowType)
		          << " }" << std::endl;

		RowType::info();
	}
};


/// \tparam T
/// \tparam nrows
/// \tparam ncols
/// \tparam q
/// \tparam packed
/// \tparam R
/// \param out
/// \param obj
/// \return
template<typename T,
		const uint32_t nrows,
		const uint32_t ncols,
		const uint32_t q,
		const bool packed = false,
		typename R=void>
std::ostream &operator<<(std::ostream &out,
                         const FqMatrix_Meta<T, nrows, ncols, q, packed, R> &obj) {
	for (uint64_t i = 0; i < nrows; ++i) {
		std::cout << "[ ";
		for (uint64_t j = 0; j < ncols; ++j) {
			const auto d = obj.get(i, j);
			std::cout << d << ' ';
		}

		std::cout << " ]\n";
	}

	return out;
}

///
#include "matrix/binary_matrix.h"
#include "matrix/fq_matrix.h"
#include "matrix/vector.h"

#endif//CRYPTANALYSISLIB_MATRIX_H
