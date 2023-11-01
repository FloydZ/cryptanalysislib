#ifndef CRYPTANALYSISLIB_BINARYMATRIX_H
#define CRYPTANALYSISLIB_BINARYMATRIX_H


#include <type_traits>
#include <utility>
#include <array>
#include <memory>

#include "helper.h"
#include "random.h"
#include "simd/simd.h"
#include "matrix/fq_matrix.h"
#include "permutation/permutation.h"


/// Most of this is taken from M4ri
/// https://github.com/malb/m4ri
namespace cryptanalysislib::internal::matrix {
	#define MATRIX_AVX_PADDING(len) (((len + 255) / 256) * 256)

	#define m4ri_radix 64
	#define m4ri_one (1ul)
	#define m4ri_ffff (-1ul)
	/**
 	 * \brief flag when ncols%64 != 0
 	 */
	#define mzd_flag_nonzero_excess 0x2

	/**
	 * \brief flag for windowed matrix
	 */
	#define mzd_flag_windowed 0x4

	/**
	 * \brief $2^i$
	 *
	 * \param i Integer.
	 */
	#define __M4RI_TWOPOW(i) ((uint64_t)1 << (i))

	/**
	 * \brief Clear the bit spot (counting from the left) in the word w
	 *
	 * \param w Word
	 * \param spot Integer with 0 <= spot < m4ri_radix
	 */
	#define __M4RI_CLR_BIT(w, spot) ((w) &= ~(m4ri_one << (spot))

	/**
	 * \brief Set the bit spot (counting from the left) in the word w
	 *
	 * \param w Word
	 * \param spot Integer with 0 <= spot < m4ri_radix
	 */
	#define __M4RI_SET_BIT(w, spot) ((w) |= (m4ri_one << (spot)))

	/**
	 * \brief Get the bit spot (counting from the left) in the word w
	 *
	 * \param w Word
	 * \param spot Integer with 0 <= spot < m4ri_radix
	 */
	#define __M4RI_GET_BIT(w, spot) __M4RI_CONVERT_TO_BIT(((w) >> (spot)) & m4ri_one)

	/**
	 * \brief Write the value to the bit spot in the word w
	 *
	 * \param w Word.
	 * \param spot Integer with 0 <= spot < m4ri_radix.
	 * \param value Either 0 or 1.
	 */
	#define __M4RI_WRITE_BIT(w, spot, value)                                                           \
	  ((w) = (((w) & ~(m4ri_one << (spot))) | (__M4RI_CONVERT_TO_WORD(value) << (spot))))

	/**
	 * \brief Flip the spot in the word w
	 *
	 * \param w Word.
	 * \param spot Integer with 0 <= spot < m4ri_radix.
	 */
	#define __M4RI_FLIP_BIT(w, spot) ((w) ^= (m4ri_one << (spot)))

		/**
	 * \brief create a bit mask to zero out all but the (n - 1) % m4ri_radix + 1 leftmost bits.
	 *
	 * This function returns 1..64 bits, never zero bits.
	 * This mask is mainly used to mask the valid bits in the most significant word,
	 * by using __M4RI_LEFT_BITMASK((M->ncols + M->offset) % m4ri_radix).
	 * In other words, the set bits represent the columns with the lowest index in the word.
	 *
	 *  Thus,
	 *
	 *  n	Output
	 *  0=64 1111111111111111111111111111111111111111111111111111111111111111
	 *  1	 0000000000000000000000000000000000000000000000000000000000000001
	 *  2    0000000000000000000000000000000000000000000000000000000000000011
	 *  .                                   ...
	 * 62    0011111111111111111111111111111111111111111111111111111111111111
	 * 63	 0111111111111111111111111111111111111111111111111111111111111111
	 *
	 * Note that n == 64 is only passed from __M4RI_MIDDLE_BITMASK, and still works
	 * (behaves the same as n == 0): the input is modulo 64.
	 *
	 * \param n Integer with 0 <= n <= m4ri_radix
	 */
	#define __M4RI_LEFT_BITMASK(n) (m4ri_ffff >> (m4ri_radix - (n)) % m4ri_radix)

		/**
	 * \brief create a bit mask to zero out all but the n rightmost bits.
	 *
	 * This function returns 1..64 bits, never zero bits.
	 * This mask is mainly used to mask the n valid bits in the least significant word
	 * with valid bits by using __M4RI_RIGHT_BITMASK(m4ri_radix - M->offset).
	 * In other words, the set bits represent the columns with the highest index in the word.
	 *
	 *  Thus,
	 *
	 *  n	Output
	 *  1	1000000000000000000000000000000000000000000000000000000000000000
	 *  2   1100000000000000000000000000000000000000000000000000000000000000
	 *  3   1110000000000000000000000000000000000000000000000000000000000000
	 *  .                                   ...
	 * 63	1111111111111111111111111111111111111111111111111111111111111110
	 * 64	1111111111111111111111111111111111111111111111111111111111111111
	 *
	 * Note that n == 0 is never passed and would fail.
	 *
	 * \param n Integer with 0 < n <= m4ri_radix
	 */
	#define __M4RI_RIGHT_BITMASK(n) (m4ri_ffff << (m4ri_radix - (n)))

		/**
	 * \brief create a bit mask that is the combination of __M4RI_LEFT_BITMASK and __M4RI_RIGHT_BITMASK.
	 *
	 * This function returns 1..64 bits, never zero bits.
	 * This mask is mainly used to mask the n valid bits in the only word with valid bits,
	 * when M->ncols + M->offset <= m4ri_radix), by using __M4RI_MIDDLE_BITMASK(M->ncols, M->offset).
	 * It is equivalent to __M4RI_LEFT_BITMASK(n + offset) & __M4RI_RIGHT_BITMASK(m4ri_radix - offset).
	 * In other words, the set bits represent the valid columns in the word.
	 *
	 * Note that when n == m4ri_radix (and thus offset == 0) then __M4RI_LEFT_BITMASK is called with n
	 * == 64.
	 *
	 * \param n Integer with 0 < n <= m4ri_radix - offset
	 * \param offset Column offset, with 0 <= offset < m4ri_radix
	 */
	#define __M4RI_MIDDLE_BITMASK(n, offset) (__M4RI_LEFT_BITMASK(n) << (offset))

	/// implementation taken from gray_code.c
	/// chooses the optimal `r` for the method of the 4 russians
	/// a = #rows, b = #cols
	constexpr int matrix_opt_k(const int a, const int b) noexcept {
		return  std::min(M4RI_MAXK, std::max(1, (int)(0.75 * (float)(1u + const_log(std::min(a, b))))));
	}

	/**
 	* \brief Data containers containing the values packed into words
 	*/
	typedef struct {
		size_t size; /*!< number of words */
		uint64_t *begin; /*!< first word */
		uint64_t *end;   /*!< last word */
	} mzd_block_t;

	/**
	 * \brief Dense matrices over GF(2).
	 *
	 * The most fundamental data type in this library.
	 */
	typedef struct mzd_t {
	private:
		uint32_t nrows; /*!< Number of rows. */
		uint32_t ncols; /*!< Number of columns. */
		uint32_t width;  /*!< Number of words with valid bits: width = ceil(ncols / m4ri_radix) */

		/**
	     * Offset in words between rows.
	     *
	     * rowstride = (width < mzd_paddingwidth || (width & 1) == 0) ? width : width + 1;
	     * where width is the width of the underlying non-windowed matrix.
	     */
		uint32_t rowstride;

		/**
	     * Booleans to speed up things.
	     *
	     * The bits have the following meaning:
	     *
	     * 1: Has non-zero excess.
	     * 2: Is windowed, but has zero offset.
	     * 3: Is windowed, but has zero excess.
	     * 4: Is windowed, but owns the blocks allocations.
	     * 5: Spans more than 1 block.
	     */
		uint8_t flags;

		/* ensures sizeof(mzd_t) == 64 */
		uint8_t padding[63 - 2 * sizeof(uint32_t) - 2 * sizeof(uint32_t) - sizeof(uint8_t) - sizeof(void *)];

		uint64_t high_bitmask;   /*!< Mask for valid bits in the word with the highest index (width - 1). */
		mzd_block_t *blocks; /*!< Pointers to the actual blocks of memory containing the values packed
							  into words. */
		/**
	     * Address of first word in each row, so the first word of row i is is m->rows[i]
	     */
		uint64_t *data;

		/// additional overlay over the `mpz_t` class for the following two reasons:
		/// 	- row size is chosen as an multiple of 256 for SIMD operations
		/// 	- custom row operations


		/// constructor
		/// \param r
		/// \param c
		mzd_t(const uint32_t r, const uint32_t c) noexcept{

		}

	} mzd_t;








	/**
 	* \brief Get pointer to first word of row.
 	*
 	* \param M Matrix
 	* \param row The row index.
 	*
 	* \return pointer to first word of the row.
 	*/

	inline uint64_t *mzd_row(mzd_t *M, const uint32_t row) noexcept {
		return M->data + M->rowstride * row;
	}

	inline uint64_t const * mzd_row_const(mzd_t const *M, const uint32_t row) noexcept {
		return mzd_row((mzd_t *)M, row);
	}


	/// z = x^y;
	/// nn = number of bytes*32 = number of uint256
	inline void xor_avx1_new(const uint8_t *__restrict__ x,
							 const uint8_t *__restrict__ y,
							 uint8_t *__restrict__ z,
							 unsigned nn) noexcept {
#ifdef USE_AVX2_SPECIAL_ALIGNMENT
		constexpr bool special_alignment = true;
#else
		constexpr bool special_alignment = false;
#endif

		LOOP_UNROLL()
		for (uint32_t i = 0; i < nn; i += 1) {
			const uint8x32_t x_avx = uint8x32_t::template load<special_alignment>(x + 32*i);
			const uint8x32_t y_avx = uint8x32_t::template load<special_alignment>(y + 32*i);
			const uint8x32_t z_avx = x_avx ^ y_avx;
			uint8x32_t::template store<special_alignment>(z + 32*i, z_avx);
		}
	}

	/// row: i ^= j
	/// \param out input/output matrix
	/// \param i input/output row
	/// \param j input row
	void mzd_row_xor(mzd_t *out,
					 const uint32_t i,
					 const uint32_t j) noexcept {
		ASSERT(out->nrows > i && out->nrows > j);
		uint32_t l = 0;

		LOOP_UNROLL()
		for (; l+4 <= uint32_t(out->width); l+=4) {
			const uint8x32_t x_avx = uint8x32_t::load(out->rows[j] + l);
			const uint8x32_t y_avx = uint8x32_t::load(out->rows[i] + l);
			const uint8x32_t z_avx = x_avx ^ y_avx;
			uint8x32_t::store(out->rows[i] + l, z_avx);
		}

		for (; l < uint32_t(out->width); ++l) {
			out->rows[i][l] ^= out->rows[j][l];
		}
	}

	/// out[i] ^= in[j]
	void mzd_row_xor(mzd_t *out,
					 const uint32_t i,
					 mzd_t *in,
					 const uint32_t j) noexcept {
		ASSERT(out->nrows > i && in->nrows > j);
		uint32_t l = 0;

		LOOP_UNROLL()
		for (; l+4 <= uint32_t(out->width); l+=4) {
			const uint8x32_t x_avx = uint8x32_t::load(out->rows[j] + l);
			const uint8x32_t y_avx = uint8x32_t::load(in->rows[i] + l);
			const uint8x32_t z_avx = x_avx ^ y_avx;
			uint8x32_t::store(out->rows[i] + l, z_avx);
		}

		for (; l < uint32_t(out->width); ++l) {
			out->rows[i][l] ^= in->rows[j][l];
		}
	}

	mzd_t *matrix_transpose(mzd_t *DST, mzd_t const *A) noexcept {
		if (DST == nullptr) {
			DST = mzd_init(A->ncols, A->nrows);
		} else if (unlikely(DST->nrows < A->ncols || DST->ncols < A->nrows)) {
			std::cout << "mzd_transpose: Wrong size for return matrix." << std::endl;
			exit(-1);
		} else {
			/** it seems this is taken care of in the subroutines, re-enable if running into problems **/
			ASSERT(false);
		}

		if (A->nrows == 0 || A->ncols == 0) {
			return mzd_copy(DST, A);
		}

		if (__M4RI_LIKELY(!mzd_is_windowed(DST) && !mzd_is_windowed(A))) {
			return _mzd_transpose(DST, A);
		}

		int A_windowed = mzd_is_windowed(A);
		if (A_windowed) A = mzd_copy(nullptr, A);
		if (__M4RI_LIKELY(!mzd_is_windowed(DST)))
			_mzd_transpose(DST, A);
		else {
			mzd_t *D = mzd_init(DST->nrows, DST->ncols);
			_mzd_transpose(D, A);
			mzd_copy(DST, D);
			mzd_free(D);
		}
		if (A_windowed) mzd_free((mzd_t *)A);
		return DST;
	}
	/// optimized version to which you have additionally pass the transposed.
	/// So its not created/freed every time
	/// \param A input matrix which should be randomly permuted
	/// \param AT
	/// \param P
	void matrix_create_random_permutation(mzd_t *__restrict__ A,
										  mzd_t *__restrict__ AT,
										  mzp_t *__restrict__ P) noexcept {
		matrix_transpose(AT, A);

		// dont permute the last column since it is the syndrome
		for (uint32_t i = 0; i < uint32_t(P->length-1); ++i) {
			uint64_t pos = fastrandombytes_uint64() % (P->length - i);

			ASSERT(i+pos < uint32_t(P->length));
			std::swap(P->values[i], P->values[i+pos]);
			mzd_row_swap(AT, i, i+pos);
			//matrix_swap_rows_new2(AT, i, i+pos);
		}
		matrix_transpose(A, AT);
	}

} // namespace cryptanalysislib::internal



using namespace cryptanalysislib::internal::matrix;
/// matrix implementation which wrapped mzd
/// \tparam T MUST be uint64_t
/// \tparam nrows number of rows
/// \tparam ncols number of columns
template<typename T, const uint32_t nrows, const uint32_t ncols>
class FqMatrix<T, nrows, ncols, 2>: private FqMatrix_Meta<T, nrows, ncols, 2> {
private:
	constexpr static uint32_t RADIX = sizeof(T) * 8u;
	constexpr static uint32_t MAX_K = 8;

	// number of limbs needed
	constexpr static uint32_t limbs = (ncols + RADIX -1u) / RADIX;

	// number of limbs actually allocated
	constexpr static uint32_t alignment = 256; // NOTE: currently that's chosen for avx
	constexpr static uint32_t padded_limbs = (ncols + alignment - 1u) / RADIX;
	constexpr static uint32_t padded_columns = ((ncols + alignment - 1u) / alignment) * alignment;

	constexpr static T high_bitmask = -1ul >> ((RADIX - (ncols%RADIX)) %RADIX);
	constexpr static uint32_t block_words = nrows * padded_limbs;


	///
	struct MatrixData {
		alignas(64) uint32_t **rev = nullptr;
		alignas(64) uint32_t **diff = nullptr;
		alignas(64) uint64_t *lookup_table = nullptr;
	};

	///
	/// \param nr_columns
	/// \return
	void init_matrix_data() noexcept {
		matrix_alloc_gray_code(&__matrix_data->rev, &__matrix_data->diff);
		matrix_build_gray_code(__matrix_data->rev, __matrix_data->diff);
		__matrix_data->lookup_table = (uint64_t *)aligned_alloc(4096, (padded_columns / 8) * (1ul<<MAX_K));
	}

	void free_matrix_data() noexcept {
		matrix_free_gray_code(__matrix_data->rev, __matrix_data->diff);
		free(__matrix_data->lookup_table);
	}

	std::array<T, block_words> __data;
	MatrixData __matrix_data;
public:
	/// that's only because I'm lazy
	static constexpr uint32_t q = 2;

	/// needed typedefs
	using RowType = T*;
	using DataType = bool;
	const uint32_t m4ri_k = matrix_opt_k(nrows, ncols);

	/// TODO write ownfunctins
	/// needed functions
	using FqMatrix_Meta<T, nrows,ncols, q>::set;
	using FqMatrix_Meta<T, nrows,ncols, q>::identity;
	using FqMatrix_Meta<T, nrows,ncols, q>::is_equal;
	using FqMatrix_Meta<T, nrows,ncols, q>::weight_column;
	using FqMatrix_Meta<T, nrows,ncols, q>::swap;

	/// simple constructor
	constexpr FqMatrix() noexcept {
		ASSERT(sizeof(mzd_t) == 64);
		static_assert(sizeof(T) == 8);
		static_assert(nrows && ncols);

		init_matrix_data(ncols);
	}

	/// simple deconstructor
	constexpr ~FqMatrix() noexcept {
		free_matrix_data();
	}

	/// copy constructor
	constexpr FqMatrix(const FqMatrix &A) noexcept {
		std::copy(A.__data.begin(), A.__data.end(), __data.begin());
		matrix_copy(__data, A.__data);
		init_matrix_data();
	}

	/// constructor reading from string
	constexpr FqMatrix(const char* data, const uint32_t cols=ncols) noexcept {
		init_matrix_data();

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

	/// copy operator
	constexpr void copy(const FqMatrix &A) noexcept {
		matrix_copy(__data, A.__data);
	}

	/// \param data data to set
	/// \param i row
	/// \param j colum
	constexpr void set(const bool data, const uint32_t i, const uint32_t j) noexcept {
		ASSERT(i < nrows);
		ASSERT(j < ncols);
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
	[[nodiscard]] constexpr const RowType get(const uint32_t i) const noexcept {
		ASSERT(i < nrows);
		return __data + padded_limbs*i;
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
				const bool data = A.get(i, j);
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
				const bool data = A.get(i, j);
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
				const bool data = A.get(row, col);
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
								uint32_t *permutation,
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

	[[nodiscard]] constexpr uint32_t fix_gaus(uint32_t *__restrict__ permutation,
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
