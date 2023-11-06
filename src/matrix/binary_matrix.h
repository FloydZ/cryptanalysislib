#ifndef CRYPTANALYSISLIB_BINARYMATRIX_H
#define CRYPTANALYSISLIB_BINARYMATRIX_H

#include <algorithm>
#include <type_traits>
#include <utility>
#include <array>
#include <memory>

#include "helper.h"
#include "matrix/fq_matrix.h"
#include "permutation/permutation.h"
#include "random.h"
#include "simd/simd.h"


/// matrix implementation which wrapped mzd
/// \tparam T MUST be uint64_t
/// \tparam nrows number of rows
/// \tparam ncols number of columns
template<typename T, const uint32_t __nrows, const uint32_t __ncols>
class FqMatrix<T, __nrows, __ncols, 2, true>: private FqMatrix_Meta<T, __nrows, __ncols, 2, true> {
private:
	constexpr static uint32_t RADIX = sizeof(T) * 8u;
	constexpr static uint32_t MAX_K = 8ul;
	constexpr static T one  = T(1ul);
	constexpr static T ffff = T(-1ul);

	constexpr static uint32_t ncols = __ncols;
	constexpr static uint32_t nrows = __nrows;

	// number of limbs needed
	constexpr static uint32_t limbs = (ncols + RADIX -1u) / RADIX;

	// number of limbs actually allocated
	constexpr static uint32_t alignment = 256; // NOTE: currently that's chosen for avx
	constexpr static uint32_t padded_limbs = (ncols + alignment - 1u) / RADIX;
	constexpr static uint32_t padded_simd_limbs = (ncols + alignment - 1u) / alignment;
	constexpr static uint32_t padded_bytes = padded_limbs * 8;
	constexpr static uint32_t padded_columns = ((ncols + alignment - 1u) / alignment) * alignment;
	constexpr static uint32_t padded_ncols = padded_columns;


	constexpr static T high_bitmask = -1ul >> ((RADIX - (ncols%RADIX)) %RADIX);
	constexpr static uint32_t block_words = nrows * padded_limbs;
	
	/// implementation taken from gray_code.c
	/// chooses the optimal `r` for the method of the 4 russians
	/// a = #rows, b = #cols
	constexpr uint32_t matrix_opt_k(const uint32_t a, const uint32_t b) noexcept {
		return  std::min(uint64_t(MAX_K), std::max(uint64_t(1ul), (uint64_t)(0.75 * (float)(1u + const_log(std::min(a, b))))));
	}

	/// matrix data
	alignas(64) uint32_t **rev = nullptr;
	alignas(64) uint32_t **diff = nullptr;
	alignas(64) uint64_t *lookup_table = nullptr;
	
	constexpr static int gray_new(const uint32_t i,
	                    const uint32_t k) noexcept {
		int lastbit = 0;
		int res = 0;
		for (int j = k; j-- > 0;) {
			int bit = i & (1 << j);
			res |= (lastbit >> 1) ^ bit;
			lastbit = bit;
		}
		return res;
	}
	
	
	///
	/// \param rev
	/// \param diff
	void matrix_alloc_gray_code(uint32_t ***__restrict__ rev, uint32_t ***__restrict__ diff) noexcept {
		constexpr uint32_t alignment = 1024;
		*rev  = (uint32_t **) aligned_alloc(alignment, std::max(alignment, (uint32_t)(MAX_K + 2) * (uint32_t)sizeof(uint32_t *)));
		*diff = (uint32_t **) aligned_alloc(alignment, std::max(alignment, (uint32_t)(MAX_K + 1) * (uint32_t)sizeof(uint32_t *)));
	
		for (size_t k = 0; k <= MAX_K; ++k) {
			(*rev)[k]  = (uint32_t *) malloc((1 << k) * sizeof(uint32_t));
			(*diff)[k] = (uint32_t *) malloc((1 << k) * sizeof(uint32_t));
		}
	}
	
	///
	/// \param rev
	/// \param diff
	void matrix_free_gray_code(uint32_t **rev, uint32_t **diff) noexcept {
		for (size_t k = 0; k <= MAX_K; ++k) {
			free(rev[k]);
			free(diff[k]);
		}
	
		free(rev);
		free(diff);
	}
	
	///
	/// \param rev
	/// \param diff
	void matrix_build_gray_code(uint32_t **rev, uint32_t **diff) noexcept {
		for (size_t k = 0; k <= MAX_K; ++k) {
			for (size_t i = 0; i < 1UL << k; ++i) {
				rev[k][gray_new(i, k)] = i;
			}
	
			for (size_t i = k + 1; i-- > 0;) {
				for (size_t j = 1; j < (1UL << i) + 1; ++j) {
					diff[k][j * (1 << (k - i)) - 1] = k - i;
				}
			}
		}
	}


	///
	/// \param nr_columns
	/// \return
	void init_matrix_data() noexcept {
		matrix_alloc_gray_code(&rev, &diff);
		matrix_build_gray_code(rev, diff);
		lookup_table = (uint64_t *)aligned_alloc(PAGE_SIZE, (padded_columns / 8) * (1ul<<MAX_K));
	}

	///
	void free_matrix_data() noexcept {
		matrix_free_gray_code(rev, diff);
		free(lookup_table);
	}

	std::array<T, block_words> __data;
public:
	/// that's only because I'm lazy
	static constexpr uint32_t q = 2;

	/// needed typedefs
	using RowType = T*;
	using DataType = bool;
	const uint32_t m4ri_k = matrix_opt_k(nrows, ncols);

	/// TODO write ownfunctins
	/// needed functions
	using FqMatrix_Meta<T, nrows,ncols, q>::identity;
	using FqMatrix_Meta<T, nrows,ncols, q>::weight_column;
	using FqMatrix_Meta<T, nrows,ncols, q>::swap;

	/// simple constructor
	constexpr FqMatrix() noexcept {
		static_assert(sizeof(T) == 8);
		static_assert(nrows && ncols);

		init_matrix_data();
	}

	/// simple deconstructor
	~FqMatrix() noexcept {
		free_matrix_data();
	}

	/// copy constructor
	constexpr FqMatrix(const FqMatrix &A) noexcept {
		std::copy(A.__data.begin(), A.__data.end(), __data.begin());
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
				ASSERT(a >= 0);

				const uint32_t aa = a;
				ASSERT(aa < q);

				set(DataType(aa), i, j);
			}
		}
	}

	/// copy operator
	constexpr void copy(const FqMatrix &A) noexcept {
		std::copy(A.__data.begin(), A.__data.end(), __data.begin());
	}

	constexpr inline T* row(const uint32_t j) noexcept {
		ASSERT(j < nrows);
	  	return __data.data() + padded_limbs*j;
	}
	
	constexpr inline T const * row(const uint32_t j) const noexcept {
		ASSERT(j < nrows);
		return __data.data() + padded_limbs*j;
	}

	/// \param data data to set
	/// \param i row
	/// \param j colum
	constexpr inline void set(const bool data, const uint32_t i, const uint32_t j) noexcept {
		ASSERT(i < nrows);
		ASSERT(j < ncols);
  		T* truerow = row(i);
		const uint32_t spot = j % RADIX;
	  	truerow[j / RADIX] = ((truerow[j / RADIX]) & ~(one << spot)) | (T(data) << (spot));
	}

	///
	/// \param j
	/// \return
	constexpr T* operator[](const uint32_t j){
		ASSERT(j < nrows);
		return __data.data() + padded_limbs*j;
	}

	/// gets the i-th row and j-th column
	/// \param i row
	/// \param j colum
	/// \return entry in this place
	[[nodiscard]] constexpr inline DataType get(const uint32_t i, const uint32_t j) const noexcept {
		ASSERT(i < nrows && j <= ncols);
		const T* truerow = row(i);
		return ((truerow[j / RADIX]) >> (j % RADIX)) & one;
	}

	/// \param i row number (zero indexed)
	/// \return a const ref to a row
	[[nodiscard]] constexpr inline RowType get(const uint32_t i) const noexcept {
		ASSERT(i < nrows);
		return __data.data() + padded_limbs*i;
	}


	/// \return the number of `T` each row is made of
	[[nodiscard]] constexpr inline uint32_t limbs_per_row() const noexcept {
		return limbs;
	}

	/// clears the matrix
	/// \return
	constexpr void clear() noexcept {
		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < limbs_per_row(); ++j) {
				__data[i*padded_limbs + j] = 0;
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
				__data[i*padded_limbs + j] = fastrandombytes_uint64();
			}

			__data[i*padded_limbs + limbs_per_row() - 1u] = fastrandombytes_uint64() & high_bitmask;
		}
	}

	constexpr void fill(const bool data) noexcept {
		if (data == 0) {
			clear();
			return;
		}

		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < limbs_per_row() - 1u; ++j) {
				__data[i*padded_limbs + j] = T(-1ull);
			}

			__data[i*padded_limbs + limbs_per_row() - 1u] = +(-1ull) & high_bitmask;
		}
	}
	
	constexpr bool is_equal(const FqMatrix &in) const noexcept {
		for (uint32_t i = 0; i < nrows; i++) {
			for (uint32_t j = 0; j < limbs_per_row(); j++) {
				if (__data[i*padded_limbs + j] != in.__data[i*padded_limbs]) {
					return false;
				}
			}
		}

		return true;
	}


	/// z = x^y;
	/// nn = number of bytes*32 = number of uint256
	static constexpr inline void xor_avx1_new(const uint8_t *__restrict__ x,
							 				  const uint8_t *__restrict__ y,
							 				  uint8_t *__restrict__ z,
							 				  const uint32_t nn) noexcept {
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
	constexpr static inline void row_xor(FqMatrix &M,
										 const uint32_t i,
										 const uint32_t j) noexcept {
		ASSERT(nrows > i && nrows > j);
		constexpr uint32_t CTR = alignment/RADIX;
		uint32_t l = 0;

		LOOP_UNROLL()
		for (; l+CTR <= padded_limbs; l+=CTR) {
			const uint8x32_t x_avx = uint8x32_t::load(M.row(j) + l);
			const uint8x32_t y_avx = uint8x32_t::load(M.row(i) + l);
			const uint8x32_t z_avx = x_avx ^ y_avx;
			uint8x32_t::store(M.row( + i) + l, z_avx);
		}

		for (; l < limbs; ++l) {
			M.__data[i*padded_limbs + l] ^= M.__data[j*padded_limbs + l];
		}
	}

	constexpr inline void row_xor(const uint32_t i,
	                              const uint32_t j) noexcept {
		row_xor(*this, i, j);
	}

	/// row: i ^= j
	/// \param out input/output matrix
	/// \param i input/output row
	/// \param j input row
	constexpr static inline void row_xor(T *out,
					 const uint32_t i,
					 const uint32_t j) noexcept {
		ASSERT(nrows > i && nrows > j);
		constexpr uint32_t CTR = alignment/RADIX;
		uint32_t l = 0;

		LOOP_UNROLL()
		for (; l+CTR <= padded_limbs; l+=CTR) {
			const uint8x32_t x_avx = uint8x32_t::load(out + j*padded_limbs + l);
			const uint8x32_t y_avx = uint8x32_t::load(out + i*padded_limbs + l);
			const uint8x32_t z_avx = x_avx ^ y_avx;
			uint8x32_t::store(out + i*padded_limbs + l, z_avx);
		}

		for (; l < limbs; ++l) {
			out[i*padded_limbs + l] ^= out[j*padded_limbs + l];
		}
	}

	/// out[i] ^= in[j]
	constexpr static inline void row_xor(FqMatrix &out,
					 const uint32_t i,
					 const FqMatrix &in,
					 const uint32_t j) noexcept {
		ASSERT(out->nrows > i && in->nrows > j);
		uint32_t l = 0;
		constexpr uint32_t CTR = alignment/RADIX;

		LOOP_UNROLL()
		for (; l+CTR <= padded_limbs; l+=CTR) {
			const uint8x32_t x_avx = uint8x32_t::load(out.row(j) + l);
			const uint8x32_t y_avx = uint8x32_t::load(in.row(i) + l);
			const uint8x32_t z_avx = x_avx ^ y_avx;
			uint8x32_t::store(out.row(i) + l, z_avx);
		}

		for (; l < uint32_t(out->width); ++l) {
			out->rows[i][l] ^= in->rows[j][l];
		}
	}

	/// simple additions
	/// \param out output
	/// \param in1 input
	/// \param in2 input
	constexpr static void add(FqMatrix&out,
							  const FqMatrix &in1,
							  const FqMatrix &in2) noexcept {
		constexpr uint32_t nr_T_in_avx = 256/RADIX;
		for (uint32_t i = 0; i < nrows; i++) {
			uint32_t j = 0;

			LOOP_UNROLL();
			for (; j+nr_T_in_avx <= padded_limbs; j+= nr_T_in_avx) {
				const uint32x8_t in1_ = uint32x8_t::load(in1.row(i) + j);
				const uint32x8_t in2_ = uint32x8_t::load(in2.row(i) + j);
				const uint32x8_t out_ = in1_ ^ in2_;

				uint32x8_t::store(out.row(i) + j, out_);
			}

			for (; j < limbs; j++) {
				*(out.row(i) + j) = *(in1.row(i) + j) ^ *(in2.row(i) + j);
			}
		}
	}

	/// simple subtraction
	/// \param out output
	/// \param in1 input
	/// \param in2 input
	constexpr static void sub(FqMatrix&out,
							  const FqMatrix &in1,
							  const FqMatrix &in2) noexcept {
		add(out, in1, in2);
	}

	constexpr static FqMatrix augment(const FqMatrix &in1,
			const FqMatrix &in2) noexcept {
		/// TODO
		return in1;
	}

	/**
	 * Transpose a 64 x 64 matrix with width 1.
	 *
	 * \param dst First word of destination matrix.
	 * \param src First word of source matrix.
	 * \param rowstride_dst Rowstride of matrix dst.
	 * \param rowstride_src Rowstride of matrix src.
	 *
	 * Rows of both matrices are expected to fit exactly in a word (offset == 0)
	 * and lay entirely inside a single block.
	 *
	 * \note This function also works when dst == src.
	 */
	constexpr static inline void _mzd_copy_transpose_64x64(T *dst, 
												 T const *src, 
												 const uint32_t rowstride_dst,
	                                             const uint32_t rowstride_src) noexcept {
	   /*
	   * m runs over the values:
	   *   0x00000000FFFFFFFF
	   *   0x0000FFFF0000FFFF
	   *   0x00FF00FF00FF00FF
	   *   0x0F0F0F0F0F0F0F0F
	   *   0x3333333333333333
	   *   0x5555555555555555,
	   * alternating j zeroes with j ones.
	   *
	   * Assume we have a matrix existing of four jxj matrices ((0,0) is in the top-right corner,
	   * this is the memory-model view, see the layout on
	   * http://m4ri.sagemath.org/doxygen/structmzd__t.html):
	   * ...[A1][B1][A0][B0]
	   * ...[C1][D1][C0][D0]
	   *          . [A2][B2]
	   *        .   [C2][B2]
	   *      .         .
	   *                .
	   * The following calulates the XOR between A and D,
	   * and subsequently applies that to A and D respectively,
	   * swapping A and D as a result.
	   * Therefore wk starts at the first row and then has rowstride
	   * added j times, running over the rows of A, then skips C
	   * by adding j * rowstride to continue with the next A below C.
	   */
	
		T m = T(0xFFFFFFFF);
		uint32_t j_rowstride_dst = rowstride_dst * 64;
		uint32_t j_rowstride_src = rowstride_src * 32;
		T *const end = dst + j_rowstride_dst;
		// We start with j = 32, and a one-time unrolled loop, where
		// we copy from src and write the result to dst, swapping
		// the two 32x32 corner matrices.
		int j = 32;
		j_rowstride_dst >>= 1;
		T *__restrict__ wk = dst;
		for (T const *__restrict__ wks = src; wk < end; wk += j_rowstride_dst, wks += j_rowstride_src) {
			for (int k = 0; k < j; ++k, wk += rowstride_dst, wks += rowstride_src) {
				T xor_ = ((*wks >> j) ^ *(wks + j_rowstride_src)) & m;
				*wk = *wks ^ (xor_ << j);
				*(wk + j_rowstride_dst) = *(wks + j_rowstride_src) ^ xor_;
			}
		}
		// Next we work in-place in dst and swap the corners of
		// each of the last matrices, all in parallel, for all
		// remaining values of j.
		m ^= m << 16;
		for (j = 16; j != 0; j = j >> 1, m ^= m << j) {
			j_rowstride_dst >>= 1;
			for (wk = dst; wk < end; wk += j_rowstride_dst) {
				for (int k = 0; k < j; ++k, wk += rowstride_dst) {
					T xor_ = ((*wk >> j) ^ *(wk + j_rowstride_dst)) & m;
					*wk ^= xor_ << j;
					*(wk + j_rowstride_dst) ^= xor_;
				}
			}
		}
	}
	
	/**
	 * Transpose two 64 x 64 matrix with width 1.
	 *
	 * \param dst1 First T of destination matrix 1.
	 * \param dst2 First T of destination matrix 2.
	 * \param src1 First T of source matrix 1.
	 * \param src2 First T of source matrix 2.
	 * \param rowstride_dst Rowstride of destination matrices.
	 * \param rowstride_src Rowstride of source matrices.
	 *
	 * Rows of all matrices are expected to fit exactly in a T (offset == 0)
	 * and lay entirely inside a single block.
	 *
	 * \note This function also works to transpose in-place.
	 */
	constexpr static inline void _mzd_copy_transpose_64x64_2(T *__restrict__ dst1, 
												   T *__restrict__ dst2,
	                                               T const *__restrict__ src1, 
												   T const *__restrict__ src2,
	                                               const uint32_t rowstride_dst, 
												   const uint32_t rowstride_src) noexcept {
		T m = T(0xFFFFFFFF);
		uint32_t j_rowstride_dst = rowstride_dst * 64;
		uint32_t j_rowstride_src = rowstride_src * 32;
		T *const end = dst1 + j_rowstride_dst;
		uint32_t j = 32;
		T *__restrict__ wk[2];
		T const *__restrict__ wks[2];
		T xor_ [2];
	
		j_rowstride_dst >>= 1;
		wk[0] = dst1;
		wk[1] = dst2;
		wks[0] = src1;
		wks[1] = src2;
	
		do {
			for (uint32_t k = 0; k < j; ++k) {
				xor_[0] = ((*wks[0] >> j) ^ *(wks[0] + j_rowstride_src)) & m;
				xor_[1] = ((*wks[1] >> j) ^ *(wks[1] + j_rowstride_src)) & m;
				*wk[0] = *wks[0] ^ (xor_[0] << j);
				*wk[1] = *wks[1] ^ (xor_[1] << j);
				*(wk[0] + j_rowstride_dst) = *(wks[0] + j_rowstride_src) ^ xor_[0];
				*(wk[1] + j_rowstride_dst) = *(wks[1] + j_rowstride_src) ^ xor_[1];
				wk[0] += rowstride_dst;
				wk[1] += rowstride_dst;
				wks[0] += rowstride_src;
				wks[1] += rowstride_src;
			}
	
			wk[0] += j_rowstride_dst;
			wk[1] += j_rowstride_dst;
			wks[0] += j_rowstride_src;
			wks[1] += j_rowstride_src;
	
		} while (wk[0] < end);
	
		m ^= m << 16;
		for (j = 16; j != 0; j = j >> 1, m ^= m << j) {
	
			j_rowstride_dst >>= 1;
			wk[0] = dst1;
			wk[1] = dst2;
	
			do {
	
				for (uint32_t k = 0; k < j; ++k) {
					xor_[0] = ((*wk[0] >> j) ^ *(wk[0] + j_rowstride_dst)) & m;
					xor_[1] = ((*wk[1] >> j) ^ *(wk[1] + j_rowstride_dst)) & m;
					*wk[0] ^= xor_[0] << j;
					*wk[1] ^= xor_[1] << j;
					*(wk[0] + j_rowstride_dst) ^= xor_[0];
					*(wk[1] + j_rowstride_dst) ^= xor_[1];
					wk[0] += rowstride_dst;
					wk[1] += rowstride_dst;
				}
	
				wk[0] += j_rowstride_dst;
				wk[1] += j_rowstride_dst;
	
			} while (wk[0] < end);
		}
	}
	
	constexpr static unsigned char log2_ceil_table[64] = {
	        0, 1, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,
	        6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6};
	
	static inline int log2_ceil(int n) { return log2_ceil_table[n - 1]; }
	
	constexpr static T const transpose_mask[6] = {
	        0x5555555555555555ULL,
	        0x3333333333333333ULL,
	        0x0F0F0F0F0F0F0F0FULL,
	        0x00FF00FF00FF00FFULL,
	        0x0000FFFF0000FFFFULL,
	        0x00000000FFFFFFFFULL,
	};
	
	/**
	 * Transpose 64/j matrices of size jxj in parallel.
	 *
	 * Where j equals n rounded up to the nearest power of 2.
	 * The input array t must be of size j (containing the rows i of all matrices in t[i]).
	 *
	 * t[0..{j-1}]  = [Al]...[A1][A0]
	 *
	 * \param t An array of j Ts.
	 * \param n The number of rows in each matrix.
	 *
	 * \return log2(j)
	 */
	static inline int _mzd_transpose_Nxjx64(T *__restrict__ t,
											int n) {
		int j = 1;
		int mi = 0;// Index into the transpose_mask array.
	
		while (j < n)// Don't swap with entirely undefined data (where [D] exists entirely of
		             // non-existant rows).
		{
			// Swap 64/j matrices of size jxj in 2j rows. Thus,
			// <---- one T --->
			// [Al][Bl]...[A0][B0]
			// [Cl][Dl]...[C0][D0], where l = 64/j - 1 and each matrix [A], [B] etc is jxj.
			// Then swap [A] and [D] in-place.
	
			// m runs over the values in transpose_mask, so that at all
			// times m exists of j zeroes followed by j ones, repeated.
			T const m = transpose_mask[mi];
			int k = 0;// Index into t[].
			do {
				// Run over all rows of [A] and [D].
				for (int i = 0; i < j; ++i, ++k) {
					// t[k] contains row i of all [A], and t[k + j] contains row i of all [D]. Swap them.
					T xor_ = ((t[k] >> j) ^ t[k + j]) & m;
					t[k] ^= xor_ << j;
					t[k + j] ^= xor_;
				}
				k += j;     // Skip [C].
			} while (k < n);// Stop if we passed all valid input.
	
			// Double the size of j and repeat this for the next 2j rows until all
			// n rows have been swapped (possibly with non-existant rows).
			j <<= 1;
			++mi;
		}
	
		return mi;
	}
	
	/**
	 * Transpose a n x 64 matrix with width 1.
	 *
	 * \param dst First T of destination matrix.
	 * \param src First T of source matrix.
	 * \param rowstride_dst Rowstride of destination matrix.
	 * \param rowstride_src Rowstride of source matrix.
	 * \param n Number of rows in source matrix, must be less than 64.
	 *
	 * Rows of all matrices are expected have offset zero
	 * and lay entirely inside a single block.
	 *
	 * \note This function also works to transpose in-place.
	 */
	constexpr static inline void _mzd_copy_transpose_lt64x64(T *__restrict__ dst, 
												   T const *__restrict__ src,
	                                               uint32_t rowstride_dst,
												   uint32_t rowstride_src, 
												   int n) noexcept {
		// Preload the n input rows into level 1, using a minimum of cache lines (compact storage).
		T t[64];
		T const *__restrict__ wks = src;
		int k;
		for (k = 0; k < n; ++k) {
			t[k] = *wks;
			wks += rowstride_src;
		}
		// see https://bitbucket.org/malb/m4ri/issues/53
		for (; k < 64; ++k) { t[k] = 0; }
		if (n > 32) {
			while (k < 64) t[k++] = 0;
			_mzd_copy_transpose_64x64(dst, t, rowstride_dst, 1);
			return;
		}
		int log2j = _mzd_transpose_Nxjx64(t, n);
		// All output bits are now transposed, but still might need to be shifted in place.
		// What we have now is 64/j matrices of size jxj. Thus,
		// [Al]...[A1][A0], where l = 64/j - 1.
		// while the actual output is:
		// [A0]
		// [A1]
		// ...
		// [Al]
		T const m = (ffff >> (RADIX - (n)) % RADIX);

		T *__restrict__ wk = dst;
		switch (log2j) {
			case 5: {
				uint32_t const j_rowstride_dst = 32 * rowstride_dst;
				for (int k = 0; k < 32; ++k) {
					wk[0] = t[k] & m;
					wk[j_rowstride_dst] = (t[k] >> 32) & m;
					wk += rowstride_dst;
				}
				break;
			}
			case 4: {
				uint32_t const j_rowstride_dst = 16 * rowstride_dst;
				for (int k = 0; k < 16; ++k) {
					wk[0] = t[k] & m;
					wk[j_rowstride_dst] = (t[k] >> 16) & m;
					wk[2 * j_rowstride_dst] = (t[k] >> 32) & m;
					wk[3 * j_rowstride_dst] = (t[k] >> 48) & m;
					wk += rowstride_dst;
				}
				break;
			}
			case 3: {
				uint32_t const j_rowstride_dst = 8 * rowstride_dst;
				for (int k = 0; k < 8; ++k) {
					wk[0] = t[k] & m;
					wk[j_rowstride_dst] = (t[k] >> 8) & m;
					wk[2 * j_rowstride_dst] = (t[k] >> 16) & m;
					wk[3 * j_rowstride_dst] = (t[k] >> 24) & m;
					wk[4 * j_rowstride_dst] = (t[k] >> 32) & m;
					wk[5 * j_rowstride_dst] = (t[k] >> 40) & m;
					wk[6 * j_rowstride_dst] = (t[k] >> 48) & m;
					wk[7 * j_rowstride_dst] = (t[k] >> 56) & m;
					wk += rowstride_dst;
				}
				break;
			}
			case 2: {
				uint32_t const j_rowstride_dst = 4 * rowstride_dst;
				for (int k = 0; k < 4; ++k) {
					T *__restrict__ wk2 = wk;
					T tk = t[k];
					for (int i = 0; i < 2; ++i) {
						wk2[0] = tk & m;
						wk2[j_rowstride_dst] = (tk >> 4) & m;
						wk2[2 * j_rowstride_dst] = (tk >> 8) & m;
						wk2[3 * j_rowstride_dst] = (tk >> 12) & m;
						wk2[4 * j_rowstride_dst] = (tk >> 16) & m;
						wk2[5 * j_rowstride_dst] = (tk >> 20) & m;
						wk2[6 * j_rowstride_dst] = (tk >> 24) & m;
						wk2[7 * j_rowstride_dst] = (tk >> 28) & m;
						wk2 += 8 * j_rowstride_dst;
						tk >>= 32;
					}
					wk += rowstride_dst;
				}
				break;
			}
			case 1: {
				uint32_t const j_rowstride_dst = 2 * rowstride_dst;
				for (int k = 0; k < 2; ++k) {
					T *__restrict__ wk2 = wk;
					T tk = t[k];
					for (int i = 0; i < 8; ++i) {
						wk2[0] = tk & m;
						wk2[j_rowstride_dst] = (tk >> 2) & m;
						wk2[2 * j_rowstride_dst] = (tk >> 4) & m;
						wk2[3 * j_rowstride_dst] = (tk >> 6) & m;
						wk2 += 4 * j_rowstride_dst;
						tk >>= 8;
					}
					wk += rowstride_dst;
				}
				break;
			}
			case 0: {
				T *__restrict__ wk2 = wk;
				T tk = t[0];
				for (int i = 0; i < 16; ++i) {
					wk2[0] = tk & m;
					wk2[rowstride_dst] = (tk >> 1) & m;
					wk2[2 * rowstride_dst] = (tk >> 2) & m;
					wk2[3 * rowstride_dst] = (tk >> 3) & m;
					wk2 += 4 * rowstride_dst;
					tk >>= 4;
				}
				break;
			}
		}
	}
	
	/**
	 * Transpose a 64 x n matrix with width 1.
	 *
	 * \param dst First T of destination matrix.
	 * \param src First T of source matrix.
	 * \param rowstride_dst Rowstride of destination matrix.
	 * \param rowstride_src Rowstride of source matrix.
	 * \param n Number of columns in source matrix, must be less than 64.
	 *
	 * Rows of all matrices are expected have offset zero
	 * and lay entirely inside a single block.
	 *
	 * \note This function also works to transpose in-place.
	 */
	constexpr static inline void _mzd_copy_transpose_64xlt64(T *__restrict__ dst,
												   T const *__restrict__ src,
	                                               uint32_t rowstride_dst, 
												   uint32_t rowstride_src, 
												   int n) noexcept {
		T t[64];
		int log2j = log2_ceil(n);
		T const *__restrict__ wks = src;
		switch (log2j) {
			case 6: {
				_mzd_copy_transpose_64x64(t, src, 1, rowstride_src);
				T *__restrict__ wk = dst;
				for (int k = 0; k < n; ++k) {
					*wk = t[k];
					wk += rowstride_dst;
				}
				return;
			}
			case 5: {
				uint32_t const j_rowstride_src = 32 * rowstride_src;
				for (int k = 0; k < 32; ++k) {
					t[k] = wks[0] | (wks[j_rowstride_src] << 32);
					wks += rowstride_src;
				}
				break;
			}
			case 4: {
				uint32_t const j_rowstride_src = 16 * rowstride_src;
				for (int k = 0; k < 16; ++k) {
					t[k] = wks[0] | (wks[j_rowstride_src] << 16);
					t[k] |= (wks[2 * j_rowstride_src] << 32) | (wks[3 * j_rowstride_src] << 48);
					wks += rowstride_src;
				}
				break;
			}
			case 3: {
				uint32_t const j_rowstride_src = 8 * rowstride_src;
				T tt;
				for (int k = 0; k < 8; ++k) {
					tt = wks[0] | (wks[j_rowstride_src] << 8);
					t[k] = (wks[2 * j_rowstride_src] << 16) | (wks[3 * j_rowstride_src] << 24);
					tt |= (wks[4 * j_rowstride_src] << 32) | (wks[5 * j_rowstride_src] << 40);
					t[k] |= (wks[6 * j_rowstride_src] << 48) | (wks[7 * j_rowstride_src] << 56);
					wks += rowstride_src;
					t[k] |= tt;
				}
				break;
			}
			case 2: {
				T const *__restrict__ wks2 = wks + 60 * rowstride_src;
				t[0] = wks2[0];
				t[1] = wks2[rowstride_src];
				t[2] = wks2[2 * rowstride_src];
				t[3] = wks2[3 * rowstride_src];
				for (int i = 0; i < 15; ++i) {
					wks2 -= 4 * rowstride_src;
					t[0] <<= 4;
					t[1] <<= 4;
					t[2] <<= 4;
					t[3] <<= 4;
					t[0] |= wks2[0];
					t[1] |= wks2[rowstride_src];
					t[2] |= wks2[2 * rowstride_src];
					t[3] |= wks2[3 * rowstride_src];
				}
				break;
			}
			case 1: {
				wks += 62 * rowstride_src;
				t[0] = wks[0];
				t[1] = wks[rowstride_src];
				for (int i = 0; i < 31; ++i) {
					wks -= 2 * rowstride_src;
					t[0] <<= 2;
					t[1] <<= 2;
					t[0] |= wks[0];
					t[1] |= wks[rowstride_src];
				}
				break;
			}
			case 0: {
				T tt[2];
				tt[0] = wks[0];
				tt[1] = wks[rowstride_src];
				for (int i = 2; i < 64; i += 2) {
					wks += 2 * rowstride_src;
					tt[0] |= wks[0] << i;
					tt[1] |= wks[rowstride_src] << i;
				}
				*dst = tt[0] | (tt[1] << 1);
				return;
			}
		}
		int j = 1 << log2j;
		_mzd_transpose_Nxjx64(t, j);
		T *__restrict__ wk = dst;
		for (int k = 0; k < n; ++k) {
			*wk = t[k];
			wk += rowstride_dst;
		}
	}
	
	/**
	 * Transpose a n x m matrix with width 1, offset 0 and m and n less than or equal 8.
	 *
	 * \param dst First T of destination matrix.
	 * \param src First T of source matrix.
	 * \param rowstride_dst Rowstride of destination matrix.
	 * \param rowstride_src Rowstride of source matrix.
	 * \param n Number of rows in source matrix, must be less than or equal 8.
	 * \param m Number of columns in source matrix, must be less than or equal 8.
	 *
	 * Rows of all matrices are expected to have offset zero
	 * and lay entirely inside a single block.
	 *
	 * \note This function also works to transpose in-place.
	 */
	constexpr static inline void _mzd_copy_transpose_le8xle8(T *__restrict__ dst, 
												   T const *__restrict__ src,
	                                               uint32_t rowstride_dst,
												   uint32_t rowstride_src,
												   int n,
												   int m,
	                                               int maxsize) noexcept {
		int end = maxsize * 7;
		T const *__restrict__ wks = src;
		T w = *wks;
		int shift = 0;
		for (int i = 1; i < n; ++i) {
			wks += rowstride_src;
			shift += 8;
			w |= (*wks << shift);
		}
		T mask = 0x80402010080402ULL;
		T w7 = w >> 7;
		shift = 7;
		--m;
		do {
			T xor_ = (w ^ w7) & mask;
			mask >>= 8;
			w ^= (xor_ << shift);
			shift += 7;
			w7 >>= 7;
			w ^= xor_;
		} while (shift < end);
		T *__restrict__ wk = dst + m * rowstride_dst;
		for (int shift = 8 * m; shift > 0; shift -= 8) {
			*wk = (unsigned char) (w >> shift);
			wk -= rowstride_dst;
		}
		*wk = (unsigned char) w;
	}
	
	/**
	 * Transpose a n x m matrix with width 1, offset 0 and m and n less than or equal 16.
	 *
	 * \param dst First T of destination matrix.
	 * \param src First T of source matrix.
	 * \param rowstride_dst Rowstride of destination matrix.
	 * \param rowstride_src Rowstride of source matrix.
	 * \param n Number of rows in source matrix, must be less than or equal 16.
	 * \param m Number of columns in source matrix, must be less than or equal 16.
	 *
	 * Rows of all matrices are expected to have offset zero
	 * and lay entirely inside a single block.
	 *
	 * \note This function also works to transpose in-place.
	 */
	constexpr static inline void _mzd_copy_transpose_le16xle16(T *__restrict__ dst,
													 T const *__restrict__ src,
	                                                 uint32_t rowstride_dst,
													 uint32_t rowstride_src, 
													 int n,
	                                                 int m, 
													 int maxsize) noexcept{
		int end = maxsize * 3;
		T const *__restrict__ wks = src;
		T t[4];
		int i = n;
		do {
			t[0] = wks[0];
			if (--i == 0) {
				t[1] = 0;
				t[2] = 0;
				t[3] = 0;
				break;
			}
			t[1] = wks[rowstride_src];
			if (--i == 0) {
				t[2] = 0;
				t[3] = 0;
				break;
			}
			t[2] = wks[2 * rowstride_src];
			if (--i == 0) {
				t[3] = 0;
				break;
			}
			t[3] = wks[3 * rowstride_src];
			if (--i == 0) break;
			wks += 4 * rowstride_src;
			for (int shift = 16;; shift += 16) {
				t[0] |= (*wks << shift);
				if (--i == 0) break;
				t[1] |= (wks[rowstride_src] << shift);
				if (--i == 0) break;
				t[2] |= (wks[2 * rowstride_src] << shift);
				if (--i == 0) break;
				t[3] |= (wks[3 * rowstride_src] << shift);
				if (--i == 0) break;
				wks += 4 * rowstride_src;
			}
		} while (0);
		T mask = 0xF0000F0000F0ULL;
		int shift = 12;
		T xor_ [4];
		do {
			xor_[0] = (t[0] ^ (t[0] >> shift)) & mask;
			xor_[1] = (t[1] ^ (t[1] >> shift)) & mask;
			xor_[2] = (t[2] ^ (t[2] >> shift)) & mask;
			xor_[3] = (t[3] ^ (t[3] >> shift)) & mask;
			mask >>= 16;
			t[0] ^= (xor_[0] << shift);
			t[1] ^= (xor_[1] << shift);
			t[2] ^= (xor_[2] << shift);
			t[3] ^= (xor_[3] << shift);
			shift += 12;
			t[0] ^= xor_[0];
			t[1] ^= xor_[1];
			t[2] ^= xor_[2];
			t[3] ^= xor_[3];
		} while (shift < end);
		_mzd_transpose_Nxjx64(t, 4);
		i = m;
		T *__restrict__ wk = dst;
		do {
			wk[0] = (uint16_t) t[0];
			if (--i == 0) break;
			wk[rowstride_dst] = (uint16_t) t[1];
			if (--i == 0) break;
			wk[2 * rowstride_dst] = (uint16_t) t[2];
			if (--i == 0) break;
			wk[3 * rowstride_dst] = (uint16_t) t[3];
			if (--i == 0) break;
			wk += 4 * rowstride_dst;
			for (int shift = 16;; shift += 16) {
				wk[0] = (uint16_t) (t[0] >> shift);
				if (--i == 0) break;
				wk[rowstride_dst] = (uint16_t) (t[1] >> shift);
				if (--i == 0) break;
				wk[2 * rowstride_dst] = (uint16_t) (t[2] >> shift);
				if (--i == 0) break;
				wk[3 * rowstride_dst] = (uint16_t) (t[3] >> shift);
				if (--i == 0) break;
				wk += 4 * rowstride_dst;
			}
		} while (0);
	}
	
	/**
	 * Transpose a n x m matrix with width 1, offset 0 and m and n less than or equal 32.
	 *
	 * \param dst First T of destination matrix.
	 * \param src First T of source matrix.
	 * \param rowstride_dst Rowstride of destination matrix.
	 * \param rowstride_src Rowstride of source matrix.
	 * \param n Number of rows in source matrix, must be less than or equal 32.
	 * \param m Number of columns in source matrix, must be less than or equal 32.
	 *
	 * Rows of all matrices are expected to have offset zero
	 * and lay entirely inside a single block.
	 *
	 * \note This function also works to transpose in-place.
	 */
	constexpr static inline void _mzd_copy_transpose_le32xle32(T *__restrict__ dst, 
													 T const *__restrict__ src,
	                                                 uint32_t rowstride_dst,
													 uint32_t rowstride_src,
													 int n,
	                                                 int m) noexcept {
		T const *__restrict__ wks = src;
		T t[16];
		int i = n;
		if (n > 16) {
			i -= 16;
			for (int j = 0; j < 16; ++j) {
				t[j] = *wks;
				wks += rowstride_src;
			}
			int j = 0;
			do {
				t[j++] |= (*wks << 32);
				wks += rowstride_src;
			} while (--i);
		} else {
			int j;
			for (j = 0; j < n; ++j) {
				t[j] = *wks;
				wks += rowstride_src;
			}
			for (; j < 16; ++j) t[j] = 0;
		}
		_mzd_transpose_Nxjx64(t, 16);
		int one_more = (m & 1);
		T *__restrict__ wk = dst;
		if (m > 16) {
			m -= 16;
			for (int j = 0; j < 16; j += 2) {
				*wk = (t[j] & 0xFFFF) | ((t[j] >> 16) & 0xFFFF0000);
				wk[rowstride_dst] = (t[j + 1] & 0xFFFF) | ((t[j + 1] >> 16) & 0xFFFF0000);
				wk += 2 * rowstride_dst;
			}
			for (int j = 1; j < m; j += 2) {
				*wk = ((t[j - 1] >> 16) & 0xFFFF) | ((t[j - 1] >> 32) & 0xFFFF0000);
				wk[rowstride_dst] = ((t[j] >> 16) & 0xFFFF) | ((t[j] >> 32) & 0xFFFF0000);
				wk += 2 * rowstride_dst;
			}
			if (one_more) { *wk = ((t[m - 1] >> 16) & 0xFFFF) | ((t[m - 1] >> 32) & 0xFFFF0000); }
		} else {
			for (int j = 1; j < m; j += 2) {
				*wk = (t[j - 1] & 0xFFFF) | ((t[j - 1] >> 16) & 0xFFFF0000);
				wk[rowstride_dst] = (t[j] & 0xFFFF) | ((t[j] >> 16) & 0xFFFF0000);
				wk += 2 * rowstride_dst;
			}
			if (one_more) { *wk = (t[m - 1] & 0xFFFF) | ((t[m - 1] >> 16) & 0xFFFF0000); }
		}
	}
	
	constexpr static inline void _mzd_copy_transpose_le64xle64(T *__restrict__ dst,
													 T const *__restrict__ src,
	                                                 const uint32_t rowstride_dst,
													 const uint32_t rowstride_src,
													 const int n,
	                                                 const int m) noexcept {
		T const *__restrict__ wks = src;
		T t[64];
		int k;
		for (k = 0; k < n; ++k) {
			t[k] = *wks;
			wks += rowstride_src;
		}
		while (k < 64) t[k++] = 0;
		_mzd_copy_transpose_64x64(t, t, 1, 1);
		T *__restrict__ wk = dst;
		for (int k = 0; k < m; ++k) {
			*wk = t[k];
			wk += rowstride_dst;
		}
		return;
	}
	
	constexpr static inline void _mzd_copy_transpose_small(T *__restrict__ fwd,
												 T const *__restrict__ fws,
	                                             const uint32_t rowstride_dst,
												 const uint32_t rowstride_src,
												 const uint32_t _nrows,
	                                             const uint32_t _ncols,
												 const uint32_t maxsize) noexcept {
		assert(maxsize < 64);
		if (maxsize <= 8) {
			_mzd_copy_transpose_le8xle8(fwd, fws, rowstride_dst, rowstride_src, _nrows, _ncols, maxsize);
		} else if (maxsize <= 16) {
			_mzd_copy_transpose_le16xle16(fwd, fws, rowstride_dst, rowstride_src, _nrows, _ncols, maxsize);
		} else if (maxsize <= 32) {
			_mzd_copy_transpose_le32xle32(fwd, fws, rowstride_dst, rowstride_src, _nrows, _ncols);
		} else {
			_mzd_copy_transpose_le64xle64(fwd, fws, rowstride_dst, rowstride_src, _nrows, _ncols);
		}
	}
	
	
	constexpr static void _mzd_transpose_base(T *__restrict__ fwd, T
							 const *__restrict__ fws,
							 const uint32_t rowstride_dst,
	                         const uint32_t rowstride_src,
							 uint32_t _nrows,
							 uint32_t _ncols,
							 uint32_t maxsize) noexcept {
		assert(maxsize >= 64);
		// Note that this code is VERY sensitive. ANY change to _mzd_transpose can easily
		// reduce the speed for small matrices (up to 64x64) by 5 to 10%.
		if (_nrows >= 64) {
			/*
	       * This is an interesting #if ...
	       * I recommend to investigate the number of instructions, and the clocks per instruction,
	       * as function of various sizes of the matrix (most likely especially the number of columns
	       * (the size of a row) will have influence; also always use multiples of 64 or even 128),
	       * for both cases below.
	       *
	       * To measure this run for example:
	       *
	       * ./bench_mzd -m 10 -x 10 -p PAPI_TOT_INS,PAPI_L1_TCM,PAPI_L2_TCM mzd_transpose 32000 32000
	       * ./bench_mzd -m 10 -x 100 -p PAPI_TOT_INS,PAPI_L1_TCM,PAPI_L2_TCM mzd_transpose 128 10240
	       * etc (increase -x for smaller sizes to get better accuracy).
	       *
	       * --Carlo Wood
	       */
	#if 1
			int js = _ncols & _nrows & 64;// True if the total number of whole 64x64 matrices is odd.
			uint32_t const rowstride_64_dst = 64 * rowstride_dst;
			T *__restrict__ fwd_current = fwd;
			T const *__restrict__ fws_current = fws;
			if (js) {
				js = 1;
				_mzd_copy_transpose_64x64(fwd, fws, rowstride_dst, rowstride_src);
				if ((_nrows | _ncols) == 64) {
					return;
				}
				fwd_current += rowstride_64_dst;
				++fws_current;
			}
			uint32_t const whole_64cols = _ncols / 64;
			// The use of delayed and even, is to avoid calling _mzd_copy_transpose_64x64_2 twice.
			// This way it can be inlined without duplicating the amount of code that has to be loaded.
			T *__restrict__ fwd_delayed = NULL;
			T const *__restrict__ fws_delayed = NULL;
			int even = 0;
			while (1) {
				for (uint32_t j = js; j < whole_64cols; ++j) {
					if (!even) {
						fwd_delayed = fwd_current;
						fws_delayed = fws_current;
					} else {
						_mzd_copy_transpose_64x64_2(fwd_delayed, fwd_current, fws_delayed, fws_current,
						                            rowstride_dst, rowstride_src);
					}
					fwd_current += rowstride_64_dst;
					++fws_current;
					even = !even;
				}
				_nrows -= 64;
				if (_ncols % 64) {
					_mzd_copy_transpose_64xlt64(fwd + whole_64cols * rowstride_64_dst, fws + whole_64cols,
					                            rowstride_dst, rowstride_src, ncols % 64);
				}
				fwd += 1;
				fws += 64 * rowstride_src;
				if (_nrows < 64) break;
				js = 0;
				fws_current = fws;
				fwd_current = fwd;
			}
	#else
			// The same as the above, but without using _mzd_copy_transpose_64x64_2.
			uint32_t const rowstride_64_dst = 64 * DST->rowstride;
			uint32_t const whole_64cols = ncols / 64;
			assert(nrows >= 64);
			do {
				for (int j = 0; j < whole_64cols; ++j) {
					_mzd_copy_transpose_64x64(fwd + j * rowstride_64_dst, fws + j, DST->rowstride,
					                          A->rowstride);
				}
				nrows -= 64;
				if (ncols % 64) {
					_mzd_copy_transpose_64xlt64(fwd + whole_64cols * rowstride_64_dst, fws + whole_64cols,
					                            DST->rowstride, A->rowstride, ncols % 64);
				}
				fwd += 1;
				fws += 64 * A->rowstride;
			} while (nrows >= 64);
	#endif
		}
	
		if (_nrows == 0) {
			return;
		}
	
		// Transpose the remaining top rows. Now 0 < nrows < 64.
	
		while (ncols >= 64) {
			_mzd_copy_transpose_lt64x64(fwd, fws, rowstride_dst, rowstride_src, _nrows);
			_ncols -= 64;
			fwd += 64 * rowstride_dst;
			fws += 1;
		}
	
		if (ncols == 0) {
			return;
		}
	
		maxsize = std::max(_nrows, _ncols);
	
		// Transpose the remaining corner. Now both 0 < nrows < 64 and 0 < ncols < 64.
		_mzd_copy_transpose_small(fwd, fws, rowstride_dst, rowstride_src, nrows, ncols, maxsize);
	}
	
	/* return the smallest multiple of k larger than n/2 */
	constexpr static inline uint32_t split_round(uint32_t n, uint32_t k) noexcept {
		uint32_t half = n / 2;
		return ((half + (k - 1)) / k) * k;
	}
	
	constexpr static void _mzd_transpose_notsmall(T *__restrict__ fwd, 
										T const *__restrict__ fws,
										const uint32_t rowstride_dst,
	                                    const uint32_t rowstride_src,
										uint32_t _nrows,
										uint32_t _ncols,
										uint32_t maxsize) noexcept {
		assert(maxsize >= 64);
	
		if (maxsize <= 512) {// just one big block
			_mzd_transpose_base(fwd, fws, rowstride_dst, rowstride_src, nrows, ncols, maxsize);
		} else {
			uint32_t large_size = split_round(maxsize, (maxsize <= 768) ? 64 : 512);
			uint32_t offset = large_size / RADIX;
			if (_nrows >= _ncols) {
				T const *__restrict__ fws_up = fws;
				T const *__restrict__ fws_down = fws + large_size * rowstride_src;
				T *__restrict__ fwd_left = fwd;
				T *__restrict__ fwd_right = fwd + offset;
				uint32_t maxsize_up = std::max(large_size, _ncols);
				uint32_t maxsize_down = std::max(nrows - large_size, _ncols);
				_mzd_transpose_notsmall(fwd_left, fws_up, rowstride_dst, rowstride_src, large_size, _ncols, maxsize_up);
				_mzd_transpose_notsmall(fwd_right, fws_down, rowstride_dst, rowstride_src, _nrows - large_size, _ncols, maxsize_down);
			} else {
				T const *__restrict__ fws_left = fws;
				T const *__restrict__ fws_right = fws + offset;
				T *__restrict__ fwd_up = fwd;
				T *__restrict__ fwd_down = fwd + large_size * rowstride_dst;
				uint32_t maxsize_left = std::max(nrows, large_size);
				uint32_t maxsize_right = std::max(nrows, ncols - large_size);
				_mzd_transpose_notsmall(fwd_up, fws_left, rowstride_dst, rowstride_src, _nrows, large_size, maxsize_left);
				_mzd_transpose_notsmall(fwd_down, fws_right, rowstride_dst, rowstride_src, _nrows, _ncols - large_size, maxsize_right);
			}
		}
	}
	
	constexpr static void _mzd_transpose(T *__restrict__ fwd, 
							   T const *__restrict__ fws, 
							   const uint32_t rowstride_dst,
	                           const uint32_t rowstride_src) noexcept {
		// rationale: small blocks corresponds to the T size
		//            two big blocks fit in L1 cache (512 --> 8KB).
	
		constexpr uint32_t maxsize = std::max(nrows, ncols);
		if constexpr (maxsize < 64) {// super-fast path for very small matrices
			_mzd_copy_transpose_small(fwd, fws, rowstride_dst, rowstride_src, nrows, ncols, maxsize);
		} else {
			_mzd_transpose_notsmall(fwd, fws, rowstride_dst, rowstride_src, nrows, ncols, maxsize);
		}
	}
	

	/// direct transpose of the full matrix
	/// NOTE: no expansion is possible
	/// \param B output
	/// \param A input
	constexpr static void transpose(FqMatrix<T, ncols, nrows, q> &B,
									FqMatrix<T, nrows, ncols, q> &A) noexcept {

		if constexpr (sizeof(T) == 8) {
			_mzd_transpose(B.__data.data(), A.__data.data(),
						  B.padded_limbs, A.padded_limbs);
			return;
		}

		for (uint32_t i = 0; i < nrows; ++i) {
			for (uint32_t j = 0; j < ncols; ++j) {
				const bool data = A.get(i, j);
				B.set(data, j, i);
			}
		}
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


		if constexpr (sizeof(T) == 8) {
			_mzd_transpose(B.__data.data(), A.__data.data(),
						   B.padded_limbs, A.padded_limbs);
			return;
		}

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

	/**
	 * \brief Swap the two columns cola and colb but only between start_row and stop_row.
	 *
	 * \param M Matrix.
	 * \param cola Column index.
	 * \param colb Column index.
	 * \param start_row Row index.
	 * \param stop_row Row index (exclusive).
	 */
	static inline void mzd_col_swap_in_rows(FqMatrix &M, 
											uint32_t const cola, 
											uint32_t const colb,
	                                        uint32_t const start_row, 
											uint32_t const stop_row) {
		if (cola == colb) { return; }
		uint32_t const _cola = cola;
		uint32_t const _colb = colb;
	
		uint64_t const a_word = _cola / RADIX;
		uint64_t const b_word = _colb / RADIX;
	
		int const a_bit = _cola % RADIX;
		int const b_bit = _colb % RADIX;
	
		T *__restrict__ ptr = M.row(start_row);
		int max_bit = std::max(a_bit, b_bit);
		int count_remaining = stop_row - start_row;
		int min_bit = a_bit + b_bit - max_bit;
		int offset = max_bit - min_bit;
		T mask = one << min_bit;
		int count = count_remaining;
	
		// Apparently we're calling with start_row == stop_row sometimes (seems a bug to me).
		if (count <= 0) { return; }
	
		if (a_word == b_word) {
			while (1) {
				count_remaining -= count;
				ASSERT(count_remaining == 0);
				ptr += a_word;
				int fast_count = count / 4;
				int rest_count = count - 4 * fast_count;
				T xor_v[4];
				uint32_t const rowstride = padded_limbs;
				while (fast_count--) {
					xor_v[0] = ptr[0];
					xor_v[1] = ptr[rowstride];
					xor_v[2] = ptr[2 * rowstride];
					xor_v[3] = ptr[3 * rowstride];
					xor_v[0] ^= xor_v[0] >> offset;
					xor_v[1] ^= xor_v[1] >> offset;
					xor_v[2] ^= xor_v[2] >> offset;
					xor_v[3] ^= xor_v[3] >> offset;
					xor_v[0] &= mask;
					xor_v[1] &= mask;
					xor_v[2] &= mask;
					xor_v[3] &= mask;
					xor_v[0] |= xor_v[0] << offset;
					xor_v[1] |= xor_v[1] << offset;
					xor_v[2] |= xor_v[2] << offset;
					xor_v[3] |= xor_v[3] << offset;
					ptr[0] ^= xor_v[0];
					ptr[rowstride] ^= xor_v[1];
					ptr[2 * rowstride] ^= xor_v[2];
					ptr[3 * rowstride] ^= xor_v[3];
					ptr += 4 * rowstride;
				}
				while (rest_count--) {
					T xor_v = *ptr;
					xor_v ^= xor_v >> offset;
					xor_v &= mask;
					*ptr ^= xor_v | (xor_v << offset);
					ptr += rowstride;
				}
				break;
			}
		} else {
			T *__restrict__ min_ptr;
			uint64_t max_offset;
			if (min_bit == a_bit) {
				min_ptr = ptr + a_word;
				max_offset = b_word - a_word;
			} else {
				min_ptr = ptr + b_word;
				max_offset = a_word - b_word;
			}
			while (1) {
				count_remaining -= count;
				assert(count_remaining == 0);
				uint32_t const rowstride = padded_limbs;
				while (count--) {
					T xor_v = (min_ptr[0] ^ (min_ptr[max_offset] >> offset)) & mask;
					min_ptr[0] ^= xor_v;
					min_ptr[max_offset] ^= xor_v << offset;
					min_ptr += rowstride;
				}
				break;
			}
		}
	}

	/// swap to columns
	/// \param i column 1
	/// \param j column 2
	constexpr inline void swap_cols(const uint16_t i, const uint16_t j) noexcept {
  		mzd_col_swap_in_rows(*this, i, j, 0, nrows);
	}

	/**
	 * \brief Swap the two rows rowa and rowb starting at startblock.
	 *
	 * \param M Matrix with a zero offset.
	 * \param rowa Row index.
	 * \param rowb Row index.
	 * \param startblock Start swapping only in this block.
	 */
	inline void _row_swap(uint32_t const rowa,
									 uint32_t const rowb,
	                                 uint32_t const startblock) noexcept {
	  if ((rowa == rowb) || (startblock >= padded_limbs)) { return; }
	
	  uint32_t width = limbs - startblock - 1;
	  T *a    = row(rowa) + startblock;
	  T *b    = row(rowb) + startblock;
	  T tmp;
	
	  for (uint32_t i = 0; i < width; ++i) {
	    tmp  = a[i];
	    a[i] = b[i];
	    b[i] = tmp;
	  }

	  tmp = (a[width] ^ b[width]) & high_bitmask;
	  a[width] ^= tmp;
	  b[width] ^= tmp;
	}

	/// swap rows
	/// \param i first row
	/// \param j second row
	constexpr static inline void swap_rows(T *out,
	                                       const uint16_t i,
	                                       const uint16_t j) noexcept {
		ASSERT(nrows > i && nrows > j);
		constexpr uint32_t CTR = alignment/RADIX;
		uint32_t l = 0;

		LOOP_UNROLL()
		for (; l+CTR <= padded_limbs; l+=CTR) {
			const uint8x32_t x_avx = uint8x32_t::load(out + i*padded_limbs + l);
			const uint8x32_t y_avx = uint8x32_t::load(out + j*padded_limbs + l);
			uint8x32_t::store(out + i*padded_limbs + l, y_avx);
			uint8x32_t::store(out + j*padded_limbs + l, x_avx);
		}

		for (; l < limbs; ++l) {
			const T tmp = out[j*padded_limbs + l];
			out[j*padded_limbs + l] = out[i*padded_limbs + l];
			out[i*padded_limbs + l] = tmp;
		}
	}

	constexpr static inline void swap_rows(FqMatrix &A,
	                                       const uint16_t i,
										   const uint16_t j) noexcept {
		swap_rows(A.__data.data(), i, j);
	}
	constexpr inline void swap_rows(const uint16_t i,
										   const uint16_t j) noexcept {
		swap_rows(__data.data(), i, j);
	}

	/// optimized version to which you have additionally pass the transposed.
	/// So its not created/freed every time
	/// \param A input matrix which should be randomly permuted
	/// \param AT
	/// \param P
	void create_random_permutation(FqMatrix<T, ncols, nrows, 2> &A,
								   FqMatrix<T, nrows, ncols, 2> &AT,
	                               Permutation &P) noexcept {
		transpose(AT, A);

		// dont permute the last column since it is the syndrome
		for (uint32_t i = 0; i < uint32_t(P.length-1); ++i) {
			uint64_t pos = fastrandombytes_uint64() % (P.length - i);

			ASSERT(i+pos < uint32_t(P->length));
			std::swap(P.values[i], P.values[i+pos]);
			swap_rows(AT, i, i+pos);
		}
		
		transpose(A, AT);
	}


	/// create A with `mzp_t *A = mzp_init(length)
	/// input permutation `A` should have initilaised with: `for(int i = 0; i < A->length; i++) A->value[i] = i` or something.
	/// \param A
	/// \param P
	void matrix_create_random_permutation(FqMatrix &A, Permutation *__restrict__ P) noexcept {
		FqMatrix<T, nrows, ncols, 2> AT;
		create_random_permutation(A, AT, P);
	}
	///
	/// \param AT
	/// \param permutation
	/// \param len
	/// \return
	constexpr inline void permute_cols(FqMatrix<T, ncols, nrows, q> &AT,
								uint32_t *permutation,
								const uint32_t len) noexcept {
		uint64_t data[2] = {0};
		Permutation *P = (Permutation *)data;
		P->length = len;
		P->values = permutation;
		create_random_permutation(*this, AT, P);
	}

	///
	/// \param M
	/// \param r
	/// \param c
	/// \param rows
	/// \param cols
	/// \param k
	/// \return
	constexpr static size_t matrix_gauss_submatrix(FqMatrix &M,
	                              		const size_t r,
	                              		const size_t c,
	                              		const size_t rows,
	                              		const size_t k) noexcept {
		size_t start_row = r, j;
		for (j = c; j < c + k; ++j) {
			int found = 0;
			for (size_t i = start_row; i < rows; ++i) {
				for (size_t l = 0; l < j - c; ++l) {
					/// check if pivot found
					if ((M[i][(c + l) / RADIX] >> ((c + l) % RADIX)) & 1u) {
						xor_avx1_new((uint8_t *) M[r + l],
						             (uint8_t *) M[i],
						             (uint8_t *) M[i],
						             padded_simd_limbs);
					}
				}

				if ((M[i][j / RADIX] >> (j % RADIX)) & 1) {
					M.swap_rows(i, start_row);
					for (size_t l = r; l < start_row; ++l) {
						if ((M.row(l)[j / RADIX] >> (j % RADIX)) & 1) {
							xor_avx1_new((uint8_t *) M.row(start_row),
							             (uint8_t *) M.row(l),
							             (uint8_t *) M.row(l),
							             padded_simd_limbs);
						}
					}
					++start_row;
					found = 1;
					break;
				}
			}

			if (found == 0) {
				break;
			}
		}
		return j - c;
	}

	/// specialised gaus which looks ahead in the l window
	/// \param M
	/// \param r
	/// \param c
	/// \param rows
	/// \param cols
	/// \param rstop
	/// \param lwin_start
	/// \param k
	/// \return
	constexpr static size_t matrix_gauss_submatrix_opt(FqMatrix &M,
	                                         const size_t r,
	                                         const size_t c,
	                                         const size_t rows,
	                                         const size_t cols,
	                                         const size_t rstop,
	                                         const size_t lwin_start,
	                                         const size_t k) noexcept {

		(void)cols;
		size_t start_row = r, j;

		// for each column
		for (j = c; j < c + k; ++j) {
			int found = 0;
			for (size_t i = start_row; i < rstop; ++i) {
				for (size_t l = 0; l < j - c; ++l) {
					if ((M.row(i)[(c + l) / RADIX] >> ((c + l) % RADIX)) & 1) {
						xor_avx1_new((uint8_t *) M.row(r + l),
						             (uint8_t *) M.row(i),
						             (uint8_t *) M.row(i),
						             padded_bytes);
					}
				}

				if ((M.row(i)[j / RADIX] >> (j % RADIX)) & 1) {
					matrix_swap_rows_new(M, i, start_row);
					for (size_t l = r; l < start_row; ++l) {
						if ((M.row(l)[j / RADIX] >> (j % RADIX)) & 1) {
							xor_avx1_new((uint8_t *) M.row(start_row),
							             (uint8_t *) M.row(l),
							             (uint8_t *) M.row(l),
							             padded_bytes);
						}
					}

					++start_row;
					found = 1;
					break;
				}
			}

			// second try: find the pivot element in the l window
			if (found == 0) {
				for (size_t i = lwin_start; i < rows; ++i) {
					for (size_t l = 0; l < j - c; ++l) {
						if ((M.row(i)[(c + l) / RADIX] >> ((c + l) % RADIX)) & 1) {
							xor_avx1_new((uint8_t *) M.row(r + l),
							             (uint8_t *) M.row(i),
							             (uint8_t *) M.row(i),
							             padded_bytes);
						}
					}

					if ((M.row(i)[j / RADIX] >> (j % RADIX)) & 1) {
						matrix_swap_rows_new(M, i, start_row);
						for (size_t l = r; l < start_row; ++l) {
							if ((M.row(l)[j / RADIX] >> (j % RADIX)) & 1) {
								xor_avx1_new((uint8_t *) M.row(start_row),
								             (uint8_t *) M.row(l),
								             (uint8_t *) M.row(l),
								             padded_bytes);
							}
						}

						++start_row;
						found = 1;
						break;
					}
				}
			}

			// okk we really couldn't find a pivot element
			if (found == 0) {
				break;
			}
		}

		return j - c;
	}

	///
	/// \param M
	/// \param r
	/// \param cols
	/// \param k
	/// \param T
	/// \param diff
	constexpr static void matrix_make_table(FqMatrix &M,
	                       const size_t r,
	                       const size_t k,
	                       T *Table,
	                       uint32_t **diff) noexcept {
		for (size_t i = 0; i < padded_limbs; ++i) {
			Table[i] = 0L;
		}

		for (size_t i = 0; i + 1 < 1UL << k; ++i) {
			xor_avx1_new((uint8_t *)M[r + diff[k][i]],
			             (uint8_t *)Table,
			             (uint8_t *)(Table + padded_limbs),
			             padded_simd_limbs);
			Table += padded_limbs;
		}
	}

	/// \param M input matrix
	/// \param x
	/// \param y
	/// \param nn
	/// \return
	constexpr static uint64_t matrix_read_bits(FqMatrix &M,
	                      const size_t x,
	                      const size_t y,
	                      const size_t nn) noexcept {
		ASSERT(x < M.nrows);
		ASSERT(y+nn <= M.ncols);
		const uint32_t spot  = y % RADIX;
		const uint64_t block = y / RADIX;

		// this must be negative...
		const int32_t spill = spot + nn - RADIX;
		uint64_t temp = (spill <= 0) ? M[x][block] << -spill
		                         : (M[x][block + 1] << (RADIX - spill)) |
		                           (M[x][block] >> spill);
		return temp >> (RADIX - nn);
	}

	///
	/// \param M
	/// \param rstart
	/// \param cstart
	/// \param rstop
	/// \param k
	/// \param cols
	/// \param T
	/// \param rev
	constexpr static void matrix_process_rows(FqMatrix &M,
	                         const size_t rstart,
	                         const size_t cstart,
	                         const size_t rstop,
	                         const size_t k,
	                         uint64_t *Table,
	                         uint32_t **rev) noexcept {
		for (size_t r = rstart; r < rstop; ++r) {
			size_t x0 = rev[k][matrix_read_bits(M, r, cstart, k)];
			if (x0) {
				xor_avx1_new((uint8_t *) (Table + x0 * padded_limbs),   // in
				             (uint8_t *) M[r],                    // in
				             (uint8_t *) M[r],                    // out
				             padded_simd_limbs);
			}
		}
	}

	///
	/// \param M 			matrix
	/// \param k 			m4ri k
	/// \param rstop 		row stop
	/// \param matrix_data 	helper data
	/// \param cstart 		column start.
	/// \return
	constexpr static size_t matrix_echelonize_partial(FqMatrix &M,
	                                 const size_t k,
	                                 const size_t rstop,
	                                 const size_t cstart) noexcept {
		alignas(64) uint32_t **rev      = M.rev;
		alignas(64) uint32_t **diff     = M.diff;
		alignas(64) uint64_t *xor_rows  = M.lookup_table;

		size_t kk = k;

		// current row
		size_t r = 0;
		// current column
		size_t c = cstart;

		while (c < rstop) {
			if (c + kk > rstop) {
				kk = rstop - c;
			}

			size_t kbar = matrix_gauss_submatrix(M, r, c, nrows, kk);
			if (kk != kbar) {
				break;
			}

			if (kbar > 0) {
				matrix_make_table  (M, r, kbar, xor_rows, diff);
				// fix everything below
				matrix_process_rows(M, r + kbar, c, nrows, kbar, xor_rows, rev);
				// fix everything over it
				matrix_process_rows(M, 0, c, r, kbar, xor_rows, rev);
			}

			r += kbar;
			c += kbar;
		}

		return r;
	}

	// specialised m4ri which is allowd to perform special look aheads
	// in the l window
	size_t matrix_echelonize_partial_opt(FqMatrix &M,
	                                 const size_t k,
	                                 const size_t rstop,
	                                 const size_t lwin_start) noexcept {
		alignas(64) uint32_t **rev      = M.rev;
		alignas(64) uint32_t **diff     = M.diff;
		alignas(64) uint64_t *xor_rows  = M.lookup_table;

		ASSERT(lwin_start <= nrows);
		ASSERT(rstop <= nrows);

		size_t kk = k;

		// current row
		size_t r = 0;
		// current column
		size_t c = 0;

		while (c < rstop) {
			if (c + kk > rstop) {
				kk = rstop - c;
			}

			size_t kbar = matrix_gauss_submatrix_opt(M, r, c, nrows, ncols, rstop, lwin_start, kk);
			if (kk != kbar)
				break;

			if (kbar > 0) {
				matrix_make_table  (M, r, ncols, kbar, xor_rows, diff);
				// fix everything below
				matrix_process_rows(M, r + kbar, c, nrows, kbar, ncols, xor_rows, rev);
				// fix everything over it
				matrix_process_rows(M, 0, c, r, kbar, ncols, xor_rows, rev);
			}

			r += kbar;
			c += kbar;
		}

		return r;
	}

	///
	/// \param M
	/// \param rang
	/// \param rstop
	/// \param fix_col
	/// \param look_ahead
	/// \param permutation
	/// \return
	constexpr static size_t matrix_fix_gaus(FqMatrix &M,
								  const size_t rang,
								  const size_t rstop,
								  const size_t fix_col,
								  const size_t look_ahead,
	                                        Permutation *__restrict__ permutation) noexcept {
		for (size_t b = rang; b < rstop; ++b) {
			bool found = false;
			// find a column where in the last row is a one
			for (size_t i = fix_col+look_ahead; i < size_t(M.ncols); ++i) {
				if (M.get(b, i)) {
					found = true;

					std::swap(permutation->values[i-look_ahead], permutation->values[b]);
					M.swap_cols(b, i);
					break;
				}
			}

			// if something found, fix this row by adding it to each row where a 1 one.
			if (found) {
				for (size_t i = 0; i < b; ++i) {
					if (M.get(i, b)) {
						M.row_xor(i, b);
					}
				}

				// and solve it below
				for (size_t i = b+1; i < (size_t)M.nrows; ++i) {
					if (M.get(i, b)) {
						M.row_xor(i, b);
					}
				}
			} else {
				// if we were not able to fix the gaussian elimination, we return
				// how far we got
				return b;
			}
		}

		return rstop;
	}

	// additionally, to the m4ri algorithm a fix is applied if m4ri fails
	/// \param M
	/// \param k			m4ri k
	/// \param rstop
	/// \param matrix_data
	/// \param cstart
	/// \param fix_col 		how many columns must be solved at the end of this function. Must be != 0
	/// \param look_ahead 	how many the algorithm must start ahead of `fix_col` to search for a pivo element. Can be zero
	/// \param permutation
	/// \return
	constexpr static inline size_t matrix_echelonize_partial_plusfix(FqMatrix &M,
	                                         const size_t k,
	                                         const size_t rstop,
	                                         const size_t cstart,
											 const size_t fix_col,
	                                         const size_t look_ahead,
	                                                                 Permutation *__restrict__ permutation) noexcept {
		size_t rang = matrix_echelonize_partial(M, k, rstop, cstart);
		return matrix_fix_gaus(M, rang, rstop, fix_col, look_ahead, permutation);
	}


	template<typename TT, std::size_t N, std::size_t... I>
	constexpr auto create_array_impl(std::index_sequence<I...>) {
		return std::array<TT, N>{ {I...} };
	}

	template<const uint32_t n, const uint32_t k, const uint32_t l>
	size_t matrix_echelonize_partial_plusfix_opt (
			FqMatrix &A,
			FqMatrix &AT,
			const size_t m4ri_k,
	        const size_t max_column,
	        Permutation *__restrict__ P) noexcept {
		constexpr uint32_t nkl  = n-k-l;
		constexpr uint16_t zero = uint16_t(-1);

	#if 1
		// create the transposed
		mzd_transpose(AT, A);

		// constexpr auto unitpos = create_array_impl<uint16_t, n-k-l>(std::make_index_sequence<n-k-l>{});
		// std::vector<uint16_t> unitpos(nkl, zero);
		// std::vector<uint16_t> posfix(n, zero);
		//auto posfix = create_array_impl<uint16_t, n>(std::make_index_sequence<n>{});

		//uint16_t unitpos[nkl] = {[0 ... nkl-1] = zero};
		//uint16_t posfix [n] = {[0 ... n-1] = zero};

		static uint16_t unitpos[nkl];
		static uint16_t posfix [n];
		for (uint32_t i = 0; i < nkl; ++i) {
			unitpos[i] = zero;
		}

		for (uint32_t i = 0; i < nkl; ++i) {
			posfix[i] = i;
		}

		for (uint32_t i = nkl; i < n; ++i) {
			posfix[i] = zero;
		}
		uint32_t unitctr = 0, switchctr = 0;

		// dont permute the last column since it is the syndrome
		for (uint32_t i = 0; i < uint32_t(P->length-1); ++i) {
			uint64_t pos = fastrandombytes_uint64() % (P->length - i);
			ASSERT(i+pos < uint64_t(P->length));

			std::swap(posfix[i], posfix[i+pos]);
			std::swap(P->values[i], P->values[pos+i]);
			// matrix_swap_rows_new2(AT, i, i+pos);
			mzd_row_swap(AT, i, i+pos);

			if (posfix[i] != zero && (i < n-k-l)) {
				unitpos[i] = posfix[i];
				unitctr += 1;
			}
		}

		// move in the matrix everything to the right
		for (uint32_t i = nkl; i > 0; i--) {
			// move the right pointer to the next free place from the right
			if (unitpos[i-1] == zero) {
				// move the left pointer to the next element
				while (unitpos[switchctr] == zero)
					switchctr += 1;

				if (i-1 <= switchctr)
					break;

				std::swap(unitpos[i-1], unitpos[switchctr]);
				std::swap(P->values[i-1], P->values[switchctr]);

				// matrix_swap_rows_new2(AT, i-1, switchctr);
				mzd_row_swap(AT, i-1, switchctr);
				switchctr += 1;
			}
		}

		// for (int i = 0; i < n-k-l; ++i) {
		// 	std::cout << i << ":" << unitpos[n-k-l-1-i] << ":" << n-k-l-1-i << "\n";
		// }
		// std::cout << "\n" << switchctr << ":" << unitctr << "\n";
		// std::cout << std::flush;

		// transpose it back
		mzd_transpose(A, AT);

		// mzd_print(A);
		// std::cout << "\n";

		// reset the array
		for (uint32_t i = 0; i < n-k-l; ++i) {
			posfix[i] = i;
		}

		// swap everything down
		for (uint32_t i = 0; i < unitctr; i++){
			// matrix_swap_rows_new2(A, n-k-l-1-i, posfix[unitpos[n-k-l-1-i]]);
			mzd_row_swap(A, nkl-1-i, posfix[unitpos[nkl-1-i]]);
			std::swap(posfix[nkl-1-i], posfix[unitpos[nkl-1-i]]);
		}

		// mzd_print(A);
		// std::cout << "\n";

		// final fix
		for (uint32_t i = nkl-unitctr; i < nkl; ++i) {
			if (mzd_read_bit(A, i, i) == 0) {
				for (uint32_t j = 0; j < nkl; ++j) {
					if (mzd_read_bit(A, j, i)) {
						// matrix_swap_rows_new2(A, i, j);
						mzd_row_swap(A, i, j);
						break;
					}
				}
			}
		}

		// mzd_print(A);
		// std::cout << "\n";

		// apply the final gaussian elimination on the first coordinates
		matrix_echelonize_partial_opt(A, m4ri_k, nkl-unitctr, nkl);
		// std::cout << unitctr << " " << nkl-unitctr << "\n";
		return nkl;
	#else
		return matrix_echelonize_partial_plusfix(A, m4ri_k, n - k - l, matrix_data, 0, n - k - l, 0, P);
	#endif
	}

	template<const uint32_t n,
		const uint32_t k,
		const uint32_t l,
		const uint32_t c>
	size_t matrix_echelonize_partial_plusfix_opt_onlyc(
			FqMatrix &A,
			FqMatrix &AT,
			const size_t m4ri_k,
	        const size_t max_column,
	        Permutation *__restrict__ P) noexcept {
		constexpr uint32_t nkl  = n-k-l;
		constexpr uint16_t zero = uint16_t(-1);
		// constexpr uint32_t mod = k+l;

		// some security checks
		static_assert(nkl >= c);
		ASSERT(c > m4ri_k);

		// create the transposed
		mzd_transpose(AT, A);

		//std::vector<uint16_t> unitpos1(c, zero);
		//std::vector<uint16_t> unitpos2(c, zero);
		//std::vector<uint16_t> posfix  (nkl, zero);
		//std::vector<uint16_t> perm    (n, zero);
		static uint16_t unitpos1[c];
		static uint16_t unitpos2[c];
		static uint16_t posfix[nkl];
		static uint16_t perm[n];

		#pragma unroll
		for (uint32_t i = 0; i < c; ++i) {
			unitpos1[i] = zero;
		}

		#pragma unroll
		for (uint32_t i = 0; i < c; ++i) {
			unitpos2[i] = zero;
		}

		for (uint32_t i = 0; i < nkl; ++i) {
			posfix[i] = i;
		}

		for (uint32_t i = 0; i < n; ++i) {
			perm[i] = i;
		}

		for (uint32_t i = 0; i < c; ++i) {
			const uint32_t pos1 = fastrandombytes_uint64() % (k+l-i);
			const uint32_t pos2 = fastrandombytes_uint64() % (n-k-l-i);
			std::swap(perm[n-k-l+i], perm[n-k-l+i+pos1]);
			std::swap(perm[i], perm[i+pos2]);

		}

		// dont permute the last column since it is the syndrome
		for (uint32_t i = 0; i < c; ++i) {
			const uint32_t pos1 = perm[i];
			const uint32_t pos2 = perm[n-k-l+i];
			ASSERT(pos1 < (uint32_t)P->length);
			ASSERT(pos2 < (uint32_t)P->length);

			std::swap(P->values[pos1], P->values[pos2]);
			mzd_row_swap(AT, pos1, pos2);

			unitpos1[i] = pos1;
			if (pos1 < c) {
				unitpos2[pos1] = 0;
			}
		}

		// move in the matrix everything to the left
		uint32_t left_ctr, right_ctr = 0;//, switch_ctr = 0;
		for (left_ctr = 0; left_ctr < c && right_ctr < c; left_ctr++) {
			if (unitpos2[left_ctr] == 0)
				continue;

			while(right_ctr < (c-1) && unitpos1[right_ctr] < c) {
				right_ctr += 1;
			}

			//if (right_ctr >= c || unitpos1[right_ctr] >= (n-k-l)) {
			//	std::cout << "kekw: ";
			//	std::cout << omp_get_thread_num() << ":";
			//	std::cout << right_ctr << "\n";
			//	for (uint32_t i = 0; i < c; i++){
			//		std::cout << unitpos1[i] << " ";
			//	}
			//}
			//ASSERT(right_ctr < c);
			//ASSERT(unitpos1[right_ctr] < n-k-l);

			mzd_row_swap(AT, left_ctr, unitpos1[right_ctr]);
			std::swap(P->values[left_ctr], P->values[unitpos1[right_ctr]]);
			posfix[unitpos1[right_ctr]] = left_ctr;

			// step to the next element
			right_ctr += 1;
		}

		// transpose it back
		mzd_transpose(A, AT);

		for (uint32_t i = nkl; i > c; i--){
			mzd_row_swap(A, i-1, posfix[i-1]);
			std::swap(posfix[i-1], posfix[posfix[i-1]]);
		}

		// apply the final gaussian elimination on the first coordinates
		const uint32_t rang = matrix_echelonize_partial_opt(A, m4ri_k, c, nkl);

		// fix the first
		for (size_t b = rang; b < c; ++b) {
			bool found = false;
			// find a column where in the last row is a one
			for (size_t i = n-k-l; i < n; ++i) {
				if (mzd_read_bit(A, (int)b, (int)i)) {
					found = true;

					std::swap(P->values[i], P->values[b]);
					mzd_col_swap(A, (int)b, (int)i);
					break;
				}
			}

			ASSERT(found);

			// if something found, fix this row by adding it to each row where a 1 one.
			if (found) {
				for (size_t i = 0; i < b; ++i) {
					if (mzd_read_bit(A, i, b)) {
						mzd_row_xor(A, i, b);
					}
				}

				// and solve it below
				for (size_t i = b+1; i < n-k; ++i) {
					if (mzd_read_bit(A, i, b)) {
						mzd_row_xor(A, i, b);
					}
				}
			}
		}

		return nkl;
	}

	constexpr inline uint32_t gaus(const uint32_t stop=nrows) noexcept {
		return matrix_echelonize_partial(*this, m4ri_k, stop, 0);
	}

	[[nodiscard]] constexpr uint32_t fix_gaus(uint32_t *__restrict__ permutation,
											  const uint32_t rang,
											  const uint32_t fix_col,
											  const uint32_t lookahead) noexcept {
		uint64_t data[2] = {0};
		Permutation *P = (Permutation *)data;
		P->length = ncols;
		P->values = permutation;
		return matrix_fix_gaus(*this, rang, fix_col, fix_col, lookahead, P);
	}

	constexpr void matrix_vector_mul(const FqMatrix<T, 1, ncols, q> &v) noexcept {
		/// TODO
	}

	constexpr static void print_matrix(const std::string &name,
							 const FqMatrix &A,
							 const uint32_t start_row=-1u,
							 const uint32_t end_row=-1u,
							 const uint32_t start_col=-1u,
							 const uint32_t end_col=-1u) noexcept {
		const uint32_t sstart_row = start_row == -1u ? 0 : start_row;
		const uint32_t eend_row = end_row == -1u ? A.nrows : end_row;

		const uint32_t sstart_col = start_col == -1u ? 0 : start_col;
		const uint32_t eend_col = end_col == -1u ? A.ncols : end_col;

		std::cout << name << "\n" << std::endl;
		for (uint32_t i = 0; i < eend_row-sstart_row; ++i) {
			std::cout << "[";
			for (uint32_t j = 0; j < eend_col-sstart_col; ++j) {
				if (A.get(i, j)) {
					std::cout << "1";
				} else {
					std::cout << " ";
				}

				if ((j % 4 == 0) && (j > 0) && (j % RADIX == 0)) {
					std::cout << "|";
				}

				if ((j % 4 == 0) && (j > 0) && (j % RADIX != 0)) {
					std::cout << ":";
				}

			}

			std::cout << "]\n";
		}

		std::cout << std::endl;
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
		print_matrix(name, *this);
	}

	constexpr bool binary() noexcept { return true; }
};

#endif//CRYPTANALYSISLIB_BINARYMATRIX_H
