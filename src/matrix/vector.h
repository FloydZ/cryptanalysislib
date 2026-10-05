#ifndef CRYPTANALYSISLIB_MATRIX_VECTOR_H
#define CRYPTANALYSISLIB_MATRIX_VECTOR_H

#include <cstdint>
#include <cstdlib>

#include "matrix/matrix.h"

/// vector implementation which is row major
/// inherits everything from `FqMatrix`, but overwrites the `get` functions
/// \tparam T base type something like `uint32_t` or `uint64_t`
/// \tparam ncols number of columns
/// \tparam q base field size
/// \tparam packed if true the rowtype to `kAryPackedContainer`
/// \tparam R helper type to overwrite the rowtype. Overwrites packed if != void
template<typename T,
		 const uint32_t ncols,
		 const uint64_t q,
		 const bool packed = true,
         typename R=void>
class FqVector : public FqMatrixMeta<T, 1, ncols, q, packed, R> {
public:
  	using M = FqMatrixMeta<T, 1, ncols, q, packed, R>;

	using typename M::DataType;
	using M::__data;

	/// gets the i-th row and j-th column
	/// \param i[in]: row
	/// \param j col: rowum
	/// \return entry in this place
	[[nodiscard]] constexpr inline DataType get(const uint32_t i, 
                                                const uint32_t j) const noexcept {
		assert(i < 1);
		assert(j <= ncols);
		return __data[i][j];
	}

	/// gets the i-th element within the first row
	/// \param j[in]; colum
	/// \return entry in this place
	[[nodiscard]] constexpr inline DataType get(const uint32_t i) const noexcept {
		assert(i < ncols);
		return __data[0][i];
	}
};
#endif//CRYPTANALYSISLIB_MATRIX_VECTOR_H
