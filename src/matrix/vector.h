#ifndef CRYPTANALYSISLIB_MATRIX_VECTOR_H
#define CRYPTANALYSISLIB_MATRIX_VECTOR_H

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
class FqVector : public FqMatrix_Meta<T, 1, ncols, q, packed, R> {
public:
  	using M = FqMatrix_Meta<T, 1, ncols, q, packed, R>;

	using typename M::DataType;
	using M::__data;

	/// gets the i-th row and j-th column
	/// \param i row
	/// \param j colum
	/// \return entry in this place
	[[nodiscard]] constexpr inline DataType get(const uint32_t i, const uint32_t j) const noexcept {
		ASSERT(i < 1 && j <= ncols);
		return __data[i][j];
	}

	[[nodiscard]] constexpr inline DataType get(const uint32_t i) const noexcept {
		ASSERT(i < ncols);
		return __data[0][i];
	}
};
#endif//CRYPTANALYSISLIB_MATRIX_VECTOR_H
