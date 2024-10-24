#ifndef CRYPTANALYSISLIB_SIMD_BITS_GENERIC_H
#define CRYPTANALYSISLIB_SIMD_BITS_GENERIC_H

#include "helper.h"

/// implementation of a binary NxM matrix
/// \tparam N number of cols
/// \tparam M number of rows
template<const uint32_t N,
		 const uint32_t M>
struct bNxM_t {
public:
	using T = std::conditional<N>=64, uint64_t, LogTypeTemplate<N>>;

	/// NOTE: think about where the difference to `BinaryMatrix`?

private:
	// main data container
	std::array<T, M> __data;
};
#endif
