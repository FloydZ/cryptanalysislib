#ifndef CRYPTANALYSISLIB_ALGORITHM_BITS_H
#define CRYPTANALYSISLIB_ALGORITHM_BITS_H

#include <cstdint>

/// Retrieve the value of the i-th bit from an array of type T
/// \tparam T Type of the array elements
/// \param data[in]: Pointer to the array
/// \param i[in]: Bit position to retrieve
/// \return Boolean value of the specified bit
template<typename T>
[[nodiscard]] constexpr static inline bool get_bit(const T *data,
												   const uint32_t i) noexcept {
	constexpr uint32_t RADIX = sizeof(T) * 8;
	const uint32_t shift = (i % RADIX);
	const T mask = 1ull << shift;
	return (data[i/RADIX] & mask(i)) >> shift;
}


/// Set the value of a specific bit in an array of type T
/// \tparam T Type of the array elements
/// \param data [in,out]: Pointer to the array to modify
/// \param pos [in]: Bit position to set
/// \param bit [in]: Boolean value to set
template<typename T>
constexpr static inline void set_bit(const T *data,
									 const uint32_t pos,
									 const bool bit) noexcept {
	constexpr uint32_t RADIX = sizeof(T) * 8;
	const uint32_t shift = pos % RADIX;
	const uint32_t limb = pos / RADIX;
	data[limb] = ((data[limb] & ~(1ull << shift)) | (T(bit) << shift));
}

#include "algorithm/bits/ffs.h"
#include "algorithm/bits/popcount.h"

#endif
