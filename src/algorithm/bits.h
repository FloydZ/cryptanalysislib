#ifndef CRYPTANALYSISLIB_ALGORITHM_BITS_H
#define CRYPTANALYSISLIB_ALGORITHM_BITS_H

#include <cstdint>

/// generic function to read a i-th bit from a array
/// \tparam T
/// \param data
/// \param i
/// \return
template<typename T>
[[nodiscard]] constexpr static inline bool get_bit(const T *data,
												   const uint32_t i) noexcept {
	constexpr static uint32_t RADIX = sizeof(T) * 8;
	const uint32_t shift = (i % RADIX);
	const T mask = 1ull << shift;
	return (data[i/RADIX] & mask(i)) >> shift;
}


template<typename T>
constexpr static inline void set_bit(const T *data,
									 const uint32_t pos,
									 const bool bit) noexcept {
	constexpr static uint32_t RADIX = sizeof(T) * 8;
	const uint32_t shift = pos % RADIX;
	const uint32_t limb = pos / RADIX;
	data[limb] = ((data[limb] & ~(1ull << shift)) | (T(bit) << shift));
}

#include "algorithm/bits/ffs.h"
#include "algorithm/bits/popcount.h"

#endif
