#ifndef CRYPTANALYSISLIB_ALGORITHM_BITS_BSR_H
#define CRYPTANALYSISLIB_ALGORITHM_BITS_BSR_H

#include <cstdint>
#include <type_traits>

#include "helper.h"
#include "clz.h"

/// namespace containing popcount algorithms
namespace cryptanalysislib::algorithm {
	/// \tparam T base data type
	/// \param data input data type
	/// \return
	template<typename T>
#if __cplusplus > 201709L
		requires std::is_integral<T>::value
#endif
	constexpr static inline uint32_t bsr(const T data) noexcept {
		constexpr uint32_t t = (sizeof(T)*8) - 1ul;
		return t - clz<T>(data);
	}
}
#endif
