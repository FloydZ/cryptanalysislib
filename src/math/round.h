#ifndef CRYPTANALYSISLIB_MATH_ROUND_H
#define CRYPTANALYSISLIB_MATH_ROUND_H

#include <cstdint>
#include "helper.h"

#include "algorithm/bits/popcount.h"

namespace cryptanalysislib::math {
	/// \param val
	/// \return
	__device__ __host__ template<typename T>
#if __cplusplus > 201709L
	    requires std::is_floating_point_v<T>
#endif
	constexpr int64_t round(const T val) {
		return (int64_t) val;
	}

	[[nodiscard]] constexpr static std::size_t round_up_to_power_of_two(std::size_t value) noexcept {
		if (cryptanalysislib::popcount::popcount(value) == 1u) {
			return value;
		}

		if (value == 0) {
			return 1;
		}

		--value;
		for (std::size_t i = 1; i < sizeof(std::size_t) * sizeof(uint8_t) * 8; i *= 2) {
			value |= value >> i;
		}

		return value + 1;
	}
} // end namespace cryptanalysislib::math
#endif //CRYPTANALYSISLIB_ROUND_H
