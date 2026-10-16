#ifndef CRYPTANALYSISLIB_MATH_ROUND_H
#define CRYPTANALYSISLIB_MATH_ROUND_H

#include <cstdint>
#include <type_traits>
#include "helper.h"

#include "algorithm/bits/popcount.h"

namespace cryptanalysislib::math {
	/// \param val
	/// \return
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_floating_point_v<T>
#endif
	__device__ __host__ constexpr int64_t round(const T val) {
		// round half away from zero (as std::round)
		const int64_t t = (int64_t) val;
		const T frac = val - T(t);
		if (frac >= T(0.5)) {
			return t + 1;
		}
		if (frac <= T(-0.5)) {
			return t - 1;
		}
		return t;
	}

	/// \return true if `x` is a power of two (false for 0)
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_integral_v<T>
#endif
	[[nodiscard]] constexpr inline bool is_power_of_two(const T x) noexcept {
		return (x > 0) && ((x & (x - 1)) == 0);
	}

	/// compile time version of `is_power_of_two`
	template<std::size_t X>
	inline constexpr bool is_power_of_two_v = is_power_of_two(X);

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
