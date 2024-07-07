#ifndef CRYPTANALYSISLIB_MATH_ABS_H
#define CRYPTANALYSISLIB_MATH_ABS_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

#include <type_traits>
#include <limits>

namespace cryptanalysislib::math {
	/// rater important, as it also works with unsigned values, without a warning
	/// \tparam T
	/// \param x
	/// \return
	template<typename T>
	    requires std::is_arithmetic<T>::value
	constexpr T abs(T x) {
		return x >= 0 ? x : -x;
	}

	///
	/// \tparam T
	/// \param x
	/// \return
	template<typename T>
	    requires std::is_floating_point<T>::value
	constexpr T fabs(T x) {
		return cryptanalysislib::math::abs(x);
	}

	/// test whether values are within machine epsilon, used for algorithm
	/// termination
	/// \tparam T
	/// \param x
	/// \param y
	/// \return
	template<typename T>
	    requires std::is_arithmetic_v<T>
	constexpr bool feq(T x, T y) {
		return abs(x - y) <= std::numeric_limits<T>::epsilon();
	}

}
#endif //CRYPTANALYSISLIB_ABS_H
