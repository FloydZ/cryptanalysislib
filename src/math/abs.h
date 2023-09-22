#ifndef CRYPTANALYSISLIB_ABS_H
#define CRYPTANALYSISLIB_ABS_H

#include <type_traits>
#include <limits>

namespace cryptanalysislib {
	/// rater important, as it also works with unsigned values, without a warning
	/// \tparam T
	/// \param x
	/// \return
	template<typename T>
	    requires std::is_arithmetic<T>::value
	constexpr T abs(T x) {
		return x >= 0 ? x : -x;
	}

}

template<typename T>
requires
    std::is_floating_point<T>::value
constexpr T fabs(T x) {
    return abs(x);
}

// test whether values are within machine epsilon, used for algorithm
// termination
template <typename T>
    requires std::is_arithmetic_v<T>
constexpr bool feq(T x, T y) {
    return abs(x - y) <= std::numeric_limits<T>::epsilon();
}
#endif //CRYPTANALYSISLIB_ABS_H
