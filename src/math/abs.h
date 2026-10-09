#ifndef CRYPTANALYSISLIB_MATH_ABS_H
#define CRYPTANALYSISLIB_MATH_ABS_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

#include <type_traits>
#include <limits>
#include <cstdint>

namespace cryptanalysislib::math {

    /// branchless
    /// Return abs(a-b)
    /// Both a and b must not have the most significant bit set
	template<typename T>
    #if __cplusplus > 201709L
    	    requires std::is_arithmetic<T>::value
    #endif
    constexpr static inline T abs_branchless(T a, T b) noexcept {
        constexpr static uint32_t BITS = sizeof(T) * 8u;
        T d1 = b - a;
        T d2 = (d1 & (T)( (long)d1 >> (BITS-1u)) ) << 1u;
        return  d1 - d2;  // == (b - d) - (a + d);
    }

	/// rater important, as it also works with unsigned values, without a warning
	/// \tparam T
	/// \param x
	/// \return
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_arithmetic<T>::value
#endif
	constexpr T abs(T x) {
		return x >= 0 ? x : -x;
	}

	///
	/// \tparam T
	/// \param x
	/// \return
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_floating_point<T>::value
#endif
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
#if __cplusplus > 201709L
	    requires std::is_arithmetic_v<T>
#endif
	constexpr bool feq(T x, T y) {
		// NOTE: relative tolerance (4 ulps); with an absolute epsilon the
		// iterations in `log`, `sqrt` and `cbrt` never terminated for |x| > 2,
		// as neighbouring doubles are further apart than epsilon there
		const T ax = abs(x), ay = abs(y);
		const T m = ax > ay ? ax : ay;
		return abs(x - y) <= T{4} * std::numeric_limits<T>::epsilon() * m;
	}

}
#endif //CRYPTANALYSISLIB_ABS_H
