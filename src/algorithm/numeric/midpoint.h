#ifndef CRYPTANALYSISLIB_ALGORITHM_NUMERIC_MIDPOINT_H
#define CRYPTANALYSISLIB_ALGORITHM_NUMERIC_MIDPOINT_H

#include <limits>
#include <type_traits>

namespace cryptanalysislib {
	/// Half-way point between `a` and `b` without overflow (like
	/// `std::midpoint`). Integers: rounded towards `a`. Floating point:
	/// `(a + b) / 2` correctly rounded, if that can not overflow.
	/// \param a[in]: first value
	/// \param b[in]: second value
	/// \return a + (b - a) / 2
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_arithmetic_v<T> &&
	             (!std::is_same_v<std::remove_cv_t<T>, bool>)
#endif
	[[nodiscard]] constexpr T midpoint(const T a,
	                                   const T b) noexcept {
		if constexpr (std::is_integral_v<T>) {
			// half of the distance, computed in the unsigned type
			using U = std::make_unsigned_t<T>;
			if (a > b) {
				return T(U(a) - U(U(U(a) - U(b)) >> 1u));
			}
			return T(U(a) + U(U(U(b) - U(a)) >> 1u));
		} else {
			constexpr T lo = std::numeric_limits<T>::min() * 2;
			constexpr T hi = std::numeric_limits<T>::max() / 2;
			const T abs_a = (a < 0) ? -a : a;
			const T abs_b = (b < 0) ? -b : b;
			if ((abs_a <= hi) && (abs_b <= hi)) {
				// no overflow, exact up to the final rounding
				return (a + b) / 2;
			}
			if (abs_a < lo) {
				// `a / 2` would lose bits
				return a + b / 2;
			}
			if (abs_b < lo) {
				return a / 2 + b;
			}
			return a / 2 + b / 2;
		}
	}

	/// Pointer to the element half-way between `a` and `b`, which point into
	/// the same array (like `std::midpoint`), rounded towards `a`
	/// \param a[in]: first pointer
	/// \param b[in]: second pointer
	/// \return a + (b - a) / 2
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_object_v<T>
#endif
	[[nodiscard]] constexpr T *midpoint(T *const a,
	                                    T *const b) noexcept {
		return a + (b - a) / 2;
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_NUMERIC_MIDPOINT_H
