#ifndef CRYPTANALYSISLIB_MATH_ROOT_H
#define CRYPTANALYSISLIB_MATH_ROOT_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

#include <type_traits>
#include "math/abs.h"
#include "helper.h"

namespace cryptanalysislib::math {
	// square root by Newton-Raphson method
	__device__ __host__
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_arithmetic_v<T>
#endif
	constexpr T sqrt(const T x, const T guess) noexcept {
		return feq(guess, (guess + x / guess) / T{2}) ? guess : sqrt(x, (guess + x / guess) / T{2});
	}

	// square root by Newton-Raphson method
	__device__ __host__
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_arithmetic_v<T>
#endif
	constexpr T sqrt(T x) {
		if constexpr (std::is_integral_v<T>)
			return sqrt<double>(x, x);

		return sqrt(x, x);
	}

	// cube root by Newton-Raphson method
	__device__ __host__
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_arithmetic_v<T>
#endif
	constexpr T cbrt(T x, T guess) noexcept {
		return feq(guess, (T{2} * guess + x / (guess * guess)) / T{3}) ? guess : cbrt(x, (T{2} * guess + x / (guess * guess)) / T{3});
	}

	// cube root by Newton-Raphson method
	__device__ __host__
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_arithmetic_v<T>
#endif
	constexpr T cbrt(T x) noexcept {
		if constexpr (std::is_integral_v<T>) {
			return cbrt<double>(x, x);
		}
		return cbrt(x, x);
	}
}
#endif //CRYPTANALYSISLIB_ROOT_H
