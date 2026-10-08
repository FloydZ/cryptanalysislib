#ifndef CRYPTANALYSISLIB_MATH_ROOT_H
#define CRYPTANALYSISLIB_MATH_ROOT_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

#include <cstdint>
#include <limits>
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
	constexpr T sqrt(const T x, T guess) noexcept {
		// NOTE: iterative with a cap, so it terminates even if the iteration
		// oscillates; sqrt(0) = 0 (the iteration divides by the guess)
		if (x == T{0}) {
			return T{0};
		}
		for (uint32_t i = 0; i < 4096; ++i) {
			const T next = (guess + x / guess) / T{2};
			if (feq(guess, next)) {
				return next;
			}
			guess = next;
		}
		return guess;
	}

	// square root by Newton-Raphson method
	__device__ __host__
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_arithmetic_v<T>
#endif
	constexpr T sqrt(T x) {
		if constexpr (std::is_integral_v<T>) {
			return sqrt<double>(x, x);
		} else {
			if (x < T{0}) {
				return std::numeric_limits<T>::quiet_NaN();
			}
			return sqrt(x, x);
		}
	}

	// cube root by Newton-Raphson method
	__device__ __host__
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_arithmetic_v<T>
#endif
	constexpr T cbrt(T x, T guess) noexcept {
		// NOTE: iterative with a cap (see `sqrt`); cbrt(0) = 0
		if (x == T{0}) {
			return T{0};
		}
		for (uint32_t i = 0; i < 4096; ++i) {
			const T next = (T{2} * guess + x / (guess * guess)) / T{3};
			if (feq(guess, next)) {
				return next;
			}
			guess = next;
		}
		return guess;
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
