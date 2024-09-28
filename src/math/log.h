#ifndef CRYPTANALYSISLIB_LOG_H
#define CRYPTANALYSISLIB_LOG_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

#include <type_traits>

#include "math/abs.h"
#include "math/exp.h"

namespace cryptanalysislib::math::internal {
	__device__ __host__
	template<typename T>
#if __cplusplus > 201709L
		requires std::is_floating_point_v<T>
#endif
	[[nodiscard]] constexpr T log_iter(T x, T y) noexcept {
		return y + T{2} * (x - exp(y)) / (x + exp(y));
	}

	__device__ __host__
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_floating_point_v<T>
#endif
	[[nodiscard]] constexpr T log(T x, T y) noexcept {
		return feq(y, log_iter(x, y)) ? y : log(x, log_iter(x, y));
	}

	__device__ __host__
	[[nodiscard]] constexpr inline long double e() noexcept {
		return 2.71828182845904523536l;
	}

	// For numerical stability, constrain the domain to be x > 0.25 && x < 1024
	// - multiply/divide as necessary. To achieve the desired recursion depth
	// constraint, we need to account for the max double. So we'll divide by
	// e^5. If you want to compute a compile-time log of huge or tiny long
	// doubles, YMMV.

	// if x <= 1, we will multiply by e^5 repeatedly until x > 1
	__device__ __host__
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_floating_point_v<T>
#endif
	[[nodiscard]] constexpr T logGT(T x) noexcept {
		return x > T{0.25} ? log(x, T{0}) : logGT<T>(x * e() * e() * e() * e() * e()) - T{5};
	}

	// if x >= 2e10, we will divide by e^5 repeatedly until x < 2e10
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_floating_point_v<T>
#endif
	[[nodiscard]] constexpr T logLT(T x) noexcept {
		return x < T{1024} ? log(x, T{0}) : logLT<T>(x / (e() * e() * e() * e() * e())) + T{5};
	}
}


namespace cryptanalysislib::math {
	__device__ __host__
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_arithmetic_v<T>
#endif
	[[nodiscard]] constexpr T log(T const x) noexcept {
		if (x > T{1024}) {
			if constexpr (std::is_integral_v<T>) {
				return cryptanalysislib::math::internal::logLT<double>(x);
			} else {
				return cryptanalysislib::math::internal::logLT<double>(x);
			}
		} else {
			if constexpr (std::is_integral_v<T>) {
				return cryptanalysislib::math::internal::logGT<double>(x);
			} else {
				return cryptanalysislib::math::internal::logGT<double>(x);
			}
		}
	}

	__device__ __host__
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_arithmetic_v<T>
#endif
	[[nodiscard]] constexpr T log2(T x) noexcept {
		return log(x) / log(2.);
	}
}

/// \param n input
/// \return ceil(log2(n)), only useful if you need the number of bits needed
__device__ __host__
[[nodiscard]] constexpr static inline uint64_t ceil_log2(uint64_t n) noexcept {
	//if constexpr (std::is_constant_evaluated()) {
		return n <= 1 ? 0 : 1 + ceil_log2((n + 1) / 2);
	//} else {
	//	return std::log2(n );
	//}
}

///
/// \param n
/// \return floor(log2(n))
__device__ __host__
[[nodiscard]] constexpr static inline uint64_t floor_log2(uint64_t n) noexcept {
	return sizeof(uint64_t ) * CHAR_BIT - 1 - __builtin_clzl((uint64_t)n);
}
#endif //CRYPTANALYSISLIB_LOG_H
