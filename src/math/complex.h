#ifndef CRYPTANALYSISLIB_MATH_COMPLEX_H
#define CRYPTANALYSISLIB_MATH_COMPLEX_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

#include <iostream>
#include <type_traits>

#include "math/root.h"

namespace cryptanalysislib::math {

	/// constexpr complex number `re + i*im`
	/// NOTE: for integral `T` this is a Gaussian integer: all operations
	/// 	except `/` and `abs` are exact.
	/// \tparam T underlying arithmetic type
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_arithmetic_v<T>
#endif
	struct complex {
		using value_type = T;

		T re{};
		T im{};

		constexpr complex() noexcept = default;
		constexpr complex(const T re, const T im = T{}) noexcept : re(re), im(im) {}

		[[nodiscard]] constexpr inline T real() const noexcept { return re; }
		[[nodiscard]] constexpr inline T imag() const noexcept { return im; }

		/// \return the complex conjugate `re - i*im`
		[[nodiscard]] constexpr inline complex conj() const noexcept {
			return complex{re, -im};
		}

		/// \return `|z|^2 = re^2 + im^2`. NOTE: same as `std::norm`
		[[nodiscard]] constexpr inline T norm() const noexcept {
			return re*re + im*im;
		}

		/// \return `|z|`, computed via `math::sqrt`
		[[nodiscard]] constexpr inline T abs() const noexcept {
			return cryptanalysislib::math::sqrt(norm());
		}

		constexpr inline complex operator-() const noexcept {
			return complex{-re, -im};
		}

		constexpr inline complex &operator+=(const complex &b) noexcept {
			re += b.re; im += b.im;
			return *this;
		}

		constexpr inline complex &operator-=(const complex &b) noexcept {
			re -= b.re; im -= b.im;
			return *this;
		}

		constexpr inline complex &operator*=(const complex &b) noexcept {
			const T r = re*b.re - im*b.im;
			im = re*b.im + im*b.re;
			re = r;
			return *this;
		}

		/// NOTE: no check for division by zero
		constexpr inline complex &operator/=(const complex &b) noexcept {
			const T d = b.norm();
			const T r = (re*b.re + im*b.im) / d;
			im = (im*b.re - re*b.im) / d;
			re = r;
			return *this;
		}

		constexpr inline complex &operator*=(const T b) noexcept {
			re *= b; im *= b;
			return *this;
		}

		/// NOTE: no check for division by zero
		constexpr inline complex &operator/=(const T b) noexcept {
			re /= b; im /= b;
			return *this;
		}

		friend constexpr inline complex operator+(complex a, const complex &b) noexcept { return a += b; }
		friend constexpr inline complex operator-(complex a, const complex &b) noexcept { return a -= b; }
		friend constexpr inline complex operator*(complex a, const complex &b) noexcept { return a *= b; }
		friend constexpr inline complex operator/(complex a, const complex &b) noexcept { return a /= b; }
		friend constexpr inline complex operator*(complex a, const T b) noexcept { return a *= b; }
		friend constexpr inline complex operator*(const T b, complex a) noexcept { return a *= b; }
		friend constexpr inline complex operator/(complex a, const T b) noexcept { return a /= b; }

		friend constexpr inline bool operator==(const complex &a, const complex &b) noexcept {
			return (a.re == b.re) && (a.im == b.im);
		}

		/// prints `(re,im)`, same format as `std::complex`
		friend std::ostream &operator<<(std::ostream &out, const complex &c) {
			return out << "(" << c.re << "," << c.im << ")";
		}
	};
} // end namespace cryptanalysislib::math

#endif//CRYPTANALYSISLIB_MATH_COMPLEX_H
