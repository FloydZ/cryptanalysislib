#ifndef CRYPTANALYSISLIB_MATH_PRIME_H
#define CRYPTANALYSISLIB_MATH_PRIME_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

/// https://stackoverflow.com/questions/1538644/c-determine-if-a-number-is-prime
/// \param n
/// \return true/false if `n` is a number
constexpr static inline bool is_prime(const size_t n) noexcept {
	if (n <= 3) {
		// as 2 and 3 are prime
		return n > 1;
	} else if ((n%2) == 0 || (n%3) == 0) {
		// check if n is divisible by 2 or 3
		return false;
	} else {
		for (size_t i=5; i*i<=n; i+=6) {
			if (n % i == 0 || n%(i + 2) == 0) {
				return false;
			}
		}

		return true;
	}
}

/// \param n base number feom which one should search
/// \return the next prime >= n
[[nodiscard]] constexpr static inline size_t next_prime(const size_t n) noexcept {
	// some safty
	if (n >= 18361375334787046697ull) {
		return 18361375334787046697ull;
	}

	if (n <= 3) {
		return n;
	}

	// round up to the next uneven number
	size_t a = n + (1 - (n&1u));
	while (!is_prime(a)) {
		++a;
	}

	return a;
}
#endif //CRYPTANALYSISLIB_MATH_PRIME_H
