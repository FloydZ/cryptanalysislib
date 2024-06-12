#ifndef CRYPTANALYSISLIB_MATH_PRIME_H
#define CRYPTANALYSISLIB_MATH_PRIME_H

#ifndef CRYPTANALYSISLIB_MATH_H
#error "do not inlcude this file directly. Use `#include <cryptanalysislib/math>`"
#endif

constexpr static inline bool is_prime(const size_t n) noexcept {
	/* https://stackoverflow.com/questions/1538644/c-determine-if-a-number-is-prime */
	if (n <= 3)
		return n > 1;     // as 2 and 3 are prime
	else if (n % 2==0 || n % 3==0)
		return false;     // check if n is divisible by 2 or 3
	else {
		for (size_t i=5; i*i<=n; i+=6) {
			if (n % i == 0 || n%(i + 2) == 0)
				return false;
		}
		return true;
	}
}

constexpr static inline size_t next_prime(size_t n) noexcept {
	while (!is_prime(n)) ++n;
	return n;
}
#endif//CRYPTANALYSISLIB_PRIME_H
