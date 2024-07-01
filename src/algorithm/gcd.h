//
// Created by duda on 01.07.24.
//

#ifndef CRYPTANALYSISLIB_GCD_H
#define CRYPTANALYSISLIB_GCD_H

template<typename T>
constexpr static T gcd(const T a, const T b) noexcept {
	if (b == 0) { return b; }
	if (a == 0) { return a; }

	// Base case
	if (a == b)
		return a;

	// a is greater
	if (a > b) {
		return gcd<T>(a - b, b);
	}

	return gcd<T>(a, b - a);
}

#endif//CRYPTANALYSISLIB_GCD_H
