#ifndef CRYPTANALYSISLIB_PRINT_H
#define CRYPTANALYSISLIB_PRINT_H

#include <type_traits>
#include <iostream>
#include <limits>
#include <inttypes.h>


/// \tparam T base type
/// \param a number to print
/// \param len number of bits to print
template<typename T>
static void print_binary(T a,
                         const size_t len = sizeof(T)*8u,
                         const char* end = "\n") {
	for (uint32_t i = 0; i < len; i++) {
		printf("%" PRIu64, uint64_t(a & 1u));
		a >>= 1u;
	}

	printf("%s", end);
}

/// \tparam T base type
/// \param a number to print
/// \param len number of bits to print
template<typename T>
static void print_binary(T *a, const size_t len) {
	constexpr uint32_t bits = sizeof(T) * 8;
	const uint32_t limbs = (len+bits-1) / bits;

	for (uint32_t i = 0; i < limbs-1u; ++i) {
		print_binary<T>(a[i], bits, "");
	}

	print_binary<T>(a[limbs - 1u], len%bits);
}
#endif//CRYPTANALYSISLIB_PRINT_H
