#ifndef CRYPTANALYSISLIB_PRINT_H
#define CRYPTANALYSISLIB_PRINT_H

#include <inttypes.h>
#include <iostream>
#include <limits>
#include <type_traits>

namespace cryptanalysislib {
	/// \tparam T base type
	/// \param a number to print
	/// \param len number of bits to print
	template<typename T>
#if __cplusplus > 201709L
		requires std::is_arithmetic_v<T>
#endif
	static void print_binary(T a,
	                         const size_t len = sizeof(T) * 8u,
	                         const bool reverse = true,
	                         const char *end = "\n") {
		if (reverse) {
			for (uint32_t i = len; i > 0; i--) {
				printf("%" PRIu64, uint64_t((a >> (i - 1u)) & 1u));
			}
		} else {
			for (uint32_t i = 0; i < len; i++) {
				printf("%" PRIu64, uint64_t(a & 1u));
				a >>= 1u;
			}
		}
		printf("%s", end);
	}

	/// print a big number
	/// \tparam T base type
	/// \param a number to print
	/// \param len number of bits to print
	template<typename T>
#if __cplusplus > 201709L
		requires std::is_arithmetic_v<T>
#endif
	static void print_binary(const T *a,
							 const size_t len,
	                         const bool reverse=true,
							 const char *end = "\n") {
		constexpr uint32_t bits = sizeof(T) * 8;
		const uint32_t limbs = (len + bits - 1) / bits;


		if (reverse) {
			for (uint32_t i = limbs; i > 0; --i) {
				print_binary<T>(a[i-1], bits, reverse, "");
			}

		}
		for (uint32_t i = 0; i < limbs - 1u; ++i) {
			print_binary<T>(a[i], bits, reverse, "");
		}

		print_binary<T>(a[limbs - 1u], len % bits, end);
	}
}
#endif//CRYPTANALYSISLIB_PRINT_H
