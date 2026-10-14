#ifndef CRYPTANALYSISLIB_SORT_DJBSORT_PORTABLE_H
#define CRYPTANALYSISLIB_SORT_DJBSORT_PORTABLE_H

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#ifdef USE_AVX2
#include "sort/djb_sort.h"
#endif

/// djbsort (https://sorting.cr.yp.to) for 8, 16, 32 and 64 bit integers:
/// the comparisons are a fixed sorting network (only depending on `n`), and
/// every comparator is branchless, so the run time does not depend on the
/// data (constant time).
/// With AVX2 the 8, 16 and 32 bit types use the AVX2 `int32_sort`.
namespace cryptanalysislib::djbsort {
	/// sets (a, b) := (min(a, b), max(a, b)) without branches
	template<typename T>
	constexpr inline void minmax(T &a, T &b) noexcept {
		static_assert(std::is_integral_v<T>);
		using U = std::make_unsigned_t<T>;
		using W = std::conditional_t<(sizeof(T) < 4), uint32_t, U>;
		constexpr uint32_t bits = 8u * sizeof(W);
		// order preserving map of `T` to the unsigned `W`
		constexpr W sign = std::is_signed_v<T> ? W(W(1) << (8u * sizeof(T) - 1u)) : W(0);
		const W x = W(U(a)) ^ sign;
		const W y = W(U(b)) ^ sign;

		// most significant bit of `lt`: y < x (Hacker's Delight 2-12)
		const W lt = W((~y & x) | ((~y | x) & W(y - x)));
		const W mask = W(0) - W(lt >> (bits - 1u));
		const U t = U((U(a) ^ U(b)) & U(mask));
		a = T(U(a) ^ t);
		b = T(U(b) ^ t);
	}

	/// sorts x[0, n) in ascending order, portable version of djbsort
	/// (int32/portable4)
	template<typename T>
	constexpr void sort(T *x, const int64_t n) noexcept {
		if (n < 2) {
			return;
		}

		int64_t top = 1;
		while (top < n - top) { top += top; }

		for (int64_t p = top; p >= 1; p >>= 1) {
			int64_t i = 0;
			while (i + 2 * p <= n) {
				for (int64_t j = i; j < i + p; ++j) {
					minmax(x[j], x[j + p]);
				}
				i += 2 * p;
			}
			for (int64_t j = i; j < n - p; ++j) {
				minmax(x[j], x[j + p]);
			}

			i = 0;
			int64_t j = 0;
			for (int64_t q = top; q > p; q >>= 1) {
				if (j != i) {
					while (true) {
						if (j == n - q) { goto done; }
						T a = x[j + p];
						for (int64_t r = q; r > p; r >>= 1) {
							minmax(a, x[j + r]);
						}
						x[j + p] = a;
						++j;
						if (j == i + p) {
							i += 2 * p;
							break;
						}
					}
				}

				while (i + p <= n - q) {
					for (j = i; j < i + p; ++j) {
						T a = x[j + p];
						for (int64_t r = q; r > p; r >>= 1) {
							minmax(a, x[j + r]);
						}
						x[j + p] = a;
					}
					i += 2 * p;
				}

				// now i + p > n - q
				j = i;
				while (j < n - q) {
					T a = x[j + p];
					for (int64_t r = q; r > p; r >>= 1) {
						minmax(a, x[j + r]);
					}
					x[j + p] = a;
					++j;
				}
			done:;
			}
		}
	}
} // end namespace cryptanalysislib::djbsort

namespace cryptanalysislib {
	/// sorts x[0, n) in ascending order in constant time (djbsort)
	/// \tparam T 8, 16, 32 or 64 bit integer
	template<typename T>
#if __cplusplus > 201709L
	    requires std::is_integral_v<T>
#endif
	inline void djb_sort(T *x, const size_t n) noexcept {
#ifdef USE_AVX2
		if constexpr (sizeof(T) == 4) {
			// order preserving map to int32_t
			constexpr uint32_t sign = std::is_signed_v<T> ? 0u : 0x80000000u;
			uint32_t *y = (uint32_t *)x;
			for (size_t i = 0; i < n; ++i) { y[i] ^= sign; }
			int32_sort((int32_t *)x, (long long)n);
			for (size_t i = 0; i < n; ++i) { y[i] ^= sign; }
			return;
		} else if constexpr (sizeof(T) < 4) {
			// the values fit into an int32_t
			std::vector<int32_t> y(n);
			for (size_t i = 0; i < n; ++i) { y[i] = int32_t(x[i]); }
			int32_sort(y.data(), (long long)n);
			for (size_t i = 0; i < n; ++i) { x[i] = T(y[i]); }
			return;
		}
#endif
		djbsort::sort(x, int64_t(n));
	}
} // end namespace cryptanalysislib
#endif
