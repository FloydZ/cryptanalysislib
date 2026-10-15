#ifndef CRYPTANALYSISLIB_ALGORITHM_RECURRENCE_H
#define CRYPTANALYSISLIB_ALGORITHM_RECURRENCE_H

#include <cassert>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "math/math.h"

// Solving linear recurrences over Z/mZ (m prime), source: KACTL.
// Given some brute-forced sequence s[0], s[1], ..., s[2n-1], `berlekamp_massey`
// finds the shortest possible recurrence relation in O(n^2). After that,
// `linear_recurrence` finds s[k] in O(n^2 log k).

namespace cryptanalysislib {
	/// Given a sequence s[0], ..., s[2n-1] finds the smallest linear recurrence
	/// of size L <= n compatible with s, i.e.
	/// 	s[i] = \sum_{j=1}^{L} C[j-1] s[i-j] mod `mod`
	/// \param s[in]: sequence, elements in [0, mod)
	/// \param mod[in]: prime modulus
	/// \return C[0, ..., L-1]
	template<std::unsigned_integral T>
	constexpr std::vector<T> berlekamp_massey(const std::vector<T> &s,
	                                          const T mod) noexcept {
		const size_t n = s.size();
		if (n == 0) {
			return {};
		}

		size_t L = 0, m = 0;
		std::vector<T> C(n, 0), B(n, 0), tmp;
		C[0] = B[0] = 1u % mod;
		T b = 1u % mod;
		for (size_t i = 0; i < n; ++i) {
			++m;
			T d = s[i] % mod;
			for (size_t j = 1; j <= L; ++j) {
				d = addmod<T>(d, mulmod<T>(C[j], s[i - j] % mod, mod), mod);
			}

			if (d == 0) {
				continue;
			}

			tmp = C;
			// b != 0 is invertible as `mod` is prime
			const T coef = mulmod<T>(d, mod_pow<T>(b, mod - 2u, mod), mod);
			for (size_t j = m; j < n; ++j) {
				C[j] = submod<T>(C[j], mulmod<T>(coef, B[j - m], mod), mod);
			}

			if (2 * L > i) {
				continue;
			}

			L = i + 1 - L;
			B = tmp;
			b = d;
			m = 0;
		}

		std::vector<T> ret(L, 0);
		for (size_t j = 0; (j < L) && (j + 1 < n); ++j) {
			ret[j] = submod<T>(0, C[j + 1], mod);
		}
		return ret;
	}

	/// Given A[0, ..., n-1] and C[0, ..., n-1] satisfying
	/// 	A[i] = \sum_{j=1}^{n} C[j-1] A[i-j] mod `mod`
	/// computes A[k] mod `mod`.
	/// \param A[in]: first n elements of the sequence
	/// \param C[in]: recurrence, e.g. from `berlekamp_massey`
	/// \param k[in]: index of the element to compute
	/// \param mod[in]: modulus
	/// \return A[k] mod `mod`
	template<std::unsigned_integral T>
	constexpr T linear_recurrence(const std::vector<T> &A,
	                              const std::vector<T> &C,
	                              uint64_t k,
	                              const T mod) noexcept {
		const size_t n = A.size();
		assert(C.size() == n);
		if (n == 0) {
			return 0;
		}

		// multiplies two polynomials mod the characteristic polynomial
		auto combine = [&](const std::vector<T> &a, const std::vector<T> &b) {
			std::vector<T> res(a.size() + b.size() - 1, 0);
			for (size_t i = 0; i < a.size(); ++i) {
				for (size_t j = 0; j < b.size(); ++j) {
					res[i + j] = addmod<T>(res[i + j], mulmod<T>(a[i], b[j], mod), mod);
				}
			}

			for (size_t i = 2 * n; i > n; --i) {
				for (size_t j = 0; j < n; ++j) {
					res[i - 1 - j] = addmod<T>(res[i - 1 - j], mulmod<T>(res[i], C[j] % mod, mod), mod);
				}
			}

			res.resize(n + 1);
			return res;
		};

		std::vector<T> pol(n + 1, 0), e(n + 1, 0);
		pol[0] = e[1] = 1u % mod;
		// NOTE: k + 1 steps, `k` must be < 2**64 - 1
		for (++k; k; k /= 2) {
			if (k % 2) {
				pol = combine(pol, e);
			}
			e = combine(e, e);
		}

		T res = 0;
		for (size_t i = 0; i < n; ++i) {
			res = addmod<T>(res, mulmod<T>(pol[i + 1], A[i] % mod, mod), mod);
		}
		return res;
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_RECURRENCE_H
