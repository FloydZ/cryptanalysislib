#pragma once

#include <cstdint>

/// Convex (unimodal) sequences, from the fxtbook (Jörg Arndt, Matters Computational).
/// Here a sequence is "convex" if it first increases and then decreases
/// (a single maximum), and "concave" if it first decreases and then increases
/// (a single minimum):
/// 	- strictly: strictly increasing, then strictly decreasing, with at
/// 		least one upstep and one downstep
/// 	- weakly: weakly increasing, then weakly decreasing
/// Tests: `tests/search/convex.cpp`
namespace cryptanalysislib {
	/// \param f[in]: sequence
	/// \param n[in]: length
	/// \return index of the maximum, if `f` is strictly convex, otherwise 0
	template<typename Type>
	[[nodiscard]] constexpr uint64_t test_strictly_convex(const Type *f,
	                                                      const uint64_t n) noexcept {
		if (n <= 2) { return 0; }

		// the following two lines make sure the searches below stop:
		if (f[0] >= f[1]) { return 0; }
		if (f[n - 2] <= f[n - 1]) { return 0; }

		uint64_t i = 0;
		do { ++i; } while (f[i - 1] < f[i]);
		--i;
		// i is second index of last rising pair from start

		uint64_t j = n - 1;
		do { --j; } while (f[j] > f[j + 1]);
		++j;
		// j is first index of last falling pair from end (going backwards)

		if (i != j) { return 0; }
		return i;
	}

	/// \param f[in]: sequence
	/// \param n[in]: length
	/// \return index of the minimum, if `f` is strictly concave, otherwise 0
	template<typename Type>
	[[nodiscard]] constexpr uint64_t test_strictly_concave(const Type *f,
	                                                       const uint64_t n) noexcept {
		if (n <= 2) { return 0; }

		// the following two lines make sure the searches below stop:
		if (f[0] <= f[1]) { return 0; }
		if (f[n - 2] >= f[n - 1]) { return 0; }

		uint64_t i = 0;
		do { ++i; } while (f[i - 1] > f[i]);
		--i;
		// i is second index of last falling pair from start

		uint64_t j = n - 1;
		do { --j; } while (f[j] < f[j + 1]);
		++j;
		// j is first index of last rising pair from end (going backwards)

		if (i != j) { return 0; }
		return i;
	}

	/// \return true if `f` is strictly convex
	template<typename Type>
	[[nodiscard]] constexpr bool is_strictly_convex(const Type *f,
	                                                const uint64_t n) noexcept {
		return 0 != test_strictly_convex(f, n);
	}

	/// \return true if `f` is strictly concave
	template<typename Type>
	[[nodiscard]] constexpr bool is_strictly_concave(const Type *f,
	                                                 const uint64_t n) noexcept {
		return 0 != test_strictly_concave(f, n);
	}

	/// \return true if `f` is weakly convex (weakly unimodal), always for n <= 2
	template<typename Type>
	[[nodiscard]] constexpr bool is_weakly_convex(const Type *f,
	                                              const uint64_t n) noexcept {
		if (n <= 2) { return true; }
		uint64_t j = 1;
		// scan over weakly increasing part
		while (j < n) {
			if (f[j - 1] > f[j]) { break; }
			++j;
		}
		++j;
		// scan over weakly decreasing part
		while (j < n) {
			if (f[j - 1] < f[j]) { break; }
			++j;
		}
		return j >= n;
	}

	/// \return true if `f` is weakly concave, always for n <= 2
	template<typename Type>
	[[nodiscard]] constexpr bool is_weakly_concave(const Type *f,
	                                               const uint64_t n) noexcept {
		if (n <= 2) { return true; }
		uint64_t j = 1;
		// scan over weakly decreasing part
		while (j < n) {
			if (f[j - 1] < f[j]) { break; }
			++j;
		}
		++j;
		// scan over weakly increasing part
		while (j < n) {
			if (f[j - 1] > f[j]) { break; }
			++j;
		}
		return j >= n;
	}

	/// Binary search for the maximum of a strongly unimodal sequence (strictly
	/// increasing, then strictly decreasing; one of the parts may be empty)
	/// in O(log n) comparisons. The result is undefined for other sequences.
	/// \param f[in]: sequence
	/// \param n[in]: length, > 0
	/// \return index of the maximum
	template<typename Type>
	[[nodiscard]] constexpr uint64_t convex_max_index(const Type *f,
	                                                  const uint64_t n) noexcept {
		uint64_t lo = 0, hi = n - 1;
		while (lo < hi) {
			const uint64_t m = lo + (hi - lo) / 2;
			if (f[m] < f[m + 1]) {
				lo = m + 1;
			} else {
				hi = m;
			}
		}
		return lo;
	}

	/// Binary search for the minimum of a strongly unimodal sequence (strictly
	/// decreasing, then strictly increasing) in O(log n) comparisons.
	/// \param f[in]: sequence
	/// \param n[in]: length, > 0
	/// \return index of the minimum
	template<typename Type>
	[[nodiscard]] constexpr uint64_t concave_min_index(const Type *f,
	                                                   const uint64_t n) noexcept {
		uint64_t lo = 0, hi = n - 1;
		while (lo < hi) {
			const uint64_t m = lo + (hi - lo) / 2;
			if (f[m] > f[m + 1]) {
				lo = m + 1;
			} else {
				hi = m;
			}
		}
		return lo;
	}
} // end namespace cryptanalysislib
