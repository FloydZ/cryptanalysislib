#ifndef CRYPTANALYSISLIB_ALGORITHM_NUMERIC_LCM_H
#define CRYPTANALYSISLIB_ALGORITHM_NUMERIC_LCM_H

#include <type_traits>

#include "math/gcd.h"

namespace cryptanalysislib {
	/// Least common multiple of |m| and |n| (like `std::lcm`), 0 if one of
	/// them is 0. Computed as |m| / gcd(m, n) * |n| in the unsigned type, the
	/// result must fit into the common type of `M` and `N`.
	/// \param m[in]: first integer
	/// \param n[in]: second integer
	/// \return lcm(|m|, |n|)
	template<typename M,
	         typename N>
#if __cplusplus > 201709L
	    requires std::is_integral_v<M> &&
	             std::is_integral_v<N> &&
	             (!std::is_same_v<std::remove_cv_t<M>, bool>) &&
	             (!std::is_same_v<std::remove_cv_t<N>, bool>)
#endif
	[[nodiscard]] constexpr std::common_type_t<M, N> lcm(const M m,
	                                                     const N n) noexcept {
		using T = std::common_type_t<M, N>;
		using U = std::make_unsigned_t<T>;
		if ((m == 0) || (n == 0)) {
			return T(0);
		}

		const T tm = T(m), tn = T(n);
		const U a = (tm < 0) ? U(U(0) - U(tm)) : U(tm);
		const U b = (tn < 0) ? U(U(0) - U(tn)) : U(tn);
		return T((a / cryptanalysislib::gcd<U>(a, b)) * b);
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_NUMERIC_LCM_H
