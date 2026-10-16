#ifndef CRYPTANALYSISLIB_ALGORITHM_CLAMP_H
#define CRYPTANALYSISLIB_ALGORITHM_CLAMP_H

#include <concepts>
#include <functional>

namespace cryptanalysislib {
	/// Clamps `v` into [lo, hi] w.r.t. `comp` (like `std::clamp`)
	/// NOTE: `hi` must not be less than `lo`
	/// \param v[in]: value to clamp
	/// \param lo[in]: lower bound
	/// \param hi[in]: upper bound
	/// \param comp[in]: less-than comparison
	/// \return `lo` if `v < lo`, `hi` if `hi < v`, otherwise `v`
	template<class T,
	         class Compare>
	[[nodiscard]] constexpr const T &clamp(const T &v,
	                                       const T &lo,
	                                       const T &hi,
	                                       Compare comp) noexcept {
		return comp(v, lo) ? lo : (comp(hi, v) ? hi : v);
	}

	/// same as above, with `<`
	/// NOTE: the requires-clause makes it more constrained than `std::clamp`,
	/// 	so unqualified calls which see both are not ambiguous.
	template<class T>
#if __cplusplus > 201709L
	    requires std::totally_ordered<T>
#endif
	[[nodiscard]] constexpr const T &clamp(const T &v,
	                                       const T &lo,
	                                       const T &hi) noexcept {
		return cryptanalysislib::clamp(v, lo, hi, std::less<>());
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_CLAMP_H
