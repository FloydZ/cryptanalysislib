#ifndef CRYPTANALYSISLIB_ALGORITHM_NUMERIC_IOTA_H
#define CRYPTANALYSISLIB_ALGORITHM_NUMERIC_IOTA_H

#include <iterator>

namespace cryptanalysislib {
	/// Fills [first, last) with `value`, `value + 1`, ... (like `std::iota`,
	/// the value is incremented with `++`)
	/// NOTE: the requires-clause makes it more constrained than `std::iota`,
	/// 	so unqualified calls which see both are not ambiguous.
	/// \param first[in]: beginning of the range
	/// \param last[in]: end of the range
	/// \param value[in]: first value
	template<class ForwardIt,
	         class T>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	constexpr void iota(ForwardIt first,
	                    ForwardIt last,
	                    T value) noexcept {
		for (; first != last; ++first, (void) ++value) {
			*first = value;
		}
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_NUMERIC_IOTA_H
