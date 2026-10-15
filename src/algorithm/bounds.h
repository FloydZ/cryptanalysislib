#ifndef CRYPTANALYSISLIB_ALGORITHM_BOUNDS_H
#define CRYPTANALYSISLIB_ALGORITHM_BOUNDS_H

#include <functional>
#include <iterator>
#include <utility>

/// `upper_bound` and `equal_range` (like the `std::` versions). `lower_bound`
/// and `binary_search` are in `search/binary.h`.
namespace cryptanalysislib {
	namespace internal {
		/// first element in the sorted [first, last) which is not less than
		/// `value` w.r.t. `comp`, or `last`
		template<class ForwardIt,
		         class T,
		         class Compare>
		constexpr ForwardIt bounds_lower_bound(ForwardIt first,
		                                       ForwardIt last,
		                                       const T &value,
		                                       Compare &comp) noexcept {
			auto count = std::distance(first, last);
			while (count > 0) {
				ForwardIt it = first;
				const auto step = count / 2;
				std::advance(it, step);
				if (comp(*it, value)) {
					first = ++it;
					count -= step + 1;
				} else {
					count = step;
				}
			}
			return first;
		}

		/// first element in the sorted [first, last) which is greater than
		/// `value` w.r.t. `comp`, or `last`
		template<class ForwardIt,
		         class T,
		         class Compare>
		constexpr ForwardIt bounds_upper_bound(ForwardIt first,
		                                       ForwardIt last,
		                                       const T &value,
		                                       Compare &comp) noexcept {
			auto count = std::distance(first, last);
			while (count > 0) {
				ForwardIt it = first;
				const auto step = count / 2;
				std::advance(it, step);
				if (!comp(value, *it)) {
					first = ++it;
					count -= step + 1;
				} else {
					count = step;
				}
			}
			return first;
		}
	} // end namespace internal

	/// \param first[in]: beginning of the range, sorted (partitioned) w.r.t.
	/// 	`comp(value, x)`
	/// \param last[in]: end of the range
	/// \param value[in]: value to compare the elements with
	/// \param comp[in]: less-than comparison
	/// \return first element `x` in [first, last) with `comp(value, x)`, or last
	template<class ForwardIt,
	         class T,
	         class Compare>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	[[nodiscard]] constexpr ForwardIt upper_bound(ForwardIt first,
	                                              ForwardIt last,
	                                              const T &value,
	                                              Compare comp) noexcept {
		return internal::bounds_upper_bound(first, last, value, comp);
	}

	/// same as above, with `<`
	template<class ForwardIt,
	         class T>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	[[nodiscard]] constexpr ForwardIt upper_bound(ForwardIt first,
	                                              ForwardIt last,
	                                              const T &value) noexcept {
		return cryptanalysislib::upper_bound(first, last, value, std::less<>());
	}

	/// Range of the elements in the sorted [first, last) which are equivalent
	/// to `value` w.r.t. `comp` (like `std::equal_range`): one binary search
	/// until the first equivalent element, then a lower and an upper bound
	/// search in the two remaining halves.
	/// \param first[in]: beginning of the sorted range
	/// \param last[in]: end of the sorted range
	/// \param value[in]: value to compare the elements with
	/// \param comp[in]: less-than comparison
	/// \return {lower bound, upper bound} of `value`
	template<class ForwardIt,
	         class T,
	         class Compare>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	[[nodiscard]] constexpr std::pair<ForwardIt, ForwardIt> equal_range(ForwardIt first,
	                                                                    ForwardIt last,
	                                                                    const T &value,
	                                                                    Compare comp) noexcept {
		auto len = std::distance(first, last);
		while (len > 0) {
			const auto half = len / 2;
			ForwardIt middle = first;
			std::advance(middle, half);
			if (comp(*middle, value)) {
				first = ++middle;
				len -= half + 1;
			} else if (comp(value, *middle)) {
				len = half;
			} else {
				ForwardIt left = internal::bounds_lower_bound(first, middle, value, comp);
				std::advance(first, len);
				ForwardIt right = internal::bounds_upper_bound(++middle, first, value, comp);
				return {left, right};
			}
		}
		return {first, first};
	}

	/// same as above, with `<`
	template<class ForwardIt,
	         class T>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt>
#endif
	[[nodiscard]] constexpr std::pair<ForwardIt, ForwardIt> equal_range(ForwardIt first,
	                                                                    ForwardIt last,
	                                                                    const T &value) noexcept {
		return cryptanalysislib::equal_range(first, last, value, std::less<>());
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_BOUNDS_H
