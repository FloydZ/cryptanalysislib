#ifndef CRYPTANALYSISLIB_ALGORITHM_PERMUTATION_H
#define CRYPTANALYSISLIB_ALGORITHM_PERMUTATION_H

#include <functional>
#include <iterator>

#include "algorithm/reverse.h"
#include "algorithm/swap.h"

namespace cryptanalysislib {
	/// Checks if [first2, last2) is a permutation of [first1, last1) w.r.t.
	/// the equivalence `p` (like `std::is_permutation`): after the common
	/// prefix, every distinct element must occur equally often in both
	/// ranges. O(n^2) comparisons in the worst case.
	/// \param first1[in]: beginning of the first range
	/// \param last1[in]: end of the first range
	/// \param first2[in]: beginning of the second range
	/// \param last2[in]: end of the second range
	/// \param p[in]: equivalence predicate
	/// \return true if the ranges are permutations of each other
	template<class ForwardIt1,
	         class ForwardIt2,
	         class BinaryPred>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt1> &&
	             std::forward_iterator<ForwardIt2>
#endif
	[[nodiscard]] constexpr bool is_permutation(ForwardIt1 first1,
	                                            ForwardIt1 last1,
	                                            ForwardIt2 first2,
	                                            ForwardIt2 last2,
	                                            BinaryPred p) noexcept {
		// skip the common prefix
		for (; (first1 != last1) && (first2 != last2) && p(*first1, *first2); ++first1, (void) ++first2) {}

		if (std::distance(first1, last1) != std::distance(first2, last2)) {
			return false;
		}

		for (ForwardIt1 i = first1; i != last1; ++i) {
			// count every distinct element only at its first occurrence
			bool seen = false;
			for (ForwardIt1 j = first1; j != i; ++j) {
				if (p(*j, *i)) {
					seen = true;
					break;
				}
			}
			if (seen) {
				continue;
			}

			std::iter_difference_t<ForwardIt2> c2 = 0;
			for (ForwardIt2 j = first2; j != last2; ++j) {
				c2 += p(*i, *j);
			}
			if (c2 == 0) {
				return false;
			}

			std::iter_difference_t<ForwardIt1> c1 = 0;
			for (ForwardIt1 j = i; j != last1; ++j) {
				c1 += p(*i, *j);
			}
			if (c1 != c2) {
				return false;
			}
		}
		return true;
	}

	/// same as above, with `==`
	template<class ForwardIt1,
	         class ForwardIt2>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt1> &&
	             std::forward_iterator<ForwardIt2>
#endif
	[[nodiscard]] constexpr bool is_permutation(ForwardIt1 first1,
	                                            ForwardIt1 last1,
	                                            ForwardIt2 first2,
	                                            ForwardIt2 last2) noexcept {
		return cryptanalysislib::is_permutation(first1, last1, first2, last2, std::equal_to<>());
	}

	/// same as above, the second range has as many elements as the first one
	template<class ForwardIt1,
	         class ForwardIt2,
	         class BinaryPred>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt1> &&
	             std::forward_iterator<ForwardIt2> &&
	             (!std::forward_iterator<BinaryPred>)
#endif
	[[nodiscard]] constexpr bool is_permutation(ForwardIt1 first1,
	                                            ForwardIt1 last1,
	                                            ForwardIt2 first2,
	                                            BinaryPred p) noexcept {
		ForwardIt2 last2 = first2;
		std::advance(last2, std::distance(first1, last1));
		return cryptanalysislib::is_permutation(first1, last1, first2, last2, p);
	}

	/// same as above, with `==`
	template<class ForwardIt1,
	         class ForwardIt2>
#if __cplusplus > 201709L
	    requires std::forward_iterator<ForwardIt1> &&
	             std::forward_iterator<ForwardIt2>
#endif
	[[nodiscard]] constexpr bool is_permutation(ForwardIt1 first1,
	                                            ForwardIt1 last1,
	                                            ForwardIt2 first2) noexcept {
		return cryptanalysislib::is_permutation(first1, last1, first2, std::equal_to<>());
	}

	/// Transforms [first, last) into the next permutation in the
	/// lexicographical order w.r.t. `comp` (like `std::next_permutation`). The
	/// last permutation (non-increasing) becomes the first one (sorted).
	/// \param first[in]: beginning of the range
	/// \param last[in]: end of the range
	/// \param comp[in]: less-than comparison
	/// \return true if the new permutation is greater than the old one,
	/// 	false if the range was the last permutation
	template<class BidirIt,
	         class Compare>
#if __cplusplus > 201709L
	    requires std::bidirectional_iterator<BidirIt>
#endif
	constexpr bool next_permutation(BidirIt first,
	                                BidirIt last,
	                                Compare comp) noexcept {
		if (first == last) {
			return false;
		}

		BidirIt i = last;
		if (first == --i) {
			return false;
		}

		while (true) {
			const BidirIt i1 = i;
			// longest non-increasing suffix [i1, last), `*i` the pivot before it
			if (comp(*--i, *i1)) {
				// rightmost element greater than the pivot
				BidirIt i2 = last;
				while (!comp(*i, *--i2)) {}
				cryptanalysislib::iter_swap(i, i2);
				cryptanalysislib::reverse(i1, last);
				return true;
			}

			if (i == first) {
				cryptanalysislib::reverse(first, last);
				return false;
			}
		}
	}

	/// same as above, with `<`
	template<class BidirIt>
#if __cplusplus > 201709L
	    requires std::bidirectional_iterator<BidirIt>
#endif
	constexpr bool next_permutation(BidirIt first,
	                                BidirIt last) noexcept {
		return cryptanalysislib::next_permutation(first, last, std::less<>());
	}

	/// Transforms [first, last) into the previous permutation in the
	/// lexicographical order w.r.t. `comp` (like `std::prev_permutation`). The
	/// first permutation (sorted) becomes the last one (non-increasing).
	/// \param first[in]: beginning of the range
	/// \param last[in]: end of the range
	/// \param comp[in]: less-than comparison
	/// \return true if the new permutation is less than the old one, false
	/// 	if the range was the first permutation
	template<class BidirIt,
	         class Compare>
#if __cplusplus > 201709L
	    requires std::bidirectional_iterator<BidirIt>
#endif
	constexpr bool prev_permutation(BidirIt first,
	                                BidirIt last,
	                                Compare comp) noexcept {
		if (first == last) {
			return false;
		}

		BidirIt i = last;
		if (first == --i) {
			return false;
		}

		while (true) {
			const BidirIt i1 = i;
			// longest non-decreasing suffix [i1, last), `*i` the pivot before it
			if (comp(*i1, *--i)) {
				// rightmost element less than the pivot
				BidirIt i2 = last;
				while (!comp(*--i2, *i)) {}
				cryptanalysislib::iter_swap(i, i2);
				cryptanalysislib::reverse(i1, last);
				return true;
			}

			if (i == first) {
				cryptanalysislib::reverse(first, last);
				return false;
			}
		}
	}

	/// same as above, with `<`
	template<class BidirIt>
#if __cplusplus > 201709L
	    requires std::bidirectional_iterator<BidirIt>
#endif
	constexpr bool prev_permutation(BidirIt first,
	                                BidirIt last) noexcept {
		return cryptanalysislib::prev_permutation(first, last, std::less<>());
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_PERMUTATION_H
