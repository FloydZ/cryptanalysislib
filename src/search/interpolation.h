#ifndef CRYPTANALYSISLIB_SEARCH_INTERPOLATION_H
#define CRYPTANALYSISLIB_SEARCH_INTERPOLATION_H

#ifndef CRYPTANALYSISLIB_SEARCH_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/search/search.h>`"
#endif

#include <algorithm>
#include <cmath>
#include <iterator>

#include "helper.h"
#include "hash/hash.h"

namespace cryptanalysislib::internal {
	/// Interpolation-guided lower bound on the hashed values. Interpolation
	/// steps alternate with bisection steps, so the search always terminates
	/// after O(log n) steps, even on non-uniform data or equal keys.
	/// \param first[in]: random access iterator/pointer to the sorted range
	/// \param n[in]: number of elements
	/// \param v[in]: hashed value to search for
	/// \param h[in]: hash function
	/// \return index of the first element `x` with `!(h(x) < v)`, or `n`
	template<typename RandIt,
	         typename V,
	         typename Hash>
	constexpr size_t interpolation_lower_bound(const RandIt first,
	                                           const size_t n,
	                                           const V &v,
	                                           Hash &h) noexcept {
		size_t lo = 0, hi = n;
		bool bisect = false;
		while (lo < hi) {
			const auto hl = h(first[lo]);
			if (!(hl < v)) {
				return lo;
			}

			const auto hh = h(first[hi - 1]);
			if (hh < v) {
				return hi;
			}

			// now: h(first[lo]) < v <= h(first[hi-1]), hence hi - lo >= 2
			size_t pos = lo + (hi - lo) / 2;
			if (!bisect) {
				const double frac = (double(v) - double(hl)) / (double(hh) - double(hl));
				const size_t off = size_t(frac * double(hi - lo - 1));
				pos = lo + (off < 1 ? 1 : off);
				pos = pos > (hi - 1) ? (hi - 1) : pos;
			}
			bisect = !bisect;

			if (h(first[pos]) < v) {
				lo = pos + 1;
			} else {
				hi = pos;
			}
		}

		return lo;
	}
} // end namespace cryptanalysislib::internal


/// Three-point interpolation search algorithm for finding lower bound
/// SRC: https://pages.cs.wisc.edu/~chronis/files/efficiently_searching_sorted_arrays.pdf
///      https://github.com/UWHustle/Efficiently-Searching-In-Memory-Sorted-Arrays/blob/master/src/algorithms/interpolation_search.h
/// 
/// \tparam ForwardIt Type of forward iterator
/// \tparam Hash Type of hash function for comparison
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param value_[in]: Value to search for
/// \param h[in]: Hash function to use for comparison
/// \return Iterator to the first element equal to value_, or last if not found
template<typename ForwardIt,
         typename Hash>
#if __cplusplus > 201709L
    requires std::forward_iterator<ForwardIt> and
             HashFunction<Hash, typename ForwardIt::value_type>
#endif
constexpr ForwardIt lower_bound_interpolation_3p_search(const ForwardIt first,
                                                        const ForwardIt last,
                                                        const typename ForwardIt::value_type &value_,
                                                        Hash h) noexcept {
	const size_t n = last - first;
	const auto v = h(value_);
	const size_t pos = cryptanalysislib::internal::interpolation_lower_bound(first, n, v, h);
	if ((pos < n) && !(v < h(first[pos]))) {
		return first + pos;
	}

	return last;
}

/// Interpolation search variant 1
/// Uses interpolation to estimate the position of a value in a sorted range
/// 
/// \tparam RandIt Type of random access iterator
/// \tparam Hash Type of hash function for comparison
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param value_[in]: Value to search for
/// \param h[in]: Hash function to use for comparison
/// \return Iterator to the first element equal to value_, or last if not found
template<typename RandIt,
         typename Hash>
#if __cplusplus > 201709L
requires std::random_access_iterator<RandIt> and
		 HashFunction<Hash, typename RandIt::value_type>
#endif
constexpr RandIt lower_bound_interpolation_search1(RandIt first,
                                                   RandIt last,
                                                   const typename RandIt::value_type &value_,
                                                   Hash h) noexcept {
	const size_t n = last - first;
	const auto v = h(value_);
	const size_t pos = cryptanalysislib::internal::interpolation_lower_bound(first, n, v, h);
	if ((pos < n) && !(v < h(first[pos]))) {
		return first + pos;
	}

	return last;
}


/// Interpolation search variant 2 for finding lower bound
/// Implementation from: https://medium.com/@vgasparyan1995/interpolation-search-a-generic-implementation-in-c-part-2-164d2c9f55fa
/// 
/// \tparam RandIt Type of random access iterator
/// \tparam Hash Type of hash function for comparison
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param value_[in]: Value to search for
/// \param h[in]: Hash function to use for comparison
/// \return Iterator to the first element equal to value_, or last if not found
template<typename RandIt,
		typename Hash>
#if __cplusplus > 201709L
requires std::random_access_iterator<RandIt> and
		 HashFunction<Hash, typename RandIt::value_type>
#endif
constexpr RandIt lower_bound_interpolation_search2(RandIt first,
                                                   RandIt last,
                                                   const typename RandIt::value_type &value_,
                                                   Hash h) noexcept {
	const size_t n = last - first;
	const auto v = h(value_);
	const size_t pos = cryptanalysislib::internal::interpolation_lower_bound(first, n, v, h);
	if ((pos < n) && !(v < h(first[pos]))) {
		return first + pos;
	}

	return last;
}

/// Array-based interpolation search implementation for lower bound
/// Implementation idea taken from https://en.wikipedia.org/wiki/Interpolation_search
/// 
/// \tparam T Type of elements in the array (must be integral)
/// \tparam Hash Type of hash function for comparison
/// \param __buckets[in]: Pointer to the sorted array to search in
/// \param key[in]: Value to search for
/// \param boffset[in]: Starting offset in the array
/// \param load[in]: Number of elements to search through
/// \param e[in]: Hash function to use for comparison
/// \return Index of the first element equal to key, or -1 if not found
template<typename T,
         typename Hash>
#if __cplusplus > 201709L
	requires HashFunction<Hash, T> and
             std::is_integral_v<T>
#endif
constexpr size_t LowerBoundInterpolationSearch(const T *__buckets,
                                               const T &key,
                                               const size_t boffset,
                                               const size_t load,
                                               Hash &&e) noexcept {
	assert(boffset < load);
	const size_t n = load - boffset;
	const auto data = e(key);
	const size_t pos = boffset + cryptanalysislib::internal::interpolation_lower_bound(__buckets + boffset, n, data, e);
	if ((pos < load) && !(data < e(__buckets[pos]))) {
		return pos;
	}

	return -1;
}

/// Iterator-based interpolation search implementation for lower bound
/// Implementation idea taken from https://en.wikipedia.org/wiki/Interpolation_search
/// 
/// \tparam RandIt Type of random access iterator
/// \tparam Hash Type of hash function for comparison
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param key_[in]: Value to search for
/// \param e[in]: Hash function to use for comparison
/// \return Iterator to the first element equal to key_, or last if not found
///
/// Note: The search assumes that the value type implements the < operator,
/// and values are distributed uniformly
template<typename RandIt,
         typename Hash>
#if __cplusplus > 201709L
requires std::random_access_iterator<RandIt> and
		 HashFunction<Hash, typename RandIt::value_type>
#endif
RandIt LowerBoundInterpolationSearch(RandIt first,
                                     RandIt last,
                                     const typename RandIt::value_type &key_,
                                     Hash e) noexcept {
	const size_t n = last - first;
	const auto v = e(key_);
	const size_t pos = cryptanalysislib::internal::interpolation_lower_bound(first, n, v, e);
	if ((pos < n) && !(v < e(first[pos]))) {
		return first + pos;
	}

	return last;
}


namespace cryptanalysislib {

	/// Perform interpolation search to find a value in a sorted range with a provided hash function
	/// 
	/// \tparam RandIt Type of random access iterator
	/// \tparam Hash Type of hash function for comparison
	/// \param first[in]: Iterator to the beginning of the range
	/// \param last[in]: Iterator to the end of the range
	/// \param key_[in]: Value to search for
	/// \param e[in]: Hash function to use for comparison
	/// \return Iterator to the matching element, or last if not found
	template<typename RandIt,
	         typename Hash>
#if __cplusplus > 201709L
	requires std::random_access_iterator<RandIt> and
			 HashFunction<Hash, typename RandIt::value_type>
#endif
	constexpr RandIt interpolation_search(RandIt first,
										  RandIt last,
										  const typename RandIt::value_type &key_,
										  Hash e) noexcept {
		static_assert(std::is_integral_v<typename decltype(std::function{e})::result_type>,
		              "the return type of the hash function must be a integral type");
		return lower_bound_interpolation_3p_search(first, last, key_, e);
	}

	/// Perform interpolation search to find a value in a sorted range using default hash function
	/// 
	/// \tparam RandIt Type of random access iterator
	/// \tparam Hash Type of hash function for comparison
	/// \param first[in]: Iterator to the beginning of the range
	/// \param last[in]: Iterator to the end of the range
	/// \param key_[in]: Value to search for
	/// \return Iterator to the matching element, or last if not found
	template<typename RandIt,
	         typename Hash>
#if __cplusplus > 201709L
	requires std::random_access_iterator<RandIt> and
			 HashFunction<Hash, typename RandIt::value_type>
#endif
	constexpr RandIt interpolation_search(RandIt first,
										  RandIt last,
										  const typename RandIt::value_type &key_) noexcept {
		using T = RandIt::value_type;
		using H = std::hash<T>;
		H e;
		static_assert(std::is_integral_v<typename decltype(std::function{e})::result_type>,
		              "the return type of the hash function must be a integral type");
		return lower_bound_interpolation_3p_search(first, last, key_, e);
	}

	namespace internal {

		/// Dispatches to the most efficient interpolation search implementation based on benchmarks
		/// 
		/// \tparam It Type of iterator
		/// \tparam Hash Type of hash function for comparison
		/// \param begin[in]: Iterator to the beginning of the range
		/// \param end[in]: Iterator to the end of the range
		/// \param value[in]: Value to search for
		/// \param h[in]: Hash function to use for comparison
		/// \return Iterator to the matching element, or end if not found
		template<typename It,
				 typename Hash>
#if __cplusplus > 201709L
			requires std::forward_iterator<It> and
					 HashFunction<Hash, typename It::value_type>
#endif
		It interpolation_search_dispatch(It begin,
										 It end,
										 const typename It::value_type &value,
										 Hash h) noexcept {
			using T = It::value_type;
			using FF = It(*)(It, It, const T&, Hash);

			// NOTE dont specify as const
			static FF functions[] = {
				LowerBoundInterpolationSearch<It, Hash>,
				lower_bound_interpolation_search2<It, Hash>,
				lower_bound_interpolation_search1<It, Hash>,
				lower_bound_interpolation_3p_search<It, Hash>,
			};

			// NOTE: the first call benchmarks all candidates once, the result
			// 	is a function local static, whose initialisation is thread safe.
			// 	Before, `set` was published before `out` (a concurrent first
			// 	call jumped to `nullptr`) and only `functions[0]` was measured.
			// 	Also, the result of the interpolation dispatch was never used:
			// 	this returned `binary_search_dispatch(...)`.
			static const FF out = [&]() noexcept {
				FF best = functions[0];
				generic_dispatch(best, functions, sizeof(functions)/sizeof(functions[0]),
				                 begin, end, value, h);
				return best;
			}();

			return std::invoke(out, begin, end, value, h);
		}
	}// end namespace internal
}//end namespace cryptanalysis

#endif
