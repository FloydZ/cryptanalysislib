#ifndef CRYPTANALYSISLIB_SEARCH_BINARY_SEARCH_H
#define CRYPTANALYSISLIB_SEARCH_BINARY_SEARCH_H

#include <cassert>
#ifndef CRYPTANALYSISLIB_SEARCH_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/search/search.h>`"
#endif

#include <bit>
#include <cstddef>
#include <cstdint>
#include <functional>

#include "dispatch.h"
#include "hash/hash.h"
#include "math/math.h"


// TODO add namespace, for all the c functions
// TODO add @lemire binary search in parallel in multiple different lists
// TODO add `constexpr` where possible
// TODO add concept to the c functions below to check that `Type` is arithmetic 


/// Binary search for an exact value in a sorted array
/// Source: https://www.jjj.de/fxt/fxtpage.html#fxtbook
///
/// NOTE: f[] must be sorted in ascending order.
/// 
/// \param f[in]: Pointer to the sorted array to search in
/// \param n[in]: Length of the array
/// \param v[in]: Value to search for
/// \return Index of first element in f[] that equals v, or n if no such element exists
template <typename Type>
size_t bsearch(const Type *f,
               const size_t n,
               const Type v) noexcept {
    if (n <= 1) [[unlikely]] {
        return 0;
    }

    size_t nlo=0, nhi=n-1;
    while ( nlo != nhi ) {
        size_t t = (nhi+nlo)/2;

        if ( f[t] < v )  nlo = t + 1;
        else             nhi = t;
    }

    if ( f[nhi]==v )  return nhi;
    else              return n;
}

/// Binary search for the first element greater than or equal to a value
/// NOTE: f[] must be sorted in ascending order.
/// 
/// \param f[in]: Pointer to the sorted array to search in
/// \param n[in]: Length of the array
/// \param v[in]: Value to compare against
/// \return Index of first element in f[] that is >= v, or n if no such element exists
template <typename Type>
size_t bsearch_geq(const Type *f, 
                   const size_t n,
                   const Type v) {
    if (n <= 1) [[unlikely]] {
        return 0;
    }

    size_t nlo=0, nhi=n-1;
    while ( nlo != nhi ) {
        size_t t = (nhi+nlo)/2;

        if ( f[t] < v )  nlo = t + 1;
        else             nhi = t;
    }

    if ( f[nhi]>=v )  return nhi;
    else              return n;
}

/// Binary search for the first element less than or equal to a value
/// NOTE: f[] must be sorted in ascending order.
/// 
/// \param f[in]: Pointer to the sorted array to search in
/// \param n[in]: Length of the array
/// \param v[in]: Value to compare against
/// \return Index of first element in f[] that is <= v, or n if no such element exists
template <typename Type>
size_t bsearch_leq(const Type *f,
                   const size_t n,
                   const Type v) noexcept {
    if (n <= 1) [[unlikely]] {
        return 0;
    }

    size_t nlo=0, nhi=n-1;
    while ( nlo != nhi ) {
        size_t t = (nhi+nlo)/2;

        if ( f[t] > v )  nlo = t + 1;
        else             nhi = t;
    }

    if ( f[nhi]<=v )  return nhi;
    else              return n;
}

/// Binary search for an exact value using a custom comparator
/// NOTE: f[] must be sorted in ascending order according to the comparator
/// 
/// \param f[in]: Pointer to the sorted array to search in
/// \param n[in]: Length of the array
/// \param v[in]: Value to search for
/// \param cmp[in]: Comparison function that returns negative if a<b, 0 if a==b, positive if a>b
/// \return Index of first element in f[] that equals v, or n if no such element exists
template <typename Type>
size_t bsearch(const Type *f,
               const size_t n, 
               const Type v,
               int (*cmp)(const Type &, const Type &)) {
    if (n <= 1) [[unlikely]] {
        return 0;
    }

    size_t nlo=0, nhi=n-1;
    while ( nlo != nhi ) {
        size_t t = (nhi+nlo)/2;
        if ( cmp(f[t], v) < 0 )  nlo = t + 1;
        else                     nhi = t;
    }

    if ( cmp(f[nhi], v)==0 )  return nhi;
    else                      return n;
}

/// Binary search for the first element greater than or equal to a value using 
/// a custom comparator
/// NOTE: f[] must be sorted in ascending order according to the comparator
/// 
/// \param f[in]: Pointer to the sorted array to search in
/// \param n[in]: Length of the array
/// \param v[in]: Value to compare against
/// \param cmp[in]: Comparison function that returns negative if a<b, 0 if a==b, positive if a>b
/// \return Index of first element in f[] that is >= v, or n if no such element exists
template <typename Type>
size_t bsearch_geq(const Type *f,
                   const size_t n,
                   const Type v,
                   int (*cmp)(const Type &, const Type &)) {
    if (n <= 1) [[unlikely]] {
        return 0;
    }

    size_t nlo=0, nhi=n-1;
    while ( nlo != nhi ) {
        size_t t = (nhi+nlo)/2;
        if ( cmp(f[t], v) < 0 )  nlo = t + 1;
        else                   nhi = t;
    }

    if ( cmp(f[nhi], v) >= 0 )  return nhi;
    else                        return n;
}

/// Binary search for the first element less than or equal to a value using a
/// custom comparator
/// NOTE: f[] must be sorted in ascending order according to the comparator
///
/// \param f[in]: Pointer to the sorted array to search in
/// \param n[in]: Length of the array
/// \param v[in]: Value to compare against
/// \param cmp[in]: Comparison function that returns negative if a<b, 0 if a==b, positive if a>b
/// \return Index of first element in f[] that is <= v, or n if no such element exists
template <typename Type>
size_t bsearch_leq(const Type *f,
                   const ulong n, 
                   const Type v,
                   int (*cmp)(const Type &, const Type &)) noexcept {
    if (n <= 1) [[unlikely]] {
        return 0;
    }

    size_t nlo=0, nhi=n-1;
    while ( nlo != nhi ) {
        size_t t = (nhi+nlo)/2;
        if ( cmp(f[t], v) > 0 )  nlo = t + 1;
        else                     nhi = t;
    }

    if ( cmp(f[nhi], v) <= 0 )  return nhi;
    else                        return n;
}

/// Binary search for elements approximately equal to a value within a given
/// tolerance.
/// 
/// NOTE: f[] must be sorted in ascending order.
/// NOTE: da must be positive.
/// NOTE: Makes sense only with inexact types (float or double).
///
/// \param f[in]: Pointer to the sorted array to search in
/// \param n[in]: Length of the array
/// \param v[in]: Value to search for
/// \param da[in]: Tolerance value - elements within v±da will match
/// \return Index of first element x in f[] for which |x-v| <= da,
///             or n if no such element exists
template <typename Type>
ulong bsearch_approx(const Type *f,
                     const ulong n,
                     const Type v,
                     const Type da) noexcept {
    if (n <= 1) [[unlikely]] {
        return 0;
    }

    size_t k = bsearch_geq(f, n, v-da);
    if (k<n) k = bsearch_leq(f+k, n-k, v+da);
    return k;
}

/// Binary search for elements approximately equal to a value within a given 
/// tolerance using a custom comparator
/// 
/// NOTE: f[] must be sorted in ascending order.
/// NOTE: da must be positive.
/// NOTE: Makes sense only with inexact types (float or double).
///
/// \param f[in]: Pointer to the sorted array to search in
/// \param n[in]: Length of the array
/// \param v[in]: Value to search for
/// \param da[in]: Tolerance value - elements within v±da will match
/// \param cmp[in]: Comparison function that returns negative if a<b, 0 if a==b, positive if a>b
/// \return Index of first element x in f[] for which |x-v| <= da according to
///         comparator, or n if no such element exists
template <typename Type>
size_t bsearch_approx(const Type *f,
                     const size_t n,
                     const Type v,
                     const Type da,
                     int (*cmp)(const Type &, const Type &)) noexcept {
    if (n == 0) [[unlikely]] {
        return 0;
    }

    size_t k = bsearch_geq(f, n, v-da, cmp);
    if (k < n) { 
        k = bsearch_leq(f+k, n-k, v+da, cmp); 
    }

    return k;
}

/// Binary search for an exact value in an indirectly sorted array
/// NOTE: f[x[]] must be (index-)sorted in ascending order: f[x[i]] <= f[x[i+1]]
///
/// \param f[in]: Pointer to the base array
/// \param n[in]: Length of the index array
/// \param x[in]: Pointer to the array of indices into f
/// \param v[in]: Value to search for
/// \return Minimal index i so that f[x[i]] == v, or n if no such i exists
template <typename Type>
size_t idx_bsearch(const Type *f,
                   const size_t n, 
                   const size_t *x,
                   const Type v) noexcept {
    if (n <= 1) [[unlikely]] {
        return 0;
    }

    ulong nlo=0, nhi=n-1;
    while ( nlo != nhi ) {
        ulong t = (nhi+nlo)/2;

        if ( f[x[t]] < v )  nlo = t + 1;
        else                nhi = t;
    }

    if ( f[x[nhi]]==v )  return nhi;
    else                 return n;
}

/// Binary search for the first element greater than or equal to a value in an 
/// indirectly sorted array
/// NOTE: f[x[]] must be (index-)sorted in ascending order: f[x[i]] <= f[x[i+1]]
/// 
/// \param f[in]: Pointer to the base array
/// \param n[in]: Length of the index array
/// \param x[in]: Pointer to the array of indices into f
/// \param v[in]: Value to compare against
/// \return Minimal index i so that f[x[i]] >= v, or n if no such i exists
template <typename Type>
size_t idx_bsearch_geq(const Type *f,
                       const size_t n,
                       const ulong *x,
                       const Type v) noexcept {
    if (n <= 1) [[unlikely]] {
        return 0;
    }

    size_t nlo=0, nhi=n-1;
    while ( nlo != nhi ) {
        size_t t = (nhi+nlo)/2;

        if ( f[x[t]] < v )  nlo = t + 1;
        else                nhi = t;
    }

    if ( f[x[nhi]]>=v )  return nhi;
    else                 return n;
}

/// Binary search for an exact value in an indirectly sorted array using a custom comparator
/// NOTE: f[x[]] must be (index-)sorted in ascending order according to the comparator: f[x[i]] <= f[x[i+1]]
/// 
/// \param f[in]: Pointer to the base array
/// \param n[in]: Length of the index array
/// \param x[in]: Pointer to the array of indices into f
/// \param v[in]: Value to search for
/// \param cmp[in]: Comparison function that returns negative if a<b, 0 if a==b, positive if a>b
/// \return Minimal index i so that f[x[i]] == v, or n if no such i exists
template <typename Type>
size_t idx_bsearch(const Type *f,
                   const size_t n, 
                   const ulong *x,
                   const Type v,
                   int (*cmp)(const Type &, const Type &)) {
    if (n <= 1) [[unlikely]] {
        return 0;
    }

    size_t nlo=0, nhi=n-1;
    while ( nlo != nhi ) {
        size_t t = (nhi+nlo)/2;
        if ( cmp(f[x[t]], v) < 0 )  nlo = t + 1;
        else                        nhi = t;
    }

    if ( cmp(f[x[nhi]], v)==0 )  return nhi;
    else                         return n;
}

/// Binary search for the first element greater than or equal to a value in an
/// indirectly sorted array using a custom comparator.
/// NOTE: f[x[]] must be (index-)sorted in ascending order according to the 
/// comparator: f[x[i]] <= f[x[i+1]]
/// 
/// \param f[in]: Pointer to the base array
/// \param n[in]: Length of the index array
/// \param x[in]: Pointer to the array of indices into f
/// \param v[in]: Value to compare against
/// \param cmp[in]: Comparison function that returns negative if a<b, 0 if a==b, positive if a>b
/// \return Minimal index i so that f[x[i]] >= v, or n if no such i exists
template <typename Type>
size_t idx_bsearch_geq(const Type *f,
                       const size_t n,
                       const ulong *x,
                       const Type v,
                      int (*cmp)(const Type &, const Type &)) {
    if (n <= 1) [[unlikely]] {
        return 0;
    }
    ulong nlo=0, nhi=n-1;
    while (nlo != nhi) {
        ulong t = (nhi+nlo)/2;
        if ( cmp(f[x[t]], v)<0 )  nlo = t + 1;
        else                      nhi = t;
    }

    if ( cmp(f[x[nhi]], v)>=0 )  return nhi;
    else                         return n;
}

/// Binary search implementation based on Paul Khuong's branch-prediction optimized algorithm
/// See: https://www.pvk.ca/Blog/2012/07/03/binary-search-star-eliminates-star-branch-mispredictions/
/// NOTE: probably wrong
/// 
/// \tparam T Type of elements in the array
/// \param list[in]: Pointer to the sorted array to search in
/// \param len_list[in]: Length of the array
/// \param value[in]: Value to search for
/// \return Index of the matching element, or -1 if not found or on error
template<typename T>
size_t Khuong_bin_search(const T *list,
                         const size_t len_list,
                         const T value) {
	if (len_list <= 1) [[unlikely]] {
		return 0;
	}

	uint32_t log = ceil_log2(len_list) - 1;
	size_t first_mid = len_list - (1UL << log);
	const T *low = (list[first_mid] < value) ? list + first_mid : list;
	size_t len = 1UL << log;

	for (uint32_t i = log; i != 0; i--) {
		len /= 2;
		T mid = low[len_list];
		if (mid < value) low += len;
	}

	return (*low == value) ? (low - list) : -1;
}

/// Eytzinger layout binary search with prefetching optimization
/// Source: https://en.algorithmica.org/hpc/data-structures/binary-search/
/// 
/// \tparam T Type of elements in the array (must be integral)
/// \param list[in]: Pointer to the sorted array in Eytzinger layout
/// \param len[in]: Length of the array
/// \param x[in]: Value to search for
/// \return Index of the first element not less than x
template<typename T>
#if __cplusplus > 201709L
	requires std::is_integral_v<T>
#endif
int lower_bound_eytzinger_prefetch(const T *list,
                const size_t len,
                const T x) {
	if (len <= 1) [[unlikely]] {
		return 0;
	}

	size_t k = 1;
	while (k <= len) {
		__builtin_prefetch(list + k * 16);
		k = 2 * k + (list[k] < x);
	}
	k >>= __builtin_ffs(~k);
	return list[k];
}


/// Standard binary search implementation to find upper bound with hash function
/// 
/// \tparam ForwardIt Forward iterator type
/// \tparam Hash Hash function type for the value type
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param key_[in]: Value to search for
/// \param h[in]: Hash function to use for comparison
/// \return Iterator to the first element greater than key_, or last if not found
template<typename ForwardIt,
         typename Hash>
#if __cplusplus > 201709L
requires std::forward_iterator<ForwardIt> and
		 HashFunction<Hash, typename ForwardIt::value_type>
#endif
ForwardIt upper_bound_standard_binary_search(ForwardIt first,
                                             ForwardIt last,
                                             const typename ForwardIt::value_type &key_,
                                             Hash h) noexcept {
	const auto count = std::distance(first, last);
	if (count <= 1) {
		return first;
	}

	const auto key = h(key_);
	auto bot = first;
	auto mid = last;
	auto top = last;
	std::advance(top, -1);

	while (bot < top) {
		const auto step = std::distance(bot, top) / 2;
		mid = top;
		std::advance(mid, -step);

		if (key < h(*mid)) {
			top = mid;
			std::advance(top, -1);
		} else {
			bot = mid;
		}
	}

	if (key == h(*top)) {
		return top;
	}

	return last;
}

/// Standard binary search implementation to find lower bound with hash function
/// 
/// \tparam ForwardIt Forward iterator type
/// \tparam Hash Hash function type for the value type
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param key_[in]: Value to search for
/// \param h[in]: Hash function to use for comparison
/// \return Iterator to the first element not less than key_, or last if not found
template<typename ForwardIt,
         typename Hash>
#if __cplusplus > 201709L
requires std::forward_iterator<ForwardIt> and
		 HashFunction<Hash, typename ForwardIt::value_type>
#endif
ForwardIt lower_bound_standard_binary_search(ForwardIt first,
                                             ForwardIt last,
                                             const typename ForwardIt::value_type &key_,
                                             Hash h) noexcept {
	ForwardIt it;
	using T = typename std::iterator_traits<ForwardIt>::difference_type;
	T count = std::distance(first, last);
	if (count <= 1) {
		return first;
	}

	const auto key = h(key_);
	while (count > 0) {
		it = first;
		const T step = count / 2;
		std::advance(it, step);
		if (h(*it) < key) {
			first = ++it;
			count -= step + 1;
		} else
			count = step;
	}
	return first;
}


/// The classic binary search implementation from textbooks
/// 
/// \tparam T Type of elements in the array
/// \param array[in]: Pointer to the sorted array to search in
/// \param array_size[in]: Length of the array
/// \param key[in]: Value to search for
/// \return Index of the element equal to key, or -1 if not found
template<typename T>
size_t standard_binary_search(const T *array,
                              const size_t array_size,
                              const T key) noexcept {
	if (array_size <= 1) {
		return 0;
	}

	size_t bot = 0, mid, top = array_size - 1;


	bot = 0;
	top = array_size - 1;

	while (bot < top) {
		mid = top - (top - bot) / 2;

		if (key < array[mid]) {
			top = mid - 1;
		} else {
			bot = mid;
		}
	}

	if (key == array[top]) {
		return top;
	}

	return -1;
}

/// Faster binary search with no upper bound check - same number of comparisons as standard
/// 
/// \tparam T Type of elements in the array
/// \param array[in]: Pointer to the sorted array to search in
/// \param array_size[in]: Length of the array
/// \param key[in]: Value to search for
/// \return Index of the element equal to key, or -1 if not found
template<typename T>
size_t boundless_binary_search(const T *array,
                               const size_t array_size,
                               const T key) noexcept {
	if (array_size <= 1) {
		return 0;
	}

	uint64_t mid = array_size,
	         bot = 0;

	while (mid > 1) {
		if (key >= array[bot + mid / 2]) {
			bot += mid++ / 2;
		}
		mid /= 2;
	}

	if (key == array[bot]) {
		return bot;
	}

	return -1;
}

/// Binary search variant that performs two comparisons at the final stages for improved performance
/// 
/// \tparam T Type of elements in the array
/// \param array[in]: Pointer to the sorted array to search in
/// \param array_size[in]: Length of the array
/// \param key[in]: Value to search for
/// \return Index of the element equal to key, or -1 if not found
template<typename T>
size_t doubletapped_binary_search(const T *array,
                                  const size_t array_size,
                                  T key) noexcept {
	if (array_size <= 1) {
		return 0;
	}

	size_t mid = array_size, bot = 0;

	while (mid > 2) {
		if (key >= array[bot + mid / 2]) {
			bot += mid++ / 2;
		}
		mid /= 2;
	}

	while (mid--) {
		if (key == array[bot + mid]) {
			return bot + mid;
		}
	}

	return -1;
}

/// Monobound binary search implementation to find upper bound with hash function
/// This variant uses a different comparison strategy that can reduce branch mispredictions
/// 
/// \tparam ForwardIt Forward iterator type
/// \tparam Hash Hash function type for the value type
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param key_[in]: Value to search for
/// \param h[in]: Hash function to use for comparison
/// \return Iterator to the first element greater than key_, or last if not found
template<typename ForwardIt,
         typename Hash>
#if __cplusplus > 201709L
requires std::forward_iterator<ForwardIt> and
		 HashFunction<Hash, typename ForwardIt::value_type>
#endif
ForwardIt upper_bound_monobound_binary_search(ForwardIt first,
                                              ForwardIt last,
                                              const typename ForwardIt::value_type &key_,
                                              Hash h) noexcept {
	auto count = std::distance(first, last);
	const auto key = h(key_);
	auto bot = first;
	auto it = first;
	auto top = last;
	std::advance(top, -1);
	if (count == 0) {
		return last;
    }

	while (count > 1) {
		const auto midc = count / 2;
		it = bot;

		std::advance(it, midc);
		if (key >= h(*it)) {
			std::advance(bot, midc);
		}

		std::advance(top, -midc);
		count = std::distance(first, top);
	}

	if (key == h(*bot))
		return bot;

	return bot;
}

/// Monobound binary search implementation to find lower bound with hash function
/// This variant uses a different comparison strategy that can reduce branch mispredictions
/// 
/// \tparam ForwardIt Forward iterator type
/// \tparam Hash Hash function type for the value type
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param key_[in]: Value to search for
/// \param h[in]: Hash function to use for comparison
/// \return Iterator to the first element not less than key_, or last if not found
template<typename ForwardIt,
         typename Hash>
#if __cplusplus > 201709L
requires std::forward_iterator<ForwardIt> and
		 HashFunction<Hash, typename ForwardIt::value_type>
#endif
ForwardIt lower_bound_monobound_binary_search(ForwardIt first,
                                              ForwardIt last,
                                              const typename ForwardIt::value_type &key_,
                                              Hash h) noexcept {
	auto count = std::distance(first, last);
	const auto key = h(key_);
	auto bot = first;
	auto it = last;
	auto top = last;
	std::advance(top, -1);

	if (count == 0) {
		return last;
    }

	while (count > 1) {
		const auto mid = count / 2;
		it = top;

		std::advance(it, -mid);
		if (key <= h(*it)) {
			std::advance(top, -mid);
		}

		std::advance(bot, mid);
		count = std::distance(bot, last);
	}

	// move the pointer down
	if (key == h(*top)) {
		while (key == h(*top) && (top != first)) {
			top -= 1;
		}
		return top += 1;
	}

	return last;
}

/// Monobound binary search - typically faster than boundless binary search despite more checks
/// Uses a different approach to mid-point calculation and range reduction
/// 
/// \tparam T Type of elements in the array
/// \param array[in]: Pointer to the sorted array to search in
/// \param array_size[in]: Length of the array
/// \param key[in]: Value to search for
/// \return Index of the element equal to key, or 0 if not found
template<typename T>
size_t monobound_binary_search(const T *array,
                               const size_t array_size,
                               const T key) noexcept {
	if (array_size == 0) {
		return 0;
	}

	uint64_t bot = 0, mid, top = array_size;

	while (top > 1) {
		mid = top / 2;

		if (key >= array[bot + mid]) {
			bot += mid;
		}
		top -= mid;
	}


	if (key == array[bot]) {
		return bot;
	}

	return -1;
}

/// TODO doc
/// \tparam ForwardIt
/// \tparam T
/// \tparam Hash
/// \param first
/// \param last
/// \param key_
/// \param h
/// \return
template<typename ForwardIt,
         typename Hash>
#if __cplusplus > 201709L
requires std::forward_iterator<ForwardIt> and
		 HashFunction<Hash, typename ForwardIt::value_type>
#endif
ForwardIt tripletapped_binary_search(ForwardIt first,
                                     ForwardIt last,
                                     const typename ForwardIt::value_type &key_,
                                     Hash h) noexcept {
	std::size_t count = std::distance(first, last);
	if (count == 0) {
		return last;
	}

	auto bot = first;
	auto top = last;
	std::advance(top, -1);

	while (count > 3ul) {
		const size_t mid = count >> 1u;
		if (key_ >= h(*(bot + mid))) {
			std::advance(bot, mid);
		}

		std::advance(top, -mid);
		count = std::distance(first, top);
	}

	while (count--) {
		if (key_ == *(bot + count)) {
			return bot + count;
		}
	}

	return last;
}

/// Triple-tapped binary search - performs three comparisons in the final stage for improved performance
/// 
/// \tparam T Type of elements in the array
/// \param array[in]: Pointer to the sorted array to search in
/// \param array_size[in]: Length of the array
/// \param key[in]: Value to search for
/// \return Index of the element equal to key, or -1 if not found
template<typename T>
size_t tripletapped_binary_search(const T *array,
                                  const size_t array_size,
                                  const T key) noexcept {
	if (array_size == 0) {
		return 0;
	}

	uint64_t bot = 0, mid, top = array_size;
	while (top > 3) {
		mid = top / 2;

		if (key >= array[bot + mid]) {
			bot += mid;
		}
		top -= mid;
	}

	while (top--) {
		if (key == array[bot + top]) {
			return bot + top;
		}
	}

	return -1;
}

/// Quaternary search - better performance on large arrays by using 4-way divisions
/// Uses quaternary divisions initially, then falls back to binary divisions for smaller ranges
/// 
/// \tparam T Type of elements in the array
/// \param array[in]: Pointer to the sorted array to search in
/// \param array_size[in]: Length of the array
/// \param key[in]: Value to search for
/// \return Index of the element equal to key, or -1 if not found
template<typename T>
size_t monobound_quaternary_search(const T *array,
                                   const size_t array_size,
                                   const T key) noexcept {
	if (array_size == 0) {
		return 0;
	}

	uint64_t bot = 0, mid, top = array_size;

	while (top >= 65536) {
		mid = top / 4;
		top -= mid * 3;

		if (key < array[bot + mid * 2]) {
			if (key >= array[bot + mid]) {
				bot += mid;
			}
		} else {
			bot += mid * 2;

			if (key >= array[bot + mid]) {
				bot += mid;
			}
		}
	}

	while (top > 3) {
		mid = top / 2;

		if (key >= array[bot + mid]) {
			bot += mid;
		}
		top -= mid;
	}

	while (top--) {
		if (key == array[bot + top]) {
			return bot + top;
		}
	}
	return -1;
}

/* Copyright Malte Skarupke 2023.
Boost Software License - Version 1.0 - August 17th, 2003
Permission is hereby granted, free of charge, to any person or organization
obtaining a copy of the software and accompanying documentation covered by
this license (the "Software") to use, reproduce, display, distribute,
execute, and transmit the Software, and to prepare derivative works of the
Software, and to permit third-parties to whom the Software is furnished to
do so, all subject to the following:
The copyright notices in the Software and this entire statement, including
the above license grant, this restriction and the following disclaimer,
must be included in all copies of the Software, in whole or in part, and
all derivative works of the Software, unless such copies or derivative
works are solely in the form of machine-executable object code generated by
a source language processor.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.*/

///  https://godbolt.org/#z:OYLghAFBqd5QCxAYwPYBMCmBRdBLAF1QCcAaPECAMzwBtMA7AQwFtMQByARg9KtQYEAysib0QXACx8BBAKoBnTAAUAHpwAMvAFYTStJg1DIApACYAQuYukl9ZATwDKjdAGFUtAK4sGErqSuADJ4DJgAcj4ARpjEIABsAQAOqAqETgwe3r7%2BpClpjgIhYZEsMXGJtpj2hQxCBEzEBFk%2BflwBdpgOGfWNBMUR0bEJHQ1NLTnttmP9oYNlw4kAlLaoXsTI7BzmAMyhyN5YANQmO27ICgT4gqfYJhoAgrv7h5gnZ5fotHhRAHQIt3uTzMewYBy8x1ObjEwBIhAQLEBj2eYNe7zcADcukRiEinsiQVgaGEjsgWKgMQB9eKSCBMUhHKJLI7oVAnADsFiOTAULCOGM8TEc9COEBOZjMZIpTHFAFYNHKuCZZW4GMq3ARzGYjiBxWZTgARXES0VMZm6rXG7UQJlLU5ckzsg1HADuCDomAgGjtOysyP9DwA9IGvKECDTKQRSeSqV4FExgJ7Q4II1H6aSBJcjsnw5JI4zmY6/UHA0cy9GKdTaemmfagcHy0diJgCOsGNy649g46DUC%2B48c6mKximBBB3m0wy0Aws%2BP81Ep5mozmdmZ8wQAhmZ8uw6v12ZmQxUJhVJsklGi/2AJw8vkCgzCt4QIGNxtagiYLOKhnmWX6lVqiqmoSte9xXlqUojnKGg/mYso7OqgEalqoEaFeFogTsRpaqadr%2BuBaEYWYVqihuKx6iREAEAesHEThNp4fiV4%2BsWjbNq2xDtkwnZPE6/YPHOUaQbQY5hkO6bTrOYkToyi7btmu5rlGG5yVJgh7spB5HEeJ5nhenLXre/KCo%2Boovq%2BZbvp%2BUbfnK/6quqwH6vhYEQTGtDQbB8GIY5KEuYReqGpRZqoehFH0WRtGUdR5GWvRtb4Sx5lluxbYdr6fZ8QGQIfiwSQPpgUIEAAnkkjCsG8ACSBAMiVZXMGwRwACq1aV5WNW4eV4pJH6qEkxBHNVjLEIYyAIPQCgKJStCoC6sSUlEawMOgEBDTEwChKQyUWTtu1lkNrhbY8e0nTtPXNeK8Qjt4mBHQ8p0PeWnVJJd5jxBmeWNJghYGcd5afCAIBpAAXpg%2Bb0EYBAIO8zquEcAC0jKYBtaoZX9ZZ4FQooQ8AUMw4aRzehyxauWhqWcUcrg8a5WX4vd/1XIDINg1GlyYC9BMAyAUSEJSVAzSQEA41DSUBo2mOimzL1gGABPC9Db1vR9SRfTayOhMqFhS8qBoMtdXjfT9rE7fLCOcx%2BHOWEcSpo/TFkS0LjC4wrWEE0Tl7%2BQRV7k%2B2VO26T4G09tjZSzDRxczzBCUpsdCO5DCCi3br7raEYdw4j2u22%2BtONvwA0QKHgYE/qvrhxbRwy279pl%2BzRxF1hRxaR7Sfi1jEBoJ9zZqyjmva7Kuv8mIBtLEboVgSn7bWOb7PU2hNO9ujr7BsJ3ehIn2cL0nPtIyj4pch3KtdwAVBPetD991NB48zPzgYBeMyAWIOCQUKCbcl1r8T23nUwXhENyYcogjTBONT8U0ZpzWIAtJaK1Qi/AnhAcicDXCIIZFwDQXgGRcwmgoV%2B0lIy3EQevcs28ub4EuKNT0cCEHkRCv7J0HAVi0E4LKXgfgOBaFIKgTgbgp5WwUGsDYbxdg8FIAQTQjCVgAGsQCSCvL8dkOx2R/g0AADlUWYdkqj2jsn0JwSQbCJFcM4LwBQIAYLiI4Yw0gcBYBIAPh6MgFB26oDyo4kAwAuCrj4HQD8xAzE2iMTzZgxBiqcFEcExoxUADyURtDYnCbwDubBBDRIYLQMJVjSBYCiF4YA0JaC0DMdwXgWAWCGGAOILJ%2BBmzdCxMUzhJ4uh/y2KIsM1QjHfCAVEjwWAjEEGIHgFgiSVj8wTAoAAangTALpon1USTIQQIgxDsCkIs%2BQSg1BGN0AEAwRgUB8JsF0sxkAVioHPBkYp8NPiGlMJYawZgFTw2iTsXgFJYiDKwCc1BnRujOAgK4CYfg4KBGWgMUo5QEi6PyOkAQQKQAgphbUcFQwKi6N%2BbUXo4xPCtARbKKoNQegzBRQsNF0w%2BjwpBRQpoJLIXxHZCsAR6xNgSCYSwwxWTuEcCOKoVR8R4Y0iOMAZAyBrY7F%2BNaXAhASDih2FwJYvBLFaBHqQGR8oJXxDMJIdkV4dFmC4Oo6QzCOAGNIOwzhXLTHmLERIlVxqzAcotSYm1ViVVYn8RkWRQA%3D%3D
/// Branchless lower bound implementation that minimizes branch mispredictions
/// Uses bit manipulation operations to reduce branching in the inner loop
/// 
/// \tparam It Iterator type
/// \tparam Cmp Comparison function type
/// \param begin[in]: Iterator to the beginning of the range
/// \param end[in]: Iterator to the end of the range
/// \param value[in]: Value to compare against
/// \param compare[in]: Comparison function
/// \return Iterator to the first element not less than value, or end if not found
template<typename It,
         typename Cmp>
#if __cplusplus > 201709L
	requires std::forward_iterator<It> and
	         CompareFunction<Cmp, typename It::value_type>
#endif
[[nodiscard]] constexpr It branchless_lower_bound(It begin,
                          It end,
                          const typename It::value_type &value,
                          Cmp compare) noexcept {
	std::size_t length = end - begin;
	if (length == 0) {
		return end;
	}

	std::size_t step = std::bit_floor(length);

	if (step != length && compare(begin[step], value)) {
		length -= step + 1;
		if (length == 0) {
			return end;
		}

		step = std::bit_ceil(length);
		begin = end - step;
	}

	for (step /= 2; step != 0; step /= 2) {
		if (compare(begin[step], value)) {
			begin += step;
		}
	}

	return begin + compare(*begin, value);
}

/// Branchless lower bound implementation with default less-than comparator
/// 
/// \tparam It Iterator type
/// \param begin[in]: Iterator to the beginning of the range
/// \param end[in]: Iterator to the end of the range
/// \param value[in]: Value to compare against
/// \return Iterator to the first element not less than value, or end if not found
template<typename It>
#if __cplusplus > 201709L
	requires std::forward_iterator<It>
#endif
[[nodiscard]] constexpr It branchless_lower_bound(It begin,
                                             It end,
                                             const typename It::value_type &value) noexcept {
	return branchless_lower_bound(begin, end, value, std::less<>{});
}


/// Branchless lower bound implementation that uses a hash function for comparison
/// 
/// \tparam It Iterator type
/// \tparam Hash Hash function type
/// \param begin[in]: Iterator to the beginning of the range
/// \param end[in]: Iterator to the end of the range
/// \param value[in]: Value to compare against
/// \param h[in]: Hash function to use for comparisons
/// \return Iterator to the first element not less than value, or end if not found
template<typename It,
         typename Hash>
#if __cplusplus > 201709L
	requires std::forward_iterator<It> and
             HashFunction<Hash, typename It::value_type>
#endif
[[nodiscard]] constexpr It branchless_lower_bound(It begin,
						It end,
						const typename It::value_type &value,
						Hash h) noexcept {
	std::size_t length = end - begin;
	if (length <= 0) {
		return end;
	}

	std::size_t step = std::bit_floor(length);
	const auto v = h(value);

	if (step != length && (h(begin[step]) < v)) {
		length -= step + 1;
		if (length == 0) {
			return end;
		}

		step = std::bit_ceil(length);
		begin = end - step;
	}

	for (step /= 2; step != 0; step /= 2) {
		if (h(begin[step]) < v) {
			begin += step;
		}
	}

	return begin + (h(*begin) < v);
}

namespace cryptanalysislib::search {

	/// Find the first element not less than a value using a hash function
	/// 
	/// \tparam It Iterator type
	/// \tparam Hash Hash function type
	/// \param begin[in]: Iterator to the beginning of the range
	/// \param end[in]: Iterator to the end of the range
	/// \param value[in]: Value to compare against
	/// \param h[in]: Hash function to use for comparison
	/// \return Iterator to the first element not less than value, or end if not found
	template<typename It,
			 typename Hash>
#if __cplusplus > 201709L
	requires std::forward_iterator<It> and
	         HashFunction<Hash, typename It::value_type>
#endif
	[[nodiscard]] constexpr inline It lower_bound(It begin,
							  					  It end,
							  					  const typename It::value_type &value,
							  					  Hash h) noexcept {
		return branchless_lower_bound(begin, end, value, h);
	}

	/// Find the first element not less than a value using a custom comparator
	/// 
	/// \tparam It Iterator type
	/// \tparam Compare Comparison function type
	/// \param begin[in]: Iterator to the beginning of the range
	/// \param end[in]: Iterator to the end of the range
	/// \param value[in]: Value to compare against
	/// \param cmp[in]: Comparison function
	/// \return Iterator to the first element not less than value, or end if not found
	template<typename It,
			 typename Compare>
#if __cplusplus > 201709L
	requires std::forward_iterator<It> and
			 CompareFunction<Compare, typename It::value_type>
#endif
	[[nodiscard]] constexpr inline It lower_bound(It begin,
												  It end,
												  const typename It::value_type &value,
												  Compare cmp) noexcept {
		return branchless_lower_bound(begin, end, value, cmp);
	}

	/// Search for a value in a sorted range using a hash function
	/// 
	/// \tparam It Iterator type
	/// \tparam Hash Hash function type
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
	[[nodiscard]] constexpr inline It binary_search(It begin,
												    It end,
												    const typename It::value_type &value,
												    Hash h) noexcept {
		return branchless_lower_bound(begin, end, value, h);
	}

	/// Search for a value in a sorted range using a custom comparator
	/// 
	/// \tparam It Iterator type
	/// \tparam Compare Comparison function type
	/// \param begin[in]: Iterator to the beginning of the range
	/// \param end[in]: Iterator to the end of the range
	/// \param value[in]: Value to search for
	/// \param cmp[in]: Comparison function
	/// \return Iterator to the matching element, or end if not found
	template<typename It,
			 typename Compare>
#if __cplusplus > 201709L
	requires std::forward_iterator<It> and
			 CompareFunction<Compare, typename It::value_type>
#endif
	[[nodiscard]] constexpr inline It binary_search(It begin,
												    It end,
												    const typename It::value_type &value,
												    Compare cmp) noexcept {
		return branchless_lower_bound(begin, end, value, cmp);
	}

	/// Search for a value in a sorted range using default less-than comparison
	/// 
	/// \tparam It Iterator type
	/// \param begin[in]: Iterator to the beginning of the range
	/// \param end[in]: Iterator to the end of the range
	/// \param value[in]: Value to search for
	/// \return Iterator to the matching element, or end if not found
	template<typename It,
			 typename Compare>
#if __cplusplus > 201709L
	requires std::forward_iterator<It>
#endif
	[[nodiscard]] constexpr inline It binary_search(It begin,
												    It end,
												    const typename It::value_type &value) noexcept {
		using T = It::value_type;
		return binary_search(begin, end, value, std::less<T>());
	}

	namespace internal {

		/// Dispatches to the most efficient binary search implementation based on hardware capabilities
		/// 
		/// \tparam It Iterator type
		/// \tparam Compare Comparison function type
		/// \param begin[in]: Iterator to the beginning of the range
		/// \param end[in]: Iterator to the end of the range
		/// \param value[in]: Value to search for
		/// \param cmp[in]: Comparison function
		/// \return Iterator to the matching element, or end if not found
		template<typename It,
				 typename Compare>
#if __cplusplus > 201709L
			requires std::forward_iterator<It> and
					 CompareFunction<Compare, typename It::value_type>
#endif
		It binary_search_dispatch(It begin,
									It end,
									const typename It::value_type &value,
									Compare cmp) noexcept {
			using T = It::value_type;
			using FF = It(*)(It, It, const T&, Compare);

			static FF out;
			static bool set = false;
			if (set) [[likely]] {
				return std::invoke(out, begin, end, value, cmp);
			}

			set = true;

			// NOTE dont specify as const
			static FF functions[] = {
				branchless_lower_bound<It, Compare>
			};

			generic_dispatch(out, functions, 1, begin, end, value, cmp);
			return binary_search_dispatch(begin, end, value, cmp);
		}

		/// Dispatches to the most efficient binary search implementation with hash function
		/// 
		/// \tparam It Iterator type
		/// \tparam Hash Hash function type
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
		It binary_search_dispatch(It begin,
									It end,
									const typename It::value_type &value,
									Hash h) noexcept {
			using T = It::value_type;
			using FF = It(*)(It, It, const T&, Hash);

			static FF out;
			static bool set = false;

			if (set) [[likely]] {
				return std::invoke(out, begin, end, value, h);
			}

			set = true;

			// NOTE dont specify as const
			static FF functions[] = {
				branchless_lower_bound<It, Hash>,
				lower_bound_standard_binary_search<It, Hash>,
				lower_bound_monobound_binary_search<It, Hash>,
				tripletapped_binary_search<It, Hash>,
			};

			const auto d = generic_dispatch(out, functions, 1, begin, end, value, h);
			return binary_search_dispatch(begin, end, value, h);
		}


	};
};

#endif
