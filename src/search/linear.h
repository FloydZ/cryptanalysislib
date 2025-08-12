#ifndef CRYPTANALYSISLIB_SEARCH_LINEAR_H
#define CRYPTANALYSISLIB_SEARCH_LINEAR_H

#ifndef CRYPTANALYSISLIB_SEARCH_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/search/search.h>`"
#endif

#include <cstdint>
#include <algorithm>
#include <iterator>

#include "hash/hash.h"

// TODO add simd implementation (probably just steal is from `src/algorithm/find.h`)
// TODO add namespace 
// TODO add dispatch function as in `src/search/binary.h`

/// Linear search to find the upper bound of a value
/// Searches backwards through the range to maintain stability
/// 
/// \tparam ForwardIt Type of forward iterator
/// \tparam Compare Type of comparison function
/// \param first[in]: Iterator to the beginning of the range
/// \param last[in]: Iterator to the end of the range
/// \param key[in]: Value to compare against
/// \param compare[in]: Comparison function that returns true if first argument is less than second
/// \return Iterator to the first element greater than key, or last if no such element exists
template<class ForwardIt,
         class Compare>
#if __cplusplus > 201709L
requires std::forward_iterator<ForwardIt> and
		 CompareFunction<Compare, typename ForwardIt::value_type>
#endif
constexpr ForwardIt upper_bound_linear_search(const ForwardIt first,
                                              const ForwardIt last,
                                              const typename ForwardIt::value_type &key,
                                              Compare compare) noexcept {
	typename std::iterator_traits<ForwardIt>::difference_type
	        count = std::distance(first, last),
	        step = -1;

	if (count == 0) {
		return first;
	}

	ForwardIt it = last;
	std::advance(it, step);

	while (--count) {
		if (compare(*it, key)) {
			return it;
		}

		std::advance(it, step);
	}

	return last;
}


/// Linear search to find the lower bound of a value
/// This algorithm iterates forward through the range to maintain stability
/// The lower bound is defined as the first element in the range not less than the key
/// 
/// \tparam ForwardIt Type of forward iterator
/// \tparam Compare Type of comparison function
/// \param first[in] Iterator to the beginning of the range
/// \param last[in] Iterator to the end of the range
/// \param key[in] Value to compare against
/// \param compare[in] Comparison function that returns true if first argument is less than second
/// \return Iterator to the first element not less than key, or last if no such element exists
template<class ForwardIt,
         class Compare>
#if __cplusplus > 201709L
requires std::forward_iterator<ForwardIt> and
		 CompareFunction<Compare, typename ForwardIt::value_type>
#endif
constexpr ForwardIt lower_bound_linear_search(const ForwardIt first,
                                              const ForwardIt last,
                                              const typename ForwardIt::value_type &key,
                                              Compare compare) noexcept {
	typename std::iterator_traits<ForwardIt>::difference_type
			count = std::distance(first, last),
			step = 1;

	if (count == 0) {
		return first;
	}

	ForwardIt it = first;
	do {
		if (compare(key, *it)) {
			return it;
		}

		std::advance(it, step);

		count -= 1;
	} while (count);

	return last;
}

/// Linear search to find the upper bound of a value using a hash function
/// Searches backwards through the range and stops when it finds an element with hash not greater than key
/// 
/// \tparam ForwardIt Type of forward iterator
/// \tparam Hash Type of hash function
/// \param first[in] Iterator to the beginning of the range
/// \param last[in] Iterator to the end of the range
/// \param key_[in] Value to compare against
/// \param h[in] Hash function that returns a comparable value
/// \return Iterator to the element with matching hash, or last if no such element exists
template<class ForwardIt,
         class Hash>
#if __cplusplus > 201709L
requires std::forward_iterator<ForwardIt> and
		 HashFunction<Hash, typename ForwardIt::value_type>
#endif
constexpr ForwardIt upper_bound_breaking_linear_search(const ForwardIt first,
                                                       const ForwardIt last,
                                                       const typename ForwardIt::value_type &key_,
                                                       Hash h) noexcept {
	auto count = std::distance(first, last);
	if (count == 0)
		return first;

	const auto key = h(key_);
	auto top = last;
	std::advance(top, -1);
	while(--count) {
		if (key >= h(*top)){
			break;
		}

		std::advance(top, -1);
	}

	if (key == h(*top)) {
		return top;
	}

	return last;
}

/// Linear search to find the lower bound of a value using a hash function
/// Searches forward through the range and returns the first element with matching hash
/// 
/// \tparam ForwardIt Type of forward iterator
/// \tparam Hash Type of hash function
/// \param first[in] Iterator to the beginning of the range
/// \param last[in] Iterator to the end of the range
/// \param key_[in] Value to compare against
/// \param h[in] Hash function that returns a comparable value
/// \return Iterator to the first element with matching hash, or last if no such element exists
template<class ForwardIt,
         class Hash>
#if __cplusplus > 201709L
requires std::forward_iterator<ForwardIt> and
		 HashFunction<Hash, typename ForwardIt::value_type>
#endif
constexpr ForwardIt lower_bound_breaking_linear_search(const ForwardIt first,
                                                       const ForwardIt last,
                                                       const typename ForwardIt::value_type &key_,
                                                       Hash h) noexcept {
	auto count = std::distance(first, last);
	if (count == 0) {
		return first;
	}

	const auto key = h(key_);
	auto bot = first;

	do {
		const auto val = h(*bot);
		if (key == val){
			return bot;
		}

		std::advance(bot, 1);
	} while(count -= 1);

	return last;
}

/// Linear search optimized for larger arrays
/// Searches backwards through the array and breaks early when a value is found
/// 
/// \tparam T Type of array elements
/// \param array[in] Pointer to the beginning of the array
/// \param array_size[in] Size of the array
/// \param key[in] Value to search for
/// \return Index of the found element, or -1 if not found
template<typename T>
constexpr uint64_t breaking_linear_search(const T *array,
                                const uint64_t array_size,
                                const T &key) noexcept {
	uint64_t top = array_size;

	if (array_size == 0) {
		return -1;
	}

	while (--top) {
		if (key >= array[top]) {
			break;
		}
	}

	if (key == array[top]) {
		return top;
	}

	return -1;
}

namespace cryptanalysislib::search {
	/// Linear search using a hash function
	/// Wrapper around lower_bound_breaking_linear_search
	/// 
	/// \tparam ForwardIt Type of forward iterator
	/// \tparam Hash Type of hash function
	/// \param first[in] Iterator to the beginning of the range
	/// \param last[in] Iterator to the end of the range
	/// \param key_[in] Value to compare against
	/// \param h[in] Hash function that returns a comparable value
	/// \return Iterator to the first element with matching hash, or last if no such element exists
	template<class ForwardIt,
	         class Hash>
#if __cplusplus > 201709L
	requires std::forward_iterator<ForwardIt> and
			 HashFunction<Hash, typename ForwardIt::value_type>
#endif
	constexpr inline ForwardIt linear_search(const ForwardIt first,
	                                  const ForwardIt last,
	                                  const typename ForwardIt::value_type &key_,
									  Hash h) noexcept {
		return lower_bound_breaking_linear_search(first, last, key_, h);
	}

	/// Linear search using a comparison function
	/// Wrapper around lower_bound_linear_search
	/// 
	/// \tparam ForwardIt Type of forward iterator
	/// \tparam Compare Type of comparison function
	/// \param first[in] Iterator to the beginning of the range
	/// \param last[in] Iterator to the end of the range
	/// \param key_[in] Value to compare against
	/// \param cmp[in] Comparison function that returns true if first argument is less than second
	/// \return Iterator to the first element not less than key, or last if no such element exists
	template<class ForwardIt,
			 class Compare>
#if __cplusplus > 201709L
	requires std::forward_iterator<ForwardIt> and
	          CompareFunction<Compare, typename ForwardIt::value_type>
#endif
	constexpr inline ForwardIt linear_search(const ForwardIt first,
											 const ForwardIt last,
											 const typename ForwardIt::value_type &key_,
											 Compare cmp) noexcept {
		return lower_bound_linear_search(first, last, key_, cmp);
	}
}
#endif
