#ifndef CRYPTANALYSISLIB_SEARCH_LINEAR_H
#define CRYPTANALYSISLIB_SEARCH_LINEAR_H

#ifndef CRYPTANALYSISLIB_SEARCH_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/search/search.h>`"
#endif

#include <cstdint>
#include <algorithm>
#include <iterator>

#include "hash/hash.h"

/// linear search, needs to run backwards so it's stable
/// \tparam ForwardIt
/// \tparam T
/// \tparam Compare
/// \param first
/// \param last
/// \param key
/// \param compare
/// \return
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


/// linear search, needs to run forward so it's stable
/// \tparam ForwardIt
/// \tparam T
/// \tparam Compare
/// \param first
/// \param last
/// \param key
/// \param compare
/// \return
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

///
/// \tparam ForwardIt
/// \tparam T
/// \tparam Hash
/// \param first
/// \param last
/// \param key_
/// \param h
/// \return
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

///
/// \tparam ForwardIt
/// \tparam T
/// \tparam Hash
/// \param first
/// \param last
/// \param key_
/// \param h
/// \return
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

	while(--count) {
		const auto val = h(*bot);
		if (key <= val){
			break;
		}

		std::advance(bot, 1);
	}

	if (key == h(*bot)) {
		return bot;
	}

	return last;
}

/// faster than linear on larger arrays
/// \tparam T
/// \param array
/// \param array_size
/// \param key
/// \return
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
