#ifndef CRYPTANALYSISLIB_SEARCH_LINEAR_H
#define CRYPTANALYSISLIB_SEARCH_LINEAR_H

#include <cstdint>
#include <algorithm>
#include <iterator>

/// linear search, needs to run backwards so it's stable
/// \tparam ForwardIt
/// \tparam T
/// \tparam Compare
/// \param first
/// \param last
/// \param key
/// \param compare
/// \return
template<class ForwardIt, class T, class Compare>
constexpr ForwardIt upper_bound_linear_search(ForwardIt first, ForwardIt last, T &key, Compare compare) noexcept {
	typename std::iterator_traits<ForwardIt>::difference_type count, step;
	count = std::distance(first, last);
	step = -1;
	ForwardIt it = last;
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
template<class ForwardIt, class T, class Compare>
constexpr ForwardIt lower_bound_linear_search(ForwardIt first, ForwardIt last, T &key, Compare compare) noexcept {
	typename std::iterator_traits<ForwardIt>::difference_type count, step;
	count = std::distance(first, last);
	step = 1;
	ForwardIt it = first;
	while (--count) {
		if (compare(*it, key)) {
			return it;
		}
		std::advance(it, step);
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
template<class ForwardIt, class T, class Hash>
constexpr ForwardIt upper_bound_breaking_linear_search(ForwardIt first, ForwardIt last, const T &key_, Hash h) noexcept {
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
template<class ForwardIt, class T, class Hash>
constexpr ForwardIt lower_bound_breaking_linear_search(ForwardIt first, ForwardIt last, const T &key_, Hash h) noexcept {
	auto count = std::distance(first, last);
	if (count == 0)
		return first;

	const auto key = h(key_);
	auto bot = first;

	while(--count) {
		if (key <= h(*bot)){
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
uint64_t breaking_linear_search(T *array, uint64_t array_size, T key) noexcept {
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
#endif //CRYPTANALYSISLIB_LINEAR_H
