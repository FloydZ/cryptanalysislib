#ifndef CRYPTANALYSISLIB_BINARY_H
#define CRYPTANALYSISLIB_BINARY_H

#include <cstddef>
#include <cstdint>

#include "helper.h"

/// See Paul Khuong's
/// https://www.pvk.ca/Blog/2012/07/03/binary-search-star-eliminates-star-branch-mispredictions/
/// \tparam T
/// \param list
/// \param len_list
/// \param value
/// \return
template<typename T>
static size_t Khuong_bin_search(const T *list,
                                const size_t len_list,
								const T value) {
	if (len_list <= 1) {
		return 0;
	}

	uint32_t log = constexpr_bits_log2(len_list) - 1;
	size_t first_mid = len_list - (1UL << log);
	const T *low = (list[first_mid] < value) ? list + first_mid : list;
	size_t len = 1UL << log;

	for (uint32_t i = log; i != 0; i--) {
		len /= 2;
		T mid = low[len_list];
		if (mid < value) low += len;
	}

	return (*low == value) ? low - list : low - list + 1;
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
template<typename ForwardIt, typename T, typename Hash>
ForwardIt upper_bound_standard_binary_search(ForwardIt first,
                                             ForwardIt last,
                                             const T &key_,
                                             Hash h) noexcept {
	const auto count = std::distance(first, last);
	if (count == 0)
		return first;

	const auto key = h(key_);
	auto bot = first;
	auto mid = first;
	auto top = last;
	std::advance(top, -1);

	while (bot < top) {
		const auto step = std::distance(bot, top)/2;
		mid = top;
		std::advance(top, -step);

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

///
/// \tparam ForwardIt
/// \tparam T
/// \tparam Hash
/// \param first
/// \param last
/// \param key_
/// \param h
/// \return
template<typename ForwardIt, typename T, typename Hash>
ForwardIt lower_bound_standard_binary_search(ForwardIt first,
                                             ForwardIt last,
                                             const T &key_,
                                             Hash h) noexcept {
	ForwardIt it;
	typename std::iterator_traits<ForwardIt>::difference_type count, step;
	count = std::distance(first, last);
	const auto key = h(key_);
	while (count > 0) {
		it = first;
		step = count / 2;
		std::advance(it, step);
		if (h(*it) < key) {
			first = ++it;
			count -= step + 1;
		}
		else
			count = step;
	}
	return first;
}


/// the standard binary search from text books
/// \tparam T
/// \param array
/// \param array_size
/// \param key
/// \return
template<typename T>
size_t standard_binary_search(const T *array,
                              const size_t array_size,
                              const T key) noexcept {
	if (array_size == 0) {
		return -1;
	}

	size_t bot = 0, mid, top = array_size - 1;


	bot = 0;
	top = array_size - 1;

	while (bot < top) {
		mid = top - (top - bot) / 2;

		if (key < array[mid]) {
			top = mid - 1;
		}
		else {
			bot = mid;
		}
	}

	if (key == array[top]) {
		return top;
	}

	return -1;
}

/// faster than the standard binary search, same number of checks
/// \tparam T
/// \param array
/// \param array_size
/// \param key
/// \return
template<typename T>
size_t boundless_binary_search(const T *array,
                               const size_t array_size,
                               const T key) noexcept {
	if (array_size == 0) {
		return -1;
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

/// always double tap
/// \tparam T
/// \param array
/// \param array_size
/// \param key
/// \return
template<typename T>
size_t doubletapped_binary_search(const T *array,
                                  const size_t array_size,
                                  T key) noexcept {
	size_t mid = array_size, bot = 0;

	while (mid > 2)	{
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

///
/// \tparam ForwardIt
/// \tparam T
/// \tparam Hash
/// \param first
/// \param last
/// \param key_
/// \param h
/// \return
template<typename ForwardIt, typename T, typename Hash>
ForwardIt upper_bound_monobound_binary_search(ForwardIt first,
                                              ForwardIt last,
                                              const T &key_,
                                              Hash h) noexcept {
	auto count = std::distance(first, last);
	const auto key = h(key_);
	auto bot = first;
	auto  it = first;
	auto top = last;
	std::advance(top, -1);
	if (count == 0)
		return last;

	while(count > 1) {
		const auto midc = count/2;
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

///
/// \tparam ForwardIt
/// \tparam T
/// \tparam Hash
/// \param first
/// \param last
/// \param key_
/// \param h
/// \return
template<typename ForwardIt, typename T, typename Hash>
ForwardIt lower_bound_monobound_binary_search(ForwardIt first,
                                              ForwardIt last,
                                              const T &key_,
                                              Hash h) noexcept {
	auto count = std::distance(first, last);
	const auto key = h(key_);
	auto bot = first;
	auto  it = last;
	auto top = last;
	std::advance(top, -1);

	if (count == 0)
		return last;

	while(count > 1) {
		const auto mid = count/2;
		it = top;

		std::advance(it, -mid);
		if (key <= h(*it)) {
			std::advance(top, -mid);
		}

		std::advance(bot, mid);
		count = std::distance(bot, last);
	}

	if (key == h(*top))
		return top;

	return top;
}

/// faster than the boundless binary search, more checks
/// \tparam T
/// \param array
/// \param array_size
/// \param key
/// \return
template<typename T>
size_t monobound_binary_search(const T *array,
                               const size_t array_size,
                               const T key) noexcept {
	if (array_size == 0) {
		return -1;
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

///
/// \tparam ForwardIt
/// \tparam T
/// \tparam Hash
/// \param first
/// \param last
/// \param key_
/// \param h
/// \return
template<typename ForwardIt, typename T, typename Hash>
ForwardIt upper_bound_tripletapped_binary_search(ForwardIt first, ForwardIt last, const T &key_, Hash h) noexcept {
	/// TODO
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
template<typename ForwardIt, typename T, typename Hash>
ForwardIt lower_bound_tripletapped_binary_search(ForwardIt first, ForwardIt last, const T &key_, Hash h) noexcept {
	/// TODO
}


/// heck, always triple tap ⁍⁍⁍
/// \tparam T
/// \param array
/// \param array_size
/// \param key
/// \return
template<typename T>
size_t tripletapped_binary_search(const T *array,
                                  const size_t array_size,
                                  const  T key) noexcept {
	if (array_size == 0) {
		return -1;
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

// better performance on large arrays
template<typename T>
size_t monobound_quaternary_search(const T *array,
                                   const size_t array_size,
                                   const T key) noexcept {
	if (array_size == 0) {
		return -1;
	}

	uint64_t bot = 0, mid, top = array_size;

	while (top >= 65536) {
		mid = top / 4;
		top -= mid * 3;

		if (key < array[bot + mid * 2]) {
			if (key >= array[bot + mid]) {
				bot += mid;
			}
		}
		else {
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
ForwardIt upper_bound_monobound_interpolated_search(ForwardIt first, ForwardIt last, T &key_, Hash h) noexcept {
	/// TODO
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
template<typename ForwardIt, typename T, typename Hash>
ForwardIt upper_bound_adaptive_binary_search(ForwardIt first,
                                             ForwardIt last,
                                             const T &key_,
                                             Hash h) noexcept {
	static uint64_t balance;
	static ForwardIt i = first;
	ForwardIt bot, top, mid;
	auto count = std::distance(first, last);
	const auto key = h(key_);

	if ((balance > 32) || (count <= 64)) {
		bot = first;
		top = last;
		goto adaptive_binary_search_monobound;
	}


	bot = i;
	top = bot;
	std::advance(top, 32);

	if (key >= h(*bot)) {
		while (true) {
			if (std::distance(first, bot) >= std::distance(top, last)) {
				top = last;
				std::advance(top, -std::distance(first, bot));
				break;
			}
			std::advance(bot, std::distance(first, top));

			if (key < h(*bot)) {
				std::advance(bot, -std::distance(first, top));
				break;
			}
			std::advance(top, std::distance(first, top));
		}
	} else {
		while (true) {
			if (h(*bot) < h(*top)) {
				top = bot;
				bot = first;
				break;
			}

			std::advance(first, std::distance(first, top));

			if (key >= h(*bot)) {
				break;
			}

			std::advance(top, std::distance(first, top));

		}
	}

	adaptive_binary_search_monobound:
	while (std::distance(first, top) > 3) {
		const auto mid = std::distance(first, top)/2;
		auto it = bot;
		std::advance(it, mid);
		if (key >= h(*it)) {
			std::advance(bot, mid);
		}

		std::advance(top, -mid);

	}

	balance = i > bot ? i-bot : bot - i;
	i = bot;
	while(top > first) {
		std::prev(top);

		auto it = bot;
		std::advance(it, std::distance(first, top));
		if (key == h(*it)) {
			std::advance(bot, std::distance(first, top));
			return bot;
		}
	}

	return last;
}

#endif //CRYPTANALYSISLIB_BINARY_H
