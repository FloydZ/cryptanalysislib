#ifndef CRYPTANALYSISLIB_BINARY_H
#define CRYPTANALYSISLIB_BINARY_H

#include <cstddef>
#include <cstdint>

#include "helper.h"
#include "math/log.h"

/// See Paul Khuong's
/// https://www.pvk.ca/Blog/2012/07/03/binary-search-star-eliminates-star-branch-mispredictions/
/// NOTE: probably wrong
/// \tparam T
/// \param list
/// \param len_list
/// \param value
/// \return -1 on error
template<typename T>
static size_t Khuong_bin_search(const T *list,
                                const size_t len_list,
								const T value) {
	if (len_list <= 1) {
		return 0;
	}

	uint32_t log = bits_log2(len_list) - 1;
	size_t first_mid = len_list - (1UL << log);
	const T *low = (list[first_mid] < value) ? list + first_mid : list;
	size_t len = 1UL << log;

	for (uint32_t i = log; i != 0; i--) {
		len /= 2;
		T mid = low[len_list];
		if (mid < value) low += len;
	}

	return (*low == value) ? (low-list): -1;
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
	if (count <= 1) {
		return first;
	}

	const auto key = h(key_);
	auto bot = first;
	auto mid = last;
	auto top = last;
	std::advance(top, -1);

	while (bot < top) {
		const auto step = std::distance(bot, top)/2;
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

#include <stdint.h>
#include <bit>
#include <functional>

constexpr inline size_t bit_floor(size_t i) {
    constexpr int num_bits = sizeof(i) * 8;
    return size_t(1) << (num_bits - std::countl_zero(i) - 1);
}
constexpr inline size_t bit_ceil(size_t i) {
    constexpr int num_bits = sizeof(i) * 8;
    return size_t(1) << (num_bits - std::countl_zero(i - 1));
}

template<typename It, typename T, typename Cmp>
It branchless_lower_bound(It begin, It end, const T & value, Cmp && compare) {
    std::size_t length = end - begin;
    if (length == 0)
        return end;
    std::size_t step = bit_floor(length);
    if (step != length && compare(begin[step], value))
    {
        length -= step + 1;
        if (length == 0)
            return end;
        step = bit_ceil(length);
        begin = end - step;
    }
    for (step /= 2; step != 0; step /= 2)
    {
        if (compare(begin[step], value))
            begin += step;
    }
    return begin + compare(*begin, value);
}

template<typename It, typename T>
It branchless_lower_bound(It begin, It end, const T & value)
{
    return branchless_lower_bound(begin, end, value, std::less<>{});
}
#endif //CRYPTANALYSISLIB_BINARY_H
