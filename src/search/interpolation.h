#ifndef CRYPTANALYSISLIB_SEARCH_INTERPOLATION_H
#define CRYPTANALYSISLIB_SEARCH_INTERPOLATION_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <iterator>
#include <vector>

#include "helper.h"


template<typename ForwardIt, typename T, typename Hash>
constexpr ForwardIt lower_bound_interpolation_search1(ForwardIt first,
                                                      ForwardIt last,
                                                      const T &value_,
                                                      Hash h) noexcept {
	auto low = first, high = last, mid = first;
	std::advance(high, -1);

	auto data = h(value_);

	while ((h(*high) >= h(*low)) &&
	       (data >= h(*low)) &&
	       (data <= h(*high))) {

		const auto count = std::distance(low, high);
		const auto midstep = std::round((data - h(*low)) / (h(*high) - h(*low)) * count);
		mid = low;
		std::advance(mid, midstep);

		if (midstep == 0) {
			while ((mid > low) && (h(*(--mid)) == h(*mid)))
				std::advance(mid, -1);

			break;
		}


		const auto middata = h(*mid);
		if (middata < data) {
			low = mid;
			std::advance(low, 1);
		} else if (data < middata) {
			high = mid;
			std::advance(high, -1);
		} else {
			// mhhh do the final walk down
			while ((mid > low) && (h(*(--mid)) == h(*mid)))
				std::advance(mid, -1);

			return mid;
		}
	}

	if (data == h(*low))
		return low;
	else
		return last;
}

// taken from:
// https://medium.com/@vgasparyan1995/interpolation-search-a-generic-implementation-in-c-part-2-164d2c9f55fa
template<typename ForwardIt, typename T, typename Hash>
constexpr ForwardIt lower_bound_interpolation_search2(ForwardIt first,
                                                      ForwardIt last,
                                                      const T &value_,
                                                      Hash h) noexcept {
	auto count = std::distance(first, last);
	auto from_iter = first;
	auto to_iter = from_iter;

	std::advance(to_iter, count - 1);
	auto value = h(value_);

	while (count > 0) {
		auto hfrom_iter = h(*from_iter);
		auto hto_iter = h(*to_iter);

		if (value < hfrom_iter) {
			return from_iter;
		} else if (!(hfrom_iter < value)) {
			return from_iter;
		}

		if (hto_iter < value) {
			return ++to_iter;
		} else if (!(value < hto_iter)) {
			return std::lower_bound(from_iter, to_iter, value_, [h](const T &e1, const T &e2) { return h(e1) < h(e2); });
		}

		const auto new_pos = std::round((value - hfrom_iter) / (hto_iter - hfrom_iter) * count);
		auto new_iter = from_iter;
		const auto hnew_iter = h(*new_iter);

		std::advance(to_iter, new_pos);
		if (value < hnew_iter) {
			to_iter = from_iter;
			std::advance(to_iter, new_pos - 1);
		} else if (hnew_iter < value) {
			std::advance(from_iter, new_pos + 1);
		} else {
			return std::lower_bound(from_iter, to_iter, value_, [h](const T &e1, const T &e2) { return h(e1) < h(e2); });
		}

		count = std::distance(from_iter, to_iter);
	}

	return to_iter;
}

/// implementation idea taken from `https://en.wikipedia.org/wiki/Interpolation_search`
template<typename T, typename Extractor>
size_t LowerBoundInterpolationSearch(const T *__buckets,
                                     const T key,
                                     const size_t boffset,
                                     const size_t load,
                                     Extractor e) noexcept {
	ASSERT(boffset < load);
	// example of the extract function#define ISAccess(x) x //((x&mask2)>>b1)

	size_t low = boffset, high = load - 1, mid;
	const T data = e(key);
	while ((e(__buckets[high]) >= e(__buckets[low])) &&
	       (data >= e(__buckets[low])) &&
	       (data <= e(__buckets[high]))) {

		const size_t div = e(__buckets[high]) - e(__buckets[low]);
		const double abc = double(high - low);
		const size_t mul = abc / double(div);
		mid = low + ((data - e(__buckets[low])) * mul);
		ASSERT(mid <= high);

		const T middata = e(__buckets[mid]);
		if (middata < data)
			low = mid + 1;
		else if (data < middata)
			high = mid - 1;
		else {
			// mhhh do the final walk down
			while ((mid > boffset) &&
			       (e(__buckets[mid - 1]) == e(__buckets[mid])))
				mid -= 1;
			return mid;
		}
	}

	if (data == e(__buckets[low]))
		return low;
	else
		return -1;
}

/// Implementation Idea taken from wikipedia: `https://en.wikipedia.org/wiki/Interpolation_search`
/// This search algorithm assumes a lot.
///		T must implement
/// 			<	Operator
/// \tparam ForwardIt	Iterator, must be random access_iterator
/// \tparam Extract		Hash/Extractor function
/// \param first		low end iterator
/// \param last			high end iterator
/// \param key_			value to look for
/// \param h			instantiation of the extractor/hash function
/// \return
template<typename ForwardIt, typename Extract>
#if __cplusplus > 201709L
    requires std::random_access_iterator<ForwardIt>
#endif
ForwardIt LowerBoundInterpolationSearch(ForwardIt first, ForwardIt last, const typename ForwardIt::value_type &key_, Extract e) noexcept {
	using diff_type = typename std::iterator_traits<ForwardIt>::difference_type;
	using T = typename ForwardIt::value_type;

	auto low = first;
	auto mid = first;
	auto high = last;
	std::advance(high, -1);
	const T data = e(key_);

	while ((e(*high) >= e(*low)) &&
	       (data >= e(*low)) &&
	       (data <= e(*high))) {

		const double div = e(*high) - e(*low);
		const double abc = std::distance(low, high);
		const diff_type mul = diff_type(abc / div);

		mid = low;
		std::advance(mid, (data - e(*low)) * mul);
		const T middata = e(*mid);
		ASSERT(middata <= e(*high));

		if (middata < data) {
			low = mid;
			std::advance(low, 1);
		} else if (data < middata) {
			high = mid;
			std::advance(high, -1);
		} else {
			// ugly, but somehow we need to catch the case, when the key is not unique in the sorted data.
			auto tmp_mid = mid;
			std::advance(tmp_mid, -1);
			while ((std::distance(first, mid) > 0) && (e(*tmp_mid) == e(*mid))) {
				std::advance(tmp_mid, -1);
				std::advance(mid, -1);
			}

			return mid;
		}
	}

	if (data == e(*low)) {
		return low;
	}

	// nothing found
	return low;
}

#endif//CRYPTANALYSISLIB_INTERPOLATION_H
