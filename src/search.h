#ifndef SMALLSECRETLWE_SEARCH_H
#define SMALLSECRETLWE_SEARCH_H

#include <algorithm>
#include <iostream>
#include <vector>

#include "helper.h"

/*
 * See Paul Khuong's
 * https://www.pvk.ca/Blog/2012/07/03/binary-search-star-eliminates-star-branch-mispredictions/
 */
template<typename T>
static size_t Khuong_bin_search(const T *list, size_t len_list,
                         T value) {
	if (len_list <= 1) return 0;
	uint32_t log = constexpr_bits_log2(len_list) - 1;
	size_t first_mid = len_list - (1UL << log);
	const T *low = (list[first_mid] < value) ? list + first_mid : list;
	len_list = 1UL << log;

	for (unsigned i = log; i != 0; i--) {
		len_list /= 2;
		T mid = low[len_list];
		if (mid < value) low += len_list;
	}

	return (*low == value) ? low - list : low - list + 1;
}

template<typename List>
static size_t Khuong_bin_search(const List &list, typename List::value_type &value){
	return Khuong_bin_search(list.data(), list.size(), value);
}


template<typename ForwardIt, typename T, typename Hash>
constexpr ForwardIt lower_bound_interpolation_search1(ForwardIt first, ForwardIt last, const T &value_, Hash h) {
	auto low = first, high = last, mid = first;
	std::advance(high, -1);

	auto data = h(value_);

	while  ((h(*high) >= h(*low)) &&
			(data >= h(*low)) &&
	        (data <= h(*high))) {

		const auto count = std::distance(low, high);
		const auto midstep = std::round((data - h(*low))/(h(*high) - h(*low)) * count);
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
		}
		else if (data < middata) {
			high = mid;
			std::advance(high, -1);
		}
		else {
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
constexpr ForwardIt lower_bound_interpolation_search2(ForwardIt first, ForwardIt last, const T &value_, Hash h) {
	auto count = std::distance(first, last);
	auto from_iter = first;
	auto to_iter = from_iter;

	std::advance(to_iter, count - 1);
	auto value = h(value_);

	while(count > 0) {
		auto hfrom_iter = h(*from_iter);
		auto hto_iter = h(*to_iter);

		if (value < hfrom_iter){
			return from_iter;
		} else if (!(hfrom_iter < value)) {
			return from_iter;
		}

		if (hto_iter < value) {
			return ++to_iter;
		} else if (!(value < hto_iter)) {
			return std::lower_bound(from_iter, to_iter, value_, [h](const T &e1, const T &e2) { return h(e1) < h(e2); });
		}

		const auto new_pos = std::round((value - hfrom_iter)/(hto_iter - hfrom_iter) * count);
		auto new_iter = from_iter;
		const auto hnew_iter = h(*new_iter);

		std::advance(to_iter, new_pos);
		if (value < hnew_iter) {
			to_iter = from_iter;
			std::advance(to_iter, new_pos - 1);
		} else if (hnew_iter  < value) {
			std::advance(from_iter, new_pos + 1);
		} else {
			return std::lower_bound(from_iter, to_iter, value_, [h](const T &e1, const T &e2) { return h(e1) < h(e2); });
		}

		count = std::distance(from_iter, to_iter);
	}

	return to_iter;
}

// linear search, needs to run backwards so it's stable
template<class ForwardIt, class T, class Compare>
constexpr ForwardIt upper_bound_linear_search(ForwardIt first, ForwardIt last, T &key, Compare compare) {
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


// linear search, needs to run forward so it's stable
template<class ForwardIt, class T, class Compare>
constexpr ForwardIt lower_bound_linear_search(ForwardIt first, ForwardIt last, T &key, Compare compare) {
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

// 
template<class ForwardIt, class T, class Hash>
constexpr ForwardIt upper_bound_breaking_linear_search(ForwardIt first, ForwardIt last, const T &key_, Hash h) {
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

// 
template<class ForwardIt, class T, class Hash>
constexpr ForwardIt lower_bound_breaking_linear_search(ForwardIt first, ForwardIt last, const T &key_, Hash h) {
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

// faster than linear on larger arrays
template<typename T>
uint64_t breaking_linear_search(T *array, uint64_t array_size, T key) {
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

template<typename ForwardIt, typename T, typename Hash>
ForwardIt upper_bound_standard_binary_search(ForwardIt first, ForwardIt last, const T &key_, Hash h) {
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

template<typename ForwardIt, typename T, typename Hash>
ForwardIt lower_bound_standard_binary_search(ForwardIt first, ForwardIt last, const T &key_, Hash h) {
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

// the standard binary search from text books
template<typename T>
uint64_t standard_binary_search(T *array, uint64_t array_size, T key) {
	uint64_t bot, mid, top;

	if (array_size == 0) {
		return -1;
	}

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

// faster than the standard binary search, same number of checks
template<typename T>
uint64_t boundless_binary_search(T *array, uint64_t array_size, T key) {
	uint64_t mid, bot;

	if (array_size == 0) {
		return -1;
	}
	bot = 0;
	mid = array_size;

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

// always double tap
template<typename T>
uint64_t doubletapped_binary_search(T *array, uint64_t array_size, T key) {
	uint64_t mid, bot;

	bot = 0;
	mid = array_size;

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

template<typename ForwardIt, typename T, typename Hash>
ForwardIt upper_bound_monobound_binary_search(ForwardIt first, ForwardIt last, const T &key_, Hash hash) {
	auto count = std::distance(first, last);
	const auto key = h(key_);
	auto bot = first;
	auto  it = first;
	auto top = last;
	std::advance(top, -1);
	if (count == 0)
		return last;

	while(count > 1) {
		// std::advance(mid, -count/2);
		const auto midc = count/2;
		it = bot;
		std::advance(it, midc);
		if (key >= h(*it)) {
			std::advance(bot, midc);
		}

		std::advance(top, -midc);
		count = std::distance(bot, top);
	}
		
	if (key == h(*bot))
		return bot;
	return last;
}

//TODO not working
template<typename ForwardIt, typename T, typename Hash>
ForwardIt lower_bound_monobound_binary_search(ForwardIt first, ForwardIt last, const T &key_, Hash h) {
	auto count = std::distance(first, last);
	const auto key = h(key_);
	auto bot = first;
	auto  it = first;
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
		count = std::distance(bot, top);
	}

	if (key == h(*top))
		return top;

	return last;
}

// faster than the boundless binary search, more checks
template<typename T>
uint64_t monobound_binary_search(T *array, uint64_t array_size, T key) {
	uint64_t bot, mid, top;

	if (array_size == 0) {
		return -1;
	}

	bot = 0;
	top = array_size;

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


template<typename ForwardIt, typename T, typename Hash>
ForwardIt upper_bound_tripletapped_binary_search(ForwardIt first, ForwardIt last, const T &key_, Hash h) {

}

template<typename ForwardIt, typename T, typename Hash>
ForwardIt lower_bound_tripletapped_binary_search(ForwardIt first, ForwardIt last, const T &key_, Hash h) {

}


// heck, always triple tap ⁍⁍⁍
template<typename T>
uint64_t tripletapped_binary_search(T *array, uint64_t array_size, T key) {
	uint64_t bot, mid, top;

	bot = 0;
	top = array_size;

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
uint64_t monobound_quaternary_search(T *array, uint64_t array_size, T key) {
	uint64_t bot, mid, top;

	if (array_size == 0) {
		return -1;
	}

	bot = 0;
	top = array_size;

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



template<class ForwardIt, class T, class Hash>
ForwardIt upper_bound_monobound_interpolated_search(ForwardIt first, ForwardIt last, T &key_, Hash h) {

}


// TODO not finished implemented
// requires an even/uniform distribution
// do not need a comparison operator
// sorted in increasing order
template<class ForwardIt, class T, class Hash>
ForwardIt lower_bound_monobound_interpolated_search(ForwardIt first, ForwardIt last, T &key_, Hash h) {
	typename std::iterator_traits<ForwardIt>::difference_type count, step, array_size;
	array_size = std::distance(first, last);
	count = array_size;
	step = 1;

	// This is stupid.
	ForwardIt llast = last;
	std::advance(llast, -1);

	const T key = h(key_);
	//unsigned int mid, top;
	int min, max;

	if (array_size == 0) {
		return last;
	}

	if (key < h(*first)) {
		return last;
	}

	ForwardIt bot = first;
	min = h(*first);
	max = h(*llast);
	ASSERT(min <= max);

	//std::advance(bot, (bot*(float) (key - min) / (max - min)));
	return last;
}

template<typename ForwardIt, typename T, typename Hash>	
ForwardIt upper_bound_adaptive_binary_search(ForwardIt first, ForwardIt last, const T &key_, Hash h) {
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
				std::advance(top, -std::distance(first, bot)); // TODO maybe -1?
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


//
//// requires in order sequential access
//template<class ForwardIt, class T, class Compare>
//ForwardIt lower_bound_adaptive_binary_search(ForwardIt first, ForwardIt last, T &key, Compare compare) {
//	typename std::iterator_traits<ForwardIt>::difference_type count, step, array_size;
//	array_size = std::distance(first, last);
//	count = array_size;
//	step = 1;
//
//	static unsigned int i, balance;
//	unsigned int bot, top, mid;
//
//	// Maybe replace these values with cacheline optimised values?
//	if (balance >= 32 || array_size <= 64) {
//		bot = 0;
//		top = array_size;
//
//		goto monobound;
//	}
//
//	bot = i;
//	top = 32;
//
//	if (key >= array[bot]) {
//		while (true) {
//			if (bot + top >= array_size) {
//				top = array_size - bot;
//				break;
//			}
//			bot += top;
//
//			if (key < array[bot]) {
//				bot -= top;
//				break;
//			}
//			top *= 2;
//		}
//	}
//	else {
//		while (true) {
//			if (bot < top) {
//				top = bot;
//				bot = 0;
//
//				break;
//			}
//			bot -= top;
//
//			if (key >= array[bot]) {
//				break;
//			}
//			top *= 2;
//		}
//	}
//
//	monobound:
//
//	while (top > 3) {
//		mid = top / 2;
//
//		if (key >= array[bot + mid]) {
//			bot += mid;
//		}
//		top -= mid;
//	}
//
//	balance = i > bot ? i - bot : bot - i;
//
//	i = bot;
//
//	while (top) {
//		if (key == array[bot + --top]) {
//			return bot + top;
//		}
//	}
//	return -1;
//}

// TODO
// Only needed for the interpolation search
#define ISAccess(x) x //((x&mask2)>>b1)


/// implementation idea taken from `https://en.wikipedia.org/wiki/Interpolation_search`
template<typename Buckets, typename BucketIndexType, typename ArgumentLimbType, typename LoadType>
BucketIndexType InterpolationSearch(const Buckets *__buckets,
									const ArgumentLimbType data2,
                                    const BucketIndexType boffset,
                                    const LoadType load) {
	ASSERT(boffset < load);
	BucketIndexType low = boffset, high = load - 1, mid;
	const ArgumentLimbType data = ISAccess(data2);
	// std::cout << data << " " << ISAccess(__buckets[low].first) << " " << ISAccess(__buckets[high].first) << "\n";


	while  ((ISAccess(__buckets[high].first) >= ISAccess(__buckets[low].first)) &&
	        (data >= ISAccess(__buckets[low].first)) &&
	        (data <= ISAccess(__buckets[high].first))) {

		uint64_t div = ISAccess(__buckets[high].first) - ISAccess(__buckets[low].first);
		double abc = double(high - low);
		uint64_t mul = abc/div;
		mid = low + ((data - ISAccess(__buckets[low].first)) * mul);
		ASSERT(mid <= high);

		const ArgumentLimbType middata = ISAccess(__buckets[mid].first);
		if (middata < data)
			low = mid + 1;
		else if (data < middata)
			high = mid - 1;
		else {
			// mhhh do the final walk down
			while ((mid > boffset) && (ISAccess(__buckets[mid-1].first) == ISAccess(__buckets[mid].first)))
				mid -= 1;
			return mid;
		}
	}

	if (data == ISAccess(__buckets[low].first))
		return low ;
	else
		return -1;
}
#endif //SMALLSECRETLWE_SEARCH_H
