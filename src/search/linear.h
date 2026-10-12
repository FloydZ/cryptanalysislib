#ifndef CRYPTANALYSISLIB_SEARCH_LINEAR_H
#define CRYPTANALYSISLIB_SEARCH_LINEAR_H

#ifndef CRYPTANALYSISLIB_SEARCH_H
#error "do not include this file directly. Use `#inluce <cryptanalysislib/search/search.h>`"
#endif

#include <cstdint>
#include <algorithm>
#include <iterator>

#include "algorithm/countr.h"
#include "dispatch.h"
#include "hash/hash.h"
#include "simd/simd.h"

namespace cryptanalysislib {

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
	// walk backwards until the first element that is not greater than `key`
	ForwardIt it = last;
	while (it != first) {
		ForwardIt prev = it;
		--prev;
		if (!compare(key, *prev)) {
			return it;
		}

		it = prev;
	}

	return first;
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
	for (ForwardIt it = first; it != last; ++it) {
		if (!compare(*it, key)) {
			return it;
		}
	}

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

namespace internal {
	/// scalar linear search for `key` in `data[0, n)`
	/// \return position of the first element == key, or n if not found
	template<typename T>
	constexpr size_t linear_search_scalar(const T *data,
	                                      const size_t n,
	                                      const T key) noexcept {
		for (size_t i = 0; i < n; ++i) {
			if (data[i] == key) { return i; }
		}
		return n;
	}

	/// SIMD linear search for `key` in `data[0, n)` (same as in
	/// `src/algorithm/find.h`, which cannot be included here, as it includes
	/// this header).
	/// \return position of the first element == key, or n if not found
	template<typename T>
#if __cplusplus > 201709L
		requires std::unsigned_integral<T>
#endif
	constexpr size_t linear_search_simd(const T *data,
	                                    const size_t n,
	                                    const T key) noexcept {
		using S = SIMDSelector<T>;
		const auto t = S::set1(key);
		size_t i = 0;
		for (; (i + S::LIMBS) <= n; i += S::LIMBS) {
			const auto d = S::template load<false>(data + i);
			// NOTE: one bit per lane
			const uint64_t m = uint64_t(d == t);
			if (m) [[unlikely]] {
				return i + cryptanalysislib::countr_zero(m);
			}
		}

		return i + linear_search_scalar(data + i, n - i, key);
	}

	/// Dispatches between the scalar and the SIMD linear search: the first
	/// call benchmarks both once (on its arguments), afterwards the faster
	/// one is used (same scheme as `binary_search_dispatch`).
	/// \return position of the first element == key, or n if not found
	template<typename T>
#if __cplusplus > 201709L
		requires std::unsigned_integral<T>
#endif
	size_t linear_search_dispatch(const T *data,
	                              const size_t n,
	                              const T key) noexcept {
		using FF = size_t(*)(const T *, const size_t, const T);

		// NOTE dont specify as const
		static FF functions[] = {
			linear_search_scalar<T>,
			linear_search_simd<T>,
		};

		static const FF out = [&]() noexcept {
			FF best = functions[0];
			generic_dispatch(best, functions, sizeof(functions)/sizeof(functions[0]),
			                 data, n, key);
			return best;
		}();

		return out(data, n, key);
	}
} // end namespace internal

	/// Linear search for an exact value, SIMD accelerated for contiguous
	/// ranges of unsigned integers.
	/// \param first[in] Iterator to the beginning of the range
	/// \param last[in] Iterator to the end of the range
	/// \param key[in] value to search for
	/// \return Iterator to the first element == key, or last if not found
	template<class ForwardIt>
#if __cplusplus > 201709L
	requires std::forward_iterator<ForwardIt>
#endif
	constexpr inline ForwardIt linear_search(const ForwardIt first,
	                                         const ForwardIt last,
	                                         const typename ForwardIt::value_type &key) noexcept {
		using T = typename ForwardIt::value_type;
		if constexpr (std::contiguous_iterator<ForwardIt> && std::is_unsigned_v<T>) {
			const size_t n = static_cast<size_t>(last - first);
			if (n == 0) { return last; }
			const size_t pos = internal::linear_search_simd<T>(&*first, n, key);
			return pos == n ? last : first + pos;
		} else {
			for (ForwardIt it = first; it != last; ++it) {
				if (*it == key) { return it; }
			}
			return last;
		}
	}

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
