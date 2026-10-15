#pragma once

#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <type_traits>
#include <utility>

#include "sort/heap/is_heap.h"
#include "sort/heap/push_heap.h"

/// Heap functions (like `std::is_heap`, `std::push_heap`, ...): a max-heap
/// w.r.t. `comp`, i.e. `comp(*(first + (i-1)/2), *(first + i)) == false`.
/// For contiguous `int32_t` ranges and `std::less<>` the SIMD versions from
/// `sort/heap` are used.
/// NOTE: call them qualified (`cryptanalysislib::make_heap`): an unqualified
/// 	call on `std::` iterators also finds the `std::` version via ADL.
namespace cryptanalysislib {
namespace internal {
	/// true if the SIMD versions in `sort/heap` can be used
	template<class RandIt, class Compare>
	constexpr static bool heap_simd_int32 =
	        std::contiguous_iterator<RandIt> &&
	        std::is_same_v<std::iter_value_t<RandIt>, int32_t> &&
	        (std::is_same_v<Compare, std::less<>> || std::is_same_v<Compare, std::less<int32_t>>);

	/// Moves `value` down from the position `hole` until the subtree below
	/// `hole` within [first, first + len) is a heap again.
	template<class RandIt, class Compare>
	constexpr void heap_sift_down(RandIt first,
	                              const std::ptrdiff_t len,
	                              std::ptrdiff_t hole,
	                              std::iter_value_t<RandIt> value,
	                              Compare &comp) noexcept {
		while (true) {
			std::ptrdiff_t child = 2 * hole + 1;
			if (child >= len) {
				break;
			}

			if ((child + 1 < len) && comp(first[child], first[child + 1])) {
				child += 1;
			}

			if (!comp(value, first[child])) {
				break;
			}

			first[hole] = std::move(first[child]);
			hole = child;
		}

		first[hole] = std::move(value);
	}
} // end namespace internal

/// \param first[in]: begin of the range
/// \param last[in]: end of the range
/// \param comp[in]: comparison, `std::less<>` creates a max-heap
/// \return the first element which violates the heap property, `last` if
/// 	[first, last) is a heap
template<class RandIt,
         class Compare = std::less<>>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
[[nodiscard]] constexpr RandIt is_heap_until(RandIt first,
                                             RandIt last,
                                             Compare comp = {}) noexcept {
	const std::ptrdiff_t n = last - first;
	for (std::ptrdiff_t i = 1; i < n; ++i) {
		if (comp(first[(i - 1) / 2], first[i])) {
			return first + i;
		}
	}

	return last;
}

/// \param first[in]: begin of the range
/// \param last[in]: end of the range
/// \param comp[in]: comparison, `std::less<>` creates a max-heap
/// \return true if [first, last) is a heap w.r.t. `comp`
template<class RandIt,
         class Compare = std::less<>>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
[[nodiscard]] constexpr bool is_heap(RandIt first,
                                     RandIt last,
                                     Compare comp = {}) noexcept {
	if constexpr (internal::heap_simd_int32<RandIt, Compare>) {
		if !consteval {
#ifdef USE_AVX512F
			return is_heap_avx512_epi32(std::to_address(first), std::to_address(first) + (last - first));
#elif defined(USE_AVX2)
			return is_heap_avx2_epi32(std::to_address(first), std::to_address(first) + (last - first));
#endif
		}
	}

	return cryptanalysislib::is_heap_until(first, last, comp) == last;
}

/// Inserts `*(last - 1)` into the heap [first, last - 1).
/// \param first[in]: begin of the heap
/// \param last[in]: end of the heap, including the new element
/// \param comp[in]: comparison, `std::less<>` creates a max-heap
template<class RandIt,
         class Compare = std::less<>>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
constexpr void push_heap(RandIt first,
                         RandIt last,
                         Compare comp = {}) noexcept {
	const std::ptrdiff_t n = last - first;
	if (n <= 1) {
		return;
	}

	if constexpr (internal::heap_simd_int32<RandIt, Compare>) {
		if !consteval {
#ifdef USE_AVX512F
			push_heap_avx512(std::to_address(first), n);
			return;
#elif defined(USE_AVX2)
			push_heap_avx2(std::to_address(first), n);
			return;
#endif
		}
	}

	std::ptrdiff_t hole = n - 1;
	std::iter_value_t<RandIt> value = std::move(first[hole]);
	while (hole > 0) {
		const std::ptrdiff_t parent = (hole - 1) / 2;
		if (!comp(first[parent], value)) {
			break;
		}

		first[hole] = std::move(first[parent]);
		hole = parent;
	}

	first[hole] = std::move(value);
}

/// Moves the top of the heap [first, last) to `last - 1`, and makes
/// [first, last - 1) a heap.
/// \param first[in]: begin of the heap
/// \param last[in]: end of the heap
/// \param comp[in]: comparison, `std::less<>` creates a max-heap
template<class RandIt,
         class Compare = std::less<>>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
constexpr void pop_heap(RandIt first,
                        RandIt last,
                        Compare comp = {}) noexcept {
	const std::ptrdiff_t n = last - first;
	if (n <= 1) {
		return;
	}

	std::iter_value_t<RandIt> value = std::move(first[n - 1]);
	first[n - 1] = std::move(first[0]);
	internal::heap_sift_down(first, n - 1, 0, std::move(value), comp);
}

/// Rearranges [first, last) into a heap.
/// \param first[in]: begin of the range
/// \param last[in]: end of the range
/// \param comp[in]: comparison, `std::less<>` creates a max-heap
template<class RandIt,
         class Compare = std::less<>>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
constexpr void make_heap(RandIt first,
                         RandIt last,
                         Compare comp = {}) noexcept {
	const std::ptrdiff_t n = last - first;
	for (std::ptrdiff_t i = n / 2; i-- > 0;) {
		internal::heap_sift_down(first, n, i, std::move(first[i]), comp);
	}
}

/// Sorts the heap [first, last) in ascending order w.r.t. `comp`.
/// \param first[in]: begin of the heap
/// \param last[in]: end of the heap
/// \param comp[in]: comparison, `std::less<>` creates a max-heap
template<class RandIt,
         class Compare = std::less<>>
#if __cplusplus > 201709L
    requires std::random_access_iterator<RandIt>
#endif
constexpr void sort_heap(RandIt first,
                         RandIt last,
                         Compare comp = {}) noexcept {
	for (; last - first > 1; --last) {
		cryptanalysislib::pop_heap(first, last, comp);
	}
}
} // end namespace cryptanalysislib
