#ifndef CRYPTANALYSISLIB_ALFORITHM_RANDOM_INDEX_H
#define CRYPTANALYSISLIB_ALFORITHM_RANDOM_INDEX_H

#include <cstdlib>
#include <vector>
#include <array>

#include "helper.h"
#include "random.h"

using namespace cryptanalysislib;
/// generates a list of rng entries mod max_entry
/// NOTE:
/// 	- As long as `max_entry_size` > `len`, every element will
/// 		be chosen uniquely. So there will be no doubles
///		- if `max_entry_size` <= `len` it wil simply choose the
/// 		the first `len` numbers and place them into the list.
///
/// \tparam T type of the entries
/// \param data output: list of entries, must be pre-allocated
/// \param len size of the output list
/// \param max_entry_size max size of the entry
/// \param min_entry_size min size of the entry
template<typename T>
constexpr void generate_random_indices(T *data,
                                       const size_t len,
                                       const T max_entry,
                                       const T min_entry=0) noexcept {
	ASSERT(len > 0);
	ASSERT(max_entry > 1);

	if (max_entry <= len) {
		// easy case, in which we have to chose certain elements often
		for (size_t i = 0; i < len; ++i) {
			data[i] = rng<T>(min_entry, max_entry);
		}
	}

	for (uint32_t i = 0; i < len; ++i) {
		while (true) {
			const T a = rng<T>(min_entry, max_entry);
			bool restart = false;
			for (uint32_t j = 0; j < i; ++j) {
				if (data[j] == a) {
					restart = true;
					break;
				}
			}

			if (restart) {
				continue;
			}

			data[i] = a ;
			break;
		}
	}
}

///
/// \tparam T
/// \param data
/// \param len
/// \param max_entry
/// \return
template<typename T>
constexpr void generate_random_mitm_indices(T *data,
									   	   const size_t len,
                                           const T max_entry) noexcept {
	ASSERT(len >= 2);
	ASSERT(max_entry >= 2);
	const size_t mitm = len/2;
	const T half = max_entry/2;
	generate_random_indices<T>(data+0, mitm, half);
	generate_random_indices<T>(data+mitm, len - mitm, max_entry, half);
}

/// just a c++ convenience wrapper
/// NOTE: the number of rng elements to generate is given
/// 	by the size of the vector `list`
/// \tparam T base type
/// \param list output: vector/list
/// \param max_entry max integer size of each element
template<typename T>
constexpr inline void generate_random_indices(std::vector<T> &list,
                                              const T max_entry) noexcept {
	if (list.empty()) { return; }
	generate_random_indices(list.data(), list.size(), max_entry);
}

template<typename T>
constexpr inline void generate_random_mitm_indices(std::vector<T> &list,
											  	   const T max_entry) noexcept {
	if (list.empty()) { return; }
	generate_random_mitm_indices(list.data(), list.size(), max_entry);
}

/// NOTE: the same convenience wrapper as above
/// \tparam T base integer type
/// \tparam s size of the array
/// \param list
/// \param max_entry
template<typename T, const size_t s>
constexpr inline void generate_random_indices(std::array<T, s> &list,
											  const T max_entry) noexcept {
	static_assert(s > 0);
	generate_random_indices(list.data(), s, max_entry);
}

template<typename T, const size_t s>
constexpr inline void generate_random_mitm_indices(std::array<T, s> &list,
                                                   const T max_entry) noexcept {
	static_assert(s > 0);
	generate_random_mitm_indices(list.data(), s, max_entry);
}

#endif
