#ifndef CRYPTANALYSISLIB_ALFORITHM_RANDOM_INDEX_H
#define CRYPTANALYSISLIB_ALFORITHM_RANDOM_INDEX_H

#include <cstdlib>
#include <vector>
#include <array>

#include "helper.h"
#include "random.h"

using namespace cryptanalysislib;

/// Generates a list of random indices within a given range
/// NOTE:
/// 	- As long as `max_entry_size` > `len`, every element will
/// 		be chosen uniquely. So there will be no doubles
///		- if `max_entry_size` <= `len` it will simply choose the
/// 		the first `len` numbers and place them into the list.
///
/// \tparam T type of the entries
/// \param data [out]: list of random indices, must be pre-allocated
/// \param len [in]: size of the output list
/// \param max_entry [in]: maximum value for generated indices (exclusive)
/// \param min_entry [in]: minimum value for generated indices (inclusive)
template<typename T>
constexpr void generate_random_indices(T *data,
                                       const size_t len,
                                       const T max_entry,
                                       const T min_entry=0) noexcept {
	assert(len > 0);
	assert(max_entry > 1);

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

/// Generates random indices for meet-in-the-middle approaches
/// Splits the range into two halves and generates unique indices for each half
/// 
/// \tparam T type of the indices
/// \param data [out]: array to store the generated indices, must be pre-allocated
/// \param len [in]: number of indices to generate
/// \param max_entry [in]: maximum value for generated indices (exclusive)
template<typename T>
constexpr void generate_random_mitm_indices(T *data,
									   	   const size_t len,
                                           const T max_entry) noexcept {
	assert(len >= 2);
	assert(max_entry >= 2);
	const size_t mitm = len/2;
	const T half = max_entry/2;
	generate_random_indices<T>(data+0, mitm, half);
	generate_random_indices<T>(data+mitm, len - mitm, max_entry, half);
}

/// C++ convenience wrapper for generate_random_indices
/// NOTE: the number of rng elements to generate is given
/// 	by the size of the vector `list`
/// 
/// \tparam T base type
/// \param list [out]: vector to store the generated random indices
/// \param max_entry [in]: maximum value for generated indices (exclusive)
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

/// C++ convenience wrapper for generate_random_indices using std::array
/// 
/// \tparam T base integer type
/// \tparam s size of the array
/// \param list [out]: array to store the generated random indices
/// \param max_entry [in]: maximum value for generated indices (exclusive)
template<typename T, const size_t s>
constexpr inline void generate_random_indices(std::array<T, s> &list,
											  const T max_entry) noexcept {
	static_assert(s > 0);
	generate_random_indices(list.data(), s, max_entry);
}

/// C++ convenience wrapper for generate_random_mitm_indices using std::array
/// Generates indices for meet-in-the-middle approaches using an array
/// 
/// \tparam T base integer type
/// \tparam s size of the array
/// \param list [out]: array to store the generated indices
/// \param max_entry [in]: maximum value for generated indices (exclusive)
template<typename T, const size_t s>
constexpr inline void generate_random_mitm_indices(std::array<T, s> &list,
                                                   const T max_entry) noexcept {
	static_assert(s > 0);
	generate_random_mitm_indices(list.data(), s, max_entry);
}

#endif
