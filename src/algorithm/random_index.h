#ifndef CRYPTANALYSISLIB_ALFORITHM_RANDOM_INDEX_H
#define CRYPTANALYSISLIB_ALFORITHM_RANDOM_INDEX_H

#include <cstdlib>
#include <vector>
#include <array>

#include "random.h"

/// generates a list of random entries mod max_entry
/// As long as `max_entry_size` < len, every element will be chosen unique
/// \tparam T type of the entries
/// \param data output: list of entries, must be pre-allocated
/// \param len size of the output list
/// \param max_entry_size max size of the entry
template<typename T>
constexpr void generate_random_indices(T *data,
                                       const size_t len,
                                       const size_t max_entry) noexcept {
	if (max_entry < len) {
		// easy case, in which we have to chose certain elements often
		for (size_t i = 0; i < len; ++i) {
			data[i] = fastrandombytes_T<T>() % max_entry;
		}
	}

	for (uint32_t i = 0; i < len; ++i) {
		while (true) {
			const T a = fastrandombytes_uint64(max_entry);
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
/// \param list
/// \param max_entry
/// \return
template<typename T>
constexpr inline void generate_random_indices(std::vector<T> &list,
                                              const size_t max_entry) noexcept {
	if (list.size() == 0) {
		return;
	}

	generate_random_indices(list.data(), list.size(), max_entry);
}

template<typename T, const size_t s>
constexpr inline void generate_random_indices(std::array<T, s> &list,
                                              const size_t max_entry) noexcept {
	static_assert(s > 0);

	generate_random_indices(list.data(), s, max_entry);
}
#endif
