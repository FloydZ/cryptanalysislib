#ifndef CRYPTANALYSISLIB_ALGORITHM_SHUFFLE_H
#define CRYPTANALYSISLIB_ALGORITHM_SHUFFLE_H

#include <cstddef>
#include <cstdint>
#include <iterator>

#include "algorithm/swap.h"
#include "random.h"

namespace cryptanalysislib {
	/// Randomly permutes the elements in [first, last) (Fisher-Yates),
	/// using the library rng. Every permutation is equally likely.
	/// Replacement for `std::random_shuffle` (removed in C++17).
	/// \param first[in]: begin of the range
	/// \param last[in]: end of the range
	template<std::random_access_iterator RandomIt>
	void random_shuffle(RandomIt first, RandomIt last) noexcept {
		const auto n = last - first;
		for (auto i = n - 1; i > 0; --i) {
			// j uniform in [0, i]
			const auto j = static_cast<decltype(n)>(rng_v2<uint64_t>(static_cast<uint64_t>(i) + 1u));
			cryptanalysislib::iter_swap(first + i, first + j);
		}
	}
} // end namespace cryptanalysislib

#endif // CRYPTANALYSISLIB_ALGORITHM_SHUFFLE_H
