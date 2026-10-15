#ifndef CRYPTANALYSISLIB_ALGORITHM_SHUFFLE_H
#define CRYPTANALYSISLIB_ALGORITHM_SHUFFLE_H

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <random>
#include <type_traits>

#include "algorithm/swap.h"
#include "random.h"

namespace cryptanalysislib {
	namespace internal {
		/// uniform random number in [0, bound] from the uniform random bit
		/// generator `g`, whose range [g.min(), g.max()] may be smaller or
		/// larger than [0, bound] (same method as libstdc++'s
		/// `uniform_int_distribution`)
		/// \param g[in]: uniform random bit generator
		/// \param bound[in]: largest possible result
		/// \return uniform in [0, bound]
		template<class URBG>
		constexpr uint64_t urbg_uniform(URBG &g,
		                                const uint64_t bound) noexcept {
			constexpr uint64_t gmin = uint64_t(URBG::min());
			constexpr uint64_t grange = uint64_t(URBG::max()) - gmin;
			if constexpr (grange < uint64_t(-1)) {
				if (grange < bound) {
					// the generator range is too small: combine two draws
					constexpr uint64_t gparts = grange + 1u;
					uint64_t r, t;
					do {
						t = gparts * urbg_uniform(g, bound / gparts);
						r = t + (uint64_t(g()) - gmin);
					} while ((r > bound) || (r < t));
					return r;
				}
			}

			if (grange == bound) {
				return uint64_t(g()) - gmin;
			}

			// rejection: [0, past) splits into bound + 1 equally large parts
			const uint64_t parts = bound + 1u;
			const uint64_t scaling = grange / parts;
			const uint64_t past = parts * scaling;
			uint64_t r;
			do {
				r = uint64_t(g()) - gmin;
			} while (r >= past);
			return r / scaling;
		}
	} // end namespace internal

	/// Randomly permutes the elements in [first, last) (Fisher-Yates) with
	/// the uniform random bit generator `g` (like `std::shuffle`). Every
	/// permutation is equally likely.
	/// \param first[in]: begin of the range
	/// \param last[in]: end of the range
	/// \param g[in]: uniform random bit generator, e.g. `std::mt19937_64`
	template<std::random_access_iterator RandomIt,
	         class URBG>
		requires std::uniform_random_bit_generator<std::remove_reference_t<URBG>>
	constexpr void shuffle(RandomIt first,
	                       RandomIt last,
	                       URBG &&g) noexcept {
		const auto n = last - first;
		for (auto i = n - 1; i > 0; --i) {
			// j uniform in [0, i]
			const auto j = static_cast<decltype(n)>(internal::urbg_uniform(g, static_cast<uint64_t>(i)));
			cryptanalysislib::iter_swap(first + i, first + j);
		}
	}

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
