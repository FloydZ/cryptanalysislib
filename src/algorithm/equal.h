#ifndef CRYPTANALYSISLIB_ALGORITHM_EQUAL_H
#define CRYPTANALYSISLIB_ALGORITHM_EQUAL_H
//https://en.cppreference.com/w/cpp/algorithm/equal

#include <numeric>

#include "thread/thread.h"
#include "algorithm/algorithm.h"
#include "memory/memcmp.h"

namespace cryptanalysislib {
	/// Configuration for equal algorithms with threading settings
	struct AlgorithmEqualConfig : public AlgorithmConfig {
		const size_t min_size_per_thread = 262144;
	};
	constexpr static AlgorithmEqualConfig algorithmEqualConfig;

	/// Checks if two ranges are equal (sequential version)
	/// \tparam InputIt1 Forward iterator type for first range
	/// \tparam InputIt2 Forward iterator type for second range
	/// \tparam config Algorithm configuration (default: algorithmEqualConfig)
	/// \param first1[in]: Iterator to the beginning of the first range
	/// \param last1[in]: Iterator to the end of the first range
	/// \param first2[in]: Iterator to the beginning of the second range
	/// \return True if ranges are equal, false otherwise
	template<class InputIt1,
			 class InputIt2,
			 const AlgorithmEqualConfig &config=algorithmEqualConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt1> &&
				 std::forward_iterator<InputIt2>
#endif
	constexpr bool equal(InputIt1 first1,
						 InputIt1 last1,
						 InputIt2 first2) noexcept {
		using T = typename std::iterator_traits<InputIt1>::value_type;
		if constexpr (std::is_same_v<InputIt1, InputIt2> &&
		              std::contiguous_iterator<InputIt1> &&
		              std::is_integral_v<T>) {
			// NOTE: `memcmp` returns `true` if the ranges differ
			const size_t size = static_cast<size_t>(last1 - first1);
			if (size == 0) {
				return true;
			}
			return !cryptanalysislib::memcmp(&(*first1), &(*first2), size);
		}

	    for (; first1 != last1; ++first1, ++first2) {
		    if (!(*first1 == *first2)) {
		    	return false;
		    }
	    }

	    return true;
	}

	/// Checks if two ranges are equal (parallel version)
	/// \tparam ExecPolicy Execution policy type for parallel execution
	/// \tparam RandIt1 Random access iterator type for first range
	/// \tparam RandIt2 Random access iterator type for second range
	/// \tparam config Algorithm configuration (default: algorithmEqualConfig)
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first1[in]: Iterator to the beginning of the first range
	/// \param last1[in]: Iterator to the end of the first range
	/// \param first2[in]: Iterator to the beginning of the second range
	/// \return True if ranges are equal, false otherwise
	template <class ExecPolicy,
			  class RandIt1,
			  class RandIt2,
			  const AlgorithmEqualConfig &config=algorithmEqualConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt1> &&
				 std::random_access_iterator<RandIt2>
#endif
	bool equal(ExecPolicy&& policy,
			   RandIt1 first1,
			   RandIt1 last1,
			   RandIt2 first2) noexcept {

		const auto size = static_cast<size_t>(std::distance(first1, last1));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			return cryptanalysislib::equal
				<RandIt1, RandIt2, config>
				(first1, last1, first2);
		}

		// every chunk compares against the matching chunk of the second range
		auto chunk = [first1, first2](RandIt1 b, RandIt1 e) noexcept -> bool {
			return cryptanalysislib::equal<RandIt1, RandIt2, config>(b, e, first2 + (b - first1));
		};

		auto futures = internal::parallel_chunk_for_1(
			std::forward<ExecPolicy>(policy),
			first1, last1, chunk,
			(bool *)0,
			1, nthreads);

		// the ranges are equal iff every chunk is equal
		bool ret = true;
		for (auto &f : futures) {
			ret &= f.get();
		}
		return ret;
	}
} // end namespace cryptanalysislib
#endif //EQUAL_H
