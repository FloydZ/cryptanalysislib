#ifndef CRYPTANALYSISLIB_ALGORITHM_EQUAL_H
#define CRYPTANALYSISLIB_ALGORITHM_EQUAL_H
//https://en.cppreference.com/w/cpp/algorithm/equal

#include <numeric>

#include "thread/thread.h"
#include "algorithm/algorithm.h"
#include "memory/memcmp.h"

namespace cryptanalysislib {
	struct AlgorithmEqualConfig : public AlgorithmConfig {
		constexpr static size_t min_size_per_thread = 1u << 10u;
	};
	constexpr static AlgorithmEqualConfig algorithmEqualConfig;

	/// @tparam InputIt1
	/// @tparam InputIt2
	/// @param first1
	/// @param last1
	/// @param first2
	/// @return
	template<class InputIt1,
			 class InputIt2,
			 const AlgorithmEqualConfig &config=algorithmEqualConfig>
#if __cplusplus > 201709L
		requires std::bidirectional_iterator<InputIt1> &&
				 std::bidirectional_iterator<InputIt2>
#endif
	constexpr bool equal(InputIt1 first1,
						 InputIt1 last1,
						 InputIt2 first2) noexcept {
		if constexpr (std::is_same_v<InputIt1, InputIt2>) {
			const auto size = static_cast<size_t>(std::distance(first1, last1));
			return cryptanalysislib::memcmp(&(*first1), &(*first2), size);
		}

	    for (; first1 != last1; ++first1, ++first2) {
		    if (!(*first1 == *first2)) {
		    	return false;
		    }
	    }

	    return true;
	}

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

		using T = uint32_t;

		auto futures = internal::parallel_chunk_for_1(
			std::forward<ExecPolicy>(policy),
			first1, last1,
			cryptanalysislib::equal<RandIt1, RandIt2, config>,
			(bool *)0,
			1, nthreads, first2);

		return (bool)std::reduce(
			internal::get_wrap(futures.begin()),
			internal::get_wrap(futures.end()), (T)0, std::plus<T>());
	}


} // end namespace cryptanalysislib
#endif //EQUAL_H
