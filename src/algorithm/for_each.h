#ifndef CRYPTANALYSISLIB_ALGORITHM_FOR_EACH_H
#define CRYPTANALYSISLIB_ALGORITHM_FOR_EACH_H

#include "algorithm/algorithm.h"
#include "thread/thread.h"

namespace cryptanalysislib {
	struct AlgorithmForEachConfig : public AlgorithmConfig {
		constexpr static size_t min_size_per_thread = 1u << 14u;
	};
	constexpr static AlgorithmForEachConfig algorithmForEachConfig;

	/// \tparam InputIt
	/// \tparam UnaryFunction
	/// \param first
	/// \param last
	/// \param f
	/// \return
	template<class InputIt,
	         class UnaryFunction,
	         const AlgorithmForEachConfig &config=algorithmForEachConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt> &&
    		     std::regular_invocable<UnaryFunction,
										const typename InputIt::value_type&>
#endif
	constexpr UnaryFunction for_each(InputIt first,
									 InputIt last,
									 UnaryFunction f) noexcept {
	    for (; first != last; ++first) {
		    f(*first);
	    }

	    return f;
	}

	/// NOTE: Iterators are expected to be rng access.
    /// See std::for_each https://en.cppreference.com/w/cpp/algorithm/for_each
    /// \tparam ExecPolicy
    /// \tparam RandIt
    /// \tparam UnaryFunction
    /// \param policy
    /// \param first
    /// \param last
    /// \param p
    /// \return
	template <class ExecPolicy,
	          class RandIt,
	          class UnaryFunction,
	          const AlgorithmForEachConfig &config=algorithmForEachConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt> &&
    		     std::regular_invocable<UnaryFunction,
										const typename RandIt::value_type&>
#endif
	void for_each(ExecPolicy &&policy,
	              RandIt first,
	              RandIt last,
	              UnaryFunction p) noexcept {

		const size_t size = static_cast<size_t>(std::distance(first, last));
		const uint32_t nthreads = should_par(policy, config, size);
		if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
			cryptanalysislib::for_each<RandIt, decltype(p), config>(first, last, p);
			return;
		}

		internal::parallel_chunk_for_1_wait(
			std::forward<ExecPolicy>(policy),
			first, last,
		    cryptanalysislib::for_each<RandIt, decltype(p), config>,
		    (void*)nullptr,
		    1,
		    nthreads,
		    p);
	}

	/// \tparam ExecPolicy
	/// \tparam RandIt
	/// \tparam Size
	/// \tparam UnaryFunction
	/// \param policy
	/// \param first
	/// \param n
	/// \param f
	/// \return
	template <class ExecPolicy,
			  class RandIt,
			  class Size,
			  class UnaryFunction,
	          const AlgorithmForEachConfig &config=algorithmForEachConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt> &&
    		     std::regular_invocable<UnaryFunction,
										const typename RandIt::value_type&>
#endif
    RandIt for_each_n(ExecPolicy &&policy,
					  RandIt first,
					  Size n,
					  UnaryFunction f) noexcept {
        RandIt last = internal::advanced(first, n);
        cryptanalysislib::for_each
			<RandIt, UnaryFunction, config>
			(std::forward<ExecPolicy>(policy), first, last, f);
        return last;
    }

    ///
    template <class RandIt,
              class ChunkConstructor,
              class UnaryFunction>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt> &&
    		     std::regular_invocable<UnaryFunction,
										const typename RandIt::value_type&>
#endif
    void for_each_chunk(RandIt first,
                        RandIt last,
                        ChunkConstructor construct, 
                        UnaryFunction f) noexcept {
        if (first == last) {
            return;
        }

        auto chunk_data = construct();
        for (; first != last; ++first) {
            f(*first, chunk_data);
        }
    }

} // end namespace cryptanalysislib

#endif
