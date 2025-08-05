#ifndef CRYPTANALYSISLIB_ALGORITHM_FOR_EACH_H
#define CRYPTANALYSISLIB_ALGORITHM_FOR_EACH_H

#include "algorithm/algorithm.h"
#include "thread/thread.h"

namespace cryptanalysislib {
	/// Configuration for for_each algorithms
	struct AlgorithmForEachConfig : public AlgorithmConfig {
		constexpr static size_t min_size_per_thread = 1u << 14u;
	};
	constexpr static AlgorithmForEachConfig algorithmForEachConfig;

	/// Applies a function to each element in a range (sequential version)
	/// \tparam InputIt Forward iterator type for the range
	/// \tparam UnaryFunction Function type to apply to each element
	/// \tparam config Algorithm configuration (default: algorithmForEachConfig)
	/// \param first[in]: Iterator to the beginning of the range
	/// \param last[in]: Iterator to the end of the range
	/// \param f[in]: Function to apply to each element
	/// \return Copy of the function object
	template<class InputIt,
	         class UnaryFunction,
	         const AlgorithmForEachConfig &config=algorithmForEachConfig>
#if __cplusplus > 201709L
		requires std::forward_iterator<InputIt> &&
    		     std::regular_invocable<UnaryFunction,
										typename InputIt::value_type&>
#endif
	constexpr UnaryFunction for_each(InputIt first,
									 InputIt last,
									 UnaryFunction f) noexcept {
	    for (; first != last; ++first) {
		    f(*first);
	    }

	    return f;
	}

	/// Applies a function to each element in a range (parallel version)
    /// NOTE: Iterators are expected to be random access.
    /// See std::for_each https://en.cppreference.com/w/cpp/algorithm/for_each
    /// \tparam ExecPolicy Execution policy type for parallel execution
    /// \tparam RandIt Random access iterator type for the range
    /// \tparam UnaryFunction Function type to apply to each element
    /// \tparam config Algorithm configuration (default: algorithmForEachConfig)
    /// \param policy[in]: Execution policy specifying parallelization strategy
    /// \param first[in]: Iterator to the beginning of the range
    /// \param last[in]: Iterator to the end of the range
    /// \param p[in]: Function to apply to each element
	template <class ExecPolicy,
	          class RandIt,
	          class UnaryFunction,
	          const AlgorithmForEachConfig &config=algorithmForEachConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt> &&
    		     std::regular_invocable<UnaryFunction,
										typename RandIt::value_type&>
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

	/// Applies a function to n elements starting from an iterator (parallel version)
	/// \tparam ExecPolicy Execution policy type for parallel execution
	/// \tparam RandIt Random access iterator type for the range
	/// \tparam Size Integral type for count
	/// \tparam UnaryFunction Function type to apply to each element
	/// \tparam config Algorithm configuration (default: algorithmForEachConfig)
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first[in]: Iterator to the beginning of the range
	/// \param n[in]: Number of elements to process
	/// \param f[in]: Function to apply to each element
	/// \return Iterator to the end of the processed range
	template <class ExecPolicy,
			  class RandIt,
			  class Size,
			  class UnaryFunction,
	          const AlgorithmForEachConfig &config=algorithmForEachConfig>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt> &&
    		     std::regular_invocable<UnaryFunction,
										typename RandIt::value_type&>
#endif
    RandIt for_each_n(ExecPolicy &&policy,
					  RandIt first,
					  const Size n,
					  UnaryFunction f) noexcept {
        RandIt last = internal::advanced(first, n);
        cryptanalysislib::for_each
			<RandIt, UnaryFunction, config>
			(std::forward<ExecPolicy>(policy), first, last, f);
        return last;
    }

    /// Applies a function to each element with chunk-local data
    /// \tparam RandIt Random access iterator type for the range
    /// \tparam ChunkConstructor Function type to construct chunk-local data
    /// \tparam UnaryFunction Function type to apply to each element
    /// \param first[in]: Iterator to the beginning of the range
    /// \param last[in]: Iterator to the end of the range
    /// \param construct[in]: Function to construct chunk-local data
    /// \param f[in]: Function to apply to each element with chunk data
    template <class RandIt,
              class ChunkConstructor,
              class UnaryFunction>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt> &&
    		     std::regular_invocable<UnaryFunction,
										typename RandIt::value_type&>
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
