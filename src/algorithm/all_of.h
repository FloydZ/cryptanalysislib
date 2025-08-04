#ifndef CRYPTANALYSISLIB_ALGORITHM_ALL_OF_H
#define CRYPTANALYSISLIB_ALGORITHM_ALL_OF_H

#include "thread/thread.h"
#include "algorithm/find.h"

// https://en.cppreference.com/w/cpp/algorithm/all_any_none_of
namespace cryptanalysislib {
	/// Configuration structure for the any_of, all_of, and none_of algorithms
	struct AlgorithmAnyOfConfig : public AlgorithmConfig {
		 /// Minimum size per thread to avoid excessive thread creation for small data sets
		 const size_t min_size_per_thread = 1048576;
	};

	/// Default configuration for the any_of, all_of, and none_of algorithms
	constexpr static AlgorithmAnyOfConfig algorithmAnyOfConfig;

	/// Checks if a predicate returns true for all elements in the range
	/// \tparam InputIt[in]: Type of the iterator for the input range
    /// \tparam UnaryPred[in]: Type of the unary predicate function
    /// \tparam config[in]: Configuration for the algorithm
    /// \param first[in]: Iterator to the first element in the range
    /// \param last[in]: Iterator to one past the last element in the range
    /// \param p[in]: Unary predicate function to apply to elements
    /// \return True if the predicate returns true for all elements in the range, false otherwise
    template<class InputIt,
             class UnaryPred,
             const AlgorithmAnyOfConfig &config=algorithmAnyOfConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt> &&
	    		 std::regular_invocable<UnaryPred,
									    const typename InputIt::value_type&>
#endif
    constexpr bool all_of(InputIt first,
                          InputIt last,
                          UnaryPred p) noexcept {
        return cryptanalysislib::find_if_not(first, last, p) == last;
    }

	/// Checks if a predicate returns true for at least one element in the range
	/// \tparam InputIt[in]: Type of the iterator for the input range
    /// \tparam UnaryPred[in]: Type of the unary predicate function
    /// \tparam config[in]: Configuration for the algorithm
    /// \param first[in]: Iterator to the first element in the range
    /// \param last[in]: Iterator to one past the last element in the range
    /// \param p[in]: Unary predicate function to apply to elements
    /// \return True if the predicate returns true for at least one element in the range, false otherwise
	template<class InputIt,
             class UnaryPred,
             const AlgorithmAnyOfConfig &config=algorithmAnyOfConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt> &&
	    		 std::regular_invocable<UnaryPred,
									    const typename InputIt::value_type&>
#endif
    constexpr bool any_of(InputIt first,
                          InputIt last,
                          UnaryPred p) noexcept {
        return std::find_if
    		<InputIt, UnaryPred>
    		(first, last, p) != last;
    }

    /// Checks if a predicate returns false for all elements in the range
    /// \tparam InputIt[in]: Type of the iterator for the input range
    /// \tparam UnaryPred[in]: Type of the unary predicate function
    /// \tparam config[in]: Configuration for the algorithm
    /// \param first[in]: Iterator to the first element in the range
    /// \param last[in]: Iterator to one past the last element in the range
    /// \param p[in]: Unary predicate function to apply to elements
    /// \return True if the predicate returns false for all elements in the range, false otherwise
    template<class InputIt,
             class UnaryPred,
             const AlgorithmAnyOfConfig &config=algorithmAnyOfConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt> &&
	    		 std::regular_invocable<UnaryPred,
									    const typename InputIt::value_type&>
#endif
    constexpr bool none_of(InputIt first,
                           InputIt last,
                           UnaryPred p) noexcept {

    	constexpr static AlgorithmFindConfig c = {
    		.min_size_per_thread = config.min_size_per_thread,
    	};
        return cryptanalysislib::find_if
    			<InputIt, UnaryPred, c>
    			(first, last, p) == last;
    }

    /// Parallel version of all_of - checks if a predicate returns true for all elements in the range
    /// \tparam ExecPolicy[in]: Type of the execution policy
    /// \tparam RandIt[in]: Type of the random access iterator
    /// \tparam Predicate[in]: Type of the predicate function
    /// \tparam config[in]: Configuration for the algorithm
    /// \param policy[in]: Execution policy to use
    /// \param first[in]: Iterator to the first element in the range
    /// \param last[in]: Iterator to one past the last element in the range
    /// \param pred[in]: Predicate function to apply to elements
    /// \return True if the predicate returns true for all elements in the range, false otherwise
    template <class ExecPolicy,
              typename RandIt,
              typename Predicate,
              const AlgorithmAnyOfConfig &config=algorithmAnyOfConfig>
#if __cplusplus > 201709L
	    requires std::bidirectional_iterator<RandIt> &&
	    		 std::regular_invocable<Predicate,
									    const typename RandIt::value_type&>
#endif
    bool all_of(ExecPolicy&& policy,
                RandIt first,
                RandIt last,
                Predicate pred) noexcept {
    	constexpr static AlgorithmFindConfig c = {
    		.min_size_per_thread = config.min_size_per_thread,
    	};

        return last == cryptanalysislib::find_if_not
    					<ExecPolicy, RandIt, Predicate, c>
    					(std::forward<ExecPolicy>(policy), first, last, pred);
    }

    /// Parallel version of none_of - checks if a predicate returns false for all elements in the range
    /// \tparam ExecPolicy[in]: Type of the execution policy
    /// \tparam Iterator[in]: Type of the iterator
    /// \tparam Predicate[in]: Type of the predicate function
    /// \tparam config[in]: Configuration for the algorithm
    /// \param policy[in]: Execution policy to use
    /// \param first[in]: Iterator to the first element in the range
    /// \param last[in]: Iterator to one past the last element in the range
    /// \param pred[in]: Predicate function to apply to elements
    /// \return True if the predicate returns false for all elements in the range, false otherwise
    template <class ExecPolicy,
              typename Iterator,
              typename Predicate,
              const AlgorithmAnyOfConfig &config=algorithmAnyOfConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<Iterator> &&
	    		 std::regular_invocable<Predicate,
									    const typename Iterator::value_type&>
#endif
    bool none_of(ExecPolicy&& policy,
                 Iterator first,
                 Iterator last,
                 Predicate pred) noexcept {
    	constexpr static AlgorithmFindConfig c = {
    		.min_size_per_thread = config.min_size_per_thread,
    	};

        return last == cryptanalysislib::find_if
						<ExecPolicy, Iterator, Predicate, c>
    					(std::forward<ExecPolicy>(policy), first, last, pred);
    }

	/// Parallel version of any_of - checks if a predicate returns true for at least one element in the range
    /// \tparam ExecPolicy[in]: Type of the execution policy
    /// \tparam RandIt[in]: Type of the random access iterator
    /// \tparam Predicate[in]: Type of the predicate function
    /// \tparam config[in]: Configuration for the algorithm
    /// \param policy[in]: Execution policy to use
    /// \param first[in]: Iterator to the first element in the range
    /// \param last[in]: Iterator to one past the last element in the range
    /// \param pred[in]: Predicate function to apply to elements
    /// \return True if the predicate returns true for at least one element in the range, false otherwise
    template <class ExecPolicy,
              typename RandIt,
              typename Predicate,
              const AlgorithmAnyOfConfig &config=algorithmAnyOfConfig>
#if __cplusplus > 201709L
	    requires std::bidirectional_iterator<RandIt> &&
	    		 std::regular_invocable<Predicate,
									    const typename RandIt::value_type&>
#endif
    bool any_of(ExecPolicy&& policy,
                RandIt first,
                RandIt last,
                Predicate pred) noexcept {
        return !cryptanalysislib::none_of
          <ExecPolicy, RandIt, Predicate, config>
          (std::forward<ExecPolicy>(policy), first, last, pred);
    }

} // end namespace cryptanalysislib
#endif 
