#ifndef CRYPTANALYSISLIB_ALGORITHM_ALL_OF_H
#define CRYPTANALYSISLIB_ALGORITHM_ALL_OF_H

#include "thread/thread.h"
#include "algorithm/find.h"

// https://en.cppreference.com/w/cpp/algorithm/all_any_none_of
namespace cryptanalysislib {
	struct AlgorithmAnyOfConfig : public AlgorithmConfig {
		 const size_t min_size_per_thread = 1048576;
	};
	constexpr static AlgorithmAnyOfConfig algorithmAnyOfConfig;

	/// \tparam InputIt
    /// \tparam UnaryPred
    /// \param first
    /// \param last
    /// \param p
    /// \return
    template<class InputIt,
             class UnaryPred,
             const AlgorithmAnyOfConfig &config=algorithmAnyOfConfig>
#if __cplusplus > 201709L
	    requires std::bidirectional_iterator<InputIt>
#endif
    constexpr bool all_of(InputIt first,
                          InputIt last,
                          UnaryPred p) noexcept {
        return cryptanalysislib::find_if_not(first, last, p) == last;
    }

	/// \tparam InputIt
    /// \tparam UnaryPred
    /// \param first
    /// \param last
    /// \param p
    /// \return
	template<class InputIt,
             class UnaryPred,
             const AlgorithmAnyOfConfig &config=algorithmAnyOfConfig>
#if __cplusplus > 201709L
	    requires std::bidirectional_iterator<InputIt>
#endif
    constexpr bool any_of(InputIt first,
                          InputIt last,
                          UnaryPred p) noexcept {
        return std::find_if
    		<InputIt, UnaryPred>
    		(first, last, p) != last;
    }

    /// \tparam InputIt
    /// \tparam UnaryPred
    /// \param first
    /// \param last
    /// \param p
    /// \return
    template<class InputIt,
             class UnaryPred,
             const AlgorithmAnyOfConfig &config=algorithmAnyOfConfig>
#if __cplusplus > 201709L
	    requires std::bidirectional_iterator<InputIt>
#endif
    constexpr bool none_of(InputIt first,
                           InputIt last,
                           UnaryPred p) noexcept {
        return cryptanalysislib::find_if
    			<InputIt, UnaryPred>
    			(first, last, p) == last;
    }

    /// @tparam ExecPolicy
    /// @tparam RandIt
    /// @tparam Predicate
    /// @param policy
    /// @param first
    /// @param last
    /// @param pred
    /// @return
    template <class ExecPolicy,
              typename RandIt,
              typename Predicate,
              const AlgorithmAnyOfConfig &config=algorithmAnyOfConfig>
#if __cplusplus > 201709L
	    requires std::bidirectional_iterator<RandIt>
#endif
    bool all_of(ExecPolicy&& policy,
                RandIt first,
                RandIt last,
                Predicate pred) noexcept {
    	constexpr static AlgorithmFindConfig c = {
    		.min_size_per_thread = config.min_size_per_thread
    	};

        return last == cryptanalysislib::find_if_not
    					<ExecPolicy, RandIt, Predicate, c>
    					(std::forward<ExecPolicy>(policy), first, last, pred);
    }

    /// @tparam ExecPolicy
    /// @tparam RandIt
    /// @tparam Predicate
    /// @param policy
    /// @param first
    /// @param last
    /// @param pred
    /// @return
    template <class ExecPolicy,
              typename RandIt,
              typename Predicate,
              const AlgorithmAnyOfConfig &config=algorithmAnyOfConfig>
#if __cplusplus > 201709L
	    requires std::bidirectional_iterator<RandIt>
#endif
    bool none_of(ExecPolicy&& policy,
                 RandIt first,
                 RandIt last,
                 Predicate pred) noexcept {
    	constexpr static AlgorithmFindConfig c = {
    		.min_size_per_thread = config.min_size_per_thread
    	};

        return last == cryptanalysislib::find_if
						<ExecPolicy, RandIt, Predicate, c>
    					(std::forward<ExecPolicy>(policy), first, last, pred);
    }

	/// @tparam ExecPolicy
    /// @tparam RandIt
    /// @tparam Predicate
    /// @param policy
    /// @param first
    /// @param last
    /// @param pred
    /// @return
    template <class ExecPolicy,
              typename RandIt,
              typename Predicate,
              const AlgorithmAnyOfConfig &config=algorithmAnyOfConfig>
#if __cplusplus > 201709L
	    requires std::bidirectional_iterator<RandIt>
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
#endif //EQUAL_H
