#ifndef CRYPTANALYSISLIB_ALGORITHM_INCLUSIVE_SCAN_H
#define CRYPTANALYSISLIB_ALGORITHM_INCLUSIVE_SCAN_H

#include "algorithm/prefixsum.h"

namespace cryptanalysislib {
	using algorithm::AlgorithmPrefixsumConfig;
	using algorithm::algorithmPrefixsumConfig;

	/// Computes inclusive prefix scan with custom binary operation 
	/// \tparam InputIt Forward iterator type for input range
	/// \tparam OutputIt Forward iterator type for output range
	/// \tparam BinaryOp Binary operation type
	/// \tparam config Algorithm configuration (default: algorithmPrefixsumConfig)
	/// \param first[in]: Iterator to the beginning of the input range
	/// \param last[in]: Iterator to the end of the input range
	/// \param d_first[out]: Iterator to the beginning of the output range
	/// \param op[in]: Binary operation to perform
	/// \return Iterator to the end of the output range
	template<class InputIt,
			 class OutputIt,
			 class BinaryOp,
			 const AlgorithmPrefixsumConfig &config=algorithmPrefixsumConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt> &&
	    		 std::forward_iterator<OutputIt>
#endif
	constexpr OutputIt inclusive_scan(InputIt first,
									  InputIt last,
									  OutputIt d_first,
									  BinaryOp op) noexcept {
		return algorithm::prefixsum
			<InputIt, OutputIt, BinaryOp, config>
			(first, last, d_first, op);
	}

	/// Computes inclusive prefix scan with initial value and custom binary operation
	/// \tparam InputIt Forward iterator type for input range
	/// \tparam OutputIt Forward iterator type for output range
	/// \tparam BinaryOp Binary operation type
	/// \tparam config Algorithm configuration (default: algorithmPrefixsumConfig)
	/// \param first[in]: Iterator to the beginning of the input range
	/// \param last[in]: Iterator to the end of the input range
	/// \param d_first[out]: Iterator to the beginning of the output range
	/// \param init[in]: Initial value for scan operation
	/// \param op[in]: Binary operation to perform
	/// \return Iterator to the end of the output range
	template<class InputIt,
			 class OutputIt,
			 class BinaryOp,
			 const AlgorithmPrefixsumConfig &config=algorithmPrefixsumConfig>
#if __cplusplus > 201709L
	    requires std::forward_iterator<InputIt> &&
	    		 std::forward_iterator<OutputIt>
#endif
	constexpr OutputIt inclusive_scan(InputIt first,
								 InputIt last,
								 OutputIt d_first,
								 const typename InputIt::value_type init,
								 BinaryOp op) noexcept {
		return algorithm::prefixsum
			<InputIt, OutputIt, BinaryOp, config>
			(first, last, d_first, init, op);
	}

	/// Computes inclusive prefix scan with initial value and custom binary operation 
	/// \tparam ExecPolicy Execution policy type for parallel execution
	/// \tparam InputIt Random access iterator type for input range
	/// \tparam OutputIt Random access iterator type for output range
	/// \tparam BinaryOp Binary operation type
	/// \tparam config Algorithm configuration (default: algorithmPrefixsumConfig)
	/// \param policy[in]: Execution policy specifying parallelization strategy
	/// \param first1[in]: Iterator to the beginning of the input range
	/// \param last1[in]: Iterator to the end of the input range
	/// \param d_first[out]: Iterator to the beginning of the output range
	/// \param init[in]: Initial value for scan operation
	/// \param op[in]: Binary operation to perform
	/// \return Iterator to the end of the output range
	template<class ExecPolicy,
			 class InputIt,
			 class OutputIt,
			 class BinaryOp,
			  const AlgorithmPrefixsumConfig &config=algorithmPrefixsumConfig>
#if __cplusplus > 201709L
    requires std::random_access_iterator<InputIt> &&
    		 std::random_access_iterator<OutputIt>
#endif
	 OutputIt inclusive_scan(ExecPolicy&& policy,
						InputIt first1,
			    		InputIt last1,
						OutputIt d_first,
						const typename InputIt::value_type init,
						BinaryOp op) noexcept {
		return algorithm::prefixsum
			<InputIt, OutputIt, BinaryOp, config>
			(std::forward<ExecPolicy>(policy), first1, last1, d_first, init, op);
	}
} // end namespace cryptanalysislib
#endif
