#ifndef CRYPTANALYSISLIB_ALGORITHM_INCLUSIVE_SCAN_H
#define CRYPTANALYSISLIB_ALGORITHM_INCLUSIVE_SCAN_H

#include "algorithm/prefixsum.h"

namespace cryptanalysislib {
	using algorithm::AlgorithmPrefixsumConfig;
	using algorithm::algorithmPrefixsumConfig;

	/// \tparam InputIt
	/// \tparam OutputIt
	/// \tparam BinaryOp
	/// \tparam config
	/// \param first
	/// \param last
	/// \param d_first
	/// \param op
	/// \return
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

	/// \tparam InputIt
	/// \tparam OutputIt
	/// \tparam BinaryOp
	/// \tparam config
	/// \param first
	/// \param last
	/// \param d_first
	/// \param init
	/// \param op
	/// \return
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

	/// \tparam ExecPolicy
	/// \tparam InputIt
	/// \tparam OutputIt
	/// \tparam BinaryOp
	/// \tparam config
	/// \param policy
	/// \param first1
	/// \param last1
	/// \param d_first
	/// \param init
	/// \param op
	/// \return
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
