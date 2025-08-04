#ifndef CRYPTANALYSISLIB_ALGORITHM_APPLY_H
#define CRYPTANALYSISLIB_ALGORITHM_APPLY_H

#include <tuple>
#include <vector>
#include <future>

#include "algorithm/algorithm.h"

namespace cryptanalysislib {
	/// Configuration for the apply algorithm operations
	struct AlgorithmApplyConfig : public AlgorithmConfig {
	    /// min_size_per_thread[in]: Minimum number of elements that should be processed per thread
	    constexpr static size_t min_size_per_thread = 1u<<10u;
	};
	/// Default configuration for apply algorithm operations
	constexpr static AlgorithmApplyConfig algorithmApplyConfig;

	/// Applies an operation to each element in a container in parallel using std::apply
	/// \tparam ExecPolicy[in]: Type of execution policy that determines parallelism behavior
	/// \tparam Op[in]: Type of operation to apply to each tuple of arguments
	/// \tparam ArgContainer[in]: Type of container holding argument tuples
	/// \tparam config[in]: Configuration for the algorithm
	/// \param policy[in]: Execution policy instance that controls thread pool access
	/// \param op[in]: Operation to apply to each tuple of arguments
	/// \param args_list[in]: Container of argument tuples to process
	/// \return Vector of futures for tracking the completion of parallel tasks
	template <class ExecPolicy,
			  class Op,
			  class ArgContainer,
			  const AlgorithmApplyConfig &config=algorithmApplyConfig>
	std::vector<std::future<void>>
	parallel_apply(ExecPolicy &&policy,
				   Op op,
				   const ArgContainer& args_list) noexcept {
		std::vector<std::future<void>> futures;
		auto& task_pool = *policy.pool();

		for (const auto& args : args_list) {
			futures.emplace_back(task_pool.submit([](Op op, const auto& args_fwd) {
					 std::apply(op, args_fwd);
				 }, op, args));
		}

		return futures;
	}
} // end namespace cryptanalysislib
#endif //APPLY_H
