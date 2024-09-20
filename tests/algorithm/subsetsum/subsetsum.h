#ifndef CRYPTANALYSISLIB_TEST_ALGORITHM_SUBSETSUM_H
#define CRYPTANALYSISLIB_TEST_ALGORITHM_SUBSETSUM_H

#include <cstdint>

/// generates a random subsetsum instance
/// NOTE:
/// 	- nr of indicies generates = n/2
/// 	- max index = n
/// \tparam Label
/// \tparam List
/// \tparam Matrix
/// \param target
/// \param weights
/// \param AT
/// \param n
/// \param mitm
/// \param debug
/// \return
template<typename Label, typename List, typename Matrix>
constexpr static void generate_subsetsum_instance(Label &target,
                                                  List &weights,
                                                  Matrix &AT,
                                                  const uint32_t n,
                                                  const bool mitm = true,
                                                  const bool debug = true) {
	weights.reserve(n/2);
	target.zero();
	if (mitm) {
		generate_random_mitm_indices(weights, n);
	} else {
		generate_random_indices(weights, n);
	}

	for (uint32_t i = 0; i < n/2; ++i) {
		Label::add(target, target, AT[0][weights[i]]);
	}

	if (debug) {
		std::cout << target << std::endl;
		for (const auto &w : weights) {
			std::cout << w << " ";
		}
		std::cout << std::endl;
	}
}

#endif