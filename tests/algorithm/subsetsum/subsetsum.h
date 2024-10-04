#ifndef CRYPTANALYSISLIB_TEST_ALGORITHM_SUBSETSUM_H
#define CRYPTANALYSISLIB_TEST_ALGORITHM_SUBSETSUM_H

#include <cstdint>

/// TODO move into algorithms
/// generates a rng subset sum instance
/// NOTE:
/// 	- nr of indicies generates = n/2
/// 	- max index = n
/// \tparam Label
/// \tparam List, std:vector, std::array
/// \tparam Matrix
/// \param target return value
/// \param weights return value
/// \param AT transposed matrix, actually vector in this case
/// \param n number of bits of the label
/// \param mitm if true: will make sure that the solution
/// 	evenly splits between each half
/// \param debug if true: will print the solution
template<typename Label,
         typename List,
         typename Matrix>
constexpr static void generate_subsetsum_instance(Label &target,
                                                  List &weights,
                                                  const Matrix &AT,
                                                  const uint32_t n,
                                                  const bool mitm = true,
                                                  const bool debug = true) noexcept {
	if (!IsStdArray<List>()) { weights.reserve(n/2);}
	target.zero();
	if (mitm) { generate_random_mitm_indices(weights, n);
	} else { generate_random_indices(weights, n); }

	for (uint32_t i = 0; i < n/2; ++i) {
		Label::add(target, target, AT[0][weights[i]]);
	}

	if (debug) {
		std::cout << target << " , subset sum target" << std::endl;
		for (const auto &w : weights) {
			std::cout << w << " ";
		}
		std::cout << std::endl;
	}
}

#endif