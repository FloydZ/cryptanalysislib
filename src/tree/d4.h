#ifndef CRYPTANALYSISLIB_TREE_D4_H
#define CRYPTANALYSISLIB_TREE_D4_H

#include "tree.h"

/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
void Tree_T<List, config>::join16lists(List &out, std::vector<List> &L,
                                       const LabelType &target,
                                       const std::vector<uint32_t> &lta) noexcept {
	assert(lta.size() == 5 && L.size() == 16);

	// limits:
	const uint32_t k_lower1 = lta[0], k_upper3 = lta[3];
	const uint32_t k_lower4 = lta[3], k_upper4 = lta[4];
	out.set_load(0);

	// the intermediate target of the left subtree, the right one gets
	// `target - R`, both on [k_lower1, k_upper3)
	LabelType R, tR;
	random_intermediate_target(R, k_lower1, k_upper3);
	tR.zero();
	LabelType::sub(tR, target, R, k_lower1, k_upper3);

	// the labels of the outputs of `join8lists` are recomputed, so they are
	// valid on [k_lower4, k_upper4)
	List iL1{out.size()}, iL2{out.size()};
	join8lists(iL1, L.data(), R, lta.data());
	if (iL1.load() == 0) { return; }
	join8lists(iL2, L.data() + 8, tR, lta.data());
	if (iL2.load() == 0) { return; }

	auto f = [this, &target, k_lower1, k_upper4](List &out, List &L1, List &L2,
	                                            const size_t i, const size_t j) __attribute__((always_inline)) {
		append_solution(out, L1[i], L2[j], target, k_lower1, k_upper4);
		return false;
	};
	join2lists(out, iL1, iL2, target, k_lower4, k_upper4, true, f);
}

#endif
