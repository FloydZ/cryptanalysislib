#ifndef CRYPTANALYSISLIB_TREE_D3_H
#define CRYPTANALYSISLIB_TREE_D3_H

#include "tree.h"

/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
void Tree_T<List, config>::append_solution(List &out,
                                           const ElementType &e1,
                                           const ElementType &e2,
                                           const LabelType &target,
                                           const uint32_t k_lower,
                                           const uint32_t k_upper) noexcept {
	const size_t b = out.load();
	out.add_and_append(e1, e2, -1u, false);
	if (out.load() == b) { return; }

	out[b].recalculate_label(matrix);
	if (!out[b].label.is_equal(target, k_lower, k_upper)) {
		out.set_load(b);
	}
}

/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
void Tree_T<List, config>::join8lists(List &out, std::vector<List> &L,
                                      const LabelType &target,
                                      const std::vector<uint32_t> &lta) noexcept {
	assert(lta.size() == 4 && L.size() == 8);
	join8lists(out, L.data(), target, lta.data());
}

/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
void Tree_T<List, config>::join8lists(List &out, List *L,
                                      const LabelType &target,
                                      const uint32_t *lta) noexcept {
	// limits:
	const uint32_t k_lower1 = lta[0], k_upper1 = lta[1];
	const uint32_t k_lower2 = lta[1], k_upper2 = lta[2];
	const uint32_t k_lower3 = lta[2], k_upper3 = lta[3];
	out.set_load(0);

	// the intermediate target of the left subtree, the right one gets
	// `target - R`, both on [k_lower1, k_upper2)
	LabelType R, tR;
	random_intermediate_target(R, k_lower1, k_upper2);
	tR.zero();
	LabelType::sub(tR, target, R, k_lower1, k_upper2);

	// the outputs of `join4lists` are only computed on [k_lower1, k_upper2),
	// so their labels are recomputed for the last join on [k_lower3, k_upper3)
	constexpr static bool sub = !LabelType::binary();
	auto f = [this, k_lower1, k_upper2](List &out, const List &iL,
	                                    ElementType &e, const size_t l) __attribute__((always_inline)) {
		const size_t b = out.load();
		out.add_and_append(iL[l], e, k_lower1, k_upper2, -1u, sub);
		if (out.load() > b) { out[b].recalculate_label(matrix); }
	};

	List iL1{out.size()}, iL2{out.size()};
	join4lists(iL1, L[0], L[1], L[2], L[3], R, k_lower1, k_upper1, k_lower2, k_upper2, true, f);
	if (iL1.load() == 0) { return; }
	join4lists(iL2, L[4], L[5], L[6], L[7], tR, k_lower1, k_upper1, k_lower2, k_upper2, true, f);
	if (iL2.load() == 0) { return; }

	auto f2 = [this, &target, k_lower1, k_upper3](List &out, List &L1, List &L2,
	                                             const size_t i, const size_t j) __attribute__((always_inline)) {
		append_solution(out, L1[i], L2[j], target, k_lower1, k_upper3);
		return false;
	};
	join2lists(out, iL1, iL2, target, k_lower3, k_upper3, true, f2);
}

#endif
