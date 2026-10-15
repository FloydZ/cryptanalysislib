#ifndef CRYPTANALYSISLIB_TREE_GENERIC_H
#define CRYPTANALYSISLIB_TREE_GENERIC_H

#include "tree.h"

/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<const uint32_t d>
void Tree_T<List, config>::join_stream_generic(List &out, std::vector<List> &L,
                                               const LabelType &target,
                                               const std::vector<uint32_t> &lta) noexcept {
	assert(L.size() == (1ull << d) && lta.size() == d + 1u);
	out.set_load(0);

	const uint32_t k_lower = lta[0], k_upper = lta[d];
	join_stream_generic<d>(L.data(), target, lta.data(),
		[&](const ElementType &a, const ElementType &b) __attribute__((always_inline)) {
		append_solution(out, a, b, target, k_lower, k_upper);
	});
}

/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<const uint32_t d, typename F>
void Tree_T<List, config>::join_stream_generic(List *L,
                                               const LabelType &target,
                                               const uint32_t *lta,
                                               F &&f) noexcept {
	static_assert(d > 0);

	if constexpr (d == 1) {
		// `L[0]` is not changed, `L[1]` is sorted on [lta[0], lta[1])
		List dummy{0};
		auto g = [&f](List &, const List &L1, List &L2,
		              const size_t i, const size_t j) __attribute__((always_inline)) {
			f(L1[i], L2[j]);
			return false;
		};
		join2lists_on_iT_v2(dummy, L[0], L[1], target, lta[0], lta[1], true, g);
	} else {
		constexpr size_t half = 1ull << (d - 1u);
		const uint32_t k_lower = lta[d - 1u], k_upper = lta[d];

		// the intermediate target of the left subtree, the right one gets
		// `target - R`, both on [lta[0], k_lower)
		LabelType R, tR;
		random_intermediate_target(R, lta[0], k_lower);
		tR.zero();
		LabelType::sub(tR, target, R, lta[0], k_lower);

		// the left subtree is materialized, sorted on [k_lower, k_upper) ...
		List iL{static_cast<size_t>(L[0].load() * config.intermediatelist_size_factor)};
		iL.set_load(0);
		join_stream_generic<d - 1u>(L, R, lta,
			[&iL](const ElementType &a, const ElementType &b) __attribute__((always_inline)) {
			iL.add_and_append(a, b, -1u, false);
		});
		if (iL.load() == 0) { return; }
		iL.sort_level(k_lower, k_upper);

		// ... and every element `r` of the right subtree is searched in it
		ElementType r;
		LabelType s;
		join_stream_generic<d - 1u>(L + half, tR, lta,
			[&](const ElementType &a, const ElementType &b) __attribute__((always_inline)) {
			ElementType::add(r, a, b);
			LabelType::sub(s, target, r.label, k_lower, k_upper);
			for (size_t j = iL.search_level(s, k_lower, k_upper);
			     (j < iL.load()) && s.is_equal(iL[j].label, k_lower, k_upper); ++j) {
				f(iL[j], r);
			}
		});
	}
}

#endif
