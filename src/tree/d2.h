#ifndef CRYPTANALYSISLIB_TREE_D2_H
#define CRYPTANALYSISLIB_TREE_D2_H

#include "tree.h"


template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
size_t Tree_T<List, config>::join4lists(List &out, List &L1, List &L2, List &L3, List &L4,
                                        const LabelType &target,
                                        const std::vector<uint32_t> &lta,
                                        const bool prepare) noexcept {
	ASSERT(lta.size() >= 3);
	// limits: k_lower1, k_upper1 for the lowest level tree. And k_lower2, k_upper2 for highest level. There are
	// only two levels..., so obviously k_upper1=k_lower2
	const uint64_t k_lower1 = lta[0], k_upper1 = lta[1];
	const uint64_t k_lower2 = lta[1], k_upper2 = lta[2];
	return join4lists(out, L1, L2, L3, L4, target, k_lower1, k_upper1, k_lower2, k_upper2, prepare);
}

template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<typename F>
size_t Tree_T<List, config>::join4lists(List &out, List &L1, List &L2, List &L3, List &L4,
	                const LabelType &target,
	                const uint32_t k_lower1, const uint32_t k_upper1,
	                const uint32_t k_lower2, const uint32_t k_upper2,
	                const bool prepare,
	                F &&f) noexcept {
	ASSERT(k_lower1 < k_upper1 &&
	       0 < k_upper1 && k_lower2 < k_upper2
	       && 0 < k_upper2 && k_lower1 <= k_lower2
	       && k_upper1 < k_upper2
	       && L1.load() > 0 && L2.load() > 0
	       && L3.load() > 0 && L4.load() > 0);

	// Intermediate Element, List, Target
	List iL{static_cast<size_t>(L1.size() * config.intermediatelist_size_factor)};
	LabelType iT; iT.zero();

	// reset everything
	out.set_load(0);

	const size_t size = std::min({L1.load(), L2.load(), L4.load(), L3.load()});
	constexpr static bool sub = !LabelType::binary();
	auto op = [](LabelType &c, const LabelType &a, const LabelType &b,
		     const uint64_t l, const uint64_t h) {
		if constexpr (sub) { LabelType::sub(c, a, b, l, h);}
		else { LabelType::add(c, a, b, l, h); }
	};

	// prepare baselists
	if ((!target.is_zero()) && prepare) {
		// TODO subsetsum only
		iT.random(0, (1ull << k_upper1) + 1);

		LabelType R2;
		LabelType::sub(R2, iT, target, k_lower1, k_upper2);

		for (size_t i = 0; i < size; ++i) {
			op(L2[i].label, iT, L2[i].label, k_lower1, k_upper2);
			LabelType::add(L4[i].label, R2, L4[i].label, k_lower1, k_upper2);
			L3[i].label.neg(k_lower1, k_upper2);
		}

		L1.sort_level(k_lower1, k_upper1);
		L2.sort_level(k_lower1, k_upper1);
	}

	// NOTE: the intermediate target `R` is ingored in this call
	join2lists(iL, L1, L2, iT, k_lower1, k_upper1, false);

	// early exit
	if (iL.load() == 0) {
		return 0;
	}

	// Now run the merge procedure for the right part of the tree.
	return twolevel_streamjoin(out, iL, L3, L4, k_lower1, k_upper1, k_lower2, k_upper2, prepare, f);
}


template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
	size_t Tree_T<List, config>::join4lists(List &out, List &L1, List &L2, List &L3, List &L4,
               const LabelType &target,
               const uint32_t k_lower1, const uint32_t k_upper1,
               const uint32_t k_lower2, const uint32_t k_upper2,
               const bool prepare) noexcept {

	auto f =
	        [k_lower1, k_upper1, k_lower2, k_upper2]
	        (List &out, const List &iL, ElementType &e, const size_t l)
	        __attribute__((always_inline)) {
		(void)k_upper1;
		(void)k_lower2;

		out.add_and_append(iL[l], e, k_lower1, k_upper2, -1u, !LabelType::binary());
	};

	return join4lists(out, L1, L2, L3, L4, target, k_lower1, k_upper1, k_lower2, k_upper2, prepare, f);
}




template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<typename F>
size_t Tree_T<List, config>::join4lists_on_iT_v2(List &out,
	                         const List &L1, List &L2,
	                         const List &L3, List &L4,
						     const LabelType &target,
						     const uint32_t k_lower1, const uint32_t k_upper1,
						     const uint32_t k_lower2, const uint32_t k_upper2,
						     const bool prepare,
						     F &&f) noexcept {
		(void)k_lower2;
		List iL{static_cast<size_t>(L1.size() * config.intermediatelist_size_factor)};
		out.set_load(0);

		// reset everything
		if (prepare) {
			L2.sort_level(k_lower1, k_upper1);
			L4.sort_level(k_lower1, k_upper1);
		}

		ASSERT(L2.is_sorted(k_lower1, k_upper1));
		ASSERT(L4.is_sorted(k_lower1, k_upper1));

		ElementType tmpe1;
		LabelType t1, iT;
		iT.random(0, 1ull << k_upper1); // TODO only correct for SubSetSum
		join2lists_on_iT_v2(iL, L1, L2, iT, k_lower1, k_upper1, prepare);
		// early exit
		if (iL.load() == 0) {
			return 0;
		}

		iL.sort_level(0, k_upper2);
		LabelType::sub(t1, target, iT);
		return twolevel_streamjoin_on_iT_v2(out, iL, L3, L4, target, t1,
		                             k_lower1, k_upper1, k_lower2, k_upper2,
		                             false, f);
	}

template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
size_t Tree_T<List, config>::join4lists_on_iT_v2(List &out,
	                         const List &L1, List &L2,
	                         const List &L3, List &L4,
						     const LabelType &target,
						     const uint32_t k_lower1, const uint32_t k_upper1,
						     const uint32_t k_lower2, const uint32_t k_upper2,
						     const bool prepare) noexcept {
	auto f = [k_lower1, k_upper1, k_lower2, k_upper2]
	        (List &out, const List &iL, ElementType &e, const size_t l)
	        __attribute__((always_inline)) {
		(void)k_upper1;
		(void)k_lower2;

		constexpr uint32_t filter = uint32_t(-1);
		constexpr bool sub = !LabelType::binary();
		out.add_and_append(iL[l], e, k_lower1, k_upper2, filter, sub);
	};
	return join4lists_on_iT_v2(out, L1, L2, L3, L4, target, k_lower1, k_upper1, k_lower2, k_upper2, prepare, f);
}


template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
	template<const uint32_t k_lower1, const uint32_t k_upper1,
			 const uint32_t k_lower2, const uint32_t k_upper2,
			 typename F>
size_t Tree_T<List, config>::join4lists_on_iT_v2(List &out,
	                         const List &L1, List &L2,
	                         const List &L3, List &L4,
	                         const LabelType &target,
	                         const bool prepare,
	                         F f) noexcept {
		(void) k_lower2;
		List iL{static_cast<size_t>(L1.size() * config.intermediatelist_size_factor)};
		out.set_load(0);

		// reset everything
		if (prepare) {
			L2.template sort_level<k_lower1, k_upper1>();
			L4.template sort_level<k_lower1, k_upper1>();
		}

		ASSERT(L2.is_sorted(k_lower1, k_upper1));
		ASSERT(L4.is_sorted(k_lower1, k_upper1));

		ElementType tmpe1;
		LabelType t1, iT;
		iT.random(0, 1ull << k_upper1);
		join2lists_on_iT_v2<k_lower1, k_upper1>(iL, L1, L2, iT, false);
		// early exit
		if (iL.load() == 0) { return 0; }

		iL.template sort_level<k_lower1, k_upper2>();
		LabelType::sub(t1, target, iT);

#ifdef DEBUG
		for (size_t i = 0; i < iL.load(); ++i) {
			ASSERT(iL[i].label.is_equal(iT, k_lower1, k_upper1));
			ASSERT(iL[i].is_correct(matrix));
		}
#endif

		return twolevel_streamjoin_on_iT_v2
		        <k_lower1, k_upper1, k_lower2, k_upper2>
		        (out, iL, L3, L4, target, t1, false, f);
	}


#endif
