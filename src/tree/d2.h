#ifndef CRYPTANALYSISLIB_TREE_D2_H
#define CRYPTANALYSISLIB_TREE_D2_H

#include "tree.h"

/// 		out HM
///        ┌─────────┐
///        └┐       ┌┘
///         └┐     ┌┘
///          └──┬──┘ match on iT
///     k_lower1│k_upper1
///     ┌───────┴───────┐
///     │  L2=iT-L1     │
/// ┌───┴───┐      ┌────┴────┐
/// │ const │      └┐       ┌┘
/// │       │       └┐     ┌┘
/// └───────┘        └─────┘  hashing
///    L1              HM2 <---------- L2
/// NOTE: v2 means that the `label` of the output elements in `out`
///		are the target. So this function actually returns targets and not
/// 	zeros.
/// NOTE: the output will a be hashmap
/// NOTE: the elemens of the output hashmap are shifted down by `k_upper`
/// \tparam k_lower lower coordinate to match on
/// \tparam k_upper upper coordinate to match on
/// \tparam HashMapIn
/// \tparam HashMapOut
/// \param out
/// \param L1 input list const, will NOT be sorted
/// \param L2 input list const, will be hashed into HM2 if prepare==true
/// \param target
/// \param hm2 hashmap of list L2
/// \param prepare if true == hashses L2 into HM2
/// \return the number of found collisions
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<const uint32_t k_lower,
         const uint32_t k_upper,
         typename HashMapIn,
         typename HashMapOut>
#if __cplusplus > 201709L
    requires HashMapAble<HashMapIn> &&
             HashMapAble<HashMapOut>
#endif
size_t Tree_T<List, config>::join2lists_on_iT_hashmap_v2(HashMapOut &out,
                                                         const List &L1, const List &L2,
                                                         HashMapIn &hm2,
                                                         const LabelType &target,
                                                         const bool prepare) noexcept {
	ASSERT(k_lower < k_upper && 0 < k_upper);
	using LoadType = typename HashMapIn::load_type;
	using HMOutValueType = HashMapOut::data_type;
	out.clear();

	if (prepare) {
		// only clear if we really need it
		hm2.clear();
		for (size_t i = 0; i < L2.load(); ++i) {
			hm2.insert(L2[i].label.value(), i);
		}
	}

#ifdef DEBUG
	for (size_t i = 0; i < HashMapIn::nrbuckets; ++i) {
		for (uint32_t j = 0; j < hm2.load_without_hash(i); ++j) {
			const size_t pos = hm2[i];
			ASSERT(pos < L2.load());
		}
	}
#endif

	LabelType sigma_t;
	LoadType load = 0;
	LabelType e;
	size_t ret = 0;
	for (size_t i = 0; i < L1.load(); ++i) {
		LabelType::template sub<k_lower, k_upper>(sigma_t, target, L1[i].label);

		size_t s = hm2.find(sigma_t.value(), load);
		for (size_t k = s; k < s + load; ++k) {
			const size_t j = hm2[k];
			ASSERT(L2[j].label.is_equal(sigma_t, k_lower, k_upper));
			ASSERT(j < L2.load());

			LabelType::add(e, L1[i].label, L2[j].label);
			ASSERT(e.is_equal(target, k_lower, k_upper));

			// NOTE: NOTE that is shifted
			out.insert(e.value(), HMOutValueType{i, j});
			ret += 1;
		}
	}

	return ret;
}
#endif
