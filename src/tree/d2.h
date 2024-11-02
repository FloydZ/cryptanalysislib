#ifndef CRYPTANALYSISLIB_TREE_D2_H
#define CRYPTANALYSISLIB_TREE_D2_H

#include "tree.h"

/// doc see tree.h
template<class List,
		const TreeConfig &config>
#if __cplusplus > 201709L
requires TreeAble<List>
#endif
template<const uint32_t weight,
         typename F>
size_t Tree_T<List, config>::twolevel_streamjoin(List &out, List &iL, List &L1, List &L2,
						 const uint32_t k_lower1, const uint32_t k_upper1,
						 const uint32_t k_lower2, const uint32_t k_upper2,
						 bool prepare,
                         F f) noexcept {
	ASSERT(k_lower1 < k_upper1 &&
		   0 < k_upper1 && k_lower2 < k_upper2
		   && 0 < k_upper2
		   && k_lower1 <= k_lower2
		   && k_upper1 <= k_upper2);
	// internal variables.
	std::pair<size_t, size_t> boundaries;
	ElementType e;

	if (prepare) {
		iL.sort_level(k_lower1, k_upper2);
		L1.sort_level(k_lower1, k_upper1);
		L2.sort_level(k_lower1, k_upper1);
	}

	auto op = [](ElementType &c, const ElementType &a, const ElementType &b,
				 const uint64_t l, const uint64_t h) __attribute__((always_inline)) {
		ElementType::sub(c, a, b, l, h, -1u);
	};

	// early exit
	if (iL.load() == 0) { return 0; }
	if (L1.load() == 0) { return 0; }
	if (L2.load() == 0) { return 0; }

	uint64_t i=0, j=0;
	size_t ret = 0;
	while (i < L1.load() && j < L2.load()) {
		if (L2[j].is_greater(L1[i], k_lower1, k_upper1)) {
			i++;
		} else if (L1[i].is_greater(L2[j], k_lower1, k_upper1)) {
			j++;
		} else {
			uint64_t i_max=i+1ull, j_max=j+1ull;
			for (; i_max < L1.load() && L1[i].is_equal(L1[i_max], k_lower1, k_upper1); i_max++) {}
			for (; j_max < L2.load() && L2[j].is_equal(L2[j_max], k_lower1, k_upper1); j_max++) {}

			const uint64_t jprev = j;

			// we have found equal elements. But this time we don't have to
			// save the result. Rather we stream join everything up to the final solution.
			for (; i < i_max; ++i) {
				for (j = jprev; j < j_max; ++j) {
					// add/sub on full length
					op(e, L1[i], L2[j], k_lower1, k_upper2);
#ifdef DEBUG
					if (!e.label.is_zero(k_lower1, k_upper1)) {
						std::cout << e;
						std::cout << L2[j];
						std::cout << L1[i];
						ASSERT(false);
					}
#endif

					boundaries = iL.search_boundaries(e, k_lower2, k_upper2);

					// finished?
					// NOTE: we cannot break out of the two loops
					// only the first one.
					if (boundaries.first == boundaries.second) { break; }

					for (size_t l = boundaries.first; l < boundaries.second; ++l) {
						f(out, iL, e, l);
						ret += 1;
					}
				}
			}
		}
	}

	return ret;
}


template<class List,
		const TreeConfig &config>
#if __cplusplus > 201709L
requires TreeAble<List>
#endif
template<const uint32_t weight>
size_t Tree_T<List, config>::twolevel_streamjoin(List &out, List &iL, List &L1, List &L2,
												 const uint32_t k_lower1, const uint32_t k_upper1,
												 const uint32_t k_lower2, const uint32_t k_upper2,
												 bool prepare) noexcept {

	auto f =
	        [k_lower1, k_upper1, k_lower2, k_upper2]
	        (List & out, const List &iL, ElementType &e, const size_t l)
	        __attribute__((always_inline)) {
		(void)k_upper1;
		(void)k_lower2;

		constexpr uint32_t filter = uint32_t(-1);
		constexpr bool sub = !LabelType::binary();

		// NOTE: it can happen that the addition here is a representation, thus
		// it will not hold any longer that value*matrix = label, if one simply
		// adds the two results.
		if constexpr (!weight) {
			const size_t b = out.load();
			ValueType::add(out[b].value, iL[l].value, e.value, k_lower1, k_upper2);
			if (out[b].value.popcnt(k_lower1, k_upper2) != weight) { return; }
			// out[b].recalculate_label(matrix);
			ValueType::add(out[b].value, iL[l].value, e.value, k_lower1, k_upper2);
			out.set_load(b + 1);
		} else {
			out.add_and_append(iL[l], e, k_lower1, k_upper2, filter, sub);
#ifdef DEBUG
			const size_t b = out.load() - 1;
			if (!out[b].label.is_zero(k_lower1, k_upper2)) {
				std::cout << iL[l] << std::endl;
				std::cout << e << std::endl;
				std::cout << out[b] << std::endl;
				ASSERT(false);
			}
#endif
		}
	};

	return twolevel_streamjoin(out, iL, L1, L2, k_lower1, k_upper1, k_lower1, k_upper2, prepare, f);
}

/// doc see tree.h
template<class List,
		const TreeConfig &config>
#if __cplusplus > 201709L
requires TreeAble<List>
#endif
template<typename F>
size_t Tree_T<List, config>::twolevel_streamjoin_on_iT(List &out, List &iL, const List &L1, List &L2,
                               const LabelType &target,
                               const uint32_t k_lower1, const uint32_t k_upper1,
                               const uint32_t k_lower2, const uint32_t k_upper2,
                               const bool prepare,
                               F f) noexcept {
	ASSERT(k_lower1 < k_upper1 &&
	       0 < k_upper1 && k_lower2 < k_upper2
	       && 0 < k_upper2
	       && k_lower1 <= k_lower2
	       && k_upper1 <= k_upper2);

	// internal variables.
	std::pair<uint64_t, uint64_t> boundaries;
	ElementType e1, e2;
	uint64_t i = 0, j = 0;
	LabelType tmp, tmp2;

	if (prepare) {
		L2.sort_level(k_lower1, k_upper1, target);
		iL.sort_level(k_lower2, k_upper2);
	}

	size_t ret=0;
	while (i < L1.load() && j < L2.load()) {
		LabelType::add(tmp, L2[j].label, target);
		if (tmp.is_greater(L1[i].label, k_lower1, k_upper1)) {
			i++;
		} else if (L1[i].label.is_greater(tmp, k_lower1, k_upper1)) {
			j++;
		} else {
			uint64_t i_max, j_max;
			for (i_max = i + 1; i_max < L1.load() && L1[i].is_equal(L1[i_max], k_lower1, k_upper1); i_max++) {}
			for (j_max = j+1;j_max < L2.load();j_max++) {
				LabelType::add(tmp2, L2[j_max].label, target);
				if (!tmp.is_equal(tmp2, k_lower1, k_upper1))  { break; }
			}

			const uint64_t jprev = j;

			// we have found equal elements. But this time we don't have to
			// save the result. Rather we stream join everything up to the final solution.
			for (; i < i_max; ++i) {
				for (j = jprev; j < j_max; ++j) {
					ElementType::add(e1, L1[i], L2[j], k_lower1, k_upper2, -1);
					ASSERT(e1.label.is_equal(target, k_lower1, k_upper1));

					LabelType::sub(e2.label, e1.label, target);
					e2.label.neg();
					boundaries = iL.search_boundaries(e2, k_lower2, k_upper2);

					// finished?
					if (boundaries.first == boundaries.second) {
						// NOTE: we cannot break out of the two loops
						// only the first one.
						break;
					}

					for (size_t l = boundaries.first; l < boundaries.second; ++l) {
						ret += 1;
						if(f(out, iL, e1, l)) { goto finish; }
					}
				}
			}
		}
	}
finish:
	return ret;
}

/// doc see tree.h
template<class List,
		const TreeConfig &config>
#if __cplusplus > 201709L
requires TreeAble<List>
#endif
size_t Tree_T<List, config>::twolevel_streamjoin_on_iT(List &out, List &iL, const List &L1, List &L2,
													   const LabelType &target,
													   const uint32_t k_lower1, const uint32_t k_upper1,
													   const uint32_t k_lower2, const uint32_t k_upper2,
													   const bool prepare) noexcept {
	auto f=[k_lower1, k_upper1, k_lower2, k_upper2]
			(List &out, const List &iL, ElementType &e, const size_t l)
			__attribute__((always_inline)) {
		out.add_and_append(e, iL[l], 0, LabelLENGTH, -1);
		return false;
	};

	return twolevel_streamjoin_on_iT(out, iL, L1, L2, target, k_lower1, k_upper1, k_lower1, k_upper2, prepare, f);
}

/// doc see tree.h
template<class List,
		const TreeConfig &config>
#if __cplusplus > 201709L
	requires TreeAble<List>
#endif
template<typename F>
size_t Tree_T<List, config>::twolevel_streamjoin_on_iT_v2(List &out, List &iL,
								  const List &L1, List &L2,
								  const LabelType &target, const LabelType &iT,
								  const uint32_t k_lower1, const uint32_t k_upper1,
								  const uint32_t k_lower2, const uint32_t k_upper2,
								  const bool prepare,
								  F f) noexcept {
	if (prepare) {
		L2.sort_level(k_lower1, k_upper1);
		iL.sort_level(k_lower1, k_upper2);
	}
	ASSERT(L2.is_sorted(k_lower1, k_upper1));
	ASSERT(iL.is_sorted(k_lower1, k_upper2));
	(void)k_lower2;

	ElementType tmpe1;
	LabelType t1, t2;
	size_t ret=0;
	for (size_t k = 0; k < L1.load(); ++k) {
		LabelType::sub(t1, iT, L1[k].label);
		size_t l = L2.search_level(t1, k_lower1, k_upper1);
		for (; (l < L2.load()) &&
			   (t1.is_equal(L2[l].label, k_lower1, k_upper1));
			   ++l) {
			LabelType::sub(t2, target, L1[k].label);
			LabelType::sub(t2, t2, L2[l].label);
			size_t o = iL.search_level(t2, k_lower1, k_upper2);
			for (; (o < iL.load()) &&
				   (t2.is_equal(iL[o].label, k_lower1, k_upper2));
				   ++o) {
				ElementType::add(tmpe1, iL[o], L1[k]);
				// out.add_and_append(tmpe1, L2[l], k_lower1, k_upper2, filter);
				f(out, L2, tmpe1, l);
				ret += 1;
			}
		}
	}

	return ret;
}

/// doc see tree.h
template<class List,
		const TreeConfig &config>
#if __cplusplus > 201709L
requires TreeAble<List>
#endif
size_t Tree_T<List, config>::twolevel_streamjoin_on_iT_v2(List &out, List &iL,
														  const List &L1, List &L2,
														  const LabelType &target, const LabelType &iT,
														  const uint32_t k_lower1, const uint32_t k_upper1,
														  const uint32_t k_lower2, const uint32_t k_upper2,
														  const bool prepare) noexcept {

	auto f=[k_lower1, k_upper1, k_lower2, k_upper2]
			(List &out, const List &iL, ElementType &e, const size_t l)
			__attribute__((always_inline)) {
		(void)k_upper1;
	    (void)k_lower2;
	  	out.add_and_append(iL[l], e, k_lower1, k_upper2, -1u);
		return false;
	};

	return twolevel_streamjoin_on_iT_v2(out, iL, L1, L2, target, iT, k_lower1, k_upper1, k_lower1, k_upper2, prepare, f);
}

/// doc see tree.h
template<class List,
		const TreeConfig &config>
#if __cplusplus > 201709L
requires TreeAble<List>
#endif
template<const uint32_t k_lower1, const uint32_t k_upper1,
		 const uint32_t k_lower2, const uint32_t k_upper2,
         const uint32_t weight,
         typename F>
size_t Tree_T<List, config>::twolevel_streamjoin_on_iT_v2(List &out, List &iL,
								  const List &L1, List &L2,
								  const LabelType &target,
								  const LabelType &iT,
								  const bool prepare,
                                  F f) noexcept {
	static_assert(k_lower1 < k_upper1);
	static_assert(k_lower2 < k_upper2);
	(void)k_lower2;

	if (prepare) {
		L2.template sort_level<k_lower1, k_upper1>();
		iL.template sort_level<k_lower1, k_upper2>();
	}

	ASSERT(L2.is_sorted(k_lower1, k_upper1));
	ASSERT(iL.is_sorted(k_lower1, k_upper2));

	ElementType tmpe1;
	LabelType t1, t2;
	size_t ret = 0;
	for (size_t k = 0; k < L1.load(); ++k) {
		LabelType::sub(t1, iT, L1[k].label);
		size_t l = L2.template search_level
				<k_lower1, k_upper1>(t1);

		for (; (l < L2.load()) &&
			   (t1.template is_equal<k_lower1, k_upper1>(L2[l].label));
			   ++l) {

			// cpmpute the collision
			ElementType::add(tmpe1, L1[k], L2[l]);
			LabelType::sub(t2, target, L1[k].label);
			LabelType::sub(t2, t2, L2[l].label);
			size_t o = iL.template search_level
					<k_lower1, k_upper2>(t2);
			for (; (o < iL.load()) &&
				   (t2.template is_equal<k_lower1, k_upper2>(iL[o].label));
				   ++o) {
				ASSERT(iL[o].is_correct(matrix));
				ASSERT(L1[k].is_correct(matrix));
				ASSERT(L2[l].is_correct(matrix));
				ASSERT(tmpe1.is_correct(matrix));

				f(out, iL, tmpe1, l);
				ret += 1;
			}
		}
	}

	return ret;
}

/// doc see tree.h
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<const uint32_t k_lower1, const uint32_t k_upper1,
         const uint32_t k_lower2, const uint32_t k_upper2,
         typename HashMap1,
         typename HashMap2,
         typename F>
#if __cplusplus > 201709L
    requires HashMapAble<HashMap1> &&
             HashMapAble<HashMap2>
#endif
size_t Tree_T<List, config>::twolevel_streamjoin_on_iT_hashmap_v2(List &out,
                                            const HashMap1 &hmiL,
                                            const List &L1,
                                            const List &L2,
                                            const HashMap2 &hmL2,
                                            const LabelType &target,
                                            const LabelType &iT,
                                            F f) noexcept {
	static_assert(k_lower1 < k_upper1);
	static_assert(k_lower2 < k_upper2);
	static_assert(k_lower1 < k_upper1);
	(void) k_lower2;
	using LoadType1 = typename HashMap1::load_type;
	using LoadType2 = typename HashMap2::load_type;

	ElementType te1, te2;
	LabelType t1, t2;
	LoadType1 load1 = 0;
	LoadType2 load2 = 0;

	size_t ret = 0;
	for (size_t k = 0; k < L1.load(); ++k) {
		LabelType::sub(t1, iT, L1[k].label);

		const size_t s2 = hmL2.find(t1.value(), load2);
		for (size_t l2 = s2; l2 < (s2 + load2); ++l2) {
			const size_t b1 = hmL2[l2];
			ASSERT(L2[b1].label.is_equal(t1, k_lower1, k_upper1));
			// ASSERT(L2[b1].is_correct(matrix));
			ASSERT(b1 < L2.load());

			const LabelType t3 = L2[b1].label;
			LabelType::sub(t2, target, L1[k].label);
			LabelType::sub(t2, t2, t3);

			ElementType::add(te1, L1[k], L2[b1]);
			ASSERT(te1.label.is_equal(iT, k_lower1, k_upper1));
			// ASSERT(te1.is_correct(matrix));

			// NOTE: its shifted
			const size_t s1 = hmiL.find(t2.value(), load1);
			for (size_t l1 = s1; l1 < (s1 + load1); ++l1) {
				ret += 1;
				const size_t a1 = hmiL[l1].first;
				const size_t a2 = hmiL[l1].second;
				ASSERT(a1 < L1.load());
				ASSERT(a2 < L2.load());
				ElementType::add(te2, L1[a1], L2[a2]);
				if (f(out, te1, te2, b1, k, a1, a2)) { goto finish; }
			}
		}
	}
finish:
	return ret;
}
#endif
