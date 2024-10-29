#ifndef CRYPTANALYSISLIB_TREE_D1_H
#define CRYPTANALYSISLIB_TREE_D1_H

#include "tree.h"


/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<typename F>
size_t Tree_T<List, config>::join2lists(List &out, List &L1, List &L2,
                                        const LabelType &target,
                                        const uint32_t k_lower,
                                        const uint32_t k_upper,
                                        bool prepare,
                                        F f) noexcept {
	ASSERT(k_lower < k_upper && 0 < k_upper);
	out.set_load(0);

	if ((!target.is_zero()) && (prepare)) {
		for (size_t s = 0; s < L2.load(); ++s) {
			// is remapped to add in the binary case
			LabelType::sub(L2[s].label, target, L2[s].label, k_lower, k_upper);
		}

		L1.sort_level(k_lower, k_upper);
		L2.sort_level(k_lower, k_upper);
	}

	// make sure everything is sorted, even if it was not prepared.
	ASSERT(L1.is_sorted(k_lower, k_upper));
	ASSERT(L2.is_sorted(k_lower, k_upper));

	uint64_t i = 0, j = 0;
	size_t ret = 0;
	while (i < L1.load() && j < L2.load()) {
		if (L2[j].is_greater(L1[i], k_lower, k_upper)) {
			i++;
		} else if (L1[i].is_greater(L2[j], k_lower, k_upper)) {
			j++;
		} else {
			uint64_t i_max = i + 1ull, j_max = j + 1ull;
			// if elements are equal find max index in each list, such that they remain equal
			for (; i_max < L1.load() && L1[i].is_equal(L1[i_max], k_lower, k_upper); i_max++) {}
			for (; j_max < L2.load() && L2[j].is_equal(L2[j_max], k_lower, k_upper); j_max++) {}

			const uint64_t jprev = j;
			for (; i < i_max; ++i) {
				for (j = jprev; j < j_max; ++j) {
					f(out, L1, L2, i, j);
					ret += 1;
				}
			}
		}
	}

	return ret;
}

/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
size_t Tree_T<List, config>::join2lists(List &out, List &L1, List &L2,
                  const LabelType &target,
                  const uint32_t k_lower,
                  const uint32_t k_upper,
                  bool prepare) noexcept {
	auto f=[k_lower, k_upper](List &out, List &L1, List &L2, const size_t i, const size_t j) __attribute__((always_inline)) {
					out.add_and_append(L1[i], L2[j], k_lower, k_upper, -1, !LabelType::binary());
#ifdef DEBUG
		const uint64_t b = out.load() - 1;
		if (!out[b].label.is_zero(k_lower, k_upper)) {
			std::cout << L1[i] << std::endl;
			std::cout << L2[j] << std::endl;
			std::cout << out[b] << std::endl;
			ASSERT(false);
		}
#endif
	};

	return join2lists(out, L1, L2, target, k_lower, k_upper, prepare, f);
}

/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<const uint32_t k_lower,
		 const uint32_t k_upper,
		 typename F>
size_t Tree_T<List, config>::join2lists(
					List &out, List &L1, List &L2,
					const LabelType &target,
					bool prepare,
					F f) noexcept {
	static_assert(k_lower < k_upper && 0 < k_upper);
	out.set_load(0);

	if ((!target.is_zero()) && (prepare)) {
		for (size_t s = 0; s < L2.load(); ++s) {
			// will be remapped to + in binary case
			LabelType::template sub
				<k_lower, k_upper>
				(L2[s].label, target, L2[s].label);
		}

		L1.template sort_level<k_lower, k_upper>();
		L2.template sort_level<k_lower, k_upper>();
	}

	ASSERT(L1.is_sorted(k_lower, k_upper));
	ASSERT(L2.is_sorted(k_lower, k_upper));

	uint64_t i = 0, j = 0;
	size_t ret = 0;
	while (i < L1.load() && j < L2.load()) {
		if (L2[j].template is_greater<k_lower, k_upper>(L1[i])) {
			i++;
		} else if (L1[i].template is_greater<k_lower, k_upper>(L2[j])) {
			j++;
		} else {
			uint64_t i_max=i+1ull, j_max=j+1ull;
			// if elements are equal find max index in each list, such that they remain equal
			for (; i_max < L1.load() && L1[i].template is_equal<k_lower, k_upper>(L1[i_max]); i_max++) {}
			for (; j_max < L2.load() && L2[j].template is_equal<k_lower, k_upper>(L2[j_max]); j_max++) {}

			const uint64_t jprev = j;
			for (; i < i_max; ++i) {
				for (j = jprev; j < j_max; ++j) {
					f(out, L1, L2, i, j);
					ret += 1;
				}
			}
		}
	}

	return ret;
}

/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<typename F>
size_t Tree_T<List, config>::join2lists_on_iT(List &out,
                      List &L1, List &L2,
                      const LabelType &target,
                      const uint32_t k_lower,
                      const uint32_t k_upper,
                      const bool prepare,
                      F f) noexcept {
	ASSERT(k_lower < k_upper && 0 < k_upper);
	out.set_load(0);

	constexpr static bool sub = !LabelType::binary();
	if (prepare) {
		L1.sort_level(k_lower, k_upper);
	}
	ASSERT(L1.is_sorted(k_lower, k_upper));

	// NOTE: will always be sorted, as we dont know the
	// target befor hand
	L2.template sort_level<sub>(k_lower, k_upper, target);

	// standard comparison oeprator
	auto op = [](LabelType &c, const LabelType &a, const LabelType &b,
	             const uint64_t l, const uint64_t h) {
		LabelType::sub(c, a, b, l, h);
	};

	LabelType tmp, tmp2;
	uint64_t i = 0, j = 0;
	size_t ret = 0;
	while ((i < L1.load()) && (j < L2.load())) {
		op(tmp, target, L2[j].label, k_lower, k_upper);

		if (tmp.is_greater(L1[i].label, k_lower, k_upper)) {
			i++;
		} else if (L1[i].label.is_greater(tmp, k_lower, k_upper)) {
			j++;
		} else {
			uint64_t i_max = i + 1ull, j_max = j + 1ull;
			// if elements are equal find max index in each list, such that they remain equal
			for (; i_max < L1.load() && L1[i].is_equal(L1[i_max], k_lower, k_upper); i_max++) {}
			for (; j_max < L2.load(); j_max++) {
				op(tmp2, target, L2[j_max].label, k_lower, k_upper);
				if (!tmp.is_equal(tmp2, k_lower, k_upper)) { break; }
			}

			const uint64_t jprev = j;
			for (; i < i_max; ++i) {
				for (j = jprev; j < j_max; ++j) {
					f(out, L1, L2, i, j);
					ret += 1;
				}
			}
		}
	}

	return ret;
}

/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
size_t Tree_T<List, config>::join2lists_on_iT(List &out,
                      List &L1, List &L2,
                      const LabelType &target,
                      const uint32_t k_lower,
                      const uint32_t k_upper,
                      const bool prepare) noexcept {

	constexpr static uint32_t filter = -1;
	auto f=[k_lower, k_upper, target]
				(List &out, List &L1, List &L2, const size_t i, const size_t j) __attribute__((always_inline)) {
		out.add_and_append(L1[i], L2[j], k_lower, k_upper, filter);

#ifdef DEBUG
		const uint64_t b = out.load() - 1;
		if (!out[b].label.is_equal(target, k_lower, k_upper)) {
			L1[i].label.print_binary();
			L2[j].label.print_binary();
			out[b].label.print_binary();
			target.print_binary();
			ASSERT(false);
		}
#endif
	};

	return join2lists_on_iT(out, L1, L2, target, k_lower, k_upper, prepare, f);
}

/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<typename F>
size_t Tree_T<List, config>::join2lists_on_iT_v2(List &out,
	                         const List &L1, List &L2,
							 const LabelType &target,
							 const uint32_t k_lower,
							 const uint32_t k_upper,
	                         const bool prepare,
	                         F f) noexcept {
	ASSERT(k_lower < k_upper && 0 < k_upper);
	out.set_load(0);
	if (prepare) { L2.sort_level(k_lower, k_upper); }
	ASSERT(L2.is_sorted(k_lower, k_upper));

	LabelType sigma_t;
	size_t ret = 0;
	for (size_t i = 0; i < L1.load(); ++i) {
		// NOTE sub will be remapped to add in the binary case
		LabelType::sub(sigma_t, target, L1[i].label, k_lower, k_upper);
		size_t j = L2.search_level(sigma_t, k_lower, k_upper);
		for (; (j < L2.load()) &&
			   (sigma_t.is_equal(L2[j].label,k_lower, k_upper));
			   ++j) {
			f(out, L1, L2, i, j);
			ret += 1;
		}
	}

	return ret;
}

/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
size_t Tree_T<List, config>::join2lists_on_iT_v2(List &out,
	                         const List &L1, List &L2,
							 const LabelType &target,
							 const uint32_t k_lower,
							 const uint32_t k_upper,
							 const bool prepare) noexcept {
	auto f=[k_lower, k_upper](List &out, const List &L1, List &L2, const size_t i, const size_t j) __attribute__((always_inline)) {
		out.add_and_append(L1[i], L2[j], k_lower, k_upper, -1u);
	};

	return join2lists_on_iT_v2(out, L1, L2, target, k_lower, k_upper, prepare, f);
}


/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<const uint32_t k_lower,
         const uint32_t k_upper,
         typename F>
size_t Tree_T<List, config>::join2lists_on_iT_v2(List &out,
						 const List &L1, List &L2,
						 const LabelType &target,
                         const bool prepare,
                         F f) noexcept {
	ASSERT(k_lower < k_upper && 0 < k_upper);
	out.set_load(0);
	if (prepare) {
		L2.template sort_level<k_lower, k_upper>();
	}

	ASSERT(L2.is_sorted(k_lower, k_upper));

	LabelType sigma_t;
	size_t ret = 0;
	for (size_t i = 0; i < L1.load(); ++i) {
		/// NOTE: sub will be add in binary
		LabelType::template sub
		        <k_lower, k_upper>
		        (sigma_t, target, L1[i].label);
		size_t j = L2.template search_level<k_lower, k_upper>(sigma_t);
		for (; (j < L2.load()) &&
			   (sigma_t.template is_equal<k_lower, k_upper>(L2[j].label)); ++j) {
			f(out, L1, L2, i, j);
		}
	}

	return ret;
}


/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<const uint32_t k_lower,
         const uint32_t k_upper,
         typename HashMap,
         typename F>
#if __cplusplus > 201709L
    requires HashMapAble<HashMap>
#endif
size_t Tree_T<List, config>::join2lists_on_iT_v2(
        List &out,
        const List &L1, const List &L2,
        HashMap &hm,
        const LabelType &target,
        const bool prepare,
        F f) noexcept {
	ASSERT(k_lower < k_upper && 0 < k_upper);
	using LoadType = typename HashMap::load_type;
	out.set_load(0);

	if (prepare) {
		hm.clear();
		for (size_t i = 0; i < L2.load(); ++i) {
			hm.insert(L2[i].label.value(), i);
		}
	}

	LabelType sigma_t;
	LoadType load = 0;
	size_t ret = 0;
	for (size_t i = 0; i < L1.load(); ++i) {
		LabelType::template sub<k_lower, k_upper>(sigma_t, target, L1[i].label);

		size_t s = hm.find(sigma_t.value(), load);
		for (size_t k = s; k < s + load; ++k) {
			ret += 1;
			const size_t j = hm[k];
			f(out, L1, L2, i, j);
			ret += 1;
		}
	}

	return ret;
}

/// see tree.h for doc
template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<const uint32_t k_lower,
         const uint32_t k_upper,
         typename HashMapIn,
         typename HashMapOut,
         typename F>
#if __cplusplus > 201709L
    requires HashMapAble<HashMapIn> &&
             HashMapAble<HashMapOut>
#endif
size_t Tree_T<List, config>::join2lists_on_iT_v2(
        HashMapOut &out,
        const List &L1, const List &L2,
        HashMapIn &hm2,
        const LabelType &target,
        const bool prepare,
        F f) noexcept {
	ASSERT(k_lower < k_upper && 0 < k_upper);
	using LoadType = typename HashMapIn::load_type;
	out.clear();

	if (prepare) {
		// only clear if we really need it
		hm2.clear();
		for (size_t i = 0; i < L2.load(); ++i) {
			hm2.insert(L2[i].label.value(), i);
		}
	}

	LabelType sigma_t;
	LoadType load = 0;
	size_t ret = 0;
	for (size_t i = 0; i < L1.load(); ++i) {
		LabelType::template sub<k_lower, k_upper>(sigma_t, target, L1[i].label);

		size_t s = hm2.find(sigma_t.value(), load);
		for (size_t k = s; k < s + load; ++k) {
			const size_t j = hm2[k];
			ASSERT(L2[j].label.is_equal(sigma_t, k_lower, k_upper));
			ASSERT(j < L2.load());

			f(out, L1, L2, i, j);
			ret += 1;
		}
	}

	return ret;
}




template<class List,
		const TreeConfig &config>
#if __cplusplus > 201709L
requires TreeAble<List>
#endif
template<const uint32_t k_lower,
		 const uint32_t k_upper,
		 const uint32_t bucketsize,
		 const uint32_t nthreads,
		 class ExecPolicy>
size_t Tree_T<List, config>::join2lists_on_iT_v2(ExecPolicy&& policy,
						   List &out,
						   const List &L1, List &L2,
						   const LabelType &target) noexcept {
	/// INIT THREADS
	auto& task_pool = *policy.pool();
	if (is_seq<ExecPolicy>(policy) || nthreads == 0) {
		return join2lists_on_iT_v2
				<k_lower, k_upper>
				(out, L1, L2, target);
	}

	/// INIT Hashmap
	using D = typename LabelType::DataType;
	constexpr static SimpleHashMapConfig simpleHashMapConfigL0 {
			bucketsize, 1ull<<(k_upper-k_lower), nthreads
	};

	using HML0 = SimpleHashMap<D, size_t, simpleHashMapConfigL0, Hash<D, k_lower, k_upper, 2>>;
	using LoadType = typename HML0::load_type;
	HML0 *hm_ = new HML0{};
	HML0 hm = *hm_;

	std::vector<std::future<void>> futures;
	for (size_t tid = 0; tid < nthreads; tid++) {
		futures.emplace_back(task_pool.enqueue([tid, &hm, &L2]() __attribute__((always_inline)) {
		  const size_t spos = L2.start_pos(tid);
		  const size_t epos = L2.end_pos(tid);
		  for (size_t i = spos; i < epos; ++i) {
			  hm.insert(L2[i].label.value(), i);
		  }
		}));
	}

	cryptanalysislib::internal::wait_futures(futures);
	futures.clear();

	for (size_t tid = 0; tid < nthreads; tid++) {
		futures.emplace_back(task_pool.enqueue([tid, &hm, &out, &L1, &L2, &target]() __attribute__((always_inline)) {
		  LabelType sigma_t;
		  LoadType load = 0;
		  const size_t spos = L1.start_pos(tid);
		  const size_t epos = L1.end_pos(tid);
		  for (size_t i = spos; i < epos; ++i) {
			  LabelType::template sub<k_lower, k_upper>(sigma_t, target, L1[i].label);

			  size_t s = hm.find(sigma_t.value(), load);
			  for (size_t k = s; k < s + load; ++k) {
				  const size_t j = hm[k];
				  // todo really atomic?
				  out.template add_and_append
						  <k_lower, k_upper, -1u, false>
						  (L1[i], L2[j]);
			  }
		  }
		}));
	}

	cryptanalysislib::internal::wait_futures(futures);
	delete hm_;
	return out.load();
}
#endif
