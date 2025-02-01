#ifndef CRYPTANALYSISLIB_TREE_D1_THREAD_H
#define CRYPTANALYSISLIB_TREE_D1_THREAD_H
#include <numeric>

template<class List,
         const TreeConfig &config>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
template<const uint32_t k_lower,
         const uint32_t k_upper,
         const uint32_t bucketsize,
         const uint32_t nthreads,
         const uint32_t chunks,
         class ExecPolicy>
size_t Tree_T<List, config>::join2lists_on_iT_v2(ExecPolicy &&policy,
                                                 List &out,
                                                 const List &L1, List &L2,
                                                 const LabelType &target) noexcept {
	/// INIT Hashmap
	using D = typename LabelType::DataType;
	constexpr static SimpleHashMapConfig simpleHashMapConfigL0{
	        bucketsize, 1ull << (k_upper - k_lower), nthreads};

	using HML0 = SimpleHashMap<D, size_t, simpleHashMapConfigL0, Hash<D, k_lower, k_upper, 2>>;
	// using LoadType = typename HML0::load_type;
	HML0 *hm = new HML0{};

	join2lists_on_iT_v2<k_lower, k_upper, bucketsize, nthreads, chunks>(policy, out, hm, L1, L2, target);

	delete hm;
	return out.load();
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
         const uint32_t chunks,
         class HashMap,
         class ExecPolicy>
size_t Tree_T<List, config>::join2lists_on_iT_v2(ExecPolicy &&policy,
                                                 List &out,
                                                 HashMap *hm,
                                                 const List &L1, List &L2,
                                                 const LabelType &target) noexcept {
	using LoadType = typename HashMap::load_type;
	static_assert(bucketsize > 0);
	static_assert(nthreads > 0);
	static_assert(chunks >= nthreads);

	/// INIT THREADS
	if (is_seq<ExecPolicy>(policy) || nthreads <= 1) {
		return join2lists_on_iT_v2<k_lower, k_upper>(out, L1, L2, *hm, target);
	}

	auto &task_pool = *policy.pool();
	std::vector<std::future<size_t>> futures;
	for (size_t tid = 0; tid < chunks; tid++) {
		futures.emplace_back(task_pool.enqueue([tid, &hm, &L2]() __attribute__((always_inline)) -> size_t {
			const size_t spos = L2.start_pos(tid);
			const size_t epos = L2.end_pos(tid);
			for (size_t i = spos; i < epos; ++i) {
				hm->insert(L2[i].label.value(), i);
			}
			return 0;
		}));
	}

	cryptanalysislib::internal::wait_futures(futures);
	futures.clear();

	for (size_t tid = 0; tid < chunks; tid++) {
		futures.emplace_back(task_pool.enqueue([tid, &hm, &out, &L1, &L2, &target]() __attribute__((always_inline)) {
			LabelType sigma_t;
			LoadType load = 0;
			size_t out_load = 0;
			const size_t out_size = out.size(tid);
			const size_t spos = L1.start_pos(tid);
			const size_t epos = L1.end_pos(tid);
			for (size_t i = spos; i < epos; ++i) {
				LabelType::template sub<k_lower, k_upper>(sigma_t, target, L1[i].label);

				size_t s = hm->find(sigma_t.value(), load);
				for (size_t k = s; k < s + load; ++k) {
					const size_t j = hm->ptr(k);

					ElementType::template sub<k_lower, k_upper, -1u>(out[out_load], L1[i], L2[j]);

					out_load += 1;
					if (out_load == out_size) { goto finish; }
				}
			}
		finish:
			return out_load;
		}));
	}

	return std::reduce(
		cryptanalysislib::internal::get_wrap(futures.begin()),
		cryptanalysislib::internal::get_wrap(futures.end()), (size_t)0, std::plus<size_t>());
}
#endif
