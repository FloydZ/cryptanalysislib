#ifndef CRYPTANALYSISLIB_TREE_DISSECTION_H
#define CRYPTANALYSISLIB_TREE_DISSECTION_H

	// SRC: https://eprint.iacr.org/2010/189.pdf
	// implementation of the modular 4-way merge
	/// same as `constexpr_dissection4`, but with runtime limits, and the
	/// 	base lists are given by the caller.
	/// NOTE: allocates the intermediate list
	/// \param out output list, elements with label = target on [0, k_upper2)
	/// \param L1 base list, const
	/// \param L2 base list, will be sorted on [k_lower1, k_upper1)
	/// \param L3 base list, const
	/// \param L4 base list, will be sorted on [k_lower1, k_upper1)
	/// \param target
	/// \param k_lower1 must be 0
	/// \param k_upper1 the intermediate targets are all values in [0, 2**k_upper1)
	/// \param k_upper2 upper coordinate to match on, on the last level
	static void dissection4(List &out,
	                        const List &L1, List &L2,
	                        const List &L3, List &L4,
	                        const LabelType &target,
	                        const uint32_t k_lower1,
	                        const uint32_t k_upper1,
	                        const uint32_t k_upper2) noexcept {
		assert(k_lower1 == 0);
		assert(k_upper1 < k_upper2);

		// reset the output list
		out.set_load(0);

		constexpr static uint32_t filter = -1u;
		constexpr static double factor = 1.5;
		// number of intermediate targets sigma_M in [0, 2**k_upper1)
		const size_t size = 1ull << k_upper1;
		List iL{(size_t) ((double) size * factor)};

		L2.sort_level(k_lower1, k_upper1);
		L4.sort_level(k_lower1, k_upper1);

		ElementType tmpe1;
		LabelType sigma_M, sigma_t, tprime;
		for (size_t sigma_M_ = 0; sigma_M_ < size; ++sigma_M_) {
			iL.set_load(0);
			sigma_M.set(sigma_M_, 0);
			for (size_t i = 0; i < L1.load(); ++i) {
				LabelType::sub(sigma_t, sigma_M, L1[i].label, 0, k_upper1);
				size_t j = L2.search_level(sigma_t, 0, k_upper1);
				for (; (j < L2.load()) &&
				       (sigma_t.is_equal(L2[j].label, 0, k_upper1)); ++j) {
					iL.add_and_append(L1[i], L2[j], 0, k_upper2, filter);
				}
			}

			if (iL.load() == 0) {
				continue;
			}

			iL.sort_level(0, k_upper2);
			for (size_t k = 0; k < L3.load(); ++k) {
				LabelType::sub(sigma_t, target, sigma_M);
				LabelType::sub(sigma_t, sigma_t, L3[k].label);
				size_t l = L4.search_level(sigma_t, 0, k_upper1);
				for (; (l < L4.load()) &&
				       (sigma_t.is_equal(L4[l].label, 0, k_upper1)); ++l) {
					LabelType::sub(tprime, target, L3[k].label);
					LabelType::sub(tprime, tprime, L4[l].label);
					size_t o = iL.search_level(tprime, 0, k_upper2);
					for (; (o < iL.load()) &&
					       (tprime.is_equal(iL[o].label, 0, k_upper2)); ++o) {
						tmpe1 = iL[o];
						ElementType::add(tmpe1, tmpe1, L3[k]);
						out.add_and_append(tmpe1, L4[l], 0, k_upper2, filter);
					}
				}
			}
		}
	}

	/// enumerates the four base lists (weight k_upper1/2 on each quarter of
	/// 	[0, 2*k_upper1) and calls `dissection4`.
	/// \tparam k_lower1 must be 0
	/// \tparam k_upper1 the intermediate targets are all values in [0, 2**k_upper1)
	/// \tparam k_upper2 upper coordinate to match on, on the last level
	/// \param out output list, elements with label = target on [0, k_upper2)
	/// \param target
	/// \param MT matrix
	template<
			 const uint32_t k_lower1=0,
	         const uint32_t k_upper1=ValueLENGTH/4u,
			 const uint32_t k_upper2=ValueLENGTH>
	static void constexpr_dissection4(List &out,
									  const LabelType &target,
									  const MatrixType &MT) noexcept {
		constexpr size_t baselist_size = sum_bc(k_upper1, k_upper1/2);
		static List L1{baselist_size}, L2{baselist_size}, L3{baselist_size}, L4{baselist_size};
		L1.set_load(0); L2.set_load(0); L3.set_load(0); L4.set_load(0);

		// enumerate the base lists
		using Enumerator = BinaryLexicographicEnumerator<List, k_upper1, k_upper1/2>;
		Enumerator e{MT};
		e.run(&L1, &L2, k_upper1);
		e.run(&L3, &L4, k_upper1, k_upper2/2);

		dissection4(out, L1, L2, L3, L4, target, k_lower1, k_upper1, k_upper2);
	}

	/// same as `constexpr_dissection4`, but `L2`, `L4` and the intermediate
	/// 	list are hashmaps instead of sorted lists.
	/// NOTE: the labels must be hashable (`LabelType::DataType`, `value()`)
	/// NOTE: allocates the hashmaps
	/// \tparam k_lower1 must be 0
	/// \tparam k_upper1 the intermediate targets are all values in [0, 2**k_upper1)
	/// \tparam k_upper2 upper coordinate to match on, on the last level
	/// \tparam bucketsize_base bucketsize of the hashmaps of `L2` and `L4`
	/// \tparam bucketsize_iL bucketsize of the intermediate hashmap
	/// \param out output list, elements with label = target on [0, k_upper2)
	/// \param target
	/// \param MT matrix
	template<
			 const uint32_t k_lower1=0,
	         const uint32_t k_upper1=ValueLENGTH/4u,
			 const uint32_t k_upper2=ValueLENGTH,
			 const uint32_t bucketsize_base=16,
			 const uint32_t bucketsize_iL=16>
	static void constexpr_dissection4_hashmap(List &out,
	                                          const LabelType &target,
	                                          const MatrixType &MT) noexcept {
		static_assert(k_lower1 == 0);
		static_assert(k_upper1 < k_upper2);

		// reset the output list
		out.set_load(0);

		constexpr static uint32_t filter = -1u;
		// number of intermediate targets sigma_M in [0, 2**k_upper1)
		constexpr static size_t size = 1ull << k_upper1;

		constexpr size_t baselist_size = sum_bc(k_upper1, k_upper1/2);
		static List L1{baselist_size}, L2{baselist_size}, L3{baselist_size}, L4{baselist_size};
		L1.set_load(0); L2.set_load(0); L3.set_load(0); L4.set_load(0);

		// enumerate the base lists
		using Enumerator = BinaryLexicographicEnumerator<List, k_upper1, k_upper1/2>;
		Enumerator e{MT};
		e.run(&L1, &L2, k_upper1);
		e.run(&L3, &L4, k_upper1, k_upper2/2);

		using D = typename LabelType::DataType;
		using E = std::pair<size_t, size_t>;
		constexpr static SimpleHashMapConfig configBase {
				bucketsize_base, 1ull << (k_upper1 - k_lower1), 1
		};
		constexpr static SimpleHashMapConfig configiL {
				bucketsize_iL, 1ull << (k_upper2 - k_upper1), 1
		};
		using HMBase = SimpleHashMap<D, size_t, configBase, Hash<D, k_lower1, k_upper1, 2>>;
		using HMiL   = SimpleHashMap<D, E, configiL, Hash<D, k_upper1, k_upper2, 2>>;
		using LoadBase = typename HMBase::load_type;
		using LoadiL = typename HMiL::load_type;
		HMBase *hm2 = new HMBase{};
		HMBase *hm4 = new HMBase{};
		HMiL *hmiL = new HMiL{};
		for (size_t i = 0; i < L2.load(); ++i) { hm2->insert(L2[i].label.value(), i); }
		for (size_t i = 0; i < L4.load(); ++i) { hm4->insert(L4[i].label.value(), i); }

		ElementType tmpe1;
		LabelType sigma_M, sigma_t, tprime, e1;
		LoadBase load = 0;
		LoadiL loadiL = 0;
		for (size_t sigma_M_ = 0; sigma_M_ < size; ++sigma_M_) {
			hmiL->clear();
			sigma_M.set(sigma_M_, 0);
			size_t iLs = 0;
			for (size_t i = 0; i < L1.load(); ++i) {
				LabelType::sub(sigma_t, sigma_M, L1[i].label);
				const size_t s = hm2->find(sigma_t.value(), load);
				for (size_t j = s; j < s + load; ++j) {
					const size_t b = (*hm2)[j];
					LabelType::add(e1, L1[i].label, L2[b].label);
					hmiL->insert(e1.value(), E{i, b});
					iLs += 1;
				}
			}

			if (iLs == 0) {
				continue;
			}

			for (size_t k = 0; k < L3.load(); ++k) {
				LabelType::sub(sigma_t, target, sigma_M);
				LabelType::sub(sigma_t, sigma_t, L3[k].label);
				const size_t s = hm4->find(sigma_t.value(), load);
				for (size_t l = s; l < s + load; ++l) {
					const size_t b = (*hm4)[l];
					LabelType::sub(tprime, target, L3[k].label);
					LabelType::sub(tprime, tprime, L4[b].label);
					const size_t s2 = hmiL->find(tprime.value(), loadiL);
					for (size_t o = s2; o < s2 + loadiL; ++o) {
						const E ij = (*hmiL)[o];
						ElementType::add(tmpe1, L1[ij.first], L2[ij.second]);
						ElementType::add(tmpe1, tmpe1, L3[k]);
						out.add_and_append(tmpe1, L4[b], 0, k_upper2, filter);
					}
				}
			}
		}

		delete hm2;
		delete hm4;
		delete hmiL;
	}
#endif//CRYPTANALYSISLIB_DISSECTION_H
