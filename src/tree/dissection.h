#ifndef CRYPTANALYSISLIB_TREE_DISSECTION_H
#define CRYPTANALYSISLIB_TREE_DISSECTION_H

	// TODO hashmap version, and non constexpr version
	// SRC: https://eprint.iacr.org/2010/189.pdf
	// implementation of the modular 4-way merge
	template<
			 const uint32_t k_lower1=0,
	         const uint32_t k_upper1=ValueLENGTH/4u,
			 const uint32_t k_upper2=ValueLENGTH>
	static void constexpr_dissection4(List &out,
									  const LabelType &target,
									  const MatrixType &MT) noexcept {
		// reset the output list
		out.set_load(0);

		constexpr static uint32_t filter = -1u;
		constexpr static double factor = 1.5;
		constexpr static size_t size = (1ull << k_upper1) - 1ull;

		constexpr size_t baselist_size = sum_bc(k_upper1, k_upper1/2);
		//constexpr size_t baselist_size = sum_bc(n/2, n/4);
		static List L1{baselist_size}, L2{baselist_size}, L3{baselist_size}, L4{baselist_size}, iL{(size_t) ((double) size * factor)};
		L1.set_load(0); L2.set_load(0); L3.set_load(0); L4.set_load(0);

		// enumerate the base lists
		// using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
		// Enumerator e{MT};
		// e.run(&L1, &L2, n/2);
		// e.run(&L3, &L4, n/2);
		using Enumerator = BinaryLexicographicEnumerator<List, k_upper1, k_upper1/2>;
		Enumerator e{MT};
		e.run(&L1, &L2, k_upper1);
		e.run(&L3, &L4, k_upper1, k_upper2/2);

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
#endif//CRYPTANALYSISLIB_DISSECTION_H
