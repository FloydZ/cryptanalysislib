#ifndef CRYPTANALYSISLIB_FQ_LIST_ENUMERATION_H
#define CRYPTANALYSISLIB_FQ_LIST_ENUMERATION_H

#include <cstdint>
#include <cstddef>

#include "helper.h"
#include "combination/fq/chase.h"

#if __cplusplus > 201709L
///
/// \tparam Container
template<class Container>
concept ValueListEnumerationFqAble =  requires(Container c) {
	//typename Container::ElementType;

	requires requires(const unsigned int i) {
		c[i];
		c.random();
		c.zero();
		c.one();
		c.clear();

		/// TODO erkennt es nihct
		//Container::add(c, c);
		//Container::sub(c, c);
		//Container::add(c, c, c);
		//Container::sub(c, c, c);
		//Container::scalar(c, c, i) -> c;
		//Container::scalar(c, i);
		//Container::set(i, i);
		//Container::set(i);
	};
};
#endif

///
/// \tparam ListType
/// \tparam LabelType
/// \tparam ValueType
/// \tparam n
/// \tparam q
/// \tparam w weight to enumerate, this is *NOT* the w of the challenge
template<class ListType,
         class LabelType,
         class ValueType,
         const uint32_t n,
         const uint32_t q,
         const uint32_t w>
#if __cplusplus > 201709L
	requires ValueListEnumerationFqAble<ValueType>
#endif
class ListEnumerationFqMultiFullLength {
public:
	/// needed type definitions
	using List = ListType;
	using ElementType = typename ListType::ElementType;

	/// needed variables
	Combinations_Fq_Chase c = Combinations_Fq_Chase(n, q, w);
	const size_t chase_size = c.chase_size;
	const size_t gray_size = c.gray_size;
	std::vector<std::pair<uint16_t, uint16_t>> chase_cl = std::vector<std::pair<uint16_t, uint16_t>>(chase_size);
	uint16_t *gray_cl = (uint16_t *)malloc(chase_size * sizeof(uint16_t));

	/// super important this is needed to recover the solution
	std::vector<uint8_t> tmp_vec;
	LabelType vec;

	/// enable the constexpt constructor
	constexpr ListEnumerationFqMultiFullLength() : tmp_vec(n, 0) {
		static_assert(n && q);

		/// this is needed, as prange can be called with w=0
		if constexpr (w == 0) {
			return;
		}
		c.changelist_mixed_radix_grey(gray_cl);
		c.changelist_chase(chase_cl.data());
	}

	constexpr ~ListEnumerationFqMultiFullLength() {
		free(gray_cl);
	}

	///
	/// \tparam MatrixT
	/// \param HT
	/// \param syndrome
	/// \param size_limit
	/// \param weight_limit
	/// \return
	template<class MatrixT>
	bool run(const MatrixT &HT,
	         const LabelType &syndrome,
	         const size_t size_limit,
	         const uint32_t weight_limit) noexcept {
		vec = syndrome;
		LabelType tmp;

		for (uint32_t i = 0; i < n; ++i) { tmp_vec[i] = 0; }

		/// compute the first element
		for (uint32_t i = 0; i < w; ++i) {
			tmp_vec[i] = 1u;
			LabelType::add(vec, vec, HT.get(i));
		}

		std::vector<uint32_t> current_set(w, 0);
		for (uint32_t i = 0; i < w; ++i) {
			current_set[i] = i;
		}

		size_t ctr = 0;

		/// iterate over all
		for (uint32_t i = 0; i < chase_size; ++i) {
			for (uint32_t j = 0; j < gray_size-1; ++j) {
				const uint32_t weight = vec.weight();
				if (weight <= weight_limit) {
					return true;
				}

				ctr += 1;
				if (ctr >= size_limit) {
					return false;
				}

				const uint32_t cs = current_set[gray_cl[j]];
				tmp_vec[cs] = (tmp_vec[cs] + 1u) % q;
				LabelType::add(vec, vec, HT.get(cs));

				/// NOTE: this is stupid, but needed. The gray code enumeration
				/// also enumerates zeros. Therefore we need to fix them
				if (tmp_vec[cs] == 0) {
					tmp_vec[cs] += 1;
					LabelType::add(vec, vec, HT.get(cs));
				}
			}

			const uint32_t weight = vec.weight();
			if (weight <= weight_limit) {
				return true;
			}

			ctr += 1;
			if (ctr >= size_limit) {
				return false;
			}

			/// advance the current set by one
			const uint32_t j = chase_cl[i].first;
			ASSERT(j < w);
			const uint32_t a = current_set[j];
			const uint32_t b = chase_cl[i].second;
			current_set[j] = b;

			LabelType::scalar(tmp, HT.get(a), q-tmp_vec[a]);
			LabelType::add(vec, vec, tmp);
			LabelType::add(vec, vec, HT.get(b));
			tmp_vec[a] = 0;
			tmp_vec[b] = 1;
		}

		return false;
	}

	/// this version is special made for fq prange
	/// q-1 Symbols on the full length
	/// the reason this function takes the list is, that different enumeration strategies
	/// need a different amount of lists
	/// \tparam MatrixT
	/// \tparam n length to enumerate (this is not the code length)
	/// \tparam q field size
	/// \tparam w max hamming weight to enumerate
	/// \param L1 output list
	/// \param HT
	/// \param size_limit
	template<class MatrixT>
	void multiFullLength(List &L1, const MatrixT &HT, const size_t size_limit) noexcept {
		LabelType vec, tmp;
		vec.zero();

		/// compute the first element
		std::vector<uint8_t> tmp_vec(n, 0);
		for (uint32_t i = 0; i < w; ++i) {
			tmp_vec[i] = 1u;
			LabelType::add(vec, vec, HT.get(i));
		}

		std::vector<uint32_t> current_set(w, 0);
		for (uint32_t i = 0; i < w; ++i) {
			current_set[i] = i;
		}

		auto check = [&]() {
#ifdef DEBUG
		  /// TEST for correctness
		  auto H = HT.transpose();
		  LabelType tmpl;
		  ValueType tmpv;
		  tmpv.clear();
		  for (uint32_t l = 0; l < n; ++l) {
			  tmpv[l] = tmp_vec[l];
		  }
		  H.matrix_row_vector_mul2(tmpl, tmpv);

		  if (!tmpl.is_equal(vec)) {
			  tmpl.print();
			  vec.print();
			  std::cout <<std::endl;
			  HT.print();
		  }

		  ASSERT(tmpl.is_equal(vec));

		  uint32_t tmp_vec_ctr = 0;
		  for (uint32_t l = 0; l < n; ++l) {
			  tmp_vec_ctr += (tmp_vec[l] > 0);
		  }
		  ASSERT(tmp_vec_ctr == w);
#endif
		};

		auto print_info = [&]() {
			for (uint32_t i = 0; i < n; i++) {
				printf("%d", tmp_vec[i]);
			}
			printf("\n");
		};

		size_t ctr = 0;

		/// iterate over all
		for (uint32_t i = 0; i < chase_size; ++i) {
			for (uint32_t j = 0; j < gray_size-1; ++j) {
				check();
				L1[ctr++] = vec;
				if (ctr >= size_limit) {
					return;
				}

				const uint32_t cs = current_set[gray_cl[j]];
				tmp_vec[cs] = (tmp_vec[cs] + 1u) % q;
				LabelType::add(vec, vec, HT.get(cs));

				/// NOTE: this is stupid, but needed. The gray code enumeration
				/// also enumerates zeros. Therefore we need to fix them
				if (tmp_vec[cs] == 0) {
					tmp_vec[cs] += 1;
					LabelType::add(vec, vec, HT.get(cs));
				}

			}

			check();
			L1[ctr++] = vec;
			if (ctr >= size_limit) {
				return;
			}

			/// advance the current set by one
			const uint32_t j = chase_cl[i].first;
			ASSERT(j < w);
			const uint32_t a = current_set[j];
			const uint32_t b = chase_cl[i].second;
			current_set[j] = b;

			LabelType::scalar(tmp, HT.get(a), q-tmp_vec[a]);
			LabelType::add(vec, vec, tmp);
			LabelType::add(vec, vec, HT.get(b));
			tmp_vec[a] = 0;
			tmp_vec[b] = 1;
		}
	}


	/// this version is special made for fq siebing
	/// q-1 Symbols on the full length
	/// the reason this function takes the list is, that different enumeration strategies
	/// need a different amount of lists
	/// \tparam MatrixT
	/// \tparam n length to enumerate (this is not the code length)
	/// \tparam q field size
	/// \tparam w max hamming weight to enumerate
	/// \param L1 output list
	/// \param HT
	/// \param size_limit
	template<class MatrixT, typename Element, class HashMap>
	void multiFullLengthSieving(List &L1,
	                            const MatrixT &HT,
	                            HashMap *hm,
	                            const size_t size_limit) noexcept {
		LabelType tmp;
		Element vec;
		vec.zero();

		/// hashmap stuff
		uint32_t tid = 0;
		using IndexType = typename HashMap::IndexType;
		using LPartType = typename HashMap::T;
		IndexType npos[1];

		/// compute the first element
		for (uint32_t i = 0; i < w; ++i) {
			vec.value[i] = 1u;
			LabelType::add(vec.label, vec.label, HT.get(i));
		}


		std::vector<uint32_t> current_set(w, 0);
		for (uint32_t i = 0; i < w; ++i) {
			current_set[i] = i;
		}

		size_t ctr = 0;

		/// iterate over all
		for (uint32_t i = 0; i < chase_size; ++i) {
			for (uint32_t j = 0; j < gray_size-1; ++j) {
				/// TODO ugly
				/// ein weg das ganze zu anbstrahiern ist eine `extractor class`
				/// es gibt dann fuer jede der 3 funktionen in dieser klasse eine eigene API
				const LPartType data = *((LPartType *)(vec.label.data().data()));
				npos[0] = ctr;
				hm->insert(data, npos, tid);
				L1[ctr++] = vec;
				if (ctr >= size_limit) {
					return;
				}

				const uint32_t cs = current_set[gray_cl[j]];
				vec.value[cs] = (tmp_vec[cs] + 1) % q;
				LabelType::add(vec.label, vec.label, HT.get(cs));

				/// NOTE: this is stupid, but needed. The gray code enumeration
				/// also enumerates zeros. Therefore we need to fix them
				if (tmp_vec[cs] == 0) {
					vec.value[cs] += 1;
					LabelType::add(vec.label, vec.label, HT.get(cs));
				}
			}

			const LPartType data = *((LPartType *)(vec.label.data().data()));
			npos[0] = ctr;
			hm->insert(data, npos, tid);
			L1[ctr++] = vec;
			if (ctr >= size_limit) {
				return;
			}

			/// advance the current set by one
			const uint32_t j = chase_cl[i].first;
			ASSERT(j < w);
			const uint32_t a = current_set[j];
			const uint32_t b = chase_cl[i].second;
			current_set[j] = b;

			LabelType::scalar(tmp, HT.get(a), q-tmp_vec[a]);
			LabelType::add(vec.label, vec.label, tmp);
			LabelType::add(vec.label, vec.label, HT.get(b));
			tmp_vec[a] = 0;
			tmp_vec[b] = 1;
		}
	}
};
#endif//CRYPTANALYSISLIB_FQ_LIST_ENUMERATION_H
