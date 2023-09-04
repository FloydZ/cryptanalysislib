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

template<class ListType, class ValueType>
#if __cplusplus > 201709L
	requires ValueListEnumerationFqAble<ValueType>
#endif
class ListEnumerationFq {
public:
	/// needed type definitions
	using List = ListType;
	using LabelType = typename ListType::ElementType;
	using ElementType = typename ListType::ElementType;

	/// needed variables

	constexpr ListEnumerationFq() {

	}

	/// q-1 Symbols on the full length
	/// the reason this function takes the list is, that different enumeration strategies
	/// need a different amount of lists
	/// \tparam MatrixT
	/// \tparam n
	/// \tparam q
	/// \tparam w
	/// \param L1
	template<class MatrixT, const uint32_t n, const uint32_t q, const uint32_t w>
	static void multiFullLength(List &L1, const MatrixT &HT, const size_t size_limit) noexcept {
		// static_assert(MatrixT::RowType ==)

		/// TODO move all of this into constructor oder so
		auto c = Combinations_Fq_Chase(n, q, w);
		const size_t chase_size = c.chase_size;
		const size_t gray_size = c.gray_size;
		auto chase_cl = std::vector<std::pair<uint16_t, uint16_t>>(chase_size);
		uint16_t *gray_cl = (uint16_t *)malloc(chase_size * sizeof(uint16_t));

		c.changelist_mixed_radix_grey(gray_cl);
		c.changelist_chase(chase_cl.data());
		L1.clear();

		std::vector<uint8_t> tmp_vec(n, 0); // TODO n is not fully correct
		LabelType vec;
		vec.zero();

		for (uint32_t i = 0; i < w; ++i) {
			vec[i] = 1u;
			tmp_vec[i] = 1u;
		}

		std::vector<uint32_t> current_set(w, 0);
		for (uint32_t i = 0; i < w; ++i) {
			current_set[i] = i;
		}

		size_t ctr = 0;
		/// iterate over all
		for (uint32_t i = 0; i < chase_size; ++i) {
			for (uint32_t j = 0; j < gray_size-1; ++j) {
				L1[ctr++] = vec;
				if (ctr >= size_limit) {
					goto finish;
				}

				tmp_vec[current_set[gray_cl[j]]] = (tmp_vec[current_set[gray_cl[j]]] + 1u) % q;
				LabelType::add(vec, vec, HT.get(current_set[gray_cl[j]]));
				if (tmp_vec[current_set[gray_cl[j]]] == 0) {
					tmp_vec[current_set[gray_cl[j]]] += 1;
					LabelType::add(vec, vec, HT.get(current_set[gray_cl[j]]));
				}
			}

			L1[ctr++] = vec;
			if (ctr > size_limit) {
				goto finish;
			}

			/// advance the current set by one
			const uint32_t j = chase_cl[i].first;
			const uint32_t a = current_set[j];
			const uint32_t b = chase_cl[i].second;
			current_set[j] = b;

			LabelType::scalar(vec, HT.get(a), q-vec[a]);
			LabelType::add(vec, vec, HT.get(b));
		}

		finish:
		free(gray_cl);
	}
};
#endif//CRYPTANALYSISLIB_FQ_LIST_ENUMERATION_H
