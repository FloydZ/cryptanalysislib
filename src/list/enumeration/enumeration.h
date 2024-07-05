#ifndef CRYPTANALYSISLIB_LIST_ENUMERATION_H
#define CRYPTANALYSISLIB_LIST_ENUMERATION_H

#include <cstddef>   // needed for std::nullptr_t
#include <functional>// needed for std::invoke

#include "container/hashmap/common.h"
#include "helper.h"
#include "list/common.h"
#include "list/enumeration/enumeration.h"

#include "combination/chase.h"

/// list enumeration type
enum ListIteration {
	SingleFullLength,       // Only one symbol on full length
	MultiFullLength,        // Multiple Symbols on the full length
	SinglePartialSingle,    // [xxxx0000|xx000000], [0000xxxx|00xx0000], [xxxx0000|0000xx00], [0000xxxx|000000xx]
	EnumSinglePartialSingle,// [xxxx0000|xx000000], [0000xxxx|00xx0000], [xxxx0000|0000xx00], [0000xxxx|000000xx] for all p=1, .., nr1
	MultiDisjointBlock,     // [xyxyxyxy|xy000000], [xyxyxyxy|00xy0000], [xyxyxyxy|0000xy00], [xyxyxyxy|000000xy]
	MITMSingle,             // [xxxxxxxx|00000000] and [xxxxxxxx|00000000]
	MITMMulti,              // [xyxyxyxy|00000000] and [00000000|xyxyxyxy]
	MITMEnumSingle,         // same as MITMSingle, but enumerating every weight
};


/// \tparam List
template<class List>
concept ListEnumeration_ListAble = requires(List c) {
	/// should return the List
	c.size()->size_t;
	c.clear();
};

///
template<class List>
concept ListEnumeration_ChangeListAble = requires(List c) {
	/// should return the ChangeList
	c.size()->size_t;
	c.clear();
};

template<typename T>
concept ListEnumerator = requires(T a) {
	typename T::Element;
	typename T::Matrix;
	typename T::Value;
	typename T::Label;
};

/// \tparam ListType BaseList Type
/// \tparam n length to enumerate
/// \tparam q field size. q-1 is the max value to enumerate
/// \tparam w weight to enumerate
/// 		if w == 2: only a chase sequence will enumerated
template<class ListType,
         const uint32_t n,
         const uint32_t q,
         const uint32_t w>
    requires ListAble<ListType>
class ListEnumeration_Meta {
public:
	/// needed typedef
	typedef ListType List;
	typedef typename ListType::ElementType Element;
	typedef typename ListType::ValueType Value;
	typedef typename ListType::LabelType Label;
	typedef typename ListType::MatrixType Matrix;

	// the transposed matrix
	const Matrix &HT;

	/// can be optionally passed:
	// the syndrome, if given, will be added into the sequence at the beginning
	const Label *syndrome = nullptr;

	/// NOTE: these elements must be public available as we need them to
	/// recover the full solution.
	Element element1, element2;

	/// needed for reconstruction
	constexpr Element &get_first() noexcept { return element1; }
	constexpr Element &get_second() noexcept { return element2; }

	/// checks for the correctness of the computed label.
	/// e.g. it checks it l == HT*e
	/// \param l computed label
	/// \param e error vector resulting in the label
	/// \return true/false
	bool check(const Label &label, const Value &error, bool add_syndrome = true) noexcept {
#ifdef DEBUG
		/// TEST for correctness
		auto H = HT.transpose();
		Label tmpl;

		H.mul(tmpl, error);
		if (syndrome != nullptr && add_syndrome) {
			Label::add(tmpl, tmpl, *syndrome);
		}

		if (!tmpl.is_equal(label)) {
			std::cout << std::endl
			          << "ERROR: (SHOULD, IS)" << std::endl;
			tmpl.print();
			label.print();
			std::cout << std::endl;
			error.print();
			std::cout << std::endl;
			HT.print();
		}

		ASSERT(tmpl.is_equal(label));

		uint32_t tmp_vec_ctr = error.popcnt();
		if (tmp_vec_ctr != w) {
			error.print();
		}

		ASSERT(tmp_vec_ctr == w);
#endif
		return true;
	}


	/// abstract insertion handler for list
	/// \param L base list to insert to
	/// \param element element to insert
	/// \param ctr position to insert it (relative to tid)
	/// \param tid thread inline
	constexpr inline void insert_list(ListType *L,
	                                  const Element &element,
	                                  const size_t ctr,
	                                  const uint32_t tid = 0) {
		L->insert(element, ctr, tid);
	}

	/// abstract insertion handler fo hashmap
	/// \tparam HashMap
	/// \tparam Extractor
	/// \param hm
	/// \param e
	template<class HashMap, typename Extractor>
#if __cplusplus > 201709L
	    requires(std::is_same_v<std::nullptr_t, HashMap> || HashMapAble<HashMap>) &&
	            (std::is_same_v<std::nullptr_t, Extractor> || std::is_invocable_v<Extractor, Label>)
#endif
	void insert_hashmap(HashMap *hm,
	                    Extractor *e,
	                    const Element &element,
	                    const size_t ctr,
	                    const uint32_t tid = 0) {
		if constexpr (!std::is_same_v<std::nullptr_t, HashMap>) {
			// nothing to inset
			if (hm == nullptr) {
				return;
			}

			using IndexType = typename HashMap::data_type;
			IndexType npos;
			npos[0] = ctr;

			const auto data = std::invoke(*e, element.label);
			hm->insert(data, npos, tid);
		}
	}

	///
	/// \param HT
	/// \param list_size
	/// \param syndrome
	constexpr ListEnumeration_Meta(const Matrix &HT,
	                               const Label *syndrome = nullptr)
	    : HT(HT), syndrome(syndrome) {
		/// some sanity checks
		/// NOTE: its allowed to call this class with `w=0`, which is needed for Prange
		static_assert(n > w);
		static_assert(q > 1);
		static_assert(n <= Value::length());
	}
};

#include "list/enumeration/binary.h"
#include "list/enumeration/fq.h"

#endif//CRYPTANALYSISLIB_LIST_ENUMERATION_H
