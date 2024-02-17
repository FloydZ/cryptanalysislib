#ifndef CRYPTANALYSISLIB_BINARY_ENUMERATION_H
#define CRYPTANALYSISLIB_BINARY_ENUMERATION_H


#ifndef CRYPTANALYSISLIB_LIST_ENUMERATION_H
#error "dont use include <list/enumeration/binary.h>, use include <list/enumeration/enumeration.h> instead"
#endif

#include <cstddef>   // needed for std::nullptr_t
#include <functional>// needed for std::invoke

#include "combination/chase.h"
#include "helper.h"
#include "list/enumeration/enumeration.h"
#include "math/bc.h"

/// This class enumerates vectors of length n and weight w, whereas each
/// nonzero position is enumerated from 1,...,q-1
/// \tparam ListType
/// \tparam n vector length
/// \tparam q field size
/// \tparam w weight
template<class ListType,
         const uint32_t n,
         const uint32_t w>
class BinaryListEnumerateMultiFullLength : public ListEnumeration_Meta<ListType, n, 2, w> {
public:
	/// Im lazy
	constexpr static uint32_t q = 2;

	/// needed typedefs
	typedef typename ListEnumeration_Meta<ListType, n, q, w>::Element Element;
	typedef typename ListEnumeration_Meta<ListType, n, q, w>::Matrix Matrix;
	typedef typename ListEnumeration_Meta<ListType, n, q, w>::Value Value;
	typedef typename ListEnumeration_Meta<ListType, n, q, w>::Label Label;
	typedef typename Label::LimbType T;

	/// needed functions
	using ListEnumeration_Meta<ListType, n, q, w>::check;
	using ListEnumeration_Meta<ListType, n, q, w>::insert_hashmap;
	using ListEnumeration_Meta<ListType, n, q, w>::insert_list;
	using ListEnumeration_Meta<ListType, n, q, w>::get_first;
	using ListEnumeration_Meta<ListType, n, q, w>::get_second;

	/// needed variables
	using ListEnumeration_Meta<ListType, n, q, w>::element1;
	using ListEnumeration_Meta<ListType, n, q, w>::element2;
	using ListEnumeration_Meta<ListType, n, q, w>::syndrome;
	using ListEnumeration_Meta<ListType, n, q, w>::HT;

	////
	Combinations_Binary_Chase<T, n, w> chase = Combinations_Binary_Chase<T, n, w>();
	constexpr static size_t LIST_SIZE = Combinations_Binary_Chase<T, n, w>::chase_size;
	const size_t list_size;
	std::array<std::pair<uint16_t, uint16_t>, bc(n, w)> cL;
	Element e11, e12, e21, e22;

	/// empty constructor
	/// \param HT transposed parity check matrix
	/// \param list_size max numbers of elements to enumerate.
	/// 			if set to 0: the complete sequence will be enumerated.
	/// \param syndrome additional element which is added to all list elements
	BinaryListEnumerateMultiFullLength(const Matrix &HT,
	                                   const size_t list_size = 0,
	                                   const Label *syndrome = nullptr) : ListEnumeration_Meta<ListType, n, q, w>(HT, syndrome),
	                                                                      list_size((list_size == size_t(0)) ? LIST_SIZE : list_size) {
		chase.template changelist<false>(cL.data(), this->list_size);
	}

	///
	/// \tparam HashMap
	/// \tparam Extractor extractor lambda
	/// 		- can be NULL
	/// \tparam Predicate Function. NOTE: can be
	///			- nullptr_t
	/// 		- std::invokable. if this returns true, the function returns
	/// \param L1 first list. NOTE:
	/// 		- the syndrome is only added into the first list
	/// 		- or nullptr_t
	/// \param L2 second list. NOTE:
	/// 		- can be nullptr_t
	/// 		- otherwise it will compute the error with and offset
	/// \param offset
	/// 		- number of position between the MITM strategy
	/// \param tid thread id
	/// \param hm hashmap
	/// \param e extractor
	/// \param p predicate function
	/// \return true/false if the golden element was found or not (only if
	///  		predicate was given)
	template<typename HashMap, typename Extractor, typename Predicate>
#if __cplusplus > 201709L
	    requires(std::is_same_v<std::nullptr_t, HashMap> || HashMapAble<HashMap>) &&
	            (std::is_same_v<std::nullptr_t, Extractor> || std::is_invocable_v<Extractor, Label>) &&
	            (std::is_same_v<std::nullptr_t, Predicate> || std::is_invocable_v<Predicate, Label>)
#endif
	bool run(ListType *L1 = nullptr,
	         ListType *L2 = nullptr,
	         const uint32_t offset = 0,
	         const uint32_t tid = 0,
	         HashMap *hm = nullptr,
	         Extractor *e = nullptr,
	         Predicate *p = nullptr) {
		/// some security checks
		ASSERT(n + offset <= Value::LENGTH);

		/// counter of how many elements already added to the list
		size_t ctr = 0;

		// check if the lists are enabled
		const bool sL1 = L1 != nullptr;
		const bool sL2 = L2 != nullptr;
		constexpr bool sHM = !std::is_same_v<std::nullptr_t, HashMap>;
		constexpr bool sP = !std::is_same_v<std::nullptr_t, Predicate>;

		/// clear stuff, needed if this functions is called multiple times
		element1.zero();
		if (sL2) { element2.zero(); }

		/// add the syndrome, if needed
		if (syndrome != nullptr) {
			element1.label = *syndrome;
		}

		/// compute the first element
		for (uint32_t i = 0; i < w; ++i) {
			/// NOTE we need to compute always this element, even if we
			/// do not save it in a list. Because otherwise we could not
			/// only use the predicate in this function.
			element1.value.set(1, i);
			Label::add(element1.label, element1.label, HT.get(i));

			if (sL2) {
				element2.value.set(1, i + offset);
				Label::add(element2.label, element2.label, HT.get(i + offset));
			}
		}

		auto chase_step = [this](Element &element,
		                         const uint32_t a,
		                         const uint32_t b,
		                         const uint32_t off) {
			/// make really sure that the the chase
			/// sequence is correct.
			ASSERT(element.value[a + off]);

			Label::add(element.label, element.label, HT.get(a + off));
			Label::add(element.label, element.label, HT.get(b + off));
			element.value.set(0, off + a);
			element.value.set(1, off + b);
		};

		/// iterate over all sequences
		for (uint32_t i = 0; i < list_size; ++i) {
			check(element1.label, element1.value);
			if (sL2) check(element2.label, element2.value, false);

			if constexpr (sP) {
				if (std::invoke(*p, element1.label)) { return true; }
			}
			if constexpr (sHM) { insert_hashmap(hm, e, element1, ctr, tid); }
			if (sL1) insert_list(L1, element1, ctr, tid);
			if (sL2) insert_list(L2, element2, ctr, tid);

			ctr += 1;

			/// advance the current set by one
			const uint32_t a = cL[i].first;
			const uint32_t b = cL[i].second;
			chase_step(element1, a, b, 0);
			if (sL2) chase_step(element2, a, b, offset);
		}

		/// make sure that all elements where generated
		ASSERT(ctr == LIST_SIZE);
		return false;
	}
};

#endif//CRYPTANALYSISLIB_BINARY_ENUMERATION_H
