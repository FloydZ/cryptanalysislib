#ifndef CRYPTANALYSISLIB_FQ_ENUMERATION_H
#define CRYPTANALYSISLIB_FQ_ENUMERATION_H

#ifndef CRYPTANALYSISLIB_LIST_ENUMERATION_H
#error "dont use include <list/enumeration/fq.h>, use include <list/enumeration/enumeration.h> instead"
#endif


#include "list/enumeration/enumeration.h"
#include "combination/chase.h"


/// This class enumerates each element of length `n`.
/// Enumerating means:
/// 	a chase sequence is enumerated to select `w` out of `n` positions
/// 	on those `w` positions a grey code is used to enumerate all q-1 symbols
/// \tparam ListType
/// \tparam n length to enumerate
/// \tparam q field size, e.g. enumeration symbols = {0, ..., q-1}
/// \tparam w weight to enumerate
template<class ListType,
		 const uint32_t n,
		 const uint32_t q,
		 const uint32_t w>
class ListEnumerateMultiFullLength: public ListEnumeration_Meta<ListType, n, q, w> {
public:
	/// needed typedefs
	typedef typename ListEnumeration_Meta<ListType, n, q, w>::Element Element;
	typedef typename ListEnumeration_Meta<ListType, n, q, w>::Matrix Matrix;
	typedef typename ListEnumeration_Meta<ListType, n, q, w>::Value Value;
	typedef typename ListEnumeration_Meta<ListType, n, q, w>::Label Label;

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

	/// needed variables
	// this generates a grey code sequence, e.g. a sequence in which
	// two consecutive elements only differ in a single position
	Combinations_Fq_Chase<n, q, w> chase = Combinations_Fq_Chase<n, q, w>{};
	constexpr static size_t chase_size = Combinations_Fq_Chase<n, q, w>::chase_size;
	constexpr static size_t gray_size  = Combinations_Fq_Chase<n, q, w>::gray_size;

	/// this can be used to specify the size of the input list
	/// e.g. its the maximum number of elements this class enumerates
	constexpr static size_t LIST_SIZE  = Combinations_Fq_Chase<n, q, w>::LIST_SIZE;

	// this can be se to something else as `LIST_SIZE`, if one wants to only
	// enumerate a part of the sequence
	const size_t list_size = 0;

	// change list for the chase sequence
	std::vector<std::pair<uint16_t, uint16_t>> chase_cl = std::vector<std::pair<uint16_t, uint16_t>>(chase_size);
	// change list for the gray code sequence
	std::vector<uint16_t> gray_cl = std::vector<uint16_t>(gray_size);

	/// empty constructor
	/// \param HT transposed parity check matrix
	/// \param list_size max numbers of elements to enumerate.
	/// 			if set to 0: the complete sequence will be enumerated.
	/// \param syndrome additional element which is added to all list elements
	ListEnumerateMultiFullLength(const Matrix &HT,
	                             const size_t list_size=0,
	                             const Label *syndrome= nullptr) :
	    		ListEnumeration_Meta<ListType, n, q, w>(HT, syndrome),
	            list_size((list_size==size_t(0)) ? LIST_SIZE : list_size) {

		static_assert(chase_size >= 0);
		static_assert(gray_size > 0);
		ASSERT(LIST_SIZE >= list_size);

		if constexpr (q>2) chase.changelist_mixed_radix_grey(gray_cl.data());
		chase.template changelist_chase<false>(chase_cl.data());
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
	requires (std::is_same_v<std::nullptr_t, HashMap> || HashMapAble<HashMap>) &&
	         (std::is_same_v<std::nullptr_t, Extractor> || std::is_invocable_v<Extractor, Label>) &&
			 (std::is_same_v<std::nullptr_t, Predicate> || std::is_invocable_v<Predicate, Label>)
#endif
	bool run(ListType *L1=nullptr,
	         ListType *L2=nullptr,
	         const uint32_t offset=0,
	         const uint32_t tid=0,
	         HashMap *hm=nullptr,
	         Extractor *e=nullptr,
	         Predicate *p=nullptr) {
		/// some security checks
		ASSERT(n+offset <= Value::LENGTH);

		/// counter of how many elements already added to the list
		size_t ctr = 0;

		// check if the lists are enabled
		const bool sL1 = L1 != nullptr;
		const bool sL2 = L2 != nullptr;
		constexpr bool sHM = !std::is_same_v<std::nullptr_t, HashMap>;
		constexpr bool sP = !std::is_same_v<std::nullptr_t, Predicate>;

		/// clear stuff, needed if this functions is called multiple times
		/// e.g. in every ISD algorithm
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
				element2.value.set(1, i+offset);
				Label::add(element2.label, element2.label, HT.get(i+offset));
			}
		}

		/// to keep track of the set positions of the grey code.
		std::vector<uint32_t> current_set(w, 0);
		for (uint32_t i = 0; i < w; ++i) {
			current_set[i] = i;
		}

		// helper lambdas
		auto gray_step = [&current_set, this](Element &element, const size_t j, const uint32_t off) {
		  const uint32_t cs = current_set[gray_cl[j]];
		  element.value.set((element.value[cs + off] + 1) % q, cs + off);
		  Label::add(element.label, element.label, HT.get(cs + off));

		  /// NOTE: this is stupid, but needed. The gray code enumeration
		  /// also enumerates zeros. Therefore we need to fix them
		  if (element.value[cs + off] == 0) {
			  element.value.set(1, cs + off);
			  Label::add(element.label, element.label, HT.get(cs + off));
		  }
		};

		auto chase_step = [this](Element &element,
				const uint32_t a,
				const uint32_t b,
				const uint32_t off) {
		  /// make really sure that the the chase
		  /// sequence is correct.
		  ASSERT(element.value[a + off]);
		  ASSERT(std::abs((int)a - (int)b) <= (int)w);

		  Label tmp;
		  Label::scalar(tmp, HT.get(a + off), (q-element.value[a + off]) % q);
		  Label::add(element.label, element.label, tmp);
		  Label::add(element.label, element.label, HT.get(b + off));
		  element.value.set(0, off + a);
		  element.value.set(1, off + b);
		};

		/// iterate over all sequences
		for (uint32_t i = 0; i < chase_size; ++i) {
			for (uint32_t j = 0; j < gray_size - 1; ++j) {
				check(element1.label, element1.value);
				if (sL2) check(element2.label, element2.value, false);

				if constexpr (sP) { if(std::invoke(*p, element1.label)) { return true; }}
				if constexpr (sHM) insert_hashmap(hm, e, element1, ctr, tid);
				if (sL1) insert_list(L1, element1, ctr, tid);
				if (sL2) insert_list(L2, element2, ctr, tid);

				ctr += 1;
				if (ctr >= list_size) {
					return false;
				}

				gray_step(element1, j, 0);
				if (sL2) gray_step(element2, j, offset);
			}

			check(element1.label, element1.value);
			if (sL2) check(element2.label, element2.value, false);

			if constexpr (sP) { if(std::invoke(*p, element1.label)) { return true; }}
			if constexpr (sHM) insert_hashmap(hm, e, element1, ctr, tid);
			if (sL1) insert_list(L1, element1, ctr, tid);
			if (sL2) insert_list(L2, element2, ctr, tid);

			ctr += 1;
			if (ctr >= list_size) {
				return false;
			}

			/// advance the current set by one
			const uint32_t a = chase_cl[i].first;
			const uint32_t b = chase_cl[i].second;
			for (uint32_t j = 0; j < w; ++j) {
				if (current_set[j] == a) {current_set[j] = b;}
			}

			chase_step(element1, a, b, 0);
			if(sL2) chase_step(element2, a, b, offset);
		}

		/// make sure that all elements where generated
		ASSERT(ctr == LIST_SIZE);
		return false;
	}
};

#endif//CRYPTANALYSISLIB_FQ_ENUMERATION_H
