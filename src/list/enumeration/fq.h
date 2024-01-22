#ifndef CRYPTANALYSISLIB_FQ_ENUMERATION_H
#define CRYPTANALYSISLIB_FQ_ENUMERATION_H

#ifndef CRYPTANALYSISLIB_LIST_ENUMERATION_H
#error "dont use include <list/enumeration/fq.h>, use include <list/enumeration/enumeration.h> instead"
#endif


#include "list/enumeration/enumeration.h"
#include "combination/chase.h"

/// only a single element is enumerated on the full length
/// \tparam ListType
/// \tparam n
/// \tparam q
/// \tparam w
template<class ListType,
		const uint32_t n,
		const uint32_t q,
		const uint32_t w>
class ListEnumerateSingleFullLength: public ListEnumeration_Meta<ListType, n, q, w> {
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

	using T = Value::ContainerLimbType;

	/// this can be used to specify the size of the input list
	/// e.g. its the maximum number of elements this class enumerates
	constexpr static size_t LIST_SIZE = Combinations_Binary_Chase<T, n, w>::chase_size;

	/// needed variables
	// this generates a grey code sequence, e.g. a sequence in which
	// two consecutive elements only differ in a single position
	Combinations_Binary_Chase<T, n, w> chase = Combinations_Binary_Chase<T, n, w>{};
	std::vector<std::pair<uint16_t, uint16_t>> chase_cl = std::vector<std::pair<uint16_t, uint16_t>>(LIST_SIZE);

	const uint32_t q_prime = 0;
	const size_t list_size = 0;

	/// \param q_prime element to enumerate
	/// \param HT transposed parity check matrix
	/// \param list_size max numbers of elements to enumerate.
	/// 			if set to 0: the complete sequence will be enumerated.
	/// \param syndrome additional element which is added to all list elements
	constexpr ListEnumerateSingleFullLength(const uint32_t q_prime,
	         					 			const Matrix &HT,
								 			const size_t list_size=0,
								 			const Label *syndrome= nullptr) :
			ListEnumeration_Meta<ListType, n, q, w>(HT, syndrome),
			q_prime(q_prime),
			list_size((list_size==size_t(0)) ? LIST_SIZE : list_size) {

		ASSERT(LIST_SIZE >= list_size);
		ASSERT(q_prime > 0);
		ASSERT(q_prime < q);
		chase.template changelist<false>(chase_cl.data(), this->LIST_SIZE);
	}

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
			element1.value.set(q_prime, i);

			Label tmp;
			Label::scalar(tmp, HT.get(i), q_prime);
			Label::add(element1.label, element1.label, tmp);

			if (sL2) {
				element2.value.set(q_prime, i+offset);
				Label::scalar(tmp, HT.get(i+offset), q_prime);
				Label::add(element2.label, element2.label, tmp);
			}
		}

		auto chase_step = [this](Element &element,
								 const uint32_t a,
								 const uint32_t b,
								 const uint32_t off) {
		  /// make really sure that the the chase
		  /// sequence is correct.
		  ASSERT(element.value[a + off]);
		  ASSERT(std::abs((int)a - (int)b) <= (int)w);

		  Label tmp;
		  Label::scalar(tmp, HT.get(a + off), q - q_prime);
		  Label::add(element.label, element.label, tmp);
		  Label::scalar(tmp, HT.get(b + off), q_prime);
		  Label::add(element.label, element.label, tmp);
		  element.value.set(0, off + a);
		  element.value.set(q_prime, off + b);
		};

		/// iterate over all sequences
		for (uint32_t i = 0; i < list_size; ++i) {
			check(element1.label, element1.value);
			if (sL2) check(element2.label, element2.value, false);

			if constexpr (sP) { if(std::invoke(*p, element1.label)) { return true; }}
			if constexpr (sHM) insert_hashmap(hm, e, element1, ctr, tid);
			if (sL1) insert_list(L1, element1, ctr, tid);
			if (sL2) insert_list(L2, element2, ctr, tid);

			ctr += 1;

			/// advance the current set by one
			const uint32_t a = chase_cl[i].first;
			const uint32_t b = chase_cl[i].second;

			if (i < list_size-1) {
				chase_step(element1, a, b, 0);
				if (sL2) chase_step(element2, a, b, offset);
			}
		}

		/// make sure that all elements where generated
		ASSERT(ctr == LIST_SIZE);
		return false;
	}
};

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

		/// TODO think of something nicer
		if constexpr (w > 0) {
			if constexpr (q > 2) chase.changelist_mixed_radix_grey(gray_cl.data());
			chase.template changelist_chase<false>(chase_cl.data());
		}
	}

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
			for (uint32_t j = 0; j < gray_size; ++j) {
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


/// this class enumerates elements of the form:
///  [xxxx0000|xx000000], [0000xxxx|00xx0000], [xxxx0000|0000xx00], [0000xxxx|000000xx]
/// In other-words you must pass 4 base lists to the function.
/// NOTE: nomenclature
/// 	<-       n       ->
/// 	[xxxx0000|xx000000]
///     0      split      n
///      mitmlen  norepslen
/// \tparam ListType
/// \tparam n length to enumerate
/// \tparam q field size, e.g. enumeration symbols = {0, ..., q-1}
/// \tparam w weight to enumerate
template<class ListType,
		 const uint32_t n,
		 const uint32_t q,
		 const uint32_t mitm_w,
         const uint32_t noreps_w,
         const uint32_t split>
class ListEnumerateSinglePartialSingle: public ListEnumeration_Meta<ListType, n, q, mitm_w + noreps_w> {
public:

	/// helper definition. Shouldnt be used.
	constexpr static uint32_t w = mitm_w + noreps_w;

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

	using T = Value::ContainerLimbType;

	const uint32_t qprime;

	///
	constexpr static uint32_t mitmlen = split;
	constexpr static uint32_t mitmlen_half   = (mitmlen + 1u) / 2u;
	constexpr static uint32_t mitmlen_offset = mitmlen - mitmlen_half;

	constexpr static uint32_t norepslen = n - split;
	constexpr static uint32_t norepslen_quarter = (norepslen + 3) / 4;
	constexpr static uint32_t norepslen_offset  = (norepslen) / 4; /// TODO not fully correct: with this the last coordinate will not be enumerated
	Combinations_Binary_Chase<T, mitmlen_half, mitm_w>        mitm_chase   = Combinations_Binary_Chase<T, mitmlen_half, mitm_w>{};
	Combinations_Binary_Chase<T, norepslen_quarter, noreps_w> noreps_chase = Combinations_Binary_Chase<T, norepslen_quarter, noreps_w>{};
	constexpr static size_t mitm_chase_size   = Combinations_Binary_Chase<T, mitmlen_half, mitm_w>::chase_size;
	constexpr static size_t noreps_chase_size = Combinations_Binary_Chase<T, norepslen_quarter, noreps_w>::chase_size;

	/// this can be used to specify the size of the input list
	/// e.g. its the maximum number of elements this class enumerates
	constexpr static size_t LIST_SIZE = mitm_chase_size * noreps_chase_size;

	// this can be se to something else as `LIST_SIZE`, if one wants to only
	// enumerate a part of the sequence
	const size_t list_size = 0;

	// change list for the chase sequence
	std::vector<std::pair<uint16_t, uint16_t>> mitm_chase_cl   = std::vector<std::pair<uint16_t, uint16_t>>(mitm_chase_size);
	std::vector<std::pair<uint16_t, uint16_t>> noreps_chase_cl = std::vector<std::pair<uint16_t, uint16_t>>(noreps_chase_size);

	/// empty constructor
	/// \param HT transposed parity check matrix
	/// \param list_size max numbers of elements to enumerate.
	/// 			if set to 0: the complete sequence will be enumerated.
	/// \param syndrome additional element which is added to all list elements
	ListEnumerateSinglePartialSingle(const uint32_t qprime,
	                                 const Matrix &HT,
	                                 const size_t list_size=0,
								     const Label *syndrome= nullptr) :
			ListEnumeration_Meta<ListType, n, q, w>(HT, syndrome),
			qprime(qprime),
			list_size((list_size==size_t(0)) ? LIST_SIZE : list_size) {
		static_assert(n > w);
		static_assert(mitmlen > mitm_w);
		static_assert(norepslen > noreps_w);

		static_assert(mitm_chase_size >= 0);
		static_assert(noreps_chase_size > 0);
		ASSERT(LIST_SIZE >= list_size);

		mitm_chase.template changelist<false>(mitm_chase_cl.data());
		noreps_chase.template changelist<false>(noreps_chase_cl.data());
	}

	/// \tparam HashMap
	/// \tparam Extractor extractor lambda
	/// 		- can be NULL
	/// \tparam Predicate Function. NOTE: can be
	///			- nullptr_t
	/// 		- std::invokable. if this returns true, the function returns
	/// \param L1 first list. NOTE:
	/// 		- the syndrome is only added into the first list
	/// 		- if a hashmap is given: this list will be hashed into the first hashmap
	/// \param L2 second list.
	/// \param L3 third list.
	/// \param L4 fourth list.
	/// \param offset
	/// 		- number of position between the MITM strategy
	/// \param tid thread id
	/// \param hm hashmap
	/// \param e extractor
	/// \param p predicate function
	/// \return true/false if the golden element was found or not (only if
	///  		predicate was given)
	template<typename HashMap, typename Extractor>
#if __cplusplus > 201709L
	requires (std::is_same_v<std::nullptr_t, HashMap> || HashMapAble<HashMap>) &&
	(std::is_same_v<std::nullptr_t, Extractor> || std::is_invocable_v<Extractor, Label>)
#endif
	bool run(ListType &L1,
			 ListType &L2,
			 ListType &L3,
			 ListType &L4,
			 const uint32_t tid=0,
			 HashMap *hm=nullptr,
			 Extractor *e=nullptr) {
		Element element3, element4;
		Label tmp;

		/// clear stuff, needed if this functions is called multiple times
		/// e.g. in every ISD algorithm
		element1.zero(); element2.zero(); element3.zero(); element4.zero();

		/// pack stuff together
		Element *elements[4] = {&element1, &element2, &element3, &element4};
		ListType *lists[4] = {&L1, &L2, &L3, &L4};

		/// counter of how many elements already added to the list
		size_t ctr = 0;

		// check if the lists are enabled
		constexpr bool sHM = !std::is_same_v<std::nullptr_t, HashMap>;

		/// add the syndrome, if needed
		if (syndrome != nullptr) {
			element1.label = *syndrome;
		}

		/// compute the first elements
		for (uint32_t i = 0; i < mitm_w; ++i) {
			for (uint32_t k = 0; k < 4; ++k) {
				elements[k]->value.set(qprime, i + (k&1u)*mitmlen_offset);
				Label::scalar(tmp, HT.get(i + (k&1u)*mitmlen_offset), qprime);
				Label::add(elements[k]->label, elements[k]->label, tmp);
			}
		}

		for (uint32_t i = 0; i < noreps_w; ++i) {
			for (uint32_t k = 0; k < 4; ++k) {
				elements[k]->value.set(qprime, i + split + k *norepslen_offset);
				Label::scalar(tmp, HT.get(i + split + k *norepslen_offset), qprime);
				Label::add(elements[k]->label, elements[k]->label, tmp);
			}
		}

		auto chase_step = [this](Element &element,
								 const uint32_t unset,
								 const uint32_t set) {
		  /// make really sure that the the chase
		  /// sequence is correct.
		  ASSERT(element.value[unset]);
		  ASSERT(!element.value[set]);
		  ASSERT(std::abs((int)unset - (int)set) <= (int)w);

		  Label tmp;
		  Label::scalar(tmp, HT.get(unset), q-qprime);
		  Label::add(element.label, element.label, tmp);
		  Label::scalar(tmp, HT.get(set), qprime);
		  Label::add(element.label, element.label, tmp);

		  element.value.set(0u, unset);
		  element.value.set(qprime, set);
		};

		/// iterate over all sequences
		for (uint32_t i = 0; i < mitm_chase_size; ++i) {
			for (uint32_t j = 0; j < noreps_chase_size ; ++j) {
				for (uint32_t k = 0; k < 4; ++k) {
					check(elements[k]->label, elements[k]->value);
					insert_list(lists[k], *elements[k], ctr, tid);

					chase_step(*elements[k],
					          split + k*norepslen_offset + noreps_chase_cl[j].first,
					          split + k*norepslen_offset + noreps_chase_cl[j].second);
				}

				// TODO also hash the 3 element
				if constexpr (sHM) insert_hashmap(hm, e, element1, ctr, tid);

				ctr += 1;
				if (ctr >= list_size) {
					return false;
				}
			}

			for (uint32_t k = 0; k < 4; ++k) {
				check(elements[k]->label, elements[k]->value);
				insert_list(lists[k], *elements[k], ctr, tid);

				chase_step(*elements[k],
						  (k&1u)*mitmlen_offset + mitm_chase_cl[i].first,
						  (k&1u)*mitmlen_offset + mitm_chase_cl[i].second);
			}

			if constexpr (sHM) insert_hashmap(hm, e, element1, ctr, tid);
			ctr += 1;
			if (ctr >= list_size) {
				return false;
			}
		}

		/// make sure that all elements where generated
		ASSERT(ctr == LIST_SIZE);
		return false;
	}
};


/// this class enumerates elements of the form:
///  [xyxyxyxy|xy000000], [xyxyxyxy|00xy0000], [xyxyxyxy|0000xy00], [xyxyxyxy|000000xy]
/// In other-words you must pass 4 base lists to the function.
/// NOTE: nomenclature
/// 	<-       n       ->
/// 	[xyxy0000|xy000000]
///     0      split      n
///      mitmlen  norepslen
/// \tparam ListType
/// \tparam n length to enumerate
/// \tparam q field size, e.g. enumeration symbols = {0, ..., q-1}
/// \tparam w weight to enumerate
template<class ListType,
		const uint32_t n,
		const uint32_t q,
		const uint32_t mitm_w,
		const uint32_t noreps_w,
		const uint32_t split>
class ListEnumerateMultiDisjointBlock: public ListEnumeration_Meta<ListType, n, q, mitm_w + noreps_w> {
public:

	/// helper definition. Shouldnt be used.
	constexpr static uint32_t w = mitm_w + noreps_w;

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

	using T = Value::ContainerLimbType;

	///
	constexpr static uint32_t mitmlen = split;
	constexpr static uint32_t mitmlen_half   = (mitmlen + 1u) / 2u;
	constexpr static uint32_t mitmlen_offset = mitmlen - mitmlen_half;

	constexpr static uint32_t norepslen = n - split;
	constexpr static uint32_t norepslen_quarter = (norepslen + 3) / 4;
	constexpr static uint32_t norepslen_offset  = (norepslen) / 4; /// TODO not fully correct: with this the last coordinate will not be enumerated
	Combinations_Fq_Chase<mitmlen_half, q, mitm_w>        mitm_chase   = Combinations_Fq_Chase<mitmlen_half, q, mitm_w>{};
	Combinations_Fq_Chase<norepslen_quarter, q, noreps_w> noreps_chase = Combinations_Fq_Chase<norepslen_quarter, q, noreps_w>{};

	constexpr static size_t mitm_chase_size   = Combinations_Fq_Chase<mitmlen_half, q, mitm_w>::chase_size;
	constexpr static size_t mitm_gray_size    = Combinations_Fq_Chase<mitmlen_half, q, mitm_w>::gray_size;
	constexpr static size_t noreps_chase_size = Combinations_Fq_Chase<norepslen_quarter, q, noreps_w>::chase_size;
	constexpr static size_t noreps_gray_size  = Combinations_Fq_Chase<norepslen_quarter, q, noreps_w>::gray_size;

	/// this can be used to specify the size of the input list
	/// e.g. its the maximum number of elements this class enumerates
	constexpr static size_t LIST_SIZE = (mitm_chase_size * mitm_gray_size) * (noreps_chase_size * noreps_gray_size);

	// this can be se to something else as `LIST_SIZE`, if one wants to only
	// enumerate a part of the sequence
	const size_t list_size = 0;

	// change list for the chase sequence
	std::vector<std::pair<uint16_t, uint16_t>> mitm_chase_cl   = std::vector<std::pair<uint16_t, uint16_t>>(mitm_chase_size);
	std::vector<std::pair<uint16_t, uint16_t>> noreps_chase_cl = std::vector<std::pair<uint16_t, uint16_t>>(noreps_chase_size);
	std::vector<uint16_t> mitm_gray_cl   = std::vector<uint16_t>(mitm_gray_size);
	std::vector<uint16_t> noreps_gray_cl = std::vector<uint16_t>(noreps_gray_size);

	/// empty constructor
	/// \param HT transposed parity check matrix
	/// \param list_size max numbers of elements to enumerate.
	/// 			if set to 0: the complete sequence will be enumerated.
	/// \param syndrome additional element which is added to all list elements
	ListEnumerateMultiDisjointBlock(const Matrix &HT,
									const size_t list_size=0,
									const Label *syndrome= nullptr) :
			ListEnumeration_Meta<ListType, n, q, w>(HT, syndrome),
			list_size((list_size==size_t(0)) ? LIST_SIZE : list_size) {
		static_assert(n > w);
		static_assert(mitmlen > mitm_w);
		static_assert(norepslen > noreps_w);

		static_assert(mitm_chase_size >= 0);
		static_assert(noreps_chase_size > 0);
		ASSERT(LIST_SIZE >= list_size);

		if constexpr (q>2) mitm_chase.changelist_mixed_radix_grey(mitm_gray_cl.data());
		if constexpr (q>2) noreps_chase.changelist_mixed_radix_grey(noreps_gray_cl.data());
		mitm_chase.template changelist_chase<false>(mitm_chase_cl.data());
		noreps_chase.template changelist_chase<false>(noreps_chase_cl.data());
	}

	/// \tparam HashMap
	/// \tparam Extractor extractor lambda
	/// 		- can be NULL
	/// \tparam Predicate Function. NOTE: can be
	///			- nullptr_t
	/// 		- std::invokable. if this returns true, the function returns
	/// \param L1 first list. NOTE:
	/// 		- the syndrome is only added into the first list
	/// 		- if a hashmap is given: this list will be hashed into the first hashmap
	/// \param L2 second list.
	/// \param L3 third list.
	/// \param L4 fourth list.
	/// \param offset
	/// 		- number of position between the MITM strategy
	/// \param tid thread id
	/// \param hm hashmap
	/// \param e extractor
	/// \param p predicate function
	/// \return true/false if the golden element was found or not (only if
	///  		predicate was given)
	template<typename HashMap, typename Extractor>
#if __cplusplus > 201709L
	requires (std::is_same_v<std::nullptr_t, HashMap> || HashMapAble<HashMap>) &&
	(std::is_same_v<std::nullptr_t, Extractor> || std::is_invocable_v<Extractor, Label>)
#endif
	bool run(ListType &L1, ListType &L2,
			 ListType &L3, ListType &L4,
			 const uint32_t tid=0,
			 HashMap *hm=nullptr,
			 Extractor *e=nullptr) {
		Element element3, element4;

		/// clear stuff, needed if this functions is called multiple times
		/// e.g. in every ISD algorithm
		element1.zero(); element2.zero(); element3.zero(); element4.zero();

		/// pack stuff together
		Element *elements[4] = {&element1, &element2, &element3, &element4};
		ListType *lists[4] = {&L1, &L2, &L3, &L4};

		/// counter of how many elements already added to the list
		size_t ctr = 0;

		// check if the lists are enabled
		constexpr bool sHM = !std::is_same_v<std::nullptr_t, HashMap>;

		/// add the syndrome, if needed
		if (syndrome != nullptr) {
			element1.label = *syndrome;
		}

		/// compute the first elements
		for (uint32_t i = 0; i < mitm_w; ++i) {
			for (uint32_t k = 0; k < 4; ++k) {
				elements[k]->value.set(1u, i + (k&1u)*mitmlen_offset);
				Label::add(elements[k]->label, elements[k]->label, HT.get(i + (k&1u)*mitmlen_offset));
			}
		}

		for (uint32_t i = 0; i < noreps_w; ++i) {
			for (uint32_t k = 0; k < 4; ++k) {
				elements[k]->value.set(1u, i + split + k *norepslen_offset);
				Label::add(elements[k]->label, elements[k]->label, HT.get(i + split + k *norepslen_offset));
			}
		}

		//
		std::vector<uint32_t> mitm_current_set(mitm_w, 0), noreps_current_set(noreps_w, 0);
		for (uint32_t i = 0; i < mitm_w; ++i) { mitm_current_set[i] = i; }
		for (uint32_t i = 0; i < noreps_w; ++i) { noreps_current_set[i] = i; }

		auto gray_step = [this](Element &element, const uint32_t pos) {
		  element.value.set((element.value[pos] + 1u) % q, pos);
		  Label::add(element.label, element.label, HT.get(pos));

		  /// NOTE: this is stupid, but needed. The gray code enumeration
		  /// also enumerates zeros. Therefore we need to fix them
		  if (element.value[pos] == 0) {
			  element.value.set(1u, pos);
			  Label::add(element.label, element.label, HT.get(pos));
		  }
		};

		auto chase_step = [this](Element &element,
								 const uint32_t unset,
								 const uint32_t set) {
		  /// make really sure that the the chase
		  /// sequence is correct.
		  ASSERT(element.value[unset]);
		  ASSERT(!element.value[set]);
		  ASSERT(std::abs((int)unset - (int)set) <= (int)w);

		  Label tmp;
		  Label::scalar(tmp, HT.get(unset), q-element.value[unset]);
		  Label::add(element.label, element.label, tmp);
		  Label::add(element.label, element.label, HT.get(set));

		  element.value.set(0u, unset);
		  element.value.set(1u, set);
		};

		/// iterate over all sequences
		for (uint32_t mc = 0; mc < mitm_chase_size; ++mc) {
			for (uint32_t mg = 0; mg < mitm_gray_size; ++mg) {
				for (uint32_t nrc = 0; nrc < noreps_chase_size; ++nrc) {
					for (uint32_t nrg = 0; nrg < noreps_gray_size; ++nrg) {
						for (uint32_t k = 0; k < 4; ++k) {
							check(elements[k]->label, elements[k]->value);
							insert_list(lists[k], *elements[k], ctr, tid);

							gray_step(*elements[k],
							           split + k*norepslen_offset + noreps_current_set[noreps_gray_cl[nrg]]);
						}

						// TODO also hash the 3 element
						if constexpr (sHM) insert_hashmap(hm, e, element1, ctr, tid);

						ctr += 1;
						if (ctr >= list_size) { return false; }
					} // end noreps gray

					for (uint32_t k = 0; k < 4; ++k) {
						check(elements[k]->label, elements[k]->value);
						insert_list(lists[k], *elements[k], ctr, tid);

						const uint32_t a = split + k*norepslen_offset + noreps_chase_cl[nrc].first;
						const uint32_t b = split + k*norepslen_offset + noreps_chase_cl[nrc].second;
						chase_step(*elements[k], a, b);
					}

					for (uint32_t j = 0; j < noreps_w; ++j) { if (noreps_current_set[j] == noreps_chase_cl[nrc].first) { noreps_current_set[j] = noreps_chase_cl[nrc].second; }}

					if constexpr (sHM) insert_hashmap(hm, e, element1, ctr, tid);
					ctr += 1;
					if (ctr >= list_size) { return false; }
				} // end noreps chase

				for (uint32_t k = 0; k < 4; ++k) {
					gray_step(*elements[k],
					          (k&1u)*mitmlen_offset + mitm_current_set[mitm_gray_cl[mg]]);
				}
			} // end mitm gray

			for (uint32_t k = 0; k < 4; ++k) {
				check(elements[k]->label, elements[k]->value);
				insert_list(lists[k], *elements[k], ctr, tid);

				const uint32_t a = (k & 1u) * mitmlen_offset + mitm_chase_cl[mc].first;
				const uint32_t b = (k & 1u) * mitmlen_offset + mitm_chase_cl[mc].second;
				chase_step(*elements[k], a, b);
			}

			for (uint32_t j = 0; j < mitm_w; ++j) { if (mitm_current_set[j] == mitm_chase_cl[mc].first) { mitm_current_set[j] = mitm_chase_cl[mc].second; }}
			if constexpr (sHM) insert_hashmap(hm, e, element1, ctr, tid);
			ctr += 1;
			if (ctr >= list_size) { return false; }
		} // end mitm chase

		/// make sure that all elements where generated
		ASSERT(ctr == LIST_SIZE);
		return false;
	}
};
#endif//CRYPTANALYSISLIB_FQ_ENUMERATION_H
