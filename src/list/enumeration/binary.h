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
/// nonzero position is enumerated in binary:w
/// \tparam ListType
/// \tparam n vector length
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

	// TODO rename to max list size
	constexpr static size_t LIST_SIZE = Combinations_Binary_Chase<T, n, w>::chase_size;
	const size_t list_size;
	std::array<std::pair<uint16_t, uint16_t>, bc(n, w)> cL;
	Element e11, e12, e21, e22;

	/// \param HT transposed parity check matrix
	/// \param list_size max numbers of elements to enumerate.
	/// 			if set to 0: the complete sequence will be enumerated.
	/// \param syndrome additional element which is added to all list elements
	constexpr BinaryListEnumerateMultiFullLength(const Matrix &HT,
	                                   const size_t list_size = 0,
	                                   const Label *syndrome = nullptr) noexcept
	    : ListEnumeration_Meta<ListType, n, q, w>(HT, syndrome),
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
			 const uint32_t base_offset = 0,
	         const uint32_t tid = 0,
	         HashMap *hm = nullptr,
	         Extractor *e = nullptr,
	         Predicate *p = nullptr) {
		/// some security checks
		ASSERT(n + offset <= Value::length());

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
		for (uint32_t i = base_offset; i < (base_offset + w); ++i) {
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

		auto chase_step = [this, base_offset](Element &element,
		                         const uint32_t a,
		                         const uint32_t b,
		                         const uint32_t off) {
			const uint32_t off2 = off + base_offset;
			/// make really sure that the the chase
			/// sequence is correct.
			ASSERT(element.value[a + off2]);

			std::cout << element;

			Label::add(element.label, element.label, HT.get(a + off2));
			Label::add(element.label, element.label, HT.get(b + off2));
			element.value.set(0, off2 + a);
			element.value.set(1, off2 + b);
			std::cout << element;
		};

		/// iterate over all sequences
		for (uint32_t i = 0; i < list_size; ++i) {
			check(element1.label, element1.value);
			if (sL2) check(element2.label, element2.value, false);

			if constexpr (sP) { if (std::invoke(*p, element1.label)) { return true; } }
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



/// This class enumerates vectors of length n and weight w, whereas
/// no changelist is used.
/// \tparam ListType
/// \tparam n vector length
/// \tparam w weight
template<class ListType,
		const uint32_t n,
		const uint32_t w>
class BinaryListEnumerateMultiFullLengthWithoutChangeList : public ListEnumeration_Meta<ListType, n, 2, w> {
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
	constexpr static size_t max_list_size = Combinations_Binary_Chase<T, n, w>::chase_size;
	const size_t list_size;

	// some security things
	static_assert(Value::length() >= w);
	static_assert(n >= w);
	static_assert(w > 0);
private:
	constexpr BinaryListEnumerateMultiFullLengthWithoutChangeList() noexcept {};

public:
	/// \param HT transposed parity check matrix
	/// \param list_size max numbers of elements to enumerate.
	/// 			if set to 0: the complete sequence will be enumerated.
	/// \param syndrome additional element which is added to all list elements
	constexpr BinaryListEnumerateMultiFullLengthWithoutChangeList(const Matrix &HT,
									   const size_t list_size = 0,
									   const Label *syndrome = nullptr) noexcept
			: ListEnumeration_Meta<ListType, n, q, w>(HT, syndrome),
			  list_size((list_size == size_t(0)) ? max_list_size : list_size)
	 		  {}

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
	requires(std::is_same_v<std::nullptr_t, HashMap> || HashMapAble<HashMap>) &&
			(std::is_same_v<std::nullptr_t, Extractor> || std::is_invocable_v<Extractor, Label>) &&
			(std::is_same_v<std::nullptr_t, Predicate> || std::is_invocable_v<Predicate, Label>)
	bool run(ListType *L1 = nullptr,
			 ListType *L2 = nullptr,
			 const uint32_t offset = 0,
			 const uint32_t base_offset = 0, // TODO add to other estimators
			 const uint32_t tid = 0,
			 HashMap *hm = nullptr,
			 Extractor *e = nullptr,
			 Predicate *p = nullptr) noexcept {
		/// some security checks
		ASSERT(n + offset <= Value::length());
		ASSERT(offset + base_offset <= Value::length());
		constexpr bool write = false;
		/// counter of how many elements already added to the list
		size_t ctr = 0;

		// check if the lists are enabled
		const bool sL1 = L1 != nullptr;
		const bool sL2 = L2 != nullptr;
		constexpr bool sHM = !std::is_same_v<std::nullptr_t, HashMap>;
		constexpr bool sP = !std::is_same_v<std::nullptr_t, Predicate>;

		std::pair<uint16_t, uint16_t> ret;
		chase.template left_step<write>(nullptr, &ret.first, &ret.second);

		/// clear stuff, needed if this functions is called multiple times
		element1.zero();
		if (sL2) { element2.zero(); }

		/// add the syndrome, if needed
		if (syndrome != nullptr) {
			element1.label = *syndrome;
		}

		/// compute the first element
		for (uint32_t i = base_offset; i < w + base_offset; ++i) {
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

		auto chase_step =
		        [this, base_offset](Element &element,
								 const uint32_t a,
								 const uint32_t b,
								 const uint32_t off) {
			const uint32_t off2 = off + base_offset;
		  /// make really sure that the the chase
		  /// sequence is correct.
		  ASSERT(element.value[a + off2]);

		  Label::add(element.label, element.label, HT.get(a + off2));
		  Label::add(element.label, element.label, HT.get(b + off2));
		  element.value.set(0, a + off2);
		  element.value.set(1, b + off2);
		};

		/// iterate over all sequences
		for (uint32_t i = 0; i < list_size; ++i) {
			check(element1.label, element1.value);
			if (sL2) check(element2.label, element2.value, false);

			if constexpr (sP) { if (std::invoke(*p, element1.label)) { return true; } }
			if constexpr (sHM) { insert_hashmap(hm, e, element1, ctr, tid); }
			if (sL1) insert_list(L1, element1, ctr, tid);
			if (sL2) insert_list(L2, element2, ctr, tid);

			ctr += 1;

			chase.template left_step<write>(nullptr, &ret.first, &ret.second);
			/// advance the current set by one
			const uint32_t a = ret.first;
			const uint32_t b = ret.second;
			chase_step(element1, a, b, 0);
			if (sL2) chase_step(element2, a, b, offset);
		}

		/// make sure that all elements where generated
		ASSERT(ctr == list_size);

		// in this case reset everything, so its recallable
		chase.reset();
		return false;
	}
};

/// This class enumerates vectors of length n and max weight w, whereas
/// no changelist is used.
/// NOTE: key difference to `RandomEnumerator` is that the the weight w
/// 	is a upper limit
/// \tparam ListType
/// \tparam n vector length
/// \tparam w max weight
template<class ListType,
         const uint32_t n,
         const uint32_t w>
class BinaryLexicographicEnumerator : public ListEnumeration_Meta<ListType, n, 2, w> {
public:
	/// Im lazy
	constexpr static uint32_t q = 2;
	using S = ListEnumeration_Meta<ListType, n, q, w>;

	/// needed typedefs
	typedef typename S::Element Element;
	typedef typename S::Matrix Matrix;
	typedef typename S::Value Value;
	typedef typename S::Label Label;
	typedef typename Label::LimbType T;

	/// needed functions
	using S::check;
	using S::insert_hashmap;
	using S::insert_list;
	using S::get_first;
	using S::get_second;
	using S::get_syndrome;
	using S::set_syndrome;

	/// needed variables
	using S::element1;
	using S::element2;
	using S::syndrome;
	using S::HT;

	////
	constexpr static size_t max_list_size = sum_bc(n, w);
	const size_t list_size;

	// some security things
	static_assert(Value::length() >= w);
	static_assert(n >= w);
	static_assert(w > 0);
private:
	constexpr BinaryLexicographicEnumerator() noexcept {};

public:
	/// \param HT transposed parity check matrix
	/// \param list_size max numbers of elements to enumerate.
	/// 			if set to 0: the complete sequence will be enumerated.
	/// \param syndrome additional element which is added to all list elements
	constexpr BinaryLexicographicEnumerator(const Matrix &HT,
	                                    const size_t list_size = 0,
	                                    const Label *syndrome = nullptr) noexcept
	    : ListEnumeration_Meta<ListType, n, q, w>(HT, syndrome),
	      list_size((list_size == size_t(0)) ? max_list_size : list_size)
	{}

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
	    requires(std::is_same_v<std::nullptr_t, HashMap> || HashMapAble<HashMap>) &&
	            (std::is_same_v<std::nullptr_t, Extractor> || std::is_invocable_v<Extractor, Label>) &&
	            (std::is_same_v<std::nullptr_t, Predicate> || std::is_invocable_v<Predicate, Label>)
	bool run(ListType *L1 = nullptr,
	         ListType *L2 = nullptr,
	         const uint32_t offset = 0,
	         const uint32_t base_offset = 0,
	         const uint32_t tid = 0,
	         HashMap *hm = nullptr,
	         Extractor *e = nullptr,
	         Predicate *p = nullptr) noexcept {
		/// some security checks
		ASSERT(n + offset <= Value::length());
		ASSERT(offset + base_offset <= Value::length());
		element1.zero(); element2.zero();

		// check if the lists are enabled
		const bool sL1 = L1 != nullptr;
		const bool sL2 = L2 != nullptr;
		constexpr bool sHM = !std::is_same_v<std::nullptr_t, HashMap>;
		constexpr bool sP = !std::is_same_v<std::nullptr_t, Predicate>;

		for (size_t ctr = 0; ctr < list_size; ++ctr) {
			element1.value.ptr()[0] += 1ull << base_offset;

			HT.mul(element1.label, element1.value);
			if (syndrome != nullptr) { Label::add(element1.label, element1.label, *syndrome); }

			if (sL2) {
				element2.value.ptr()[0] += 1ull << (offset + base_offset);
				HT.mul(element2.label, element2.value);
			}

			// stupid security checks
			const uint32_t wt = element1.value.popcnt();
			if ((wt > w) || (wt == 0)) { ctr -= (wt > w); continue; }

			check(element1.label, element1.value, true, false);
			if (sL2) check(element2.label, element2.value, false, false);

			if constexpr (sP) { if (std::invoke(*p, element1.label)) { return true; } }
			if constexpr (sHM) { insert_hashmap(hm, e, element1, ctr, tid); }
			if (sL1) { insert_list(L1, element1, ctr, tid); }
			if (sL2) { insert_list(L2, element2, ctr, tid); }
		}

		if (sL1) { L1->set_load(list_size); }
		if (sL2) { L2->set_load(list_size); }

		return false;
	}

	bool run(ListType *L1 = nullptr,
			 ListType *L2 = nullptr,
			 const uint32_t offset = 0,
			 const uint32_t base_offset = 0,
			 const uint32_t tid = 0){
		return run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
		        (L1, L2, offset, base_offset, tid);
	}
};

#endif//CRYPTANALYSISLIB_BINARY_ENUMERATION_H
