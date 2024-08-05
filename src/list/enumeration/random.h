#ifndef CRYPTANALYSISLIB_RANDOM_ENUMERATION_H
#define CRYPTANALYSISLIB_RANDOM_ENUMERATION_H


#ifndef CRYPTANALYSISLIB_LIST_ENUMERATION_H
#error "dont use include <list/enumeration/random.h>, use #include <list/enumeration/enumeration.h> instead"
#endif

#include <cstddef>   // needed for std::nullptr_t
#include <functional>// needed for std::invoke

#include "combination/chase.h"
#include "helper.h"
#include "list/enumeration/enumeration.h"
#include "math/bc.h"

/// This class enumerates vectors of length n and weight w, whereas
/// no changelist is used.
/// \tparam ListType
/// \tparam n vector length
/// \tparam w weight
template<class ListType,
		const uint32_t n,
		const uint32_t w>
class BinaryRandomEnumerator : public ListEnumeration_Meta<ListType, n, 2, w> {
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
	constexpr static size_t max_list_size = 1ull << (n - w);
	const size_t list_size;

	// some security things
	static_assert(Value::length() >= w);
	static_assert(n >= w);
	static_assert(w > 0);
private:
	constexpr BinaryRandomEnumerator() noexcept {};

public:
	/// \param HT transposed parity check matrix
	/// \param list_size max numbers of elements to enumerate.
	/// 			if set to 0: the complete sequence will be enumerated.
	/// \param syndrome additional element which is added to all list elements
	constexpr BinaryRandomEnumerator(const Matrix &HT,
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
		const auto H = HT.transpose();

		// check if the lists are enabled
		const bool sL1 = L1 != nullptr;
		const bool sL2 = L2 != nullptr;
		constexpr bool sHM = !std::is_same_v<std::nullptr_t, HashMap>;
		constexpr bool sP = !std::is_same_v<std::nullptr_t, Predicate>;

		for (size_t ctr = 0; ctr < list_size; ++ctr) {
			element1.value.random_with_weight(w, n/2, base_offset);
			H.mul(element1.label, element1.value);
			if (syndrome != nullptr) { Label::add(element1.label, element1.label, *syndrome); }

			if (sL2) {
				Value::sll(element2.value, element1.value, offset);
				H.mul(element2.label, element2.value);
			}

			check(element1.label, element1.value);
			if (sL2) check(element2.label, element2.value, false);

			if constexpr (sP) { if (std::invoke(*p, element1.label)) { return true; } }
			if constexpr (sHM) { insert_hashmap(hm, e, element1, ctr, tid); }
			if (sL1) { insert_list(L1, element1, ctr, tid); }
			if (sL2) { insert_list(L2, element2, ctr, tid); }
		}

		if (sL1) { L1->set_load(list_size); }
		if (sL2) { L2->set_load(list_size); }

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
class MaxBinaryRandomEnumerator : public ListEnumeration_Meta<ListType, n, 2, w> {
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
	constexpr static size_t max_list_size = 1ull << (n - w);
	const size_t list_size;

	// some security things
	static_assert(Value::length() >= w);
	static_assert(n >= w);
	static_assert(w > 0);
private:
	constexpr MaxBinaryRandomEnumerator() noexcept {};

public:
	/// \param HT transposed parity check matrix
	/// \param list_size max numbers of elements to enumerate.
	/// 			if set to 0: the complete sequence will be enumerated.
	/// \param syndrome additional element which is added to all list elements
	constexpr MaxBinaryRandomEnumerator(const Matrix &HT,
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
		const auto H = HT.transpose();
		element1.zero();
		element2.zero();

		// check if the lists are enabled
		const bool sL1 = L1 != nullptr;
		const bool sL2 = L2 != nullptr;
		constexpr bool sHM = !std::is_same_v<std::nullptr_t, HashMap>;
		constexpr bool sP = !std::is_same_v<std::nullptr_t, Predicate>;

		for (size_t ctr = 0; ctr < list_size; ++ctr) {
			const uint32_t w1 = fastrandombytes_uint64(1, w);
			element1.value.random_with_weight(w1, n/2, base_offset);
			H.mul(element1.label, element1.value);

			if (syndrome != nullptr) { Label::add(element1.label, element1.label, *syndrome); }

			if (sL2) {
				const uint32_t w2 = fastrandombytes_uint64(1, w);
				element2.value.random_with_weight(w2, n/2, base_offset+offset);
				H.mul(element2.label, element2.value);
			}

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
};

#endif//CRYPTANALYSISLIB_BINARY_ENUMERATION_H
