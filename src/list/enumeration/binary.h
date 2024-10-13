#ifndef CRYPTANALYSISLIB_BINARY_ENUMERATION_H
#define CRYPTANALYSISLIB_BINARY_ENUMERATION_H


#ifndef CRYPTANALYSISLIB_LIST_ENUMERATION_H
#error "dont use include <list/enumeration/binary.h>, use include <list/enumeration/enumeration.h> instead"
#endif

#include <cstddef>   // needed for std::nullptr_t
#include <functional>// needed for std::invoke

#include "list/enumeration/enumeration.h"
#include "combination/chase.h"
#include "helper.h"
#include "math/bc.h"
#include "alloc/alloc.h"

/// enumerates: [11111111]
/// e.g. starts with [110...0]
///		   ends with [00..011]
/// This class enumerates vectors of length n and weight w, whereas each
/// nonzero position is enumerated in binary
/// \tparam ListType
/// \tparam n vector length
/// \tparam w weight
template<class ListType,
         const uint32_t n,
         const uint32_t w>
#if __cplusplus > 201709L
	requires ListAble<ListType>
#endif
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
	std::vector<std::pair<uint16_t, uint16_t>> cL{bc(n, w)};

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
		  chase.template changelist<false>(cL, this->list_size);
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

			Label::sub(element.label, element.label, HT.get(a + off2));
			Label::add(element.label, element.label, HT.get(b + off2));
			element.value.set(0, off2 + a);
			element.value.set(1, off2 + b);
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

		if (sL1) { L1->set_load(list_size); }
		if (sL2) { L2->set_load(list_size); }

		return false;
	}

	/// \param L1
	/// \param L2
	/// \param offset
	/// \param base_offset
	/// \param tid
	/// \return
	bool run(ListType *L1 = nullptr,
			 ListType *L2 = nullptr,
			 const uint32_t offset = 0,
			 const uint32_t base_offset = 0,
			 const uint32_t tid = 0){
		return run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
		        (L1, L2, offset, base_offset, tid);
	}

	///
	constexpr void info() noexcept {
		std::cout << " { name: \"BinaryListEnumerateMultiFullLength\""
				  << ", n: " << n
				  << ", w: " << w
				  << ", max_list_size: " << LIST_SIZE
				  << ", list_size: " << list_size
		          << " }\n";
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
#if __cplusplus > 201709L
requires ListAble<ListType>
#endif
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
	const size_t list_size{};

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
			 const uint32_t base_offset = 0,
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

		if (sL1) { L1->set_load(list_size); }
		if (sL2) { L2->set_load(list_size); }

		return false;
	}


	/// \param L1
	/// \param L2
	/// \param offset
	/// \param base_offset
	/// \param tid
	/// \return
	bool run(ListType *L1 = nullptr,
			 ListType *L2 = nullptr,
			 const uint32_t offset = 0,
			 const uint32_t base_offset = 0,
			 const uint32_t tid = 0){
		return run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
		        (L1, L2, offset, base_offset, tid);
	}

	///
	constexpr void info() noexcept {
		std::cout << " { name: \"BinaryListEnumerateMultiFullLengthWithoutChangeList\""
				  << ", n: " << n
				  << ", w: " << w
				  << ", max_list_size: " << max_list_size
				  << ", list_size: " << list_size
		          << " }\n";
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
#if __cplusplus > 201709L
	requires ListAble<ListType>
#endif
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

	/// \return the max size of the list, which can be enumerated
	[[nodiscard]] constexpr static size_t size() noexcept {
		return max_list_size;
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
			// TODO: this is only correct for values `BinaryContainer` and `PackedContainer` as otherwise there is an overflow
			// 	MAYBE: fix this via if constexpr(is_good_value_type)
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

	///
	/// \param L1
	/// \param L2
	/// \param offset
	/// \param base_offset
	/// \param tid
	/// \return
	bool run(ListType *L1 = nullptr,
			 ListType *L2 = nullptr,
			 const uint32_t offset = 0,
			 const uint32_t base_offset = 0,
			 const uint32_t tid = 0){
		return run <std::nullptr_t, std::nullptr_t, std::nullptr_t>
		        (L1, L2, offset, base_offset, tid);
	}

	///
	constexpr void info() noexcept {
		std::cout << " { name: \"BinaryLexicographicEnumerator\""
				  << ", n: " << n
				  << ", w: " << w
				  << ", max_list_size: " << max_list_size
				  << ", list_size: " << list_size
		          << " }\n";
	}
};




/// this class enumerates elements of the form:
///  [11110000|11000000], [0000111|00110000], [11110000|0000001100], [00001111|00000011]
/// In other-words you must pass 4 base lists to the function.
/// NOTE: nomenclature
/// 	<-       n       ->
/// 	[11110000|11000000] L1
/// 	[00001111|00110000] L2
/// 	[11110000|00001100] L3
/// 	[00001111|00000011] L4
///     0      split      n
///      mitmlen  norepslen
/// \tparam ListType
/// \tparam n length to enumerate
/// \tparam q field size, e.g. enumeration symbols = {0, ..., q-1}
/// \tparam w weight to enumerate
/// \tparam mitm_w hamming weight to enumerate on the mitm part
/// \tparam noreps_w hamming weight to enumerate on the no representations part
/// \tparam split were the split between the two is
template<class ListType,
		 const uint32_t n,
		 const uint32_t mitm_w,
		 const uint32_t noreps_w,
		 const uint32_t split>
#if __cplusplus > 201709L
	requires ListAble<ListType>
#endif
class BinarySinglePartialSingleEnumerator :
    public ListEnumeration_Meta<ListType, n, 2, mitm_w + noreps_w> {
public:
	/// if set to `true` the noreps part will be increased, s.t.
	/// it will be divisible by 4; Therefore the noreps part of L1 will
	/// be overlapping with the reps/mitm part of L2
	constexpr static bool align_noreps = true;

	/// helper definition. Shouldnt be used.
	constexpr static uint32_t w = mitm_w + noreps_w;
	constexpr static uint64_t q = 2;

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

	using T = typename Value::ContainerLimbType;

	/// Helper definitions for the mitm part
	/// Total length to enumerate
	constexpr static uint32_t mitmlen = split;
	constexpr static uint32_t mitmlen_half 		= (mitmlen + 1u) / 2u;
	constexpr static uint32_t mitmlen_offset 	= mitmlen - mitmlen_half;

	constexpr static uint32_t norepslen 		= align_noreps ? roundToAligned<4>(n - split) : n - split;
	constexpr static uint32_t noreps_base 		= align_noreps ? n - norepslen : split;
	constexpr static uint32_t norepslen_quarter = (norepslen + 3) / 4;
	constexpr static uint32_t norepslen_offset 	= (norepslen) / 4;

	using mitm_enumerator = BinaryChaseEnumerator<mitmlen_half, mitm_w>;
	using noreps_enumerator = BinaryChaseEnumerator<norepslen_quarter, noreps_w>;

	/// NOTE: the +1ull is needed, as the computation of `noreps` does not count
	/// for the first element, which needs to be created by the caller function
	constexpr static size_t mitm_chase_size   = mitm_enumerator::size();
	constexpr static size_t noreps_chase_size = noreps_enumerator::size() + 1ull;

	static_assert(n > w);
	static_assert(mitmlen > mitm_w);
	static_assert(norepslen > noreps_w);
	static_assert(mitm_chase_size >= 0);
	static_assert(noreps_chase_size > 0);

	/// this can be used to specify the size of the input list
	/// e.g. its the maximum number of elements this class enumerates
	constexpr static size_t LIST_SIZE = mitm_chase_size * noreps_chase_size;

	// this can be se to something else as `LIST_SIZE`, if one wants to only
	// enumerate a part of the sequence
	const size_t list_size = 0;

	// change list for the chase sequence
	using changelist = std::vector<std::pair<uint16_t, uint16_t>>;
	changelist mitm_chase_cl;
	changelist noreps_chase_cl;

	BinarySinglePartialSingleEnumerator(const Matrix &HT,
									 	const size_t list_size = 0,
									 	const Label *syndrome = nullptr) :
			ListEnumeration_Meta<ListType, n, q, w>(HT, syndrome),
			list_size((list_size == size_t(0)) ? LIST_SIZE : list_size) {
		ASSERT(LIST_SIZE >= list_size);

		mitm_enumerator::changelist(mitm_chase_cl);
		noreps_enumerator::changelist(noreps_chase_cl);
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
	template<typename HashMap,
			typename Extractor,
			typename Predicate>
#if __cplusplus > 201709L
	requires(std::is_same_v<std::nullptr_t, HashMap> || HashMapAble<HashMap>) &&
	(std::is_same_v<std::nullptr_t, Extractor> || std::is_invocable_v<Extractor, Label>) &&
	(std::is_same_v<std::nullptr_t, Predicate> || std::is_invocable_v<Predicate, Label>)
#endif
	bool run(ListType &L1,
			 ListType &L2,
			 ListType &L3,
			 ListType &L4,
			 const uint32_t tid = 0,
			 HashMap *hm = nullptr,
			 Extractor *e = nullptr,
			 Predicate *p = nullptr) {
		Element element3, element4;

		/// clear stuff, needed if this functions is called multiple times
		/// e.g. in every ISD algorithm
		element1.zero();
		element2.zero();
		element3.zero();
		element4.zero();

		/// pack stuff together
		Element *elements[4] = {&element1, &element2, &element3, &element4};
		ListType *lists[4] = {&L1, &L2, &L3, &L4};

		/// counter of how many elements already added to the list
		size_t ctr = 0;

		// check if the lists are enabled
		constexpr bool sHM = !std::is_same_v<std::nullptr_t, HashMap>;
		constexpr bool sP = !std::is_same_v<std::nullptr_t, Predicate>;
		(void)sP; (void)p;

		/// add the syndrome, if needed
		if (syndrome != nullptr) {
			element1.label = *syndrome;
		}

		/// compute the first elements
		/// set the mitm part
		for (uint32_t i = 0; i < mitm_w; ++i) {
			for (uint32_t k = 0; k < 4; ++k) {
				elements[k]->value.set(1u, i + (k & 1u) * mitmlen_offset);
				Label::add(elements[k]->label, elements[k]->label,
				           HT.get(i + (k & 1u) * mitmlen_offset));
			}
		}

		/// set the no representations part
		for (uint32_t i = 0; i < noreps_w; ++i) {
			for (uint32_t k = 0; k < 4; ++k) {
				elements[k]->value.set(1u, i + noreps_base + k * norepslen_offset);
				Label::add(elements[k]->label, elements[k]->label,
				           HT.get(i + noreps_base + k * norepslen_offset));
			}
		}

		auto chase_step = [this](Element &element,
								 const uint32_t unset,
								 const uint32_t set) __attribute__((always_inline)) {
			if (unset == set) {
				// this quirk can happen if `noreps_w` is even.
				return;
			}

			/// make really sure that the the chase
			/// sequence is correct.
			ASSERT(element.value[unset]);
			ASSERT(!element.value[set]);
			ASSERT(std::abs((int) unset - (int) set) <= (int) w);

			Label::sub(element.label, element.label, HT.get(unset));
			Label::add(element.label, element.label, HT.get(set));

			// TODO optimize
			element.value.set(0u, unset);
			element.value.set(1, set);
		};

		/// iterate over all sequences
		for (size_t i = 0; i < mitm_chase_size; ++i) {
			for (size_t j = 0; j < noreps_chase_size-1ull; ++j) {
				for (uint32_t k = 0; k < 4; ++k) {
					check(elements[k]->label, elements[k]->value);
					insert_list(lists[k], *elements[k], ctr, tid);

					chase_step(*elements[k],
							   noreps_base + k * norepslen_offset + noreps_chase_cl[j].first,
							   noreps_base + k * norepslen_offset + noreps_chase_cl[j].second);
				}

				// TODO also hash the 3 element
				if constexpr (sHM) insert_hashmap(hm, e, element1, ctr, tid);

				ctr += 1;
				if (ctr >= list_size) {
					goto finish;
				}
			}

			for (uint32_t k = 0; k < 4; ++k) {
				check(elements[k]->label, elements[k]->value);
				insert_list(lists[k], *elements[k], ctr, tid);

				chase_step(*elements[k],
						   ((k & 1u)*mitmlen_offset) + mitm_chase_cl[i].first,
						   ((k & 1u)*mitmlen_offset) + mitm_chase_cl[i].second);

				// due to easieness reasons, we simply reset the no reps part,
				// and do not walk backwards
				for (uint32_t j = 0; j < noreps_w; ++j) {
					chase_step(*elements[k],
							   noreps_base + k*norepslen_offset + norepslen_quarter-j-1,
							   noreps_base + k*norepslen_offset + j);
				}
			}

			ctr += 1;
			if (ctr >= list_size) {
				goto finish;
			}
		}

	finish:

		/// make sure that all elements where generated
		ASSERT(ctr == LIST_SIZE);
		L1.set_load(list_size);
		L2.set_load(list_size);
		L3.set_load(list_size);
		L4.set_load(list_size);
		return false;
	}

	/// \param L1
	/// \param L2
	/// \param L3
	/// \param L4
	/// \param tid
	/// \return
	bool run(ListType &L1,
			 ListType &L2,
			 ListType &L3,
			 ListType &L4,
			 const uint32_t tid = 0) noexcept {
		return run<std::nullptr_t, std::nullptr_t, std::nullptr_t>
			      (L1, L2, L3, L4, tid, nullptr, nullptr, nullptr);
	}

	///
	constexpr static void info() noexcept {
		std::cout << " { name: \"BinarySinglePartialSingleEnumerator\""
				  << ", n: " << n
				  << ", mitm_w: " << mitm_w
				  << ", noreps_w: " << noreps_w
				  << ", split: " << split
				  << ", mitmlen: " << mitmlen
				  << ", mitmlen_half: " << mitmlen_half
				  << ", mitmlen_offset: " << mitmlen_offset
				  << ", noreps_base: " << noreps_base
				  << ", norepslen: " << norepslen
				  << ", norepslen_quarter: " << norepslen_quarter
				  << ", norepslen_offset: " << norepslen_offset
				  << ", mitm_chase_size: " << mitm_chase_size
				  << ", noreps_chase_size: " << noreps_chase_size
				  << ", LIST_SIZE: " << LIST_SIZE
		          << " }\n";
	}
};
#endif//CRYPTANALYSISLIB_BINARY_ENUMERATION_H
