#ifndef CRYPTANALYSISLIB_FQ_NEW_H
#define CRYPTANALYSISLIB_FQ_NEW_H

/// needed for std::nullptr_t
#include <cstddef>

#include "helper.h"
#include "list/common.h"
#include "container/hashmap/common.h"

///
/// \tparam ListType BaseList Type
/// \tparam n length to enumerate
/// \tparam q field size. q-1 is the max value to enumerate
/// \tparam w weight to enumerate
/// 		if w == 2: only a chase sequence will enumerated
template<class ListType,
		const uint32_t n,
		const uint32_t q,
		const uint32_t w>
#if __cplusplus > 201709L
requires ListAble<ListType>
#endif
class ListEnumeration_Meta {
public:
	/// needed typedef
	typedef ListType List;
	typedef typename ListType::ElementType Element;
	typedef typename ListType::ValueType Value;
	typedef typename ListType::LabelType Label;
	typedef typename ListType::MatrixType Matrix;

	/// needed variables
	// this generates a grey code sequence, e.g. a sequence in which
	// two consecutive elements only differ in a single position
	Combinations_Fq_Chase<n, q, w> chase = Combinations_Fq_Chase<n, q, w>();
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

	// the transposed matrix
	const Matrix &HT;

	/// can be optionally passed:
	// the syndrome, if given, will be added into the sequence at the beginning
	const Label *syndrome = nullptr;



	/// checks for the correctness of the computed label.
	/// e.g. it checks it l == HT*e
	/// \param l computed label
	/// \param e error vector resulting in the label
	/// \return true/false
	bool check(const Label &label, const Value &error, bool add_syndrome=true) {
#ifdef DEBUG
	  	/// TEST for correctness
		auto H = HT.transpose();
		Label tmpl;

		H.matrix_row_vector_mul2(tmpl, error);
		if (syndrome != nullptr && add_syndrome) {
			Label::add(tmpl, tmpl, *syndrome);
		}

		if (!tmpl.is_equal(label)) {
			std::cout << std::endl << "ERROR: (SHOULD, IS)" << std::endl;
			tmpl.print();
			label.print();
			std::cout <<std::endl;
			error.print();
			std::cout <<std::endl;
			HT.print();
		}

		ASSERT(tmpl.is_equal(label));

		uint32_t tmp_vec_ctr = error.weight();
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
	void insert_list(ListType *L,
	                 const Element &element,
	                 const size_t ctr,
	                 const uint32_t tid) {
		// nothing to inset
		if (L == nullptr) {
			return;
		}

		L->insert(element, ctr, tid);
	}

	/// abstract insertion handler fo hashmap
	/// \tparam HashMap
	/// \tparam Extractor
	/// \param hm
	/// \param e
	template<class HashMap, typename Extractor>
#if __cplusplus > 201709L
	requires (std::is_same_v<std::nullptr_t, HashMap> || HashMapAble<HashMap>) &&
			 (std::is_same_v<std::nullptr_t, Extractor> || std::is_invocable_v<Extractor, Label>)
#endif
	void insert_hashmap(HashMap *hm,
	                    Extractor *e,
	                    const Element &element,
	                    const size_t ctr,
	                    const uint32_t tid) {
		if constexpr (! std::is_same_v<std::nullptr_t , HashMap>) {
			// nothing to inset
			if (hm == nullptr) {
				return;
			}

			using IndexType = typename HashMap::IndexType;
			using LPartType = typename HashMap::T;
			IndexType npos[1];
			npos[0] = ctr;

			const LPartType data = std::invoke(*e, element.label);
			hm->insert(data, npos, tid);
		}
	}

	///
	/// \param HT
	/// \param list_size
	/// \param syndrome
	ListEnumeration_Meta(const Matrix &HT,
	                     const size_t list_size=0,
	                     const Label *syndrome= nullptr) :
	    HT(HT), syndrome(syndrome), list_size((list_size==size_t(0)) ? LIST_SIZE : list_size)  {
		/// some sanity checks
		/// NOTE: its allowed to call this class with `w=0`, which is needed for Prange
		static_assert(n > w);
		static_assert(q > 1);
		static_assert(n <= Value::LENGTH);

		ASSERT(chase_size > 0);
		ASSERT(gray_size > 0);
		ASSERT(LIST_SIZE >= list_size);

		chase.changelist_mixed_radix_grey(gray_cl.data());
		chase.changelist_chase(chase_cl.data());
	}
};


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

	/// needed variables
	using ListEnumeration_Meta<ListType, n, q, w>::syndrome;
	using ListEnumeration_Meta<ListType, n, q, w>::HT;
	using ListEnumeration_Meta<ListType, n, q, w>::chase_size;
	using ListEnumeration_Meta<ListType, n, q, w>::gray_size;
	using ListEnumeration_Meta<ListType, n, q, w>::gray_cl;
	using ListEnumeration_Meta<ListType, n, q, w>::chase_cl;
	using ListEnumeration_Meta<ListType, n, q, w>::list_size;
	using ListEnumeration_Meta<ListType, n, q, w>::LIST_SIZE;

	/// empty constructor
	///
	/// \param HT
	/// \param list_size
	/// \param syndrome
	ListEnumerateMultiFullLength(const Matrix &HT,
	                             const size_t list_size=0,
	                             const Label *syndrome= nullptr) :
	    ListEnumeration_Meta<ListType, n, q, w>(HT, list_size, syndrome) {
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
	template<typename HashMap, typename Extractor, typename Predicate>
#if __cplusplus > 201709L
	requires (std::is_same_v<std::nullptr_t, HashMap> || HashMapAble<HashMap>) &&
	         (std::is_same_v<std::nullptr_t, Extractor> || std::is_invocable_v<Extractor, Label>) &&
			 (std::is_same_v<std::nullptr_t, Predicate> || std::is_invocable_v<Predicate, Label>)
#endif
	void run(ListType *L1=nullptr,
	         ListType *L2=nullptr,
	         const uint32_t offset=0,
	         const uint32_t tid=0,
	         HashMap *hm=nullptr,
	         Extractor *e=nullptr,
	         Predicate *p= nullptr) {
		/// some security checks
		ASSERT(n+offset <= Value::LENGTH);

		/// counter of how many elements already added to the list
		size_t ctr = 0;
		Element element, element2;

		// check if the second list is enabled
		const bool sL1 = L1 != nullptr;
		const bool sL2 = L2 != nullptr;
		constexpr bool sHM = !std::is_same_v<std::nullptr_t, HashMap>;
		constexpr bool sP = !std::is_same_v<std::nullptr_t, Predicate>;

		/// add the syndrome, if needed
		if (syndrome != nullptr) {
			element.label = *syndrome;
		}

		/// compute the first element
		for (uint32_t i = 0; i < w; ++i) {
			if (sL1) {
				element.value.set(1, i);
				Label::add(element.label, element.label, HT.get(i));
			}
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

		auto chase_step = [&current_set, this](Element &element,
				const uint32_t a,
				const uint32_t b,
				const uint32_t off) {
		  /// make really sure that the the chase
		  /// sequence is correct.
		  ASSERT(element.value[a + off]);

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
				if (sL1) check(element.label, element.value);
				if (sL2) check(element2.label, element2.value, false);

				if constexpr (sP) if(std::invoke(p, element.label)) return;
				if constexpr (sHM) insert_hashmap(hm, e, element, ctr, tid);
				if (sL1) insert_list(L1, element, ctr, tid);
				if (sL2) insert_list(L2, element2, ctr, tid);

				ctr += 1;
				if (ctr >= list_size) {
					return;
				}

				if (sL1) gray_step(element, j, 0);
				if (sL2) gray_step(element2, j, offset);
			}

			if (sL1) check(element.label, element.value);
			if (sL2) check(element2.label, element2.value, false);

			if constexpr (sP) if(std::invoke(p, element.label)) return;
			if constexpr (sHM) insert_hashmap(hm, e, element, ctr, tid);
			if (sL1) insert_list(L1, element, ctr, tid);
			if (sL2) insert_list(L2, element2, ctr, tid);

			ctr += 1;
			if (ctr >= list_size) {
				return;
			}

			/// advance the current set by one
			const uint32_t j = chase_cl[i].first;
			ASSERT(j < w);
			const uint32_t a = current_set[j];
			const uint32_t b = chase_cl[i].second;
			current_set[j] = b;

			if(sL1) chase_step(element, a, b, 0);
			if(sL2) chase_step(element2, a, b, offset);
		}

		/// make sure that all elements where generated
		ASSERT(ctr == LIST_SIZE);
	}
};

#endif//CRYPTANALYSISLIB_FQ_NEW_H
