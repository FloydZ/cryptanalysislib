#ifndef DECODING_LIST_PARALLEL_FULL_H
#define DECODING_LIST_PARALLEL_FULL_H

#include <iterator>
#include <vector>           // main data container
#include <algorithm>        // search/find routines
#include <cassert>

#include "list/common.h"


/// same the class
///		`Parallel_List_T`
/// with the big difference that Elements = <Label, Value> are stored within the same array.
/// Additionally the thread parallel access is further implemented
/// In comparison to different approach this dataset does not track its load factors. It leaves this work completely to
///		the calling function.
template<class Element>
class Parallel_List_FullElement_T : public MetaListT<Element> {
public:
	/// needed typedefs
	using typename MetaListT<Element>::ElementType;
	using typename MetaListT<Element>::ValueType;
	using typename MetaListT<Element>::LabelType;
	using typename MetaListT<Element>::ValueContainerLimbType;
	using typename MetaListT<Element>::LabelContainerLimbType;
	using typename MetaListT<Element>::ValueContainerType;
	using typename MetaListT<Element>::LabelContainerType;
	using typename MetaListT<Element>::ValueDataType;
	using typename MetaListT<Element>::LabelDataType;
	using typename MetaListT<Element>::MatrixType;
	using LoadType = uint64_t;

	/// needed values
	using MetaListT<Element>::__load;
	using MetaListT<Element>::__size;
	using MetaListT<Element>::__data;
	using MetaListT<Element>::__threads;
	using MetaListT<Element>::__thread_block_size;

	using MetaListT<Element>::ValueLENGTH;
	using MetaListT<Element>::LabelLENGTH;

	using MetaListT<Element>::ElementBytes;
	using MetaListT<Element>::ValueBytes;
	using MetaListT<Element>::LabelBytes;

	/// needed functions
	using MetaListT<Element>::size;
	using MetaListT<Element>::set_size;
	using MetaListT<Element>::threads;
	using MetaListT<Element>::set_threads;
	using MetaListT<Element>::thread_block_size;
	using MetaListT<Element>::set_thread_block_size;
	using MetaListT<Element>::resize;
	using MetaListT<Element>::load;
	using MetaListT<Element>::set_load;
	using MetaListT<Element>::start_pos;
	using MetaListT<Element>::end_pos;
	using MetaListT<Element>::data;
	using MetaListT<Element>::data_value;
	using MetaListT<Element>::data_label;
	using MetaListT<Element>::at;
	using MetaListT<Element>::set;
	using MetaListT<Element>::print;
	using MetaListT<Element>::print_binary;
	using MetaListT<Element>::begin;
	using MetaListT<Element>::end;
	using MetaListT<Element>::zero;
	using MetaListT<Element>::erase;

private:
	// disable the empty constructor. So you have to specify a rough size of the list. This is for optimisations reasons.
	Parallel_List_FullElement_T() : MetaListT<Element>() {};

public:


	// somehow make such flags configurable
	constexpr static bool USE_STD_SORT = false;

	/// multithreaded constructor
	constexpr explicit Parallel_List_FullElement_T(const size_t size,
	                                               const uint32_t threads=1) noexcept :
	   MetaListT<Element>(size, threads)
	{}



#ifdef  _OPENMP
	/// src: https://www.cs.rutgers.edu/~venugopa/parallel_summer2012/bitonic_openmp.html
	/// needs to be benchmarked against std::sort()
	/// \param start 	start position in `seq` to start from
	/// \param length 	number of elements to sort
	/// \param seq 		array to sort
	/// \param flag 	sorting direction.
	void bitonic_sort_par(int start, int length, int *seq, int flag) {
		int i;
		int split_length;

		if (length == 1)
			return;

		if (length % 2 !=0 ) {
			printf("The length of a (sub)sequence is not divided by 2.\n");
			exit(0);
		}

		split_length = length / 2;

		// bitonic split
#pragma omp parallel for default(none) shared(seq, flag, start, split_length) private(i)
		for (i = start; i < start + split_length; i++) {
			if (flag == 1) {
				if (seq[i] > seq[i + split_length])
					std::swap(seq[i], seq[i + split_length]);
			}
			else {
				if (seq[i] < seq[i + split_length])
					std::swap(seq[i], seq[i + split_length]);
			}
		}


		// if (split_length > m) {
		// m is the size of sub part-> n/numThreads
		if (threads > 2) {
			bitonic_sort_par(start, split_length, seq, flag);
			bitonic_sort_par(start + split_length, split_length, seq, flag);
		}
	}
#endif

	constexpr void sort() noexcept {
		ASSERT(0);
	}

	/// generic hash function
	/// \tparam Hash a function/lambda which hashes an element of the list down to a comparable number
	/// 			[](const Label &l) -> uint32_t { return l.data().data()[0]; }
	/// \param s	start position to start sorting
	/// \param e	upper bound of the sorting algorithm
	/// \param hash
	template<typename Hash>
	void sort(LoadType s, LoadType e, Hash hash) noexcept {
		ska_sort(__data.begin() + s,
				 __data.begin() + e,
				 hash
		);
	}

	/// the same as the function above but it will sort on all coordinates.`
	/// \tparam Hash
	/// \param tid thread id
	/// \param hash hash function
	template<typename Hash>
	void sort(uint32_t tid, Hash hash, const size_t size=-1) noexcept {
		ASSERT(tid < __threads);

		const auto e = size == size_t(-1) ? end_pos(tid) : size;
		sort(start_pos(tid), e, hash);
	}

	/// \param i lower coordinate in the label used as the sorting index
	/// \param j upper   .....
	void sort_level(const uint32_t i, const uint32_t j) noexcept {
		ASSERT(i < j);
		using T = LabelContainerType;

		//constexpr uint64_t upper = T::round_down_to_limb(j);
		constexpr uint64_t lower = T::round_down_to_limb(i);
		const uint64_t mask = T::higher_mask(i) & T::lower_mask(j);
		if constexpr (USE_STD_SORT) {
			std::sort(__data.begin(),
					  __data.end(),
					  [lower, mask](const auto &e1, const auto &e2) {
						return e1.get_label().data().is_lower_simple2(e2.get_label().data(), lower, mask);
					  }
			);
		} else {
			ska_sort(__data.begin(),
					 __data.end(),
					 [lower, mask](const Element &e) {
					   return e.get_label_container_ptr()[lower] & mask;
					 }
			);
		}

	}

	/// \param i lower coordinate in the label used as the sorting index
	/// \param j upper   .....
	/// \param tid thread id
	void sort_level(const uint32_t i, const uint32_t j, const uint32_t tid) noexcept {
		ASSERT(i < j);
		using T = LabelContainerType;

		const uint64_t lower = T::round_down_to_limb(i);
		//const uint64_t upper = T::round_down_to_limb(j);
		//ASSERT(lower == upper); // TODO

		const uint64_t mask = T::higher_mask(i) & T::lower_mask(j);

		if constexpr (USE_STD_SORT) {
			std::sort(__data.begin() + start_pos(tid),
					  __data.begin() + end_pos(tid),
					  [lower, mask](const auto &e1, const auto &e2) {
						return e1.get_label().data().is_lower_simple2(e2.get_label().data(), lower, mask);
					  }
			);
		} else {
			ska_sort(__data.begin() + start_pos(tid),
					 __data.begin() + end_pos(tid),
					 [lower, mask](const Element &e) {
					   return e.get_label_container_ptr()[lower] & mask;
					 }
			);
		}

	}

	/// does what the name suggest.
	/// \param e element we want to search
	/// \param k_lower lower coordinate on which the element must be equal
	/// \param k_higher higher coordinate the elements must be equal
	/// \return the position of the first (lower) element which is equal to e. -1 if nothing found
	size_t search_level(const Element &e,
						const uint32_t k_lower,
						const uint32_t k_higher,
						bool sort=false) noexcept {
		auto r = std::find_if(__data.begin(),
							  __data.end(),
							  [&e, k_lower, k_higher](const Element &c) {
								return e.is_equal(c, k_lower, k_higher);
							  }
		);

		const auto dist = distance(__data.begin(), r);

		if (r == __data.end())
			return -1; // nothing found

		if (!__data[dist].is_equal(e, k_lower, k_higher))
			return -1;

		return dist;
	}

	/// \param e	element to search for
	/// \return	a tuple indicating the start and end indices within the list. start == end == load indicating nothing found,
	std::pair<uint64_t, uint64_t> search_boundaries(const Element &e,
													const uint32_t k_lower,
													const uint32_t k_higher,
													const uint32_t tid) noexcept {
		uint64_t end_index;
		uint64_t start_index = search_level(e, k_lower, k_higher, tid);
		if (start_index == uint64_t(-1))
			return std::pair<uint64_t, uint64_t>(-1, -1);

		// get the upper index
		end_index = start_index + 1;
		while (end_index < size() && (__data[start_index].is_equal(__data[end_index], k_lower, k_higher)))
			end_index += 1;

		return std::pair<uint64_t, uint64_t>(start_index, end_index);
	}

	/// zero out the i-th element.
	/// \param i
	void zero_element(size_t i) noexcept {
		ASSERT(i < size());
		__data[i].zero();
	}

	/// set L[load] = e1 + e2 and updated the load factor. Note this is usefull, because every thread can so maintain
	/// its own list size.
	/// \param e1	first element
	/// \param e2	second element to add
	/// \param load load factor = number of elements currently in the list.
	/// \param tid thread number
	void add_and_append(const Element &e1,
						const Element &e2,
						LoadType &load,
						const uint32_t tid) noexcept {
		ASSERT(tid < __threads);

		if (load >= thread_block_size())
			return;

		Element::add(__data[start_pos(tid) + load], e1, e2);
		load += 1;
	}

	/// same function as above, but the label and value split into two separate function parameters.
	/// This is needed for lists which do not save an element in one variable.
	/// \param l1 label of the first element
	/// \param v1 value of the first element
	/// \param l2 label of the second element
	/// \param v2 value of the second element
	/// \param load current load of the list
	/// \param tid thread id inserting this element.
	void add_and_append(const LabelType &l1, const ValueType &v1,
						const LabelType &l2, const ValueType &v2,
						LoadType &load, const uint32_t tid) noexcept {
		ASSERT(tid < threads);

		if (load >= thread_block_size())
			return;

		ValueType::add(data_value(start_pos(tid) + load), v1, v2);
		LabelType::add(data_label(start_pos(tid) + load), l1, l2);
		load += 1;
	}


	// returning the start pointer for each thread
	inline LabelType* start_label_ptr(const uint32_t tid) noexcept { return (LabelType *)(((uint64_t)&__data[start_pos(tid)]) + ValueBytes); };

};


///
/// \tparam Element
/// \param out
/// \param obj
/// \return
template<class Element>
std::ostream &operator<<(std::ostream &out, const Parallel_List_FullElement_T<Element> &obj) {
	for (uint64_t i = 0; i < obj.size(); ++i) {
		out << i << " " << obj.data(i) << std::flush;
	}

	return out;
}

#endif//DECODING_PARALLEL_FULL_H