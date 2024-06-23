#ifndef DECODING_LIST_PARALLEL_FULL_H
#define DECODING_LIST_PARALLEL_FULL_H

#include <algorithm>// search/find routines
#include <cassert>
#include <iterator>
#include <vector>// main data container

#include "list/common.h"
#include "sort/sort.h"


/// same the class
///		`Parallel_List_T`
/// with the big difference that Elements = <Label, Value> are stored within the same const_array.
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
	using typename MetaListT<Element>::ValueLimbType;
	using typename MetaListT<Element>::LabelLimbType;
	using typename MetaListT<Element>::ValueContainerType;
	using typename MetaListT<Element>::LabelContainerType;
	using typename MetaListT<Element>::ValueDataType;
	using typename MetaListT<Element>::LabelDataType;
	using typename MetaListT<Element>::MatrixType;
	using LoadType = size_t;

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
	using MetaListT<Element>::random;
	using MetaListT<Element>::is_correct;

private:
	// disable the empty constructor. So you have to specify a rough size of the list. This is for optimisations reasons.
	Parallel_List_FullElement_T() : MetaListT<Element>(){};
	
public:
	// somehow make such flags configurable
	constexpr static bool USE_STD_SORT = true;

	/// multithreaded constructor
	constexpr explicit Parallel_List_FullElement_T(const size_t _size,
	                                               const uint32_t threads = 1) noexcept :
	   MetaListT<Element>(_size, threads) {}

	constexpr void random() noexcept {
		for (size_t i = 0; i < size(); ++i) {
			__data[i].random();
		}
	}

	constexpr void random(const uint32_t tid) noexcept {
		const size_t start = start_pos(tid);
		const size_t end = end_pos(tid);
		for (size_t i = start; i < end; ++i) {
			__data[i].random();
		}
	}

	constexpr void random(MatrixType &m) noexcept {
		for (size_t i = 0; i < size(); ++i) {
			__data[i].random(&m);
		}
	}

	///
	constexpr void sort(const size_t s=0, const size_t e=size()) noexcept {
		ASSERT(e <= size());
		std::sort(begin() + s, begin() + e);
	}

	/// generic hash function
	/// \tparam Hash a function/lambda which hashes an element of the list down to a comparable number
	/// 			[](const Label &l) -> uint32_t { return l.data().data()[0]; }
	/// \param s	start position to start sorting
	/// \param e	upper bound of the sorting algorithm
	/// \param hash
	template<typename Hash>
	constexpr void sort(Hash &hash, const size_t s, const size_t e) noexcept {
		ASSERT(e <= size());
		ska_sort(__data.begin() + s,
		         __data.begin() + e,
		         hash);
	}

	/// \param i lower coordinate in the label used as the sorting index
	/// \param j upper   .....
	/// \param tid thread id
	void sort_level(const uint32_t i, const uint32_t j) noexcept {
		ASSERT(i < j);
		using T = LabelContainerType;
		using Limb = LabelLimbType;

		const uint64_t lower = T::round_down_to_limb(i);
		const uint64_t upper = T::round_down_to_limb(j);

		if (lower == upper) {
			const uint64_t mask = T::higher_mask(i) & T::lower_mask(j);

			if constexpr (USE_STD_SORT) {
				std::sort(__data.begin(),
				          __data.end(),
				          [lower, mask](const auto &e1, const auto &e2) {
					          return (e1.label_ptr(lower) & mask) < (e2.label_ptr(lower) & mask);
				          });
			} else {
				ska_sort(__data.begin(),
				         __data.end(),
				         [lower, mask](const Element &e) {
					         return e.label_ptr(lower) & mask;
				         });
			}
		} else {
			const Limb j_mask = T::lower_mask(j);
			const Limb i_mask = T::higher_mask(i);
			const uint32_t i_shift = i % (sizeof(Limb) * 8u);
			const uint32_t j_shift = (sizeof(Limb) * 8u) - i_shift;


			if constexpr (USE_STD_SORT) {
				std::sort(__data.begin(),
				          __data.end(),
				          [lower, upper, i_mask, j_mask, i_shift, j_shift](const auto &e1, const auto &e2) {
					          const Limb data1 = ((e1.label_ptr(lower) & i_mask) >> i_shift) ^
					                             ((e1.label_ptr(upper) & j_mask) >> j_shift);
					          const Limb data2 = ((e2.label_ptr(lower) & i_mask) >> i_shift) ^
					                             ((e2.label_ptr(upper) & j_mask) >> j_shift);

					          return data1 < data2;
				          });
			} else {
				ska_sort(__data.begin(),
				         __data.end(),
				         [lower, upper, i_mask, j_mask, i_shift, j_shift](const Element &e) {
					         return ((e.label_ptr(lower) & i_mask) >> i_shift) ^
					                ((e.label_ptr(upper) & j_mask) >> j_shift);
				         });
			}
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
	                    bool sort = false) noexcept {
		if (sort) {
			sort_level(k_lower, k_higher);
		}

		auto r = std::find_if(__data.begin(),
		                      __data.end(),
		                      [&e, k_lower, k_higher](const Element &c) {
			                      return e.is_equal(c, k_lower, k_higher);
		                      });

		const auto dist = distance(__data.begin(), r);

		if (r == __data.end()) {
			return -1;// nothing found
		}

		if (!__data[dist].is_equal(e, k_lower, k_higher)) {
			return -1;
		}

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
		if (start_index == uint64_t(-1)) {
			return std::pair<uint64_t, uint64_t>(-1, -1);
		}

		// get the upper index
		end_index = start_index + 1;
		while (end_index < size() && (__data[start_index].is_equal(__data[end_index], k_lower, k_higher)))
			end_index += 1;

		return std::pair<uint64_t, uint64_t>(start_index, end_index);
	}

	/// zero out the i-th element.
	/// \param i
	void zero(size_t i) noexcept {
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
};

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
