#ifndef DECODING_LIST_H
#define DECODING_LIST_H

#include "list/common.h"
#include "list/enumeration/enumeration.h"
#include "list/parallel.h"
#include "list/parallel_full.h"
#include "list/parallel_index.h"
#include "list/simple.h"
#include "list/simple_limb.h"
#include "matrix/matrix.h"

#include <algorithm>// search/find routines
#include <cassert>
#include <iterator>
#include <vector>// main data container


/// Mother of all lists
/// \tparam Element
template<class Element>
class List_T : public MetaListT<Element> {
public:
	/// needed typedefs
	typedef typename MetaListT<Element>::ElementType ElementType;
	typedef typename MetaListT<Element>::ValueType ValueType;
	typedef typename MetaListT<Element>::LabelType LabelType;
	typedef typename MetaListT<Element>::ValueLimbType ValueLimbType;
	typedef typename MetaListT<Element>::LabelLimbType LabelLimbType;
	typedef typename MetaListT<Element>::ValueContainerType ValueContainerType;
	typedef typename MetaListT<Element>::LabelContainerType LabelContainerType;
	typedef typename MetaListT<Element>::ValueDataType ValueDataType;
	typedef typename MetaListT<Element>::LabelDataType LabelDataType;
	typedef typename MetaListT<Element>::MatrixType MatrixType;

	using typename MetaListT<Element>::LoadType;
	using List = List_T<Element>;

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
	using MetaListT<Element>::start_pos;
	using MetaListT<Element>::end_pos;
	using MetaListT<Element>::set_threads;
	using MetaListT<Element>::thread_block_size;
	using MetaListT<Element>::set_thread_block_size;
	using MetaListT<Element>::resize;
	using MetaListT<Element>::load;
	using MetaListT<Element>::set_load;
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
	using MetaListT<Element>::insert;
	using MetaListT<Element>::is_correct;
	using MetaListT<Element>::is_sorted;

	///
	/// \param e element to search for
	/// \param k_lower lower limit
	/// \param k_higher higher limit
	/// \return the position within the
	constexpr inline uint64_t search_level_binary_simple(
			const Element &e,
	        const uint64_t k_lower,
	        const uint64_t k_higher) const noexcept {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t mask = T::higher_mask(k_lower) & T::lower_mask(k_higher);
		ASSERT(lower == upper);

		auto r = std::lower_bound(__data.begin(),
				__data.begin() + load(), e,
		                          [lower, mask](const Element &c1, const Element &c2) {
			                          return c1.label.is_lower_simple2(c2.label, lower, mask);
		                          });

		const auto dist = distance(__data.begin(), r);
		if (r == __data.begin() + load()) { return -1; }
		if (!__data[dist].is_equal(e, k_lower, k_higher)) return -1;
		return dist;
	}

	/// custom written binary search. Idea Taken from `https://academy.realm.io/posts/how-we-beat-cpp-stl-binary-search/`
	/// simple means: k_lower, k_upper must be in the same limb
	/// return -1 if nothing found
	constexpr inline size_t search_level_binary_custom_simple(const Element &e,
	                                                          const uint64_t k_lower,
	                                                          const uint64_t k_higher) const noexcept {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t mask = T::higher_mask(k_lower) & T::lower_mask(k_higher);
		ASSERT(lower == upper);

		size_t size = load();
		size_t low = 0;

		LabelContainerType v;
		const LabelContainerType s = e.label;

		while (size > 0) {
			size_t half = size / 2;
			size_t other_half = size - half;
			size_t probe = low + half;
			size_t other_low = low + other_half;
			v = __data[probe].label;
			size = half;
			low = v.is_lower_simple2(s, lower, mask) ? other_low : low;
		}

		return (low != load()) ? low : -1ul;
	}

private:
	// disable the empty constructor. So you have to specify a rough size of the list.
	// This is for optimisations reasons.
	List_T() : MetaListT<Element>(){};

public:
	/// Base constructor. The empty one is disabled.
	/// \param nr_element of elements in the list
	/// \param threads  number of threads, which work on parallel on the list
	constexpr List_T(const size_t nr_element,
	                 const uint32_t threads = 1) noexcept
	    : MetaListT<Element>(nr_element, threads) {}


	/// Andres Code
	constexpr void static odl_merge(std::vector<std::pair<uint64_t, uint64_t>> &target,
	                                const List_T &L1,
	                                const List_T &L2,
	                                int klower = 0,
	                                int kupper = -1) noexcept {
		if (kupper == -1 && L1.get_load() > 0)
			kupper = L1[0].label_size();
		uint64_t i = 0, j = 0;
		target.resize(0);
		while (i < L1.get_load() && j < L2.get_load()) {
			if (L2[j].is_greater(L1[i], klower, kupper))
				i++;

			else if (L1[i].is_greater(L2[j], klower, kupper))
				j++;

			else {
				uint64_t i_max, j_max;
				// if elements are equal find max index in each list, such that they remain equal
				for (i_max = i + 1; i_max < L1.get_load() && L1[i].is_equal(L1[i_max], klower, kupper); i_max++) {}
				for (j_max = j + 1; j_max < L2.get_load() && L2[j].is_equal(L2[j_max], klower, kupper); j_max++) {}

				// store each matching tuple
				int jprev = j;
				for (; i < i_max; ++i) {
					for (j = jprev; j < j_max; ++j) {
						std::pair<uint64_t, uint64_t> a;
						a.first = L1[i].get_value();
						a.second = L2[j].get_value();
						target.push_back(a);
					}
				}
			}
		}
	}

	/// TODO: rename to random
	/// generate/initialize this list as a random base list.
	/// for each element a new value is randomly choosen and the label is calculated by a matrix-vector-multiplication.
	/// \param k amount of 'Elements' to add to the list.
	/// \param m base matrix to calculate the 'Labels' corresponding to a 'Value'
	constexpr void generate_base_random(const uint64_t k,
	                                    const MatrixType &m) noexcept {
		/// somehow we should be sure that the list is big enough
		__data.resize(k);
		set_size(k);
		set_load(0);

		for (size_t i = 0; i < k; ++i) {
			// by default this creates a complete random 'Element' with 'Value' coordinates \in \[0,1\]
			Element e{};
			e.random(m);
			append(e);
		}
	}


	/// \param level				current lvl within the tree.
	/// \param level_translation
	constexpr void sort_level(const uint32_t level,
	                          const std::vector<uint64_t> &level_translation) noexcept {
		uint64_t k_lower, k_higher;
		translate_level(&k_lower, &k_higher, level, level_translation);
		return sort_level(k_lower, k_higher);
	}

	/// sort the list
	///
	constexpr void sort_level(const uint32_t k_lower,
	                          const uint32_t k_higher) noexcept {
		std::sort(__data.begin(), __data.begin() + load(),
		          [k_lower, k_higher](const auto &e1, const auto &e2) {

#if !defined(SORT_INCREASING_ORDER)
			          return e1.is_lower(e2, k_lower, k_higher);
#else
			        return e1.is_greater(e2, k_lower, k_higher);
#endif
		          });

		ASSERT(is_sorted(k_lower, k_higher));
	}

	/// NOTE: this does not search the FULL list, only each segmeent
	/// \param k_lower lower dimendion to sort on (inclusive)
	/// \param k_higher upper dimesnsion to sort (not included)
	/// \param tid thread id
	constexpr void sort_level(const uint32_t k_lower,
	                          const uint32_t k_higher,
	                          const uint32_t tid) noexcept {
		std::sort(__data.begin() + start_pos(tid), __data.begin() + end_pos(tid),
		          [k_lower, k_higher](const auto &e1, const auto &e2) {

#if !defined(SORT_INCREASING_ORDER)
			          return e1.is_lower(e2, k_lower, k_higher);
#else
			        return e1.is_greater(e2, k_lower, k_higher);
#endif
		          });

		ASSERT(is_sorted(k_lower, k_higher));
	}

	/// sort the list. only valid in the binary case
	constexpr inline void sort_level_binary(const uint32_t k_lower,
	                                        const uint32_t k_higher) noexcept {
		ASSERT(load() > 0);

		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);

		// choose the optimal implementation.
		if (lower == upper) {
			sort_level_sim_binary(k_lower, k_higher);
		} else {
			sort_level_ext_binary(k_lower, k_higher);
		}

		ASSERT(is_sorted(k_lower, k_higher));
	}

private:
	/// IMPORTANT: DO NOT CALL THIS FUNCTION directly. Use `sort_level` instead.
	/// special implementation of the sorting function using specialised compare
	/// functions of the `BinaryContainer` class, which uses precomputed masks.
	constexpr inline void sort_level_ext_binary(const uint64_t k_lower,
	                                            const uint64_t k_higher) noexcept {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t lmask = T::higher_mask(k_lower);
		const uint64_t umask = T::lower_mask(k_higher);

		std::sort(__data.begin(), __data.begin() + load,
		          [lower, upper, lmask, umask](const auto &e1, const auto &e2) {
#if !defined(SORT_INCREASING_ORDER)
			          return e1.get_label().data().is_lower_ext2(e2.get_label().data(), lower, upper, lmask, umask);
#else
			        return e1.get_label().data().is_greater_ext2(e2.get_label().data(), lower, upper, lmask, umask);
#endif
		          });
	}

	/// IMPORTANT: DO NOT CALL THIS FUNCTION directly. Use `sort_level` instead.
	/// special implementation of the sort function using highly optimized special compare routines from the data class
	/// `BinaryContainer` if one knows that k_lower, k_higher are in the same limb.
	constexpr inline void sort_level_sim_binary(const uint32_t k_lower,
	                                            const uint32_t k_higher) noexcept {
		static_assert(Element::binary());
		using T = LabelContainerType;

		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);
		ASSERT(lower == upper);

		const uint64_t mask = T::higher_mask(k_lower) & T::lower_mask(k_higher);
		std::sort(__data.begin(), __data.begin() + load,
		          [lower, mask](const auto &e1, const auto &e2) {
#if !defined(SORT_INCREASING_ORDER)
			          return e1.get_label().is_lower_simple2(e2.get_label(), lower, mask);
#else
			        return e1.get_label().is_greater_simple2(e2.get_label(), lower, mask);
#endif
		          });
	}

public:
	// implements a binary search on the given data.
	// if the boolean flag `sort` is set to true, the underlying list is sorted.
	// USAGE:
	// can be found: "tests/binary/list.h TEST(SerchBinary, Simple) "
	constexpr inline size_t search_level_binary(const Element &e,
	                                            const uint32_t k_lower,
	                                            const uint32_t k_higher,
	                                            const bool sort = false) noexcept {
		static_assert(Element::binary());
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);

		if (sort) {
			sort_level(k_lower, k_higher);
		}

		if (lower == upper)
			return search_level_binary_simple(e, k_lower, k_higher);
		else
			return search_level_binary_extended(e, k_lower, k_higher);
	}

private:
	///
	/// \param e
	/// \param k_lower
	/// \param k_higher
	/// \return
	constexpr inline uint64_t search_level_binary_extended(const Element &e,
	                                                       const uint64_t k_lower,
	                                                       const uint64_t k_higher) const noexcept {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t lmask = T::higher_mask(k_lower);
		const uint64_t umask = T::lower_mask(k_higher);
		ASSERT(lower != upper);

		auto r = std::lower_bound(__data.begin(), __data.begin() + load(), e,
		                          [lower, upper, lmask, umask](const Element &c1, const Element &c2) {
			                          return c1.label.is_lower_ext2(c2.label, lower, upper, lmask, umask);
		                          });

		const auto dist = distance(__data.begin(), r);
		if (r == __data.begin() + load()) { return -1; }
		if (!__data[dist].is_equal(e, k_lower, k_higher)) return -1;
		return dist;
	}

public:
	///
	/// \param e
	/// \param k_lower
	/// \param k_higher
	/// \param sort
	/// \return
	constexpr inline uint64_t search_level_binary_custom(const Element &e,
	                                                     const uint64_t k_lower,
	                                                     const uint64_t k_higher,
	                                                     const bool sort = false) noexcept {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);

		if (sort) {
			sort_level(k_lower, k_higher);
		}

		if (lower == upper)
			return search_level_binary_custom_simple(e, k_lower, k_higher);
		else
			return search_level_binary_custom_extended(e, k_lower, k_higher);
	}

private:
	///
	/// \param e
	/// \param k_lower
	/// \param k_higher
	/// \return
	constexpr inline uint64_t search_level_binary_custom_extended(const Element &e,
	                                                              const uint64_t k_lower,
	                                                              const uint64_t k_higher) const noexcept {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t lmask = T::higher_mask(k_lower);
		const uint64_t umask = T::lower_mask(k_higher);
		ASSERT(lower != upper);

		size_t size = load();
		size_t low = 0;

		LabelContainerType v;
		const LabelContainerType s = e.label;

		while (size > 0) {
			size_t half = size / 2;
			size_t other_half = size - half;
			size_t probe = low + half;
			size_t other_low = low + other_half;
			v = __data[probe].label;
			size = half;
			low = v.is_lower_ext2(s, lower, upper, lmask, umask) ? other_low : low;
		}
		return (low != load()) ? low : -1;
	}

public:
	/// does what the name suggest.
	/// \param e element we want to search
	/// \param k_lower lower coordinate on which the element must be equal
	/// \param k_higher higher coordinate the elements must be equal
	/// \return the position of the first (lower) element which is equal to e. -1 if nothing found
	constexpr size_t search_level(const Element &e,
	                              const uint64_t k_lower,
	                              const uint64_t k_higher,
	                              bool sort = false) noexcept {
		if constexpr (Element::binary()) {
			return search_level_binary(e, k_lower, k_higher, sort);
		} else {
			if (sort) {
				sort_level(k_lower, k_higher);
			}

			auto r = std::lower_bound(__data.begin(), __data.begin() + load(), e,
			                            [k_lower, k_higher](const Element &a1, const Element &a2) {
				// return e.is_equal(c, k_lower, k_higher);
				return a1.is_lower(a2, k_lower, k_higher);
			});

			const auto dist = std::distance(__data.begin(), r);

			if (r == __data.begin() + load()) {
				return -1;// nothing found
			}

			if (!__data[dist].is_equal(e, k_lower, k_higher)) {
				return -1;
			}

			return dist;
		}
	}

	/// NOTE: this is a stupid linear search
	/// \param e
	/// \return
	constexpr size_t search(const Element &e) {
		for (size_t i = 0; i < load(); ++i) {
			if (__data[i] == e) {
				return i;
			}
		}
		return -1u;
	}

	/// \param e
	/// \return	a tuple indicating the start and end indices within the list.
	// 		start = end = load indicating nothing found,
	constexpr std::pair<size_t, size_t>
	search_boundaries(const Element &e,
	                  const uint64_t k_lower,
	                  const uint64_t k_higher) noexcept {
		uint64_t end_index;
		uint64_t start_index;
		if constexpr (!Element::binary()) {
			start_index = search_level(e, k_lower, k_higher);
		} else {
			start_index = search_level_binary(e, k_lower, k_higher);
		}


		if (start_index == uint64_t(-1)) {
			return std::pair<uint64_t, uint64_t>(load(), load());
		}

		// get the upper index
		end_index = start_index + 1;
		while (end_index < load() && (__data[start_index].is_equal(__data[end_index], k_lower, k_higher))) {
			end_index += 1;
		}

		return std::pair<size_t, size_t>{start_index, end_index};
	}

	///
	constexpr void sort() noexcept {
		std::sort(__data.begin(), __data.end());
	}

	/// appends the element e to the list. Note that the list keeps track of its load. So you dont have to do anyting.
	/// Note: if the list is full, every new element is silently discarded.
	/// \param e	Element to add
	constexpr void append(Element &e) noexcept {
		if (load() < size()) {
			__data[load()] = e;
		} else {
			__data.push_back(e);
		}

		set_load(load() + 1);
	}

	/// append e1+e2|full_length to list
	/// \param e1 first element.
	/// \param e2 second element
	/// \param norm filter out all elements above the norm
	/// \param sub: if true the e1-e2 will be stored, instead of e1+e2
	constexpr void add_and_append(const Element &e1,
	                              const Element &e2,
	                              const uint32_t norm = -1,
	                              const bool sub = false) noexcept {
		add_and_append(e1, e2, 0, LabelLENGTH, norm, sub);
	}

	/// Same as the function above, but with a `constexpr` size factor.
	/// This may allow further optimisations to the compiler. Additionally,
	/// this functions adds e1 and e2 together and daves the result.
	/// \tparam approx_size Size of the list
	/// \param e1 first element
	/// \param e2 second element
	/// \param norm norm of the element e1+e2. If the calculated norm is bigger than `norm` the element is discarded.
	/// \return
	template<const uint64_t approx_size>
	constexpr int add_and_append(const Element &e1,
	                             const Element &e2,
	                             const uint32_t norm = -1) noexcept {
		if (load() < approx_size) {
			Element::add(__data[load()], e1, e2, 0, LabelLENGTH, norm);
			__load += 1;
		}
		return approx_size - load();
		// ignore every element which could not be added to the list.
	}

	/// same as above, but the label is only added between k_lower and k_higher
	/// \param e1 first element to add
	/// \param e2 second element
	/// \param k_lower lower dimension to add the label on
	/// \param k_higher higher dimension to add the label on
	/// \param norm filter norm. If the norm of the resulting element is higher than `norm` it will be discarded.
	constexpr void add_and_append(const Element &e1,
	                              const Element &e2,
	                              const uint32_t k_lower,
	                              const uint32_t k_higher,
	                              const uint32_t norm = -1,
	                              const bool sub = false) noexcept {
		if (load() < this->size()) {
			auto b = Element::add(__data[load()], e1, e2, k_lower, k_higher, norm);
			// 'add' returns true if a overflow, over the given norm occurred. This means that at least coordinate 'r'
			// exists for which it holds: |data[load].value[r]| >= norm
			if (b == true) {
				return;
			}
		} else {
			Element t{};
			bool b = false;
			if (sub) {
				// TODO extend the api for the norm factor
				Element::sub(t, e1, e2);
			} else {
				b = Element::add(t, e1, e2, k_lower, k_higher, norm);
			}

			if (b) {
				return;
			}

			// this __MUST__ be a copy.
			__data.push_back(t);
			__size += 1;
		}

		// we do not increase the 'load' of our internal data structure if one of the add functions above returns true.
		set_load(load() + 1);
	}

private:
};

#endif//DECODING_LIST_H
