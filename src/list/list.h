#ifndef DECODING_LIST_H
#define DECODING_LIST_H

#include "list/common.h"
#include "list/simple.h"
#include "list/simple_limb.h"
#include "list/parallel.h"
#include "list/parallel_full.h"
#include "list/parallel_index.h"

#include "combinations.h"

#include <iterator>
#include <vector>           // main data container
#include <algorithm>        // search/find routines
#include <cassert>

#ifdef DUSE_FPLLL
// external includes
#include "fplll/nr/matrix.h"
#endif



/// Mother of all lists
/// \tparam Element
template<class Element>
#if __cplusplus > 201709L
requires ListElementAble<Element>
#endif
class List_T {
private:
	// disable the empty constructor. So you have to specify a rough size of the list.
	// This is for optimisations reasons.
	List_T() : load(0), threads(1) {};

public:
	typedef Element ElementType;
	typedef typename Element::ValueType ValueType;
	typedef typename Element::LabelType LabelType;

	typedef typename Element::ValueType::ContainerLimbType ValueContainerLimbType;
	typedef typename Element::LabelType::ContainerLimbType LabelContainerLimbType;

	typedef typename Element::ValueContainerType ValueContainerType;
	typedef typename Element::LabelContainerType LabelContainerType;

	typedef typename Element::ValueDataType ValueDataType;
	typedef typename Element::LabelDataType LabelDataType;

	typedef typename Element::MatrixType MatrixType;

	// internal data types lengths
	constexpr static uint32_t ValueLENGTH = ValueType::LENGTH;
	constexpr static uint32_t LabelLENGTH = LabelType::LENGTH;

	/// Base constructor. The empty one is disabled.
	/// \param number of elements in the list
	constexpr List_T(const size_t nr_element, const uint32_t threads=1) noexcept
	    : load(0), threads(threads) {
		__data.resize(nr_element);
	}

	/// \return size the size of the list
	[[nodiscard]] constexpr size_t size() const noexcept { return __data.size(); }
	/// \return the number of elements each thread enumerates
	[[nodiscard]] constexpr size_t size(const uint32_t tid) const noexcept { return __data.size(); } // TODO not correct

	/// set/get the load factor
	[[nodiscard]] size_t get_load() const noexcept { return load; }
	void set_load(size_t l) noexcept { load = l; }

	/// returning the range in which one thread is allowed to operate
	[[nodiscard]] constexpr inline size_t start_pos(const uint32_t tid) const noexcept { return tid*(__data.size()/threads); };
	[[nodiscard]] constexpr inline size_t end_pos(const uint32_t tid) const noexcept { return( tid+1)*(__data.size()/threads); };

	/// Get a const pointer. Sometimes useful if one ones to tell the kernel how to access memory.
	constexpr inline auto* data() noexcept{ return __data.data(); }
	const auto* data() const noexcept { return __data.data(); }

	/// wrapper
	constexpr inline ValueType& data_value(const size_t i) noexcept {  ASSERT(i < __data.size()); return __data[i].get_value(); }
	constexpr inline const ValueType& data_value(const size_t i) const noexcept { ASSERT(i < __data.size()); return __data[i].get_value(); }
	constexpr inline LabelType& data_label(const size_t i) noexcept { ASSERT(i < __data.size()); return __data[i].get_label(); }
	constexpr inline const LabelType& data_label(const size_t i) const noexcept { ASSERT(i < __data.size()); return __data[i].get_label(); }

	/// operator overloading
	constexpr inline Element &at(const size_t i) noexcept {
		ASSERT(i < load);
		return __data[i];
	}
	constexpr inline const Element &at(const size_t i) const noexcept {
		ASSERT(i < load);
		return __data[i];
	}
	constexpr inline Element &operator[](const size_t i) noexcept {
		ASSERT(i < load);
		return __data[i];
	}
	constexpr inline const Element &operator[](const size_t i) const noexcept {
		ASSERT(i < load);
		return __data[i];
	}

	void set_data(Element &e, const uint64_t i) { ASSERT(i <size()); __data[i] = e; }

	/// print the `pos` element
	/// 	label between [label_k_lower, label_k_upper)
	/// 	value between [value_k_lower, value_k_upper)
	/// \param pos position of the element in the list to print
	/// \param value_k_lower inclusive
	/// \param value_k_higher exclusive
	/// \param label_k_lower inclusive
	/// \param label_k_higher exclusive
	void print_binary(const uint64_t pos,
					  const uint32_t value_k_lower,
					  const uint32_t value_k_higher,
					  const uint32_t label_k_lower,
					  const uint32_t label_k_higher) const noexcept {
		ASSERT(value_k_lower < value_k_higher);
		ASSERT(value_k_higher <= ValueLENGTH);
		ASSERT(label_k_lower < label_k_higher);
		ASSERT(label_k_higher <= LabelLENGTH);

		data_value(pos).print_binary(value_k_lower, value_k_higher);
		data_label(pos).print_binary(label_k_lower, label_k_higher);
	}

	/// print the `pos` element
	/// 	label between [label_k_lower, label_k_upper)
	/// 	value between [value_k_lower, value_k_upper)
	/// \param pos position of the element in the list to print
	/// \param value_k_lower inclusive
	/// \param value_k_higher exclusive
	/// \param label_k_lower inclusive
	/// \param label_k_higher exclusive
	void print(const uint64_t pos,
			   const uint32_t value_k_lower,
			   const uint32_t value_k_higher,
			   const uint32_t label_k_lower,
			   const uint32_t label_k_higher) const noexcept {
		ASSERT(value_k_lower < value_k_higher);
		ASSERT(value_k_higher <= ValueLENGTH);
		ASSERT(label_k_lower < label_k_higher);
		ASSERT(label_k_higher <= LabelLENGTH);

		data_value(pos).print(value_k_lower, value_k_higher);
		data_label(pos).print(label_k_lower, label_k_higher);
	}

	/// print the element between [start, end) s.t.:
	/// 	label between [label_k_lower, label_k_upper)
	/// 	value between [value_k_lower, value_k_upper)
	/// \param pos position of the element in the list to print
	/// \param value_k_lower inclusive
	/// \param value_k_higher exclusive
	/// \param label_k_lower inclusive
	/// \param label_k_higher exclusive
	void print(const uint32_t value_k_lower,
			   const uint32_t value_k_higher,
			   const uint32_t label_k_lower,
			   const uint32_t label_k_higher,
			   const size_t start,
			   const size_t end) const noexcept {
		ASSERT(start < end);
		ASSERT(end <= __data.size());
		ASSERT(value_k_lower < value_k_higher);
		ASSERT(value_k_higher <= ValueLENGTH);
		ASSERT(label_k_lower < label_k_higher);
		ASSERT(label_k_higher <= LabelLENGTH);

		for (size_t i = start; i < end; ++i) {
			print(i, value_k_lower, value_k_higher,
				  label_k_lower, label_k_higher);
		}
	}

	/// print the element binary between [start, end) s.t.:
	/// 	label between [label_k_lower, label_k_upper)
	/// 	value between [value_k_lower, value_k_upper)
	/// \param pos position of the element in the list to print
	/// \param value_k_lower inclusive
	/// \param value_k_higher exclusive
	/// \param label_k_lower inclusive
	/// \param label_k_higher exclusive
	void print_binary(const uint32_t value_k_lower,
					  const uint32_t value_k_higher,
					  const uint32_t label_k_lower,
					  const uint32_t label_k_higher,
					  const size_t start,
					  const size_t end) const noexcept {
		ASSERT(start < end);
		ASSERT(end <= __data.size());
		ASSERT(value_k_lower < value_k_higher);
		ASSERT(value_k_higher <= ValueLENGTH);
		ASSERT(label_k_lower < label_k_higher);
		ASSERT(label_k_higher <= LabelLENGTH);

		for (size_t i = start; i < end; ++i) {
			print_binary(i, value_k_lower, value_k_higher,
						 label_k_lower, label_k_higher);
		}
	}

	/// checks if all elements in the list fullfil the equation: 	label == value*m
	/// \param m 		the matrix.
	/// \param rewrite 	if set to true, all labels within each element will we overwritten by the recalculated.
	/// \return 		true if ech element is correct.
	bool is_correct(const Matrix_T<MatrixType> &m, const bool rewrite=false) {
		bool ret = false;
		for (int i = 0; i < get_load(); ++i) {
			ret |= __data[i].is_correct(m, rewrite);
			if ((ret) && (!rewrite))
				return ret;
		}

		return ret;
	}

	/// Andres Code
	void static odl_merge(std::vector<std::pair<uint64_t,uint64_t>>&target, const List_T &L1, const List_T &L2, int klower = 0,
						  int kupper = -1) {
		if (kupper == -1 && L1.get_load() > 0)
			kupper = L1[0].label_size();
		uint64_t i = 0, j = 0;
		target.resize(0);
		while (i < L1.get_load() && j < L2.get_load()) {
			if (L2[j].is_greater(L1[i], klower,kupper))
				i++;

			else if (L1[i].is_greater(L2[j], klower,kupper))
				j++;

			else {
				uint64_t i_max, j_max;
				// if elements are equal find max index in each list, such that they remain equal
				for (i_max = i + 1; i_max < L1.get_load() && L1[i].is_equal(L1[i_max], klower,kupper); i_max++) {}
				for (j_max = j + 1; j_max < L2.get_load() && L2[j].is_equal(L2[j_max], klower,kupper); j_max++) {}

				// store each matching tuple
				int jprev = j;
				for (; i < i_max; ++i) {
					for (j = jprev; j < j_max; ++j) {
						std::pair<uint64_t,uint64_t > a;
						a.first=L1[i].get_value();
						a.second=L2[j].get_value();
						target.push_back(a);
					}

				}
			}
		}
	}

	/// generate/initialize this list as a random base list.
	/// for each element a new value is randomly choosen and the label is calculated by a matrix-vector-multiplication.
	/// \param k amount of 'Elements' to add to the list.
	/// \param m base matrix to calculate the 'Labels' corresponding to a 'Value'
	void generate_base_random(const uint64_t k, const Matrix_T<MatrixType> &m) {
		for (int i = 0; i < k; ++i) {
			// by default this creates a complete random 'Element' with 'Value' coordinates \in \[0,1\]
			Element e{};
			e.random(m);    // this 'random' function takes care of the vector-matrix multiplication.
			append(e);
		}
	}

	/// create a list of lexicographical ordered elements (where the values are lexicographical ordered).
	/// IMPORTANT; No special trick is applied, so every Element needs a fill Matrix-vector multiplication.
	/// \param number_of_elements
	/// \param ones 				hamming weight of the 'Value' of all elements
	/// \param m
	void generate_base_lex(const uint64_t number_of_elements, const uint64_t ones, const Matrix_T<MatrixType> &m) {
		ASSERT(internal_counter == 0 && "already initialised");
		Element e{}; e.zero();

		__data.resize(number_of_elements);
		load = number_of_elements;

		const uint64_t n = e.value_size();
		Combinations_Lexicographic<decltype(e.get_value().data().get_type())> c{n, ones};
		c.left_init(e.get_value().data().data().data());
		while(c.left_step(e.get_value().data().data().data()) != 0) {
			if (internal_counter >= size())
				return;

			new_vector_matrix_product<LabelType, ValueType, MatrixType>(e.get_label(), e.get_value(), m);

			__data[internal_counter] = e;
			internal_counter += 1;
		}
	}

	/// \param level				current lvl within the tree.
	/// \param level_translation
	void sort_level(const uint32_t level, const std::vector<uint64_t> &level_translation) {
		uint64_t k_lower, k_higher;
		translate_level(&k_lower, &k_higher, level, level_translation);
		return sort_level(k_lower, k_higher);
	}

	/// sort the list
	void sort_level(const uint32_t k_lower, const uint32_t k_higher) {
		std::sort(__data.begin(), __data.begin()+load,
				  [k_lower, k_higher](const auto &e1, const auto &e2) {

#if !defined(SORT_INCREASING_ORDER)
					return e1.is_lower(e2, k_lower, k_higher);
#else
					return e1.is_greater(e2, k_lower, k_higher);
#endif
				  }
		);

		ASSERT(is_sorted(k_lower, k_higher));
	}


	void sort_level(const uint32_t k_lower, const uint32_t k_higher, const uint32_t tid) {
		std::sort(__data.begin()+start_pos(tid), __data.begin()+end_pos(tid),
				  [k_lower, k_higher](const auto &e1, const auto &e2) {

#if !defined(SORT_INCREASING_ORDER)
					return e1.is_lower(e2, k_lower, k_higher);
#else
					return e1.is_greater(e2, k_lower, k_higher);
#endif
				  }
		);

		ASSERT(is_sorted(k_lower, k_higher));
	}

	/// sort the list. only valid in the binary case
	inline void sort_level_binary(const uint32_t k_lower, const uint32_t k_higher) {
		ASSERT(get_load() > 0);

		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);

		// choose the optimal implementation.
		if (lower == upper)
			sort_level_sim_binary(k_lower, k_higher);
		else
			sort_level_ext_binary(k_lower, k_higher);

		ASSERT(is_sorted(k_lower, k_higher));
	}

private:
	/// IMPORTANT: DO NOT CALL THIS FUNCTION directly. Use `sort_level` instead.
	/// special implementation of the sorting function using specialised compare functions of the `BinaryContainer` class
	/// which uses precomputed masks.
	inline void sort_level_ext_binary(const uint64_t k_lower, const uint64_t k_higher) {
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
				  }
		);

	}

	/// IMPORTANT: DO NOT CALL THIS FUNCTION directly. Use `sort_level` instead.
	/// special implementation of the sort function using highly optimized special compare routines from the data class
	/// `BinaryContainer` if one knows that k_lower, k_higher are in the same limb.
	inline void sort_level_sim_binary(const uint32_t k_lower, const uint32_t k_higher) {
		using T = LabelContainerType;

		const uint64_t lower = T::round_down_to_limb(k_lower);
		//const uint64_t upper = T::round_down_to_limb(k_higher);
		// TODO ASSERT(lower == upper);

		const uint64_t mask = T::higher_mask(k_lower) & T::lower_mask(k_higher);
		std::sort(__data.begin(), __data.begin() + load,
				  [lower, mask](const auto &e1, const auto &e2) {
#if !defined(SORT_INCREASING_ORDER)
					return e1.get_label().data().is_lower_simple2(e2.get_label().data(), lower, mask);
#else
					return e1.get_label().data().is_greater_simple2(e2.get_label().data(), lower, mask);
#endif
				  }
		);
	}

public:
	// implements a binary search on the given data.
	// if the boolean flag `sort` is set to true, the underlying list is sorted.
	// USAGE:
	// can be found: "tests/binary/list.h TEST(SerchBinary, Simple) "
	inline size_t search_level_binary(const Element &e, const uint32_t k_lower, const uint32_t k_higher,
									  const bool sort = false) {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);

		if (sort)
			sort_level(k_lower, k_higher);

		if (lower == upper)
			return search_level_binary_simple(e, k_lower, k_higher);
		else
			return search_level_binary_extended(e, k_lower, k_higher);
	}

private:
	///
	/// \param e element to search for
	/// \param k_lower lower limit
	/// \param k_higher higher limit
	/// \return the position within the
	inline uint64_t search_level_binary_simple(const Element &e, const uint64_t k_lower, const uint64_t k_higher) {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		//const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t mask = T::higher_mask(k_lower) & T::lower_mask(k_higher);
		// TODO ASSERT(lower == upper);

		auto r = std::lower_bound(__data.begin(), __data.begin() + load, e,
								  [lower, mask](const Element &c1, const Element &c2) {
									return c1.get_label().data().is_lower_simple2(c2.get_label().data(), lower, mask);
								  }
		);

		const auto dist = distance(__data.begin(), r);
		if (r == __data.begin() + load) { return -1; }
		if (!__data[dist].is_equal(e, k_lower, k_higher)) return -1;
		return dist;
	}

	///
	/// \param e
	/// \param k_lower
	/// \param k_higher
	/// \return
	inline uint64_t search_level_binary_extended(const Element &e, const uint64_t k_lower, const uint64_t k_higher) {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t lmask = T::higher_mask(k_lower);
		const uint64_t umask = T::lower_mask(k_higher);
		ASSERT(lower != upper);

		auto r = std::lower_bound(__data.begin(), __data.begin() + load, e,
								  [lower, upper, lmask, umask](const Element &c1, const Element &c2) {
									return c1.get_label().data().is_lower_ext2(c2.get_label().data(), lower, upper, lmask, umask);
								  }
		);

		const auto dist = distance(__data.begin(), r);
		if (r == __data.begin() + load) { return -1; }
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
	inline uint64_t search_level_binary_custom(const Element &e, const uint64_t k_lower, const uint64_t k_higher,
											   const bool sort = false) {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);

		if (sort)
			sort_level(k_lower, k_higher);

		if (lower == upper)
			return search_level_binary_custom_simple(e, k_lower, k_higher);
		else
			return search_level_binary_custom_extended(e, k_lower, k_higher);
	}

private:
	/// custom written binary search. Idea Taken from `https://academy.realm.io/posts/how-we-beat-cpp-stl-binary-search/`
	inline uint64_t search_level_binary_custom_simple(const Element &e, const uint64_t k_lower, const uint64_t k_higher) {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		//const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t mask = T::higher_mask(k_lower) & T::lower_mask(k_higher);
		// TODO  ASSERT(lower == upper);

		size_t size = load;
		size_t low = 0;

		LabelContainerType v;
		const LabelContainerType s = e.get_label().data();

		while (size > 0) {
			size_t half = size / 2;
			size_t other_half = size - half;
			size_t probe = low + half;
			size_t other_low = low + other_half;
			v = __data[probe].get_label().data();
			size = half;
			low = v.is_lower_simple2(s, lower, mask) ? other_low : low;
		}
		return low!=load ? low : -1;
	}

	///
	/// \param e
	/// \param k_lower
	/// \param k_higher
	/// \return
	inline uint64_t search_level_binary_custom_extended(const Element &e, const uint64_t k_lower, const uint64_t k_higher) {
		using T = LabelContainerType;
		const uint64_t lower = T::round_down_to_limb(k_lower);
		const uint64_t upper = T::round_down_to_limb(k_higher);
		const uint64_t lmask = T::higher_mask(k_lower);
		const uint64_t umask = T::lower_mask(k_higher);
		ASSERT(lower == upper);

		size_t size = load;
		size_t low = 0;

		LabelContainerType v;
		const LabelContainerType s = e.get_label().data();

		while (size > 0) {
			size_t half = size / 2;
			size_t other_half = size - half;
			size_t probe = low + half;
			size_t other_low = low + other_half;
			v = __data[probe].get_label().data();
			size = half;
			low = v.is_lower_ext2(s, lower, upper, lmask, umask) ? other_low : low;
		}
		return low!=load ? low : -1;
	}

public:
	/// does what the name suggest.
	/// \param e element we want to search
	/// \param k_lower lower coordinate on which the element must be equal
	/// \param k_higher higher coordinate the elements must be equal
	/// \return the position of the first (lower) element which is equal to e. -1 if nothing found
	size_t search_level(const Element &e, const uint64_t k_lower, const uint64_t k_higher, bool sort=false) {
		if constexpr (Element::binary()) {
			return search_level_binary(e, k_lower, k_higher, sort);
		} else {
			auto r = std::find_if(__data.begin(), __data.begin() + load, [&e, k_lower, k_higher](const Element &c) {
									return e.is_equal(c, k_lower, k_higher);
								  }
			);

			const auto dist = distance(__data.begin(), r);

			if (r == __data.begin() + load)
				return -1; // nothing found

			if (!__data[dist].is_equal(e, k_lower, k_higher))
				return -1;

			return dist;
		}
	}

	// TODO vll als klasse extended
	//	size_t search_parallel(const Element &e) {
	//        auto r = std::find_if(std::execution::par, __data.begin(), __data.end(),
	//                              [&e](const Element *c) { return c->get_value() == e.get_value(); });
	//        if (r == std::end(__data)) { // nothing found
	//            return -1;
	//        }
	//
	//        return distance(__data.begin(), r);
	//    }


	//    uint64_t search_parallel_level(const Element &e, const uint64_t level, const std::vector<uint64_t> &vec) {
	//        uint64_t k_lower, k_higher;
	//        translate_level(&k_lower, &k_higher, level, vec);
	//
	//        auto r = std::find_if(std::execution::par, __data.begin(), __data.end(),
	//                              [e, k_lower, k_higher](const Element *c) {
	//                                    return e.is_equal(*c, k_lower, k_higher);
	//                                });
	//        if (r == std::end(__data)) { // nothing found
	//            return -1;
	//        }
	//
	//        return distance(__data.begin(), r);
	//    }

	/// \param e
	/// \return	a tuple indicating the start and end indices within the list. start == end == load indicating nothing found,
	std::pair<uint64_t, uint64_t> search_boundaries(const Element &e, const uint64_t k_lower, const uint64_t k_higher) {
		uint64_t end_index;
		uint64_t start_index;
		if constexpr (!Element::binary()){
			start_index = search_level(e, k_lower, k_higher);
		} else {
			start_index = search_level_binary(e, k_lower, k_higher);
		}


		if (start_index == uint64_t(-1))
			return std::pair<uint64_t, uint64_t>(load, load);

		// get the upper index
		end_index = start_index + 1;
		while (end_index < load && (__data[start_index].is_equal(__data[end_index], k_lower, k_higher)))
			end_index += 1;

		return std::pair<uint64_t, uint64_t>(start_index, end_index);
	}

	/// A little helper function to check if a list is sorted. This is very useful to assert specific states within
	/// complex cryptanalytic algorithms.
	/// \param k_lower lower bound
	/// \param k_higher upper bound
	/// \return if its sorted
	bool is_sorted(const uint64_t k_lower,
				   const uint64_t k_higher) const {
		for (uint64_t i = 1; i < get_load(); ++i) {
			if (__data[i-1].is_equal(__data[i], k_lower, k_higher)) {
				continue;
			}

#if !defined(SORT_INCREASING_ORDER)
			if (!__data[i-1].is_lower(__data[i], k_lower, k_higher)){
				return false;
			}
#else
			if (!__data[i-1].is_greater(__data[i], k_lower, k_higher)){
				return false;
			}
#endif
		}

		return true;
	}

	/// appends the element e to the list. Note that the list keeps track of its load. So you dont have to do anyting.
	/// Note: if the list is full, every new element is silently discarded.
	/// \param e	Element to add
	void append(Element &e) {
		if (load < size()) {
			// wrong, increases effective size of container, use custom element copy function instead
			__data[load] = e;
		} else {
			__data.push_back(e);
		}

		load++;
	}

	/// append e1+e2|full_length to list
	/// \param e1 first element.
	/// \param e2 second element
	void add_and_append(const Element &e1, const Element &e2, const uint32_t norm=-1) {
		add_and_append(e1, e2, 0, LabelLENGTH, norm);
	}

	/// Same as the function above, but with a `constexpr` size factor. This may allow further optimisations to the
	/// compiler. Additionaly this functions adds e1 and e2 together and daves the result.
	/// \tparam approx_size Size of the list
	/// \param e1 first element
	/// \param e2 second element
	/// \param norm norm of the element e1+e2. If the calculated norm is bigger than `norm` the element is discarded.
	/// \return
	template<const uint64_t approx_size>
	int add_and_append(const Element &e1, const Element &e2, const uint32_t norm=-1) {
		if (load < approx_size) {
			Element::add(__data[load], e1, e2, 0, LabelLENGTH, norm);
			load += 1;
		}
		return approx_size-load;
		// ignore every element which could not be added to the list.
	}

	/// same as above, but the label is only added between k_lower and k_higher
	/// \param e1 first element to add
	/// \param e2 second element
	/// \param k_lower lower dimension to add the label on
	/// \param k_higher higher dimension to add the label on
	/// \param norm filter norm. If the norm of the resulting element is higher than `norm` it will be discarded.
	void add_and_append(const Element &e1, const Element &e2,
						const uint32_t k_lower, const uint32_t k_higher, const uint32_t norm=-1) {
		if (load < this->size()) {
			auto b = Element::add(__data[load], e1, e2, k_lower, k_higher, norm);
			// 'add' returns true if a overflow, over the given norm occurred. This means that at least coordinate 'r'
			// exists for which it holds: |data[load].value[r]| >= norm
			if (b == true)
				return;
		} else {
			Element t{};
			auto b = Element::add(t, e1, e2, k_lower, k_higher, norm);
			if (b == true)
				return;

			// this __MUST__ be a copy.
			__data.push_back(t);
		}

		// we do not increase the 'load' of our internal data structure if one of the add functions above returns true.
		load++;
	}

	/// set the element at position i to zero.
	/// \param i
	void zero(const size_t i) {
		ASSERT(i < get_load()); __data[i].zero();
	}

	/// remove the element at pos i.
	/// \param i
	void erase(const size_t i) {
		ASSERT(i < get_load()); __data.erase(__data.begin()+i); load -= 1;
	}

	/// resize the list. Note only the sanity check, if the list is already big enough, is made. If not enough mem
	/// is allocatable it will through an exception.
	/// \param size		new size
	void resize(const size_t size) {
		if (size > __data.size() && size > load) {
			__data.resize(size);
		}
	}

	/// some useful stuff
	auto begin() noexcept { return __data.begin(); }
	auto end() noexcept { return __data.end(); }

private:
	// just a small counter for the '_generate' function. __MUST__ be ignored.
	size_t internal_counter = 0;

	/// load factor of the list
	size_t load;

	uint32_t threads;

	/// internal data representation of the list.
	alignas(PAGE_SIZE) std::vector<Element> __data;
};

#endif//DECODING_LIST_H
