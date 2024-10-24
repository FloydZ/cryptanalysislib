#ifndef CRYPTANALYSISLIB_LIST_H
#define CRYPTANALYSISLIB_LIST_H

#include "list/enumeration/enumeration.h"
#include "list/common.h"
#include "list/parallel.h"
#include "list/parallel_full.h"
#include "list/parallel_index.h"
#include "list/simple.h"
#include "list/simple_limb.h"
#include "matrix/matrix.h"
#include "search/search.h"
#include "hash/hash.h"

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

	using typename MetaListT<Element>::value_type;
	using typename MetaListT<Element>::allocator_type;
	using typename MetaListT<Element>::size_type;
	using typename MetaListT<Element>::difference_type;
	using typename MetaListT<Element>::reference;
	using typename MetaListT<Element>::const_reference;
	using typename MetaListT<Element>::pointer;
	using typename MetaListT<Element>::const_pointer;
	using typename MetaListT<Element>::iterator;
	using typename MetaListT<Element>::const_iterator;
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

	/// needed configs
	using MetaListT<Element>::use_std_sort;
	using MetaListT<Element>::use_std_binary_search;
	using MetaListT<Element>::use_interpolation_search;
	using MetaListT<Element>::sort_increasing_order;
	using MetaListT<Element>::allow_resize;

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

	[[nodiscard]] constexpr inline auto begin() noexcept { return __data.begin(); }
	[[nodiscard]] constexpr inline auto begin() const noexcept { return __data.begin(); }
	[[nodiscard]] constexpr inline auto end() noexcept { return __data.end(); }
	[[nodiscard]] constexpr inline auto end() const noexcept { return __data.end(); }

	/// NOTE: assumes sorted
	/// @param e
	/// @return
	[[nodiscard]] constexpr inline auto begin(const Element &e,
											  const uint32_t k_lower,
											  const uint32_t k_upper) noexcept {
		return begin() + search_level(e, k_lower, k_upper);
	}

	/// NOTE: you need to pass the current iterator to optimize the process
	/// of finding the last element which is equal to e.
	/// @param e
	/// @param current current iterator.
	/// @param e find equal elements to e
	/// @param k_lower lower part
	/// @param k_upper upper part
	/// @return  first element in the list which is != e on [k_lower, k_upper)
	[[nodiscard]] constexpr inline auto end(const Element &e,
											const Element &current,
											const uint32_t k_lower,
											const uint32_t k_upper) noexcept {
		if (e.is_equal(current, k_lower, k_upper)) {
			return e;
		}

		return current;
	}

	/// NOTE: this function needs to do a forward search to find the last
	///		element which is equal to `e`.
	///		So technically dont use this do enumerate over equal elements, as you
	///		will do it twice
	/// @param e find equal elements to e
	/// @param k_lower lower part
	/// @param k_upper upper part
	/// @return  first element in the list which is != e on [k_lower, k_upper)
	[[nodiscard]] constexpr inline auto end(const Element &e,
											const uint32_t k_lower,
											const uint32_t k_upper) noexcept {
		auto b = begin(e, k_lower, k_upper)++;
		while (e.is_equal(*b, k_lower, k_upper)) {
			b += 1;
		}

		return b;
	}


	// using TMP = decltype(e.label.hash());
	constexpr static bool use_hash_operator = (sizeof(LabelType) * 8u) <= 64u;

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
	    : MetaListT<Element>(nr_element, threads) {
		ASSERT(threads > 0);
		for (uint32_t i = 0; i < threads; ++i) {
			set_load(0, i);
		}
	}


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

	///
	constexpr void sort() noexcept {
		std::sort(__data.begin(), __data.end());
	}


	/// \param level current lvl within the tree.
	/// \param level_translation
	constexpr void sort_level(const uint32_t level,
	                          const std::vector<uint32_t> &level_translation) noexcept {
		uint32_t k_lower, k_higher;
		translate_level(&k_lower, &k_higher, level, level_translation);
		return sort_level(k_lower, k_higher);
	}

	/// sort the whole list between [k_lower, k_higher)
	/// \param k_lower
	/// \param k_higher
	template<const bool sub=false>
	constexpr inline void sort_level(const uint32_t k_lower,
							         const uint32_t k_higher) noexcept {
		sort_level<sub>(k_lower, k_higher, 0);
	}

	template<const uint32_t k_lower,
	         const uint32_t k_upper,
	         const bool sub=false>
	constexpr inline void sort_level() noexcept {
		sort_level<k_lower, k_upper, sub>(0);
	}

	/// \param k_lower
	/// \param k_higher
	/// \param target
	/// \param tid: thread id currently unsupported
	/// \return
	template<const bool sub = false>
	constexpr inline void sort_level(const uint32_t k_lower,
							  		 const uint32_t k_higher,
							  		 const LabelType &target) noexcept {
		sort_level<sub>(k_lower, k_higher, target, 0);
	}

	/// NOTE: this does not search the FULL list, only each segment
	/// NOTE: if `Element` is hashable a fast radix sort will be used.
	/// \param k_lower lower dimension to sort on (inclusive)
	/// \param k_higher upper dimensions to sort (not included)
	/// \param tid thread id
	template<const uint32_t sub = false>
	constexpr void sort_level(const uint32_t k_lower,
							  const uint32_t k_higher,
							  const uint32_t tid) noexcept {
		ASSERT(k_lower < k_higher);
		const size_t sp = start_pos(tid), ep = end_pos(tid);

		if (use_std_sort || (!Element::is_hashable(k_lower, k_higher))) {
			sort_level_std_sort(sp, ep, [k_lower, k_higher](const auto &e1,
			                                                const auto &e2)  __attribute__((always_inline)) {
			    if constexpr (sort_increasing_order) {
			        return e1.is_lower(e2, k_lower, k_higher);
			    } else {
			        return e1.is_greater(e2, k_lower, k_higher);
			    }
			});
		} else {
			sort_level_radix_sort(sp, ep, [k_lower, k_higher]
					(const auto &e1) __attribute__((always_inline)) {
			    return e1.hash(k_lower, k_higher);
			});

		}
		ASSERT(is_sorted(k_lower, k_higher));
	}

	/// NOTE: this does not search the FULL list, only each segment
	/// \param k_lower lower dimension to sort on (inclusive)
	/// \param k_higher upper dimensions to sort (not included)
	/// \param tid thread id
	template<const uint32_t k_lower,
	         const uint32_t k_higher,
	         const uint32_t sub = false>
	constexpr void sort_level(const uint32_t tid) noexcept {
		static_assert( k_lower < k_higher);
		const size_t sp = start_pos(tid), ep = end_pos(tid);

		if constexpr (use_std_sort || (!Element::is_hashable(k_lower, k_higher))) {
			sort_level_std_sort(sp, ep, [](const auto &e1,
			                               const auto &e2)  __attribute__((always_inline)) {
			  if constexpr (sort_increasing_order) {
				  return e1.template is_lower<k_lower, k_higher>(e2);
			  } else {
				  return e1.template is_greater<k_lower, k_higher>(e2);
			  }
			});
		} else {
			sort_level_radix_sort(sp, ep, []
					(const auto &e1) __attribute__((always_inline)) {
			  return e1.template hash<k_lower, k_higher>();
			});
		}

		ASSERT(is_sorted(k_lower, k_higher));
	}

	/// \param k_lower
	/// \param k_higher
	/// \param target
	/// \return
	template<const bool sub = false>
	constexpr void sort_level(const uint32_t k_lower,
							  const uint32_t k_higher,
	                          const LabelType &target,
	                          const uint32_t tid) noexcept {
		ASSERT(k_lower < k_higher);
		const size_t sp = start_pos(tid), ep = end_pos(tid);

		if (use_std_sort || (!Element::is_hashable(k_lower, k_higher))) {
			sort_level_std_sort(sp, ep, [k_lower, k_higher, target]
			                    (const auto &e1,
								 const auto &e2)  __attribute__((always_inline)) {
			    LabelType tmp1, tmp2;
				if constexpr (sub) {
					  LabelType::sub(tmp1, target, e1.label, k_lower, k_higher);
					  LabelType::sub(tmp2, target, e2.label, k_lower, k_higher);
				} else {
					  LabelType::add(tmp1, target, e1.label, k_lower, k_higher);
					  LabelType::add(tmp2, target, e2.label, k_lower, k_higher);
				}

				if constexpr (sort_increasing_order) {
					  return tmp1.is_lower(tmp2, k_lower, k_higher);
				} else {
					  return tmp1.is_greater(tmp2, k_lower, k_higher);
				}
			});
		} else {
			sort_level_radix_sort(sp, ep, [k_lower, k_higher, target]
					(const auto &e1) __attribute__((always_inline)) {
				LabelType tmp;
				if constexpr (sub) {
					  LabelType::sub(tmp, target, e1.label, k_lower, k_higher);
				} else {
					  LabelType::add(tmp, e1.label, target, k_lower, k_higher);
				}
				return tmp.hash(k_lower, k_higher);
			});
		}

		ASSERT(is_sorted(target, sub, k_lower, k_higher));
	}

	/// \param k_lower
	/// \param k_higher
	/// \param target
	/// \return
	template<const uint32_t k_lower,
			 const uint32_t k_higher,
			 const bool sub = false>
	constexpr inline void sort_level(const LabelType &target) noexcept {
		return sort_level<k_lower, k_higher, sub> (target, 0);
	}

	/// \param k_lower
	/// \param k_higher
	/// \param target
	/// \return
	template<const uint32_t k_lower,
	         const uint32_t k_higher,
	         const bool sub = false>
	constexpr void sort_level(const LabelType &target,
							  const uint32_t tid) noexcept {
		ASSERT(k_lower < k_higher);
		const size_t sp = start_pos(tid), ep = end_pos(tid);

		if constexpr (use_std_sort || (!Element::template is_hashable<k_lower, k_higher>())) {
			sort_level_std_sort(sp, ep, [&target]
					(const auto &e1,
					 const auto &e2)  __attribute__((always_inline)) {
				LabelType tmp1, tmp2;
				if constexpr (sub) {
					  LabelType::template sub<k_lower, k_higher>(tmp1, target, e1.label);
					  LabelType::template sub<k_lower, k_higher>(tmp2, target, e2.label);
				} else {
					  LabelType::template add<k_lower, k_higher>(tmp1, target, e1.label);
					  LabelType::template add<k_lower, k_higher>(tmp2, target, e2.label);
				}

				if constexpr (sort_increasing_order) {
					  return tmp1.template is_lower<k_lower, k_higher>(tmp2);
				} else {
					  return tmp1.template is_greater<k_lower, k_higher>(tmp2);
				}
			});
		} else {
			sort_level_radix_sort(sp, ep, [&target]
					(const auto &e1) __attribute__((always_inline)) {
			  LabelType tmp;
			  if constexpr (sub) {
				  LabelType::template sub<k_lower, k_higher>(tmp, target, e1.label);
			  } else {
				  LabelType::template add<k_lower, k_higher>(tmp, target, e1.label);
			  }

			  return tmp.template hash<k_lower, k_higher>();
			});
		}

		ASSERT(is_sorted(target, sub, k_lower, k_higher));
	}


	/// apply standard sort
	/// \tparam F
	/// \param sp
	/// \param ep
	/// \param f
	/// \return
	template<class F>
	constexpr inline void sort_level_std_sort(const size_t sp,
	                                          const size_t ep,
	                                          F &&f) noexcept {
		std::sort(__data.begin() + sp, __data.begin() + ep, f);
	}

	///
	/// \tparam F
	/// \param sp
	/// \param ep
	/// \param f
	/// \return
	template<class F>
	constexpr inline void sort_level_radix_sort(const size_t sp,
									   			const size_t ep,
									   			F &&f) noexcept {
		ska_sort(__data.begin() + sp, __data.begin() + ep, f);
	}


public:

	/// tried to mimic the api of the hashmap
	constexpr inline size_t find(size_t &load,
	                             const Element &e,
	                             const uint32_t k_lower,
								 const uint32_t k_upper) const noexcept {
		const size_t a = search_level(e, k_lower, k_upper);
		if (a == -1ull) {return -1ull; }

		load = 1;
		while ((a + load) < load &&
		        e.is_equal(__data[a + load], k_lower, k_upper)) {
			load += 1;
		}

		return a;
	}

	/// generic search function, which depending on your configuration
	/// does different things. If
	/// \param e element to search
	/// \param tid thread which is searching
	/// \return either the position of the element or -1ull
	constexpr inline size_t search(const Element &e,
	                               const uint32_t tid=0) const noexcept {
		if constexpr (use_interpolation_search) {
			return interpolation_search(e, tid);
		}

		return binary_search(e, tid);
	}


	/// does what the name suggest.
	/// \param e element we want to search
	/// \param k_lower lower coordinate on which the element must be equal
	/// \param k_higher higher coordinate the elements must be equal
	/// \return the position of the first (lower) element which is equal to e. -1 if nothing found
	constexpr inline size_t search_level(const Element &e,
								  const uint32_t k_lower,
								  const uint32_t k_higher) const noexcept {
		ASSERT(is_sorted(k_lower, k_higher));
		if constexpr (use_interpolation_search) {
			return interpolation_search(e, k_lower, k_higher);
		} else {
			return binary_search(e, k_lower, k_higher);
		}
	}

	/// small helper function
	constexpr inline size_t search_level(const LabelType &l,
								  const uint32_t k_lower,
								  const uint32_t k_higher) const noexcept {
		Element e;
		e.label = l;
		return search_level(e, k_lower, k_higher);
	}

	///
	/// \tparam k_lower
	/// \tparam k_higher
	/// \param e
	/// \return
	template<const uint32_t k_lower, const uint32_t k_higher>
	constexpr inline size_t search_level(const Element &e) const noexcept {
		ASSERT(is_sorted(k_lower, k_higher));
		if constexpr (use_interpolation_search) {
			return interpolation_search<k_lower, k_higher>(e);
		} else {
			return binary_search<k_lower, k_higher>(e);
		}
	}

	template<const uint32_t k_lower, const uint32_t k_higher>
	constexpr inline size_t search_level(const LabelType &l) const noexcept {
		Element e;
		e.label = l;
		return search_level<k_lower, k_higher>(e);
	}

	///
	/// \tparam k_lower
	/// \tparam k_upper
	/// \param e
	/// \param tid
	/// \return
	constexpr inline size_t linear_search(const Element &e,
										  const uint32_t tid = 0) const noexcept {
		return linear_search<0, LabelLENGTH>(e, tid);
	}

	///
	/// \tparam k_lower
	/// \tparam k_upper
	/// \param e
	/// \param tid
	/// \return
	constexpr inline size_t linear_search(const Element &e,
	                                      const uint32_t k_lower,
	                                      const uint32_t k_upper,
										  const uint32_t tid = 0) const noexcept {
		ASSERT(k_upper > k_lower);

		// the linear search, doesn't need the data to be sorted
		if (use_hash_operator && Element::is_hashable(k_lower, k_upper)) {
			return linear_search(e, tid,
				[k_lower, k_upper](const Element &a)  __attribute__((always_inline)) {
				    return a.hash(k_lower, k_upper);
				});
		} else {
			return linear_search(e, tid, [](const Element &a,
											const Element &b)  __attribute__((always_inline)) {
				return a == b;
			});
		}
	}

	///
	/// \tparam k_lower
	/// \tparam k_upper
	/// \param e
	/// \param tid
	/// \return
	template<const uint32_t k_lower, const uint32_t k_upper>
	constexpr inline size_t linear_search(const Element &e,
								          const uint32_t tid = 0) const noexcept {
		// the linear search, doesn't need the data to be sorted
		if constexpr (use_hash_operator || (k_lower != k_upper)) {
			// NOTE: the checks if `k_lower` and `k_upper` are valid, are done
			// within the `hash` function
			if constexpr (Element::template is_hashable<k_lower, k_upper>()) {
				return linear_search(e, tid,
					 [](const Element &a)  __attribute__((always_inline)) {
					   return a.template hash<k_lower, k_upper>();
					 });
			} else {
				// fall back implementation
				return linear_search(e, tid,
					[](const Element &a)  __attribute__((always_inline)) {
					  return a.hash();
					});
			}
		} else {
			return linear_search(e, tid, [](const Element &a,
			                                const Element &b)  __attribute__((always_inline)) {
			    return a == b;
			});
		}
	}

	/// NOTE: this is a linear search
	/// \param e element to find
	/// \return the position of the element or -1
	template <class F>
	constexpr inline size_t linear_search(const Element &e,
	                               		  const uint32_t tid,
	                                      F &&f) const noexcept {
		const size_t sp = start_pos(tid), ep = end_pos(tid);
		const auto it = cryptanalysislib::search::linear_search(__data.begin() + sp, __data.begin() + ep, e, f);
		if (it == (__data.begin() + ep)) {
			return -1ull;
		} else {
			return std::distance(__data.begin()+sp, it);
		}
	}



	///
	/// \tparam k_lower
	/// \tparam k_upper
	/// \param e
	/// \param tid
	/// \return
	constexpr inline size_t binary_search(const Element &e,
										  const uint32_t k_lower,
										  const uint32_t k_upper,
										  const uint32_t tid=0) const noexcept {
		ASSERT(k_upper > k_lower);

		// the linear search, doesn't need the data to be sorted
		if constexpr (!use_std_binary_search && use_hash_operator) {
			if (Element::is_hashable(k_lower, k_upper)) {
				return binary_search(e, tid,
					 [k_lower, k_upper](const Element &a)  __attribute__((always_inline)) {
					   return a.hash(k_lower, k_upper);
					 });
			} else {
				ASSERT(false);
			}
		} else {
			return binary_search(e, tid, [k_lower, k_upper](const Element &a,
													   		const Element &b)  __attribute__((always_inline)) {
			  return a.is_lower(b, k_lower, k_upper);
			});
		}
	}


	/// NOTE: this function tries to use a hashbase searching implementation,
	/// but if its not possible, (either the hash range is to big, or element is to big)
	/// then a normal comparison based approach is chosen.
	/// \param e element to search for
	/// \param tid thread id
	/// \return position of the element within the list or -1
	template<const uint32_t k_lower=0, const uint32_t k_upper=0>
	constexpr inline size_t binary_search(const Element &e,
										  const uint32_t tid = 0) const noexcept {
		if constexpr (!use_std_binary_search &&
		              (use_hash_operator || (k_lower != k_upper))) {
			return binary_search(e, tid, [](const Element &a)  __attribute__((always_inline)) {
			    // NOTE: the checks if `k_lower` and `k_upper` are valid, are done
			    // within the `hash` function
			    if constexpr ((k_lower != k_upper) and
				              (Element::template is_hashable<k_lower, k_upper>())) {
			    	  return a.template hash<k_lower, k_upper>();
			    } else {
					return a.hash();
				}
			});
		} else {
			return binary_search(e, tid, [](const Element &a,
											const Element &b)  __attribute__((always_inline)) -> bool {
			  if constexpr (k_lower != k_upper){
				  return a.template is_lower<k_lower, k_upper>(b);
			  } else {
				  return a < b;
			  }
			});
		}
	}

	///
	/// \tparam F
	/// \param e
	/// \param tid
	/// \param f
	/// \return
	template<class F>
	constexpr size_t binary_search(const Element &e,
								   const uint32_t tid,
	                               F &&f) const noexcept {
		const size_t sp = start_pos(tid), ep = end_pos(tid);
		const_iterator it;
		if constexpr (use_std_binary_search) {
			it = std::lower_bound(__data.begin() + sp,
			                      __data.begin() + ep,
			                      e, f);
		} else {
			it = cryptanalysislib::search::binary_search(__data.begin() + sp,
			                                             __data.begin() + ep, e, f);
		}
		if (it == (__data.begin() + ep)) {
			return -1ull;
		} else {
			return std::distance(__data.begin()+sp, it);
		}

	}

	///
	/// \param e
	/// \param tid
	/// \return
	template<const uint32_t l, const uint32_t h>
	constexpr inline size_t interpolation_search(const Element &e,
												 const uint32_t tid = 0) const noexcept {
		if constexpr (not Element::template is_hashable<l, h>()) {
			return binary_search<l, h>(e, tid);
		} else {
			return interpolation_search(e, tid,
				[](const Element &a) __attribute__((always_inline)) {
				  return a.template hash<l, h>();
				});
		}
	}

	/// WARNING: if the element is not hashable, a standard binary search is conducted
	/// \tparam F hash function
	/// \param e element to seach for
	/// \param tid
	/// \param f
	/// \return
	template<class F>
	constexpr size_t interpolation_search(const Element &e,
	                                      const uint32_t tid,
	                                      F &&f) const noexcept {
		if constexpr (!use_hash_operator) {
			return binary_search(e, tid);
		}

		const size_t sp = start_pos(tid), ep = end_pos(tid);
		const auto it = cryptanalysislib::search::interpolation_search(__data.begin() + sp, __data.begin() + ep, e, f);
		if (it == (__data.begin() + ep)) {
			return -1ull;
		} else {
			return std::distance(__data.begin()+sp, it);
		}
	}

	/// \param e
	/// \return	a tuple indicating the start and end indices within the list.
	// 		start = end = load indicating nothing found,
	constexpr std::pair<size_t, size_t>
	search_boundaries(const Element &e,
					  const uint32_t k_lower,
					  const uint32_t k_higher) const noexcept {
		const size_t start_index = search_level(e, k_lower, k_higher);
		size_t end_index;

		if (start_index == uint64_t(-1)) {
			return std::pair<uint64_t, uint64_t>(load(), load());
		}

		// get the upper index
		end_index = start_index + 1;
		while (end_index < load() &&
		       (__data[start_index].is_equal(__data[end_index], k_lower, k_higher))) {
			end_index += 1;
		}

		return std::pair<size_t, size_t>{start_index, end_index};
	}

	/// \param e
	/// \return	a tuple indicating the start and end indices within the list.
	// 		start = end = load indicating nothing found,
	template<const uint32_t k_lower, const uint32_t k_higher>
	constexpr std::pair<size_t, size_t>
	search_boundaries(const Element &e) const noexcept {
		const size_t start_index = search_level<k_lower, k_higher>(e);
		size_t end_index;

		if (start_index == uint64_t(-1)) {
			return std::pair<uint64_t, uint64_t>(load(), load());
		}

		// get the upper index
		end_index = start_index + 1;
		while (end_index < load() &&
		       (__data[start_index].template is_equal<k_lower, k_higher>(__data[end_index]))) {
			end_index += 1;
		}

		return std::pair<size_t, size_t>{start_index, end_index};
	}

public:

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
	                              const uint32_t norm,
	                              const bool sub=false,
	                              const uint32_t tid=0) noexcept {
		auto op1 = [&e1, &e2, k_lower, k_higher, sub, norm](Element &c) __attribute__((always_inline)) {
			if (sub) {
			    return Element::sub(c, e1, e2, k_lower, k_higher, norm);
			} else {
			    return Element::add(c, e1, e2, k_lower, k_higher, norm);
			}
		};

		// norm  == -1:
		// 'add' returns true if a overflow, over the given norm occurred. This means that at least coordinate 'r'
		// exists for which it holds: |data[load].value[r]| >= norm
		if (load() < size()) {
			const bool b = op1(__data[load(tid)]);
			if ((norm != uint32_t(-1u)) && b) { return; }
		} else {
			if constexpr (! allow_resize) {
				return;
			}

			Element t{};
			const bool b = op1(t);
			if ((norm != uint32_t(-1u)) && b) { return; }

			// this __MUST__ be a copy.
			__data.push_back(t);
			__size += 1;
		}

		// we do not increase the 'load' of our internal data structure if one of the add functions above returns true.
		set_load(load(tid) + 1, tid);
	}

	template<const uint32_t k_lower,
	         const uint32_t k_higher,
	         const uint32_t norm,
	         const bool sub>
	constexpr void add_and_append(const Element &e1,
								  const Element &e2,
	                              const uint32_t tid=0) noexcept {

		auto op1 = [&e1, &e2](Element &c) __attribute__((always_inline)) {
		  if constexpr (sub) {
			  return Element::template sub<k_lower, k_higher, norm>(c, e1, e2);
		  } else {
			  return Element::template add<k_lower, k_higher, norm>(c, e1, e2);
		  }
		};

		if (load() < size()) {
			const bool b = op1(__data[load(tid)]);
			if constexpr (norm != uint32_t(-1u)) { if (b) { return; } }
		} else {
			if constexpr (! allow_resize) {
				return;
			}

			Element t{};
			const bool b = op1(t);
			if constexpr (norm != uint32_t(-1u)) { if (b) { return; } }

			// this __MUST__ be a copy.
			__data.push_back(t);
			__size += 1;
		}
		// we do not increase the 'load' of our internal data structure if one of the add functions above returns true.
		set_load(load(tid) + 1, tid);
	}

	static void info()  noexcept{
		std::cout << " { name=\"List\""
				  << " , sizeof(LoadType):" << sizeof(LoadType)
				  << " , ValueLENGTH:" << ValueLENGTH
				  << " , LabelLENGTH:" << LabelLENGTH
				  << " }" << std::endl;
		ListConfig::info();
		ElementType::info();
	}
};

#endif//DECODING_LIST_H
