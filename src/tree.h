#ifndef SMALLSECRETLWE_TREE_H
#define SMALLSECRETLWE_TREE_H

#include <utility>
#include <vector>

#ifdef __unix__
#include <sys/mman.h>
#endif

// internal includes
#include "container/kAry_type.h"
#include "container/vector.h"
#include "element.h"
#include "list/list.h"
#include "matrix/binary_matrix.h"
#include "matrix/fq_matrix.h"
#include "thread/thread.h"

// needed for `ExtendedTree`
#include "container/hashmap.h"
#include "random.h"

#if __cplusplus > 201709L
/// IDEA: extend the tree syntax to arbitrary hashmaps.
template<class HM>
concept TreeAbleHashMap = requires(HM hm) {
	// Base Datatype which is hash down
	typename HM::T;

	// base datatype for the hash type. So H = Hash(T)
	typename HM::H;

	// indextype within the hashmap
	typename HM::IndexType;

	// List data type
	typename HM::List;

	// Hasher and extractor. Note that extractor is needed as an input for the hash function
	typename HM::Hash;
	typename HM::Extractor;

	///
	requires requires(const typename HM::List &list,
	                  typename HM::Hash hash,
	                  typename HM::Extractor extractor,
	                  const typename HM::T t,
					  const typename HM::IndexPos i,
	 		   		  const size_t s,
	                  const uint32_t u32,
	                  const uint64_t u64) {
		hash(list, u32);
		hash(list, u64, u32, hash);

	 	insert(t, *i, u32);
		{ find(t, i) } -> std::convertible_to<typename HM::IndexType>;
	};
};

template<class List>
concept TreeAble = requires(List l) {
	// the following types are needed
	typename List::ValueType;
	typename List::LabelType;
	typename List::ElementType;
	typename List::MatrixType;

	// make sure the underlying data structs are useful
	requires ListAble<List>;
	requires ListElementAble<typename List::ElementType>;

	// function requirements
	requires requires(const uint32_t u32,
	                  const uint64_t u64,
	                  const size_t s,
	                  typename List::ElementType &e) {
		l[u32];
		List(u64);// constructor
		l.sort_level(u32, u32);
		l.sort();

		l.search_level(e, u64, u64);
		l.linear_search(e);
		l.binary_search(e);
		l.append(e);
		l.add_and_append(e, e, u32);
		l.add_and_append(e, e, u64, u64, u32);

		l.size();
		l.zero(s);
		l.erase(s);
		l.resize(s);
		l.data_value(s);
		l.data_label(s);
		l.start_pos(u32);
		l.end_pos(u32);
	};
};
#endif


struct TreeConfig : public AlignmentConfig {
	/// if a tree algorithm needs an intermediate lists, which is not
	/// passed as an argument
	constexpr static double intermediatelist_size_factor = 1.1;
} treeConfig;

template<class List, const TreeConfig &config=treeConfig>
#if __cplusplus > 201709L
    requires TreeAble<List>
#endif
class Tree_T {
public:
	typedef typename List::ElementType ElementType;
	typedef typename List::ValueType ValueType;
	typedef typename List::LabelType LabelType;

	typedef typename List::ValueContainerType ValueContainerType;
	typedef typename List::LabelContainerType LabelContainerType;

	typedef typename List::ValueDataType ValueDataType;
	typedef typename List::ValueDataType LabelDataType;

	typedef typename List::MatrixType MatrixType;

	// internal data types lengths
	constexpr static uint32_t ValueLENGTH = ValueType::length();
	constexpr static uint32_t LabelLENGTH = LabelType::length();

private:
	/// We allow additionally two baselists. So for the Stream join we have in total d + 2 lists we have to save in
	/// memory.
	constexpr static unsigned int additional_baselists = 2;

	/// Remember that we follow a streaming merge approach. By this we only need save in LEVEL 0 the two baselists.
	///		Example: k = 4 (4 baselists only two are saved.)
	///			Level 1: only 1 list will be saved (the one on the left two baselists)
	///			Level 2: the final list MUST be saved
	///
	///		Example k = 8
	///			Level 1: the one lists for the first 4 baselists MUST be saved AND on list for the second 4 baselists.
	///			Level 2: only one list must be saved, because we can stream merge the generated list for the right 4 baselists.
	///			Level 3: final list must be saved.
	/// \param i 	current level
	/// \return number of lists needed (-1 as the last one is streamed)
	[[nodiscard]] constexpr uint32_t intermediat_level_limit(const uint32_t i) const noexcept {
		// the first -1 half the amount of lists.
		return (1ULL << (depth - i - 1ull)) - 1ull;
	}

	/// count how many carries would occur of one add 2 to `in`
	/// \param in
	/// \return
	[[nodiscard]] constexpr static uint32_t count_carry_propagates(const uint32_t in) noexcept {
		ASSERT(in >= 2 && "insert bigger than 2");

		const uint32_t prev = in - 2u;
		uint32_t mask = 1ULL << 1u;
		uint32_t amount_negates = 1;
		while (mask & prev) {
			amount_negates++;
			mask <<= 1u;
		}

		return amount_negates;
	}

public:
	explicit Tree_T(const uint32_t d,
					const MatrixType &A,
					const size_t baselist_size) noexcept :
			matrix(A),
			level_translation_array(),
			level_filter_array() {
		ASSERT(d > 0 && "at least level 1");
		ASSERT(level_translation_array.size() >= d);
		for (uint32_t i = 0; i < d + additional_baselists; ++i) {
			List a{1u << baselist_size};
			lists.push_back(a);
		}

		depth = d;
		base_size = 1u << baselist_size;
	}

	explicit Tree_T(const uint32_t d,
					const MatrixType &A,
					const List &b1,
					const List &b2) noexcept :
			matrix(A),
			level_translation_array() {
		ASSERT(d > 0 && "at least level 1");
		lists.push_back(b1);
		lists.push_back(b2);

		for (uint32_t i = 2; i < d + additional_baselists; ++i) {
			List a{0};
			a.set_load(0);
			lists.push_back(a);
		}

		depth = d;
		base_size = b1.size();
	}

	///	On normal usage only d+2 (two baselists) __MUST__ be allocated and saved in memory. So technically that implements
	/// a non-recursive k-List algorithm.
	/// \param d depth of the tree.
	/// \param A Matrix
	/// \param baselist_size size of the baselists to generate (log)
	/// \param level_translation_array const_array containing the limits of the coorditates to merge on each lvl.
	///         Example:
	///				[0, 10, 20, ... 100]
	///			means: that on the first lvl the lists are matched on the coordinates 0-9 (9 inclusive)
	///         and so on
	explicit Tree_T(const uint32_t d,
	                const MatrixType &A,
	                const size_t baselist_size,
	                const std::vector<uint64_t> &level_translation_array,
	                const std::vector<std::vector<uint8_t>> &level_filter_array) noexcept :
	    matrix(A),
	    level_translation_array(level_translation_array),
	    level_filter_array(level_filter_array) {
		ASSERT(d > 0 && "at least level 1");
		ASSERT(level_translation_array.size() >= d);
		for (uint32_t i = 1; i < d; ++i) {
			ASSERT(level_translation_array[i - 1] <
			       level_translation_array[i]);
		}
		for (uint32_t i = 0; i < d + additional_baselists; ++i) {
			List a{1u << baselist_size};
			lists.push_back(a);
		}

		depth = d;
		base_size = 1u << baselist_size;
	}

	/// Same constructor as above, with additional b1, b2 as the two baselists
	/// \param d				depth of the tree
	/// \param A				Matrix. Connection between label and value.
	/// \param b1				first baselist
	/// \param b2				second baselist
	/// \param baselist_size	size of the baselists in log2
	/// \param level_translation_array const_array containing the limits of the coorditates to merge on each lvl.
	///         Example:
	///				[0, 10, 20, ... 100]
	///			means: that on the first lvl the lists are matched on the coordinates 0-9 (9 inclusive)
	///         and so on
	explicit Tree_T(const uint32_t d,
	                const MatrixType &A,
	                const List &b1,
	                const List &b2,
	                const uint32_t baselist_size,
	                std::vector<uint64_t> &level_translation_array) noexcept :
	    matrix(A),
	    level_translation_array(level_translation_array) {
		ASSERT(d > 0 && "at least level 1");
		lists.push_back(b1);
		lists.push_back(b2);

		for (uint32_t i = 2; i < d + additional_baselists; ++i) {
			List a{0};
			a.set_load(0);
			lists.push_back(a);
		}

		depth = d;
		base_size = 1u << baselist_size;
	}

	// Andre: this just saves in default target list (lists[level+2])
	/// \param level
	void join_stream(const uint64_t level) noexcept {
		ASSERT(lists.size() >= level + 1);
		join_stream_internal(level, lists[level + 2]);
	}

	/// use this function if you only want one final solution/target list.							nomenclature:
	///										[ TARGET LIST ]														LEVEL 2
	///											| Again direct merge with list 1 and list 2
	///						  ---------------------------------------								MERGE ON LEVEL 1
	///						  |										|
	///					[ LIST	1	]	(Save) 					[ LIST	2	] (dont save directly merge)		LEVEL 1
	///						  |										|
	///				---------------------					---------------------					MERGE ON LEVEL 0
	///				|					|					|					|
	///		[ BASE LIST 1]		[ BASE LIST 2]		[ BASE LIST 3]		[ BASE LIST 4]							LEVEL 0
	void join_stream(const uint64_t level, List &target) noexcept {
		join_stream_internal(level, target);
	}


	/// runs the full tree algorithm, indepentend of the depth
	/// \param target
	/// \param gen_lists
	/// \param two_result_lists
	void build_tree(LabelType &target, 
					const bool gen_lists = true,
					const bool two_result_lists = false) noexcept {
		uint64_t k_lower, k_higher;
		if (gen_lists) {
			lists[0].set_load(0);
			lists[1].set_load(0);
			lists[0].random(this->base_size, matrix);
			lists[1].random(this->base_size, matrix);
		}

		// generate the intermediate targets for all levels
		std::vector<std::vector<LabelType>> intermediate_targets(depth);
		for (uint32_t i = 0; i < depth; ++i) {
			// +1 to have enough space.
			intermediate_targets[i].resize(intermediat_level_limit(i) + 1);

			// set random intermediate targets
			for (uint32_t j = 0; j < intermediat_level_limit(i); ++j) {
				intermediate_targets[i][j].random();
			}

			intermediate_targets[i][intermediat_level_limit(i)] = target;
		}

		// adjust last intermediate target of each level, such that targets of
		// each level add up the true target.
		for (uint32_t i = 0; i < depth - 1; ++i) {
			for (uint32_t j = 0; j < intermediat_level_limit(i); ++j) {
				translate_level(&k_lower, &k_higher, -1, level_translation_array);

				LabelType::sub(intermediate_targets[i][intermediat_level_limit(i)],
				               intermediate_targets[i][intermediat_level_limit(i)],
							   intermediate_targets[i][j], k_lower, k_higher);
			}
		}


		// start joining the lists
		for (uint32_t i = 0; i < (1ULL << (depth - 1)); ++i) {
			// calc number of trailing zeros of i, which indicates to which 
			// level we have to join
			const int join_to_level = __builtin_ctz(i + 1);

			// prepare the base-lists, as the join itself just checks for 
			// *equality* on all levels and performs only *addition* of labels
			prepare_lists(i, intermediate_targets);

			// sort the baselists
			lists[0].sort_level(0, level_translation_array);
			lists[1].sort_level(0, level_translation_array);

			// if `two_result_lists` is set, the tree saves the last two lists 
			// and does not joint to the final level
			if (!two_result_lists || join_to_level != depth - 1)
				join_stream(join_to_level);
			else
				join_stream(join_to_level - 1, lists[depth + 1]);

			// Empty List.
			ASSERT(lists[join_to_level + 2].get_load() != 0 && "list empty");

			// if not finished: sort the resulting list
			if (likely(i != (1ULL << (this->depth - 1)) - 1)) {
				lists[join_to_level + 2].sort_level(join_to_level + 1, level_translation_array);
			} else {
				// else restore the original baselists for the next run
				restore_baselists(i + 1, intermediate_targets);
				restore_label(intermediate_targets, two_result_lists);
			}
		}
	}

	///
	/// \param i
	/// \param intermediate_targets
	void restore_label(std::vector<std::vector<LabelType>> &intermediate_targets, 
					   const bool two_result_lists) noexcept {
		uint64_t k_lower, k_higher;

		// if one result list was created, the label matches the target on bla coordinates
		// list preparation procedure leads to inconsistency on the already matched coordinates
		// hence we have to set the label to the target manually (faster than label recomputation)
		if (!two_result_lists) {
			for (uint64_t i = 0; i < lists[depth + 1].get_load(); ++i) {
				for (int j = 0; j < depth; ++j) {
					translate_level(&k_lower, &k_higher, j, level_translation_array);
					LabelType::set(lists[depth + 1][i].get_label(), intermediate_targets[depth - 1][0], k_lower, k_higher);
				}
			}
		} else {
			// if two result lists were created the labels of the first list match the sum of the first half of intermediate targets
			// the second matches the corresponding second half of intermediate targets
			for (uint32_t a = 0; a < 2; ++a) {
				for (uint64_t i = 0; i < lists[depth + a].get_load(); ++i) {
					for (int j = 0; j < depth - 1; ++j) {
						translate_level(&k_lower, &k_higher, j, level_translation_array);

						int offset = a * (intermediat_level_limit(j) + 1) / 2;
						LabelType::set(lists[depth + a][i].get_label(), intermediate_targets[j][0 + offset], k_lower,
						               k_higher);

						for (int k = 1; k < (intermediat_level_limit(j) + 1) / 2; ++k) {
							LabelType::add(lists[depth + a][i].get_label(), lists[depth + a][i].get_label(),
							               intermediate_targets[j][k + offset], k_lower, k_higher);
						}
					}
				}
			}
		}

		//NOTE: Before applying BDD-Solver the label has actually to be negated 
		// on all coordinates, is here the right place to do so?
	}

	///
	/// \param i
	/// \param intermediate_targets
	void restore_baselists(int i, std::vector<std::vector<LabelType>> &intermediate_targets) noexcept {
		/// first, second are the positions of the lists to prepare
		int first = 2 * i;
		int second = 2 * i + 1;

		uint64_t k_lower, k_higher;

		// determine which parts of the labels of first baselist have to be negated
		unsigned int amount_negates = count_carry_propagates(first) - 1;

		// negate corresponding parts of the labels
		for (uint64_t ind = 0; ind < lists[0].get_load(); ++ind) {
			for (uint32_t k = 0; k < amount_negates; ++k) {
				translate_level(&k_lower, &k_higher, k + 1, level_translation_array);
				lists[0][ind].get_label().neg(k_lower, k_higher);
			}
		}

		// calculate number of old targets to eliminate from baselists
		const uint32_t old_targets = __builtin_ctz(second - 1);

		// determine which parts of the labels have to be negate
		amount_negates = count_carry_propagates(second);

		for (uint64_t ind = 0; ind < lists[1].get_load(); ++ind) {
			//get rid of intermediate targets from previous levels
			for (unsigned int j = 0; j < old_targets; ++j) {
				translate_level(&k_lower, &k_higher, j, level_translation_array);
				LabelType::sub(lists[1][ind].get_label(), lists[1][ind].get_label(),
				               intermediate_targets[j][((second - 1u) >> (j + 1u)) - 1], k_lower, k_higher);
			}

			for (uint32_t k = 0; k < amount_negates; ++k) {
				translate_level(&k_lower, &k_higher, k, level_translation_array);
				lists[1][ind].get_label().neg(k_lower, k_higher);
			}
		}
	}

	///
	/// \param i
	/// \param intermediate_targets
	void prepare_lists(const int i, std::vector<std::vector<LabelType>> &intermediate_targets) noexcept {
		/// first, second are the positions of the lists to prepare
		const int first = 2 * i;
		const int second = 2 * i + 1;

		uint64_t k_lower, k_higher;

		// determine which parts of the labels of first baselist have to be negated
		if (first != 0) {
			// IMPORTANT: the first-2 is within the function. Is this intuitively?
			unsigned int amount_negates = count_carry_propagates(first);

			// negate corresponding parts of the labels
			for (uint64_t ind = 0; ind < lists[0].load(); ++ind) {
				for (uint32_t k = 0; k < amount_negates; ++k) {
					translate_level(&k_lower, &k_higher, k + 1, level_translation_array);
					lists[0][ind].label.neg(k_lower, k_higher);
				}
			}
		}

		if (second == 1) {
			translate_level(&k_lower, &k_higher, 0, level_translation_array);

			// if we work on the first two baselists (which means the ones on the far left side of the tree.) We need to
			// negate the label of each element within first list.
			for (uint64_t ind = 0; ind < lists[0].load(); ++ind) {
				lists[1][ind].label.neg(k_lower, k_higher);
				LabelType::add(lists[1][ind].label,
				               lists[1][ind].label,
				               intermediate_targets[0][0],
				               k_lower, k_higher);
			}
		} else {
			// calculate number of old targets to eliminate from baselists and number of new intermediate targets to add
			const uint32_t old_targets = __builtin_ctz(second - 1);
			const uint32_t new_targets = __builtin_ctz(second + 1);

			// determine which parts of the labels have to be negated
			// IMPORTANT: the second-2 is within the function. Is this intuitively?
			unsigned int amount_negates = count_carry_propagates(second);

			for (uint64_t ind = 0; ind < lists[1].load(); ++ind) {
				//get rid of intermediate targets from previous levels
				for (unsigned int j = 0; j < old_targets; ++j) {
					const uint32_t index = ((second - 1u) >> (j + 1u)) - 1;
					translate_level(&k_lower, &k_higher, j, level_translation_array);
					LabelType::sub(lists[1][ind].label,
					               lists[1][ind].label,
					               intermediate_targets[j][index],
					               k_lower, k_higher);
				}

				for (uint32_t k = 0; k < amount_negates; ++k) {
					translate_level(&k_lower, &k_higher, k + 1, level_translation_array);
					lists[1][ind].label.neg(k_lower, k_higher);
				}

				//add new intermediate targets
				for (uint32_t j = 0; j < new_targets; ++j) {
					const uint32_t index = ((second + 1u) >> (j + 1u)) - 1;
					translate_level(&k_lower, &k_higher, j, level_translation_array);
					LabelType::add(lists[1][ind].label,
					               lists[1][ind].label,
					               intermediate_targets[j][index],
					               k_lower, k_higher);
				}
			}
		}
	}

	/// NOTE: should not be a tree function
	/// Usage:
	///		using BDD_element = Element_T<Value_T<kAryContainer_T<uint64_t, 1>>, Label_T<kAryContainer_T<Label_Type, G_n + G_l>>, fplll::ZZ_mat<Label_Type>>;
	///		using BDD_list = List_T<BDD_element>;
	///		using Odlyzko_element = Element_T<Value_T<kAryContainer_T<uint64_t, 1>>, Label_T<BinaryContainer<G_n + G_l>>, fplll::ZZ_mat<Label_Type>>;
	///		using Odlyzko_list = List_T<Odlyzko_element>;
	template<class BDD_list, class Odlyzko_list, class Odlyzko_element, const uint32_t q>
	static void create_odlyzko_list(BDD_list &in, Odlyzko_list &out, const uint32_t n, const uint32_t l) noexcept {
		Odlyzko_element el{};
		//NOTE: 10 ok?
		std::vector<uint64_t> border_indices(10);

		//NOTE: resize odlzyko list to expected size
		for (uint64_t i = 0; i < in.get_load(); ++i) {
			uint64_t num_border_cases = 0;
			//remember index of current bdd-element via value field of odl-list
			el.get_value().data()[0] = i;
			for (uint32_t j = 0; j < (n + l); ++j) {
				auto coordinate = in[i].get_label().data()[j].get_value();

				//if border case, remember position
				if ((coordinate == 0) || (coordinate == q - 1) ||
				    (coordinate == q / 2 - 1) || (coordinate == q / 2)) {
					border_indices[num_border_cases] = j;
					num_border_cases++;
				}
				//otherwise hash to the sign
				else if (coordinate < q / 2)
					el.get_label().data()[j] = 0u;
				else
					el.get_label().data()[j] = 1u;
			}
			//store if no border case
			if (num_border_cases == 0)
				out.append(el);
			//otherwise iterate over all possible combinations of border cases and store
			else {
				for (uint64_t k = 0; k < 1 << num_border_cases; ++k) {
					uint32_t mask = 1ULL;
					for (uint64_t l = 0; l < num_border_cases; ++l) {
						el.get_label().data()[border_indices[l]] = (k & mask);
						mask <<= 1u;
					}
					out.append(el);
				}
			}
		}
	}

	/// IMPORTANT: The two lists L1, L2 __MUST__ be prepared with any intermediate target before calling this function.
	///                                                 Out
	///                                      +----------------------+
	///                                      |           |          |
	///                                      +----------------------+
	/// Merge on k_upper                                 |
	///                        +-------------------------+-------------------------+  Stream Merge
	///                        |                Merge on target                    |
	///            +-----------------------+                          + - - - - - --- - - - - +
	///            |           |           | Intermediate List        |            |          | NOT Saved
	///            +-----------+-----------+                          +- - - - - - | - - - - -+
	///                       i_L                                                  |  Merge on k_lower
	///                                                              +-------------+-----------+
	///                                                              |    merge on equal       |
	///                                                   +-----------------------+ +-----------------------+
	///                                                   |          |            | |          |            |
	///                                                   +----------+------------+ +----------+------------+
	///                                                  k_lower1   L_1      k_upper2         L_2      k_upper2
	///                                                            k_upper1=k_lower2
	/// \param out			Output list
	/// \param iL			Input list, will be sorted on [k_lower1, k_upper2]
	/// \param L1			Input list, will be sorted on [k_lower1, k_upper1]
	/// \param L2			Input list, will be sorted on [k_lower1, k_upper1]
	/// \param k_lower1 lower coordinate to match on, on the first level
	/// \param k_upper1 upper coordinate to match on, on the first level
	/// \param k_lower2 NOTE: ignored, assumed == k_upper1
	/// \param k_upper2 upper coordinate to match on, on the last level
	/// \param prepare if true: will only sort the lists.
	static void twolevel_streamjoin(List &out, List &iL, List &L1, List &L2,
	                                const uint32_t k_lower1, const uint32_t k_upper1,
									const uint32_t k_lower2, const uint32_t k_upper2,
	                                bool prepare = true) noexcept {
		ASSERT(k_lower1 < k_upper1 &&
		       0 < k_upper1 && k_lower2 < k_upper2
		       && 0 < k_upper2
		       && k_lower1 <= k_lower2
		       && k_upper1 <= k_upper2);
		// internal variables.
		constexpr uint32_t filter = uint32_t(-1);
		constexpr bool sub = !LabelType::binary();
		std::pair<size_t, size_t> boundaries;
		ElementType e;

		if (prepare) {
			iL.sort_level(k_lower1, k_upper2);
			L1.sort_level(k_lower1, k_upper1);
			L2.sort_level(k_lower1, k_upper1);
		}


		auto op = [](ElementType &c, const ElementType &a, const ElementType &b,
					 const uint64_t l, const uint64_t h) {
		  if constexpr (sub) {
			  ElementType::sub(c, a, b, l, h, filter);
		  } else {
			  ElementType::add(c, a, b, l, h, filter);
		  }
		};

		// early exit
		if (iL.load() == 0) { return; }
		if (L1.load() == 0) { return; }
		if (L2.load() == 0) { return; }

		uint64_t i=0, j=0;
		while (i < L1.load() && j < L2.load()) {
			if (L2[j].is_greater(L1[i], k_lower1, k_upper1)) {
				i++;
			} else if (L1[i].is_greater(L2[j], k_lower1, k_upper1)) {
				j++;
			} else {
				uint64_t i_max=i+1ull, j_max=j+1ull;
				for (; i_max < L1.load() && L1[i].is_equal(L1[i_max], k_lower1, k_upper1); i_max++) {}
				for (; j_max < L2.load() && L2[j].is_equal(L2[j_max], k_lower1, k_upper1); j_max++) {}

				const uint64_t jprev = j;

				// we have found equal elements. But this time we don't have to
				// save the result. Rather we stream join everything up to the final solution.
				for (; i < i_max; ++i) {
					for (j = jprev; j < j_max; ++j) {
						// add/sub on full length
						op(e, L1[i], L2[j], k_lower1, k_upper2);
#ifdef DEBUG
						if (!e.label.is_zero(k_lower1, k_upper1)) {
							std::cout << e;
							std::cout << L2[j];
							std::cout << L1[i];
							ASSERT(false);
						}
#endif

						boundaries = iL.search_boundaries(e, k_lower2, k_upper2);

						// finished?
						if (boundaries.first == boundaries.second) {
							// NOTE: we cannot break out of the two loops
							// only the first one.
							break;
						}

						for (size_t l = boundaries.first; l < boundaries.second; ++l) {
							out.add_and_append(iL[l], e, k_lower1, k_upper2, filter, sub);

#ifdef DEBUG
							const size_t lb = out.load() - 1;
							if (!out[lb].label.is_zero(k_lower1, k_upper2)) {
								std::cout << out[lb] << std::endl;
								std::cout << iL[l] << std::endl;
								std::cout << e << std::endl;
								ASSERT(false);
							}
#endif
						}
					}
				}
			}
		}
	}

	// TODO test
	/// doc: see function above /
	/// NOTE: on_iT means, that the input lists, will only be sorted, not altered
	static void twolevel_streamjoin_on_iT(List &out, List &iL, List &L1, List &L2,
										  const LabelType &target,
										  const uint64_t k_lower1, const uint64_t k_upper1,
										  const uint64_t k_lower2, const uint64_t k_upper2) noexcept {
		ASSERT(k_lower1 < k_upper1 &&
			   0 < k_upper1 && k_lower2 < k_upper2
			   && 0 < k_upper2
			   && k_lower1 <= k_lower2
			   && k_upper1 <= k_upper2);

		// internal variables.
		std::pair<uint64_t, uint64_t> boundaries;
		ElementType e1, e2;
		uint64_t i = 0, j = 0;
		LabelType tmp, tmp2;

		L2.sort_level(k_lower1, k_upper1, target);
		iL.sort_level(k_lower2, k_upper2);

		while (i < L1.load() && j < L2.load()) {
			LabelType::add(tmp, L2[j].label, target);
			if (tmp.is_greater(L1[i].label, k_lower1, k_upper1)) {
				i++;
			} else if (L1[i].label.is_greater(tmp, k_lower1, k_upper1)) {
				j++;
			} else {
				uint64_t i_max, j_max;
				for (i_max = i + 1; i_max < L1.load() && L1[i].is_equal(L1[i_max], k_lower1, k_upper1); i_max++) {}
				for (j_max = j+1;j_max < L2.load();j_max++) {
					LabelType::add(tmp2, L2[j_max].label, target);
					if (!tmp.is_equal(tmp2, k_lower1, k_upper1))  { break; }
				}

				const uint64_t jprev = j;

				// we have found equal elements. But this time we don't have to
				// save the result. Rather we stream join everything up to the final solution.
				for (; i < i_max; ++i) {
					for (j = jprev; j < j_max; ++j) {
						ElementType::add(e1, L1[i], L2[j], k_lower1, k_upper2, -1);
						if (!e1.label.is_equal(target, k_lower1, k_upper1)) {
							L1[i].label.print_binary();
							L2[j].label.print_binary();
							e1.label.print_binary();
							target.print_binary();
							ASSERT(false);
						}



						LabelType::sub(e2.label, e1.label, target);
						e2.label.neg();
						boundaries = iL.search_boundaries(e2, k_lower2, k_upper2);

						// finished?
						if (boundaries.first == boundaries.second) {
							// NOTE: we cannot break out of the two loops
							// only the first one.
							break;
						}

						for (size_t l = boundaries.first; l < boundaries.second; ++l) {
							out.add_and_append(e1, iL[l], 0, LabelLENGTH, -1);
						}
					}
				}
			}
		}
	}

	/// 			out list
	///              ┌───────┐
	///              │  out  │
	///              └───┬───┘
	///          k_lower2│k_upper2
	///     ┌──────────── ───────────┐
	///     │                        │
	/// ┌───┴───┐                ┌───┴───┐ this intermediate list is not saved
	/// │       │                │
	/// │  iL   │
	/// └───────┘                └───┬───┘
	///                              │
	///                      k_lower1│k_upper1
	///                       ┌──────┴──────┐
	///                       │             │
	///                   ┌───┴───┐     ┌───┴───┐
	///                   │const  │     │sorted │
	///                   │   L1  │     │   L2  │
	///                   └───────┘     └───────┘
	///
	/// NOTE: make sure that iT = iT_ + target.
	///		otherwise it this implementation wont find a valid solution
	/// \param out out list: will contain the target and not zeros:w
	/// \param iL sorted on [0, k_upper2)
	/// \param L1 const: dont care
	/// \param L2 sorted on [0, k_upper1)
	/// \param target full target
	/// \param iT intermediate target, make sure that its related to the actual target
	/// 	like: iT=target-iT
	/// \param k_lower1 lower coordinate to match on, on the first level
	/// \param k_upper1 upper coordinate to match on, on the first level
	/// \param k_lower2 NOTE: ignored, assumed == k_upper1
	/// \param k_upper2 upper coordinate to match on, on the last level
	/// \param prepare if true: sort L2 and iL. nothing else.
	static void twolevel_streamjoin_on_iT_v2(List &out, List &iL,
	                                         const List &L1, List &L2,
										     const LabelType &target, const LabelType &iT,
										     const uint32_t k_lower1, const uint32_t k_upper1,
										     const uint32_t k_lower2, const uint32_t k_upper2,
	                                         const bool prepare = true) noexcept {
		constexpr uint32_t filter = -1u;

		if (prepare) {
			L2.sort_level(k_lower1, k_upper1);
			iL.sort_level(k_lower1, k_upper2);
		}
		ASSERT(L2.is_sorted(k_lower1, k_upper1));
		ASSERT(iL.is_sorted(k_lower1, k_upper2));
		(void)k_lower2;

		ElementType tmpe1;
		LabelType t1, t2;
		for (size_t k = 0; k < L1.load(); ++k) {
			LabelType::sub(t1, iT, L1[k].label);
			size_t l = L2.search_level(t1, k_lower1, k_upper1);
			for (; (l < L2.load()) &&
			       (t1.is_equal(L2[l].label, k_lower1, k_upper1));
			     ++l) {
				LabelType::sub(t2, target, L1[k].label);
				LabelType::sub(t2, t2, L2[l].label);
				size_t o = iL.search_level(t2, k_lower1, k_upper2);
				for (; (o < iL.load()) &&
				       (t2.is_equal(iL[o].label, k_lower1, k_upper2));
				     ++o) {
					ElementType::add(tmpe1, iL[o], L1[k]);
					out.add_and_append(tmpe1, L2[l], k_lower1, k_upper2, filter);
				}
			}
		}
	}

	/// 			out list
	///              ┌───────┐
	///              │  out  │
	///              └───┬───┘
	///          k_lower2│k_upper2
	///     ┌──────────── ───────────┐
	///     │                        │
	/// ┌───┴───┐                ┌───┴───┐ this intermediate list is not saved
	/// │       │                │
	/// │  iL   │
	/// └───────┘                └───┬───┘
	///                              │
	///                      k_lower1│k_upper1
	///                       ┌──────┴──────┐
	///                       │             │
	///                   ┌───┴───┐     ┌───┴───┐
	///                   │       │     │       │
	///                   │   L1  │     │   L2  │
	///                   └───────┘     └───────┘
	///
	/// NOTE: the name appendix `_v2` means that the target label is
	/// 	computed in the the outlist. So the out list will contain
	///		the solutions and not zeros.
	/// NOTE: make sure that iT = iT_ + target.
	///		otherwise it this implementation wont find a valid solution
	/// \tparam k_lower1 lower coordinate to match on, on the first level
	/// \tparam k_upper1 upper coordinate to match on, on the first level
	/// \tparam k_lower2 NOTE: ignored, assumed == k_upper1
	/// \tparam k_upper2 upper coordinate to match on, on the last level
	/// \param out outlist
	/// \param iL sorted on [k_lower1, k_upper2), will be sorted if prepare==true
	/// \param L1 dont care
	/// \param L2 sorted on [k_lower1, k_upper1), will be sorted if prepare==true
	/// \param target full target
	/// \param iT intermediate target: should be related to the target
	template<const uint32_t k_lower1, const uint32_t k_upper1,
	         const uint32_t k_lower2, const uint32_t k_upper2>
	static void twolevel_streamjoin_on_iT_v2(List &out, List &iL,
											 const List &L1, List &L2,
											 const LabelType &target,
	                                         const LabelType &iT,
	                                         const bool prepare=true) noexcept {
		static_assert(k_lower1 < k_upper1);
		static_assert(k_lower2 < k_upper2);
		(void)k_lower2;

		if (prepare) {
			L2.template sort_level<k_lower1, k_upper1>();
			iL.template sort_level<k_lower1, k_upper2>();
		}

		ASSERT(L2.is_sorted(k_lower1, k_upper1));
		ASSERT(iL.is_sorted(k_lower1, k_upper2));

		constexpr uint32_t filter = -1u;
		constexpr bool sub = false;

		ElementType tmpe1;
		LabelType t1, t2;
		for (size_t k = 0; k < L1.load(); ++k) {
			LabelType::sub(t1, iT, L1[k].label);
			size_t l = L2.template search_level<k_lower1, k_upper1>(t1);
			for (; (l < L2.load()) &&
				   (t1.template is_equal<0, k_upper1>(L2[l].label));
				   ++l) {
				LabelType::sub(t2, target, L1[k].label);
				LabelType::sub(t2, t2, L2[l].label);
				size_t o = iL.template search_level<k_lower1, k_upper2>(t2);
				for (; (o < iL.load()) &&
					   (t2.template is_equal<k_lower1, k_upper2>(iL[o].label));
					   ++o) {
					ElementType::add(tmpe1, iL[o], L1[k]);
					out.template add_and_append
					        <k_lower1, k_upper2, filter, sub>
					        (tmpe1, L2[l]);
				}
			}
		}
	}

	/// TODO bild + test not finished, doc
	/// NOTE: v2 means that the `label` of the output elements in `out`
	///		are the target. So this function actually returns targets and not
	/// 	zeros.
	/// \tparam k_lower1
	/// \tparam k_upper1
	/// \tparam k_lower2
	/// \tparam k_upper2
	/// \tparam HashMap
	/// \param out
	/// \param iL
	/// \param L1
	/// \param L2
	/// \param target
	/// \param iT
	template<const uint32_t k_lower1, const uint32_t k_upper1,
			 const uint32_t k_lower2, const uint32_t k_upper2,
			 typename HashMap1,
	         typename HashMap2>
#if __cplusplus > 201709L
	requires HashMapAble<HashMap1> &&
	 		 HashMapAble<HashMap2>
#endif
	static void twolevel_streamjoin_on_iT_hashmap_v2(List &out,
	                                                 const HashMap1 &hmiL,
											 		 const List &L1,
													 const List &L2,
	                                                 const HashMap2 &hmL2,
	                                                 const LabelType &target,
	                                                 const LabelType &iT) {
		static_assert(k_lower1 < k_upper1);
		static_assert(k_lower2 < k_upper2);
		static_assert(k_lower1 < k_upper1);
		(void)k_lower1;
		(void)k_lower2;
		using LoadType1 = typename HashMap1::load_type;
		using LoadType2 = typename HashMap2::load_type;
		constexpr uint32_t filter = -1u;
		constexpr bool sub = false;

		ElementType te1, te2;
		LabelType t1, t2;
		LoadType1 load1 = 0;
		LoadType2 load2 = 0;

		for (size_t k = 0; k < L1.load(); ++k) {
			LabelType::sub(t1, iT, L1[k].label);
			// size_t l = L2.template search_level<0, k_upper1>(t1);

			const size_t s2 = hmL2.find(t1.value(), load2);
			for (size_t l2 = s2; l2 < (s2 + load2); ++l2) {
			// for (; (l < L2.load()) &&
			// 	   (t1.template is_equal<0, k_upper1>(L2[l].label));
			// 	   ++l) {

				const size_t b1 = hmL2[l2];
				ASSERT(b1 < L2.load());
				const LabelType t3 = L2[b1].label;
				LabelType::sub(t2, target, L1[k].label);
				LabelType::sub(t2, t2, t3);

				ElementType::add(te1, L1[k], L2[b1]);

				// NOTE: its shifted
				const size_t s1 = hmiL.find(t2.value(), load1);
				for (size_t l1 = s1; l1 < (s1 + load1); ++l1) {
				// size_t o = iL.template search_level<0, k_upper2>(t2);
				// for (; (o < iL.load()) &&
				// 	   (t2.template is_equal<0, k_upper2>(iL[o].label));
				// 	   ++o) {
					const size_t a1 = hmiL[l1].first;
					const size_t a2 = hmiL[l1].second;
					ASSERT(a1 < L1.load());
					ASSERT(a2 < L2.load());
					ElementType::add(te2, L1[a1], L2[a2]);
					out.template add_and_append
							<k_lower1, k_upper2, filter, sub>
							(te1, te2);
				}
			}
		}
		std::cout << "kek" << std::endl;
	}

	/// TODO bild
	/// NOTE: v2 means that the `label` of the output elements in `out`
	///		are the target. So this function actually returns targets and not
	/// 	zeros.
	/// \tparam k_lower1
	/// \tparam k_upper1
	/// \tparam k_lower2
	/// \tparam k_upper2
	/// \tparam HashMap1
	/// \tparam HashMap2
	/// \param out
	/// \param iL
	/// \param L1
	/// \param L2
	/// \param target
	/// \param iT
	template<const uint32_t k_lower1, const uint32_t k_upper1,
			 const uint32_t k_lower2, const uint32_t k_upper2,
			 typename HashMap1,
	         typename HashMap2,
	         typename HashMap3>
#if __cplusplus > 201709L
	requires HashMapAble<HashMap1> &&
	         HashMapAble<HashMap2> &&
			 HashMapAble<HashMap3>
#endif
	static void twolevel_streamjoin_on_iT_hashmap_v2(HashMap3 &out, const HashMap2 &iL,
											 		const List &L1, const HashMap1 &L2,
											 		const LabelType &target, const LabelType &iT) {
		static_assert(k_lower1 < k_upper1);
		static_assert(k_lower2 < k_upper2);
		static_assert(k_lower1 < k_upper1);
		(void)k_lower1;
		(void)k_lower2;

		constexpr uint32_t filter = -1u;
		constexpr bool sub = false;

		ElementType tmpe1;
		LabelType t1, t2;
		for (size_t k = 0; k < L1.load(); ++k) {
			LabelType::sub(t1, iT, L1[k].label);
			size_t l = L2.template search_level<0, k_upper1>(t1);
			for (; (l < L2.load()) &&
				   (t1.template is_equal<0, k_upper1>(L2[l].label));
				   ++l) {
				LabelType::sub(t2, target, L1[k].label);
				LabelType::sub(t2, t2, L2[l].label);
				size_t o = iL.template search_level<0, k_upper2>(t2);
				for (; (o < iL.load()) &&
					   (t2.template is_equal<0, k_upper2>(iL[o].label));
					   ++o) {
					ElementType::add(tmpe1, iL[o], L1[k]);
					out.template add_and_append
							<0, k_upper2, filter, sub>
							(tmpe1, L2[l]);
				}
			}
		}
	}

	///  Schematic View on how the algorithm Works.
	///  The algorithm does not care what weight is on each element, nor does it care about the output size.
	///  IMPORTANT: The Output size IS NOT GUESSED ahead. Because we dont know hom many representations of the solution there are.
	///                         Out
	///              +-----------------------+
	///              |                       |
	///              +-----------^-----------+
	///                          |
	///             +------------+------------+
	///             |        Target           |   The merge is on [k_lower, k_upper] coordinates.
	/// +-----------+-----------+ +-----------+-----------+
	/// |        sorted         | |        sorted         |
	/// +-----------------------+ +-----------------------+
	///            L_1                       L_2
	/// given two lists L1, L2. It outputs every elements which are the same on the specified coordinates in 'out'
	/// \param out 	Output List
	/// \param L1 	Input List (sorted)
	/// \param L2 	Input List (will be changed. Target is added to every element in it.)
	/// \param ta __MUST__ be an uint64_t const_array containing two elements. The first will be used as the lower cooridnate bound to match the elements on, where as the latter one will be the upper bound.
	static void join2lists(List &out,
	                       List &L1,
	                       List &L2,
	                       const LabelType &target,
	                       const std::vector<uint64_t> &lta,
	                       const bool prepare = true) noexcept {
		ASSERT(lta.size() >= 2);
		join2lists(out, L1, L2, target, lta[0], lta[1], prepare);
	}

	///                         Out
	///              +-----------------------+
	///              |                       |
	///              +-----------^-----------+
	///                          |
	///             +------------+------------+
	///             |        Target           |   The merge is on [k_lower, k_upper) coordinates.
	/// +-----------+-----------+ +-----------+-----------+
	/// |        sorted         | |        sorted         |
	/// +-----------------------+ +-----------------------+
	///            L_1                       L_2
	/// NOTE: the output list will only contain zeros in the label
	/// NOTE: the output list will contains zeros on the dimensions of the target
	/// \param out the values will be s.t. out[i].value * Matrix = target
	/// 	the labels will be zero. You need to recompute the label on you own
	/// \param L1 first list, will be sorted on [k_lower, k_upper) if prepare==true
	/// \param L2 second list,will be sorted on [k_lower, k_upper) if prepare==true
	/// \param target
	/// \param k_lower lower coordinate to match on
	/// \param k_upper upper coordinate to match on
	/// \param prepare if true: the target will be added/subtracted into the left input list L2.
	/// 			and the two lists will be sorted
	static void join2lists(List &out, List &L1, List &L2,
						   const LabelType &target,
						   const uint32_t k_lower,
						   const uint32_t k_upper,
						   bool prepare = true) noexcept {
		ASSERT(k_lower < k_upper && 0 < k_upper);
		out.set_load(0);

		constexpr bool sub = !LabelType::binary();

		if ((!target.is_zero()) && (prepare)) {
			for (size_t s = 0; s < L2.load(); ++s) {
				if constexpr (sub) {
					LabelType::sub(L2[s].label, target, L2[s].label, k_lower, k_upper);
				} else {
					LabelType::add(L2[s].label, target, L2[s].label, k_lower, k_upper);
				}
			}

			L1.sort_level(k_lower, k_upper);
			L2.sort_level(k_lower, k_upper);
		}
		ASSERT(L1.is_sorted(k_lower, k_upper));
		ASSERT(L2.is_sorted(k_lower, k_upper));

		uint64_t i = 0, j = 0;
		while (i < L1.load() && j < L2.load()) {
			if (L2[j].is_greater(L1[i], k_lower, k_upper)) {
				i++;
			} else if (L1[i].is_greater(L2[j], k_lower, k_upper)) {
				j++;
			} else {
				uint64_t i_max=i+1ull, j_max=j+1ull;
				// if elements are equal find max index in each list, such that they remain equal
				for (; i_max < L1.load() && L1[i].is_equal(L1[i_max], k_lower, k_upper); i_max++) {}
				for (; j_max < L2.load() && L2[j].is_equal(L2[j_max], k_lower, k_upper); j_max++) {}

				const uint64_t jprev = j;
				for (; i < i_max; ++i) {
					for (j = jprev; j < j_max; ++j) {
						out.add_and_append(L1[i], L2[j], k_lower, k_upper, -1, sub);
#ifdef DEBUG
						const uint64_t b = out.load() - 1;
						if (!out[b].label.is_zero(k_lower, k_upper)) {
							L1[i].print_binary();
							L2[j].print_binary();
							out[b].print_binary();
							ASSERT(false);
						}
#endif
					}
				}
			}
		}
	}

	///                         Out
	///              +-----------------------+
	///              |                       |
	///              +-----------^-----------+
	///                          |
	///             +------------+------------+
	///             |        Target           |   The merge is on [k_lower, k_upper) coordinates.
	/// +-----------+-----------+ +-----------+-----------+
	/// |        sorted         | |        sorted         |
	/// +-----------------------+ +-----------------------+
	///            L_1                       L_2
	/// NOTE: the output list will only contain zeros in the label
	/// NOTE: the output list will contains zeros on the dimensions of the target
	/// \tparam k_lower lower coordinate to match on
	/// \tparam k_upper upper coordinate to match on
	/// \param out the values will be s.t. out[i].value * Matrix = target
	/// 	the labels will be zero. You need to recompute the label on you own
	/// \param L1 first list, will be sorted on [k_lower, k_upper) if prepare==true
	/// \param L2 second list,will be sorted on [k_lower, k_upper) if prepare==true
	/// \param target
	/// \param prepare if true: the target will be added/subtracted into the left input list L2.
	/// 			and the two lists will be sorted
	template<const uint32_t k_lower, const uint32_t k_upper>
	static void join2lists(List &out, List &L1, List &L2,
						   const LabelType &target,
						   bool prepare = true) noexcept {
		static_assert(k_lower < k_upper && 0 < k_upper);
		out.set_load(0);

		constexpr bool sub = !LabelType::binary();
		constexpr uint32_t filter = -1u;

		if ((!target.is_zero()) && (prepare)) {
			for (size_t s = 0; s < L2.load(); ++s) {
				if constexpr (sub) {
					LabelType::template sub<k_lower, k_upper>(L2[s].label, target, L2[s].label);
				} else {
					LabelType::template add<k_lower, k_upper>(L2[s].label, target, L2[s].label);
				}
			}

			L1.template sort_level<k_lower, k_upper>();
			L2.template sort_level<k_lower, k_upper>();
		}

		ASSERT(L1.is_sorted(k_lower, k_upper));
		ASSERT(L2.is_sorted(k_lower, k_upper));

		uint64_t i = 0, j = 0;
		while (i < L1.load() && j < L2.load()) {
			if (L2[j].template is_greater<k_lower, k_upper>(L1[i])) {
				i++;
			} else if (L1[i].template is_greater<k_lower, k_upper>(L2[j])) {
				j++;
			} else {
				uint64_t i_max=i+1ull, j_max=j+1ull;
				// if elements are equal find max index in each list, such that they remain equal
				for (; i_max < L1.load() && L1[i].template is_equal<k_lower, k_upper>(L1[i_max]); i_max++) {}
				for (; j_max < L2.load() && L2[j].template is_equal<k_lower, k_upper>(L2[j_max]); j_max++) {}

				const uint64_t jprev = j;
				for (; i < i_max; ++i) {
					for (j = jprev; j < j_max; ++j) {
						out.template add_and_append<k_lower, k_upper, filter, sub>(L1[i], L2[j]);
					}
				}
			}
		}
	}

	///                         Out
	///              +-----------------------+
	///              |                       |
	///              +-----------^-----------+
	///                          |
	///             +------------+------------+
	///             |        Target           |   The merge is on [k_lower, k_upper) coordinates.
	/// +-----------+-----------+ +-----------+-----------+
	/// |         const         | |sorted/but not altered |
	/// +-----------------------+ +-----------------------+
	///            L_1                       L_2
	/// NOTE:
	/// 	in contrast to `join2lists` does this function not alter the
	/// 	values of L2 (except for sorting them). This can be useful in cases
	/// 	in which we do not want to recover the original values from a list and
	/// 	at the same time its cheap to add in the target into the values which
	/// 	are compared (SubSetSum Setting).
	/// NOTE: the output list will contain the target between [k_lower, k_upper)
	/// \param out output list
	/// \param L1 const assumed to be sorted
	/// \param L2 cannot be const, as its sorted
	/// \param target is added
	/// \param k_lower lower limit
	/// \param k_upper upper limit
	/// \param prepare if true: will sort L1
	static void join2lists_on_iT(List &out,
	                             List &L1, List &L2,
	                       		 const LabelType &target,
	                       		 const uint32_t k_lower,
	                       		 const uint32_t k_upper,
	                             const bool prepare=true) noexcept {
		ASSERT(k_lower < k_upper && 0 < k_upper);
		out.set_load(0);

		constexpr static bool sub = !LabelType::binary();
		constexpr static uint32_t filter = -1;
		if (prepare) {
			L1.sort_level(k_lower, k_upper);
		}
		ASSERT(L1.is_sorted(k_lower, k_upper));

		// NOTE: will always be sorted, as we dont know the
		// target befor hand
		L2.template sort_level<sub>(k_lower, k_upper, target);

		// standard comparison oeprator
		auto op = [](LabelType &c, const LabelType &a, const LabelType &b,
					 const uint64_t l, const uint64_t h) {
			if constexpr (sub) {
				  LabelType::sub(c, a, b, l, h);
			} else {
				  LabelType::add(c, a, b, l, h);
			}
		};

		LabelType tmp, tmp2;
		uint64_t i=0, j=0;
		while ((i < L1.load()) && (j < L2.load())) {
			op(tmp, target, L2[j].label, k_lower, k_upper);

			if (tmp.is_greater(L1[i].label, k_lower, k_upper)) {
				i++;
			} else if (L1[i].label.is_greater(tmp, k_lower, k_upper)) {
				j++;
			} else {
				uint64_t i_max=i+1ull, j_max=j+1ull;
				// if elements are equal find max index in each list, such that they remain equal
				for (;i_max < L1.load() && L1[i].is_equal(L1[i_max], k_lower, k_upper); i_max++) {}
				for (;j_max < L2.load(); j_max++) {
					op(tmp2, target, L2[j_max].label, k_lower, k_upper);
					if (!tmp.is_equal(tmp2, k_lower, k_upper))  { break; }
				}

				const uint64_t jprev = j;
				for (; i < i_max; ++i) {
					for (j = jprev; j < j_max; ++j) {
						out.add_and_append(L1[i], L2[j], k_lower, k_upper, filter);

#ifdef DEBUG
						const uint64_t b = out.load() - 1;
						if (!out[b].label.is_equal(target, k_lower, k_upper)) {
							L1[i].label.print_binary();
							L2[j].label.print_binary();
							out[b].label.print_binary();
							target.print_binary();
							ASSERT(false);
						}
#endif
					}
				}
			}
		}
	}

	///            out
	///         ┌───────┐
	///         │       │
	///         └───┬───┘
	///     k_lower1│k_upper1
	///     ┌───────┴───────┐
	///     │               │
	/// ┌───┴───┐       ┌───┴───┐
	/// │const  │       │sorted │
	/// └───────┘       └───────┘
	///    L1              L2
	/// NOTE: v2 means that the `label` of the output elements in `out`
	///		are the target. So this function actually returns targets and not
	/// 	zeros.
	/// NOTE: the input lists will only be sorted, not altered. So no need
	/// 	to reocver the lists
	/// NOTE: L1 const
	/// NOTE: L2 needs to be sorted, will be sorted if prepare==true
	/// \param out output list
	/// \param L1 first input list, doesnt need to be sorted
	/// \param L2 second input list, needs to be sorted
	/// \param target
	/// \param k_lower lower coordinate to match on
	/// \param k_upper upper coordinate to match on
	/// \param prepare if true L1 and L2 will bie
	static void join2lists_on_iT_v2(List &out,
	                                const List &L1, List &L2,
							  		const LabelType &target,
							  		const uint32_t k_lower,
							  		const uint32_t k_upper,
	                                const bool prepare=true) noexcept {
		ASSERT(k_lower < k_upper && 0 < k_upper);
		if (prepare) {
			L2.sort_level(k_lower, k_upper);
		}

		ASSERT(L2.is_sorted(k_lower, k_upper));

		constexpr uint32_t filter = -1u;
		LabelType sigma_t;
		for (size_t i = 0; i < L1.load(); ++i) {
			// NOTE sub will be remapped to add in the binary case
			LabelType::sub(sigma_t, target, L1[i].label, k_lower, k_upper);
			size_t j = L2.search_level(sigma_t, k_lower, k_upper);
			for (; (j < L2.load()) &&
				   (sigma_t.is_equal(L2[j].label,k_lower, k_upper));
				   ++j) {
				out.add_and_append(L1[i], L2[j], k_lower, k_upper, filter);
			}
		}
	}

	///            out
	///         ┌───────┐
	///         │       │
	///         └───┬───┘
	///     k_lower1│k_upper1
	///     ┌───────┴───────┐
	///     │               │
	/// ┌───┴───┐       ┌───┴───┐
	/// │sorted │       │sorted │
	/// └───────┘       └───────┘
	///    L1              L2
	/// NOTE: L2 needs to be sorted
	/// \tparam k_lower lower coordinate to match on
	/// \tparam k_upper upper coordinate to match on
	/// \param out
	/// \param L1 first input list, doesnt need to be sorted
	/// \param L2 second input list, needs to be sorted
	/// \param target
	/// \param prepare if true L2 will be sorted
	template<const uint32_t k_lower,
	         const uint32_t k_upper>
	static void join2lists_on_iT_v2(List &out,
								    const List &L1, List &L2,
								    const LabelType &target,
	                                const bool prepare=true) noexcept {
		ASSERT(k_lower < k_upper && 0 < k_upper);
		out.set_load(0);
		constexpr uint32_t filter = -1u;

		if (prepare) {
			L2.template sort_level<k_lower, k_upper>();
		}

		ASSERT(L2.is_sorted(k_lower, k_upper));

		LabelType sigma_t;
		for (size_t i = 0; i < L1.load(); ++i) {
			/// NOTE: sub will be add in binary
			LabelType::template sub
			        <k_lower, k_upper>
			        (sigma_t, target, L1[i].label);
			size_t j = L2.template search_level<k_lower, k_upper>(sigma_t);
			for (; (j < L2.load()) &&
				   (sigma_t.template is_equal<k_lower, k_upper>(L2[j].label)); ++j) {
				out.template add_and_append
				        <k_lower, k_upper, filter, false>
				        (L1[i], L2[j]);
			}
		}
	}

	///         ┌───────┐
	///         │       │
	///         └───────┘
	///     k_lower1│k_upper1
	///     ┌───────┴───────┐
	///     │               │
	/// ┌───┴───┐      ┌────┴────┐
	/// │ const │      └┐       ┌┘
	/// │       │       └┐     ┌┘
	/// └───────┘        └─────┘  hashed
	///    L1              HM <-------------L2
	/// NOTE: v2 means that the `label` of the output elements in `out`
	///		are the target. So this function actually returns targets and not
	/// 	zeros.
	/// NOTE: the output will a be list
	/// \tparam k_lower lower coordinate to match on
	/// \tparam k_upper upper coordinate to match on
	/// \tparam HashMap
	/// \param out output list
	/// \param L1 const: Does not need to be sorted
	/// \param L2 const: Does not need to be sorted
	/// \param target
	/// \param hm if prepare == true: L2 will be hashed into hm
	/// \param prepare
	template<const uint32_t k_lower,
			 const uint32_t k_upper,
	         typename HashMap>
#if __cplusplus > 201709L
		requires HashMapAble<HashMap>
#endif
	static void join2lists_on_iT_hashmap_v2(List &out,
											const List &L1, const List &L2,
											HashMap &hm,
											const LabelType &target,
											const bool prepare=true) noexcept {
		ASSERT(k_lower < k_upper && 0 < k_upper);
		constexpr uint32_t filter = -1u;
		using LoadType = typename HashMap::load_type;
		out.set_load(0);

		if (prepare) {
			hm.clear();
			for (size_t i = 0; i < L2.load(); ++i) {
				hm.insert(L2[i].label.value(), i);
			}
		}

		LabelType sigma_t;
		LoadType load = 0;
		for (size_t i = 0; i < L1.load(); ++i) {
			LabelType::template sub<k_lower, k_upper>(sigma_t, target, L1[i].label);

			size_t s = hm.find(sigma_t.value(), load);
			for (size_t k = s; k < s + load; ++k) {
				const size_t j = hm[k];
				out.template add_and_append
				        <k_lower, k_upper, filter, false>
				        (L1[i], L2[j]);
			}
		}
	}

	/// 		out HM
	///        ┌─────────┐
	///        └┐       ┌┘
	///         └┐     ┌┘
	///          └──┬──┘
	///     k_lower1│k_upper1
	///     ┌───────┴───────┐
	///     │               │
	/// ┌───┴───┐      ┌────┴────┐
	/// │ const │      └┐       ┌┘
	/// │       │       └┐     ┌┘
	/// └───────┘        └─────┘  hashing
	///    L1              HM2 <---------- L2
	/// NOTE: v2 means that the `label` of the output elements in `out`
	///		are the target. So this function actually returns targets and not
	/// 	zeros.
	/// NOTE: the output will a be hashmap
	/// NOTE: the elemens of the output hashmap are shifted down by `k_upper`
	/// \tparam k_lower lower coordinate to match on
	/// \tparam k_upper upper coordinate to match on
	/// \tparam HashMapIn
	/// \tparam HashMapOut
	/// \param out
	/// \param L1 input list const, will NOT be sorted
	/// \param L2 input list const, will be hashed into HM2 if prepare==true
	/// \param target
	/// \param hm2 hashmap of list L2
	/// \param prepare if true == hashses L2 into HM2
	template<const uint32_t k_lower,
			 const uint32_t k_upper,
			 typename HashMapIn,
	         typename HashMapOut>
#if __cplusplus > 201709L
	requires HashMapAble<HashMapIn> &&
	         HashMapAble<HashMapOut>
#endif
	static void join2lists_on_iT_hashmap_v2(HashMapOut &out,
											const List &L1, const List &L2,
											HashMapIn &hm2,
											const LabelType &target,
											const bool prepare=true) noexcept {
		ASSERT(k_lower < k_upper && 0 < k_upper);
		// constexpr uint32_t filter = -1u;
		using LoadType = typename HashMapIn::load_type;
		using HMOutValueType = HashMapOut::data_type;
		out.clear();

		if (prepare) {
			hm2.clear();
			for (size_t i = 0; i < L2.load(); ++i) {
				hm2.insert(L2[i].label.value(), i);
			}
		}

		LabelType sigma_t;
		LoadType load = 0;
		LabelType e;
		for (size_t i = 0; i < L1.load(); ++i) {
			LabelType::template sub<k_lower, k_upper>(sigma_t, target, L1[i].label);

			size_t s = hm2.find(sigma_t.value(), load);
			for (size_t k = s; k < s + load; ++k) {
				const size_t j = hm2[k];
				LabelType::add(e, L1[i].label, L2[j].label);
				// NOTE: NOTE that is shifted
				out.insert(e.value(), HMOutValueType{i, j});
				//out.template add_and_append
				//        <k_lower, k_upper, filter, false>
				//        (L1[i], L2[j]);
			}
		}
	}


	// Schematic view on the Algorithm.
	// The algorithm ignores any facts about th weight, nor does it guess the output size.
	// The Intermediate List is created every time you call this function. So if called repeatedly, maybe write it yourself. or set the list to static.
	//                                                  Out
	//                                       +-----------+----------+
	//                                       | k_lower   |  k_higher|                                         Level 2
	//                                       +----------------------+
	// Merge on k_upper                                  |
	//                         +-------------------------+-------------------------+  Stream Merge
	//                         |                Merge on target                    |
	//             +-----------------------+                          + - - - - - - - - - - - +
	//             | k_lower   |  k_higher | Intermediate List        |            |          |               Level 1
	//             +-----------------------+ Will be save.            +- - - - - - - - - - - -+ NOT Saved
	// Merge on k_lower        |                                                   |
	//            +------------+------------+                        +-------------+-----------+
	//            |      Merge on random R  |                        |    merge on R+target    |
	//+-----------------------+ +-----------------------+ +-----------------------+ +-----------------------+
	//| k_lower   |  k_higher | | k_lower   |  k_higher | |          |            | |          |            | Level 0
	//+-----------+-----------+ +-----------+-----------+ +----------+------------+ +----------+------------+
	//           L_1                       L_2                      L_3                       L_4
	///
	/// \param out output list. All targets which are a solution to the 4 kxor problem.
	/// \param L1 	(sorted)
	/// \param L2 	(sorted+R) intermediate target is added.
	/// \param L3 	(sorted)
	/// \param L4 	(sorted+R+target) intermediate target and target is added
	/// \param target
	/// \param lta
	static void join4lists(List &out, List &L1, List &L2, List &L3, List &L4,
	                             const LabelType &target,
	                             const std::vector<uint64_t> &lta,
	                             const bool prepare = true) noexcept {
		ASSERT(lta.size() >= 3);
		// limits: k_lower1, k_upper1 for the lowest level tree. And k_lower2, k_upper2 for highest level. There are
		// only two levels..., so obviously k_upper1=k_lower2
		const uint64_t k_lower1 = lta[0], k_upper1 = lta[1];
		const uint64_t k_lower2 = lta[1], k_upper2 = lta[2];
		join4lists(out, L1, L2, L3, L4, target, k_lower1, k_upper1, k_lower2, k_upper2, prepare);
	}

	///                     ┌───────┐
	///                     │       │
	///                     │ out   │
	///                     └───┬───┘
	///                 k_lower2│k_upper2
	///            ┌────────────┼───────────┐
	///            │                        │
	///        ┌───┴───┐                ┌───┴───┐
	///        │  iL   │will be                 │ wont be
	///        │       │allocated       │         allocated
	///        └───┬───┘                └───┬───┘
	///    k_lower1│k_upper1        k_lower1│k_upper1
	///     ┌──────┴──────┐          ┌──────┴──────┐
	///     │             │          │             │
	/// ┌───┴───┐     ┌───┴───┐  ┌───┴───┐     ┌───┴───┐
	/// │sorted │     │sorted │  │sorted │     │sorted │
	/// └───────┘     └───────┘  └───────┘     └───────┘
	///    L1             L2         L3            L4
	///
	/// fully computes the 4-tree algorithm in a stream join fashion. That means only
	/// a single intermediate list is needed.
	/// finds x1,x2,x3,x4 \in L1,L2,L3,L4 s.t. x1+x2+x3+x4 = target
	/// \param out out List, each solution will be zero
	/// \param L1 must be sorted (if prepare==true: will be sorted)
	/// \param L2 must be sorted (if prepare==true: will be sorted)
	/// \param L3 must be sorted (if prepare==true: will be sorted)
	/// \param L4 must be sorted (if prepare==true: will be sorted)
	/// \param target finds
	/// \param k_lower1 lower coordinate to match on in the base lists
	/// \param k_upper1 upper coordinate to match on in the base lists
	/// \param k_lower2 lower coordinate to match on in the intermediate lists
	/// \param k_upper2 upper coordinate to match on in the intermediate lists
	/// \param prepare if true the intermediate targets will be added into the base lists
	/// 			and all lists will be sorted
	static void join4lists(List &out, List &L1, List &L2, List &L3, List &L4,
	                             const LabelType &target,
	                             const uint64_t k_lower1, const uint64_t k_upper1,
	                             const uint64_t k_lower2, const uint64_t k_upper2,
	                             const bool prepare = true) noexcept {
		ASSERT(k_lower1 < k_upper1 &&
		       0 < k_upper1 && k_lower2 < k_upper2
		       && 0 < k_upper2 && k_lower1 <= k_lower2
		       && k_upper1 < k_upper2
		       && L1.load() > 0 && L2.load() > 0
		       && L3.load() > 0 && L4.load() > 0);

		// Intermediate Element, List, Target
		List iL{static_cast<size_t>(L1.size() * config.intermediatelist_size_factor)};
		LabelType iT; iT.zero();

		// reset everything
		out.set_load(0);

		const size_t size = std::min({L1.load(), L2.load(), L4.load(), L3.load()});
		constexpr static bool sub = !LabelType::binary();
		auto op = [](LabelType &c, const LabelType &a, const LabelType &b,
		             const uint64_t l, const uint64_t h) {
			if constexpr (sub) { LabelType::sub(c, a, b, l, h);}
			else { LabelType::add(c, a, b, l, h); }
		};

		// TODO document these changes, what is added into what list in the picture above
		// prepare baselists
		if ((!target.is_zero()) && prepare) {
			iT.random(0, (1ull << k_upper1) + 1);

			LabelType R2;
			LabelType::sub(R2, iT, target, k_lower1, k_upper2);

			for (size_t i = 0; i < size; ++i) {
				op(L2[i].label, iT, L2[i].label, k_lower1, k_upper2);
				LabelType::add(L4[i].label, R2, L4[i].label, k_lower1, k_upper2);
				L3[i].label.neg(k_lower1, k_upper2);
			}

			L1.sort_level(k_lower1, k_upper1);
			L2.sort_level(k_lower1, k_upper1);
		}

		// NOTE: the intermediate target `R` is ingored in this call
		join2lists(iL, L1, L2, iT, k_lower1, k_upper1, false);

		// early exit
		if (iL.load() == 0) {
			return;
		}

		// Now run the merge procedure for the right part of the tree.
		twolevel_streamjoin(out, iL, L3, L4, k_lower1, k_upper1, k_lower2, k_upper2, prepare);
	}

	///                     ┌───────┐
	///                     │ out   │
	///                     └───┬───┘
	///                 k_lower2│k_upper2
	///            ┌────────────┼───────────┐
	///            │                        │
	///        ┌───┴───┐                ┌───┴───┐
	///        │  iL   │will be                 │ wont be
	///        │       │allocated       │         allocated
	///        └───┬───┘                └───┬───┘
	///    k_lower1│k_upper1        k_lower1│k_upper1
	///     ┌──────┴──────┐          ┌──────┴──────┐
	///     │             │          │             │
	/// ┌───┴───┐     ┌───┴───┐  ┌───┴───┐     ┌───┴───┐
	/// │const  │     │sorted │  │const  │     │sorted │
	/// └───────┘     └───────┘  └───────┘     └───────┘
	///    L1             L2         L3            L4
	/// NOTE: v2 means that the `label` of the output elements in `out`
	///		are the target. So this function actually returns targets and not
	/// 	the zeros.
	/// NOTE: allocates iL
	/// \param out output list
	/// \param L1 dont care: can be sorted, but dont have to
	/// \param L2 must be sorted (if prepare==true: will be sorted)
	/// \param L3 dont care: can be sorted, but dont have to
	/// \param L4 must be sorted (if prepare==true: will be sorted)
	/// \param target
	/// \param k_lower1 lower coordinate to match on in the base lists
	/// \param k_upper1 upper coordinate to match on in the base lists
	/// \param k_lower2 lower coordinate to match on in the intermediate lists
	/// \param k_upper2 upper coordinate to match on in the intermediate lists
	/// \param prepare if `true` L2 and L4 will sorted. NO intermediate target
	/// 		will be added into the lists
	static void join4lists_on_iT_v2(List &out,
	                                const List &L1, List &L2,
	                                const List &L3, List &L4,
						      		const LabelType &target,
						      		const uint32_t k_lower1, const uint32_t k_upper1,
						      		const uint32_t k_lower2, const uint32_t k_upper2,
						      		const bool prepare = true) noexcept {
		(void)k_lower2;
		List iL{static_cast<size_t>(L1.size() * config.intermediatelist_size_factor)};
		out.set_load(0);

		// reset everything
		if (prepare) {
			L2.sort_level(k_lower1, k_upper1);
			L4.sort_level(k_lower1, k_upper1);
		}

		ASSERT(L2.is_sorted(k_lower1, k_upper1));
		ASSERT(L4.is_sorted(k_lower1, k_upper1));

		ElementType tmpe1;
		LabelType t1, iT;
		iT.random(0, 1ull << k_upper1); // TODO only correct for SubSetSum
		join2lists_on_iT_v2(iL, L1, L2, iT, k_lower1, k_upper1, false);
		// early exit
		if (iL.load() == 0) {
			return;
		}

		iL.sort_level(0, k_upper2);
		LabelType::sub(t1, target, iT);
		twolevel_streamjoin_on_iT_v2(out, iL, L3, L4, target, t1,
		                             k_lower1, k_upper1, k_lower2, k_upper2,
		                             false);
	}

	///                     ┌───────┐
	///                     │ out   │
	///                     └───┬───┘
	///                 k_lower2│k_upper2
	///            ┌────────────┼───────────┐
	///            │                        │
	///        ┌───┴───┐                ┌───┴───┐
	///        │  iL   │will be                 │ wont be
	///        │       │allocated       │         allocated
	///        └───┬───┘                └───┬───┘
	///    k_lower1│k_upper1        k_lower1│k_upper1
	///     ┌──────┴──────┐          ┌──────┴──────┐
	///     │             │          │             │
	/// ┌───┴───┐     ┌───┴───┐  ┌───┴───┐     ┌───┴───┐
	/// │const  │     │sorted │  │const  │     │sorted │
	/// └───────┘     └───────┘  └───────┘     └───────┘
	///    L1             L2         L3            L4
	/// NOTE: v2 means that the `label` of the output elements in `out`
	///		are the target. So this function actually returns targets and not
	/// 	the zeros.
	/// NOTE: allocates iL
	/// \param out output list
	/// \param L1 dont care: can be sorted, but dont have to
	/// \param L2 must be sorted (if prepare==true: will be sorted)
	/// \param L3 dont care: can be sorted, but dont have to
	/// \param L4 must be sorted (if prepare==true: will be sorted)
	/// \param target
	/// \param k_lower1 lower coordinate to match on in the base lists
	/// \param k_upper1 upper coordinate to match on in the base lists
	/// \param k_lower2 lower coordinate to match on in the intermediate lists
	/// \param k_upper2 upper coordinate to match on in the intermediate lists
	/// \param prepare if `true` L2 and L4 will sorted. NO intermediate target
	/// 		will be added into the lists
	template<const uint32_t k_lower1, const uint32_t k_upper1,
			 const uint32_t k_lower2, const uint32_t k_upper2>
	static void join4lists_on_iT_v2(List &out,
	                                const List &L1, List &L2,
	                                const List &L3, List &L4,
	                                const LabelType &target,
	                                const bool prepare = true) noexcept {
		(void) k_lower2;
		List iL{static_cast<size_t>(L1.size() * config.intermediatelist_size_factor)};
		out.set_load(0);

		// reset everything
		if (prepare) {
			L2.template sort_level<k_lower1, k_upper1>();
			L4.template sort_level<k_lower1, k_upper1>();
		}

		ElementType tmpe1;
		LabelType t1, iT;
		iT.random(0, 1ull << k_upper1);
		join2lists_on_iT_v2<k_lower1, k_upper1>(iL, L1, L2, iT, false);
		// early exit
		if (iL.load() == 0) {
			return;
		}

		iL.template sort_level<k_lower1, k_upper2>();
		LabelType::sub(t1, target, iT);
		twolevel_streamjoin_on_iT_v2
		        <k_lower1, k_upper1, k_lower2, k_upper2>
		        (out, iL, L3, L4, target, t1, false);
	}

	///                     ┌───────┐
	///                     │ out   │
	///                     └───┬───┘
	///                 k_lower2│k_upper2
	///            ┌────────────┼───────────┐
	///            │                        │
	///        ┌───┴───┐                ┌───┴───┐
	///        │  iL   │will be                 │ wont be
	///        │       │allocated       │         allocated
	///        └───┬───┘                └───┬───┘
	///    k_lower1│k_upper1        k_lower1│k_upper1
	///     ┌──────┴──────┐          ┌──────┴──────┐
	///     │             │          │             │
	/// ┌───┴───┐     ┌───┴───┐  ┌───┴───┐     ┌───┴───┐ both L3 and L4
	/// │const  │     │sorted │     					 are simply L1 and
	/// └───────┘     └───────┘  └───────┘     └───────┘ L2. Wont be allocated
	///    L1             L2     L3 = L1 		L4 = L2
	/// NOTE: v2 means that the `label` of the output elements in `out`
	///		are the target. So this function actually returns targets and not
	/// 	the zeros.
	/// NOTE: allocates iL
	/// \param out output list
	/// \param L1 dont care: can be sorted, but dont have to
	/// \param L2 must be sorted (if prepare==true: will be sorted)
	/// \param target
	/// \param k_lower1 lower coordinate to match on in the base lists
	/// \param k_upper1 upper coordinate to match on in the base lists
	/// \param k_lower2 lower coordinate to match on in the intermediate lists
	/// \param k_upper2 upper coordinate to match on in the intermediate lists
	/// \param prepare if `true` L2 sorted. NO intermediate target
	/// 		will be added into the lists
	static void join4lists_twolists_on_iT_v2(List &out,
									const List &L1, List &L2,
									const LabelType &target,
									const uint32_t k_lower1, const uint32_t k_upper1,
									const uint32_t k_lower2, const uint32_t k_upper2,
									const bool prepare = true) noexcept {
		(void)k_lower2;
		List iL{static_cast<size_t>(L1.size() * config.intermediatelist_size_factor)};
		out.set_load(0);

		// reset everything
		if (prepare) {
			L2.sort_level(k_lower1, k_upper1);
		}

		ElementType tmpe1;
		LabelType t1, iT;
		iT.random(0, 1ull << k_upper1);
		join2lists_on_iT_v2(iL, L1, L2, iT, k_lower1, k_upper1, false);
		if (iL.load() == 0) { return; }

		iL.sort_level(k_lower1, k_upper2);
		LabelType::sub(t1, target, iT);
		twolevel_streamjoin_on_iT_v2(out, iL, L1, L2, target, t1,
									 k_lower1, k_upper1, k_lower2, k_upper2,
		                             false);
	}


	///                     ┌───────┐
	///                     │ out   │
	///                     └───┬───┘
	///                 k_lower2│k_upper2
	///            ┌────────────┼───────────┐
	///            │                        │
	///        ┌───┴───┐                ┌───┴───┐
	///        │  iL   │will be                 │ wont be
	///        │       │allocated       │         allocated
	///        └───┬───┘                └───┬───┘
	///    k_lower1│k_upper1        k_lower1│k_upper1
	///     ┌──────┴──────┐          ┌──────┴──────┐
	///     │             │          │             │
	/// ┌───┴───┐     ┌───┴───┐  ┌───┴───┐     ┌───┴───┐ both L3 and L4
	/// │const  │     │sorted │     					 are simply L1 and
	/// └───────┘     └───────┘  └───────┘     └───────┘ L2. Wont be allocated
	///    L1             L2     L3 = L1 		L4 = L2
	/// NOTE: v2 means that the `label` of the output elements in `out`
	///		are the target. So this function actually returns targets and not
	/// 	the zeros.
	/// NOTE: allocates iL
	/// \tparam k_lower1 lower coordinate to match on in the base lists
	/// \tparam k_upper1 upper coordinate to match on in the base lists
	/// \tparam k_lower2 lower coordinate to match on in the intermediate lists
	/// \tparam k_upper2 upper coordinate to match on in the intermediate lists
	/// \param out output list
	/// \param L1 dont care: can be sorted, but dont have to
	/// \param L2 must be sorted (if prepare==true: will be sorted)
	/// \param target
	/// \param prepare if `true` L2 sorted. NO intermediate target
	/// 		will be added into the lists
	template<const uint32_t k_lower1, const uint32_t k_upper1,
			 const uint32_t k_lower2, const uint32_t k_upper2>
	static void join4lists_twolists_on_iT_v2(List &out,
											 const List &L1, List &L2,
											 const LabelType &target,
											 const bool prepare = true) noexcept {
		(void)k_lower2;
		List iL{static_cast<size_t>(L1.size() * config.intermediatelist_size_factor)};
		out.set_load(0);

		// sort everything
		if (prepare) {
			L2.template sort_level<k_lower1, k_upper1>();
		}

		ElementType tmpe1;
		LabelType t1, iT;
		iT.random(0, 1ull << k_upper1);
		join2lists_on_iT_v2<k_lower1, k_upper1>(iL, L1, L2, iT, false);
		if (iL.load() == 0) {
			// early exit
			return;
		}

		iL.template sort_level<k_lower1, k_upper2>();
		LabelType::sub(t1, target, iT);
		twolevel_streamjoin_on_iT_v2
		        <k_lower1, k_upper1, k_lower2, k_upper2>
		        (out, iL, L1, L2, target, t1, false);

	}


	/// TODO test, doc, img
	///
	/// \tparam k_lower1
	/// \tparam k_upper1
	/// \tparam k_lower2
	/// \tparam k_upper2
	/// \tparam HashMap
	/// \param out
	/// \param L1
	/// \param L2
	/// \param hm
	/// \param target
	/// \param prepare
	template<const uint32_t k_lower1, const uint32_t k_upper1,
	         const uint32_t k_lower2, const uint32_t k_upper2,
			 typename HashMapL0, typename HashMapL1>
#if __cplusplus > 201709L
	requires HashMapAble<HashMapL0> &&
			 HashMapAble<HashMapL1>
#endif
	static void join4lists_twolists_on_iT_hashmap_v2(List &out,
											 		 const List &L1, const List &L2,
	                                                 HashMapL0 &hmL0,
													 HashMapL1 &hmL1,
											 		 const LabelType &target,
	                                                 const bool prepare = true) noexcept {
		(void)k_lower2;
		hmL0.clear(); hmL1.clear();

		// reset everything
		if (prepare) {
			for (size_t i = 0; i < L2.load(); ++i) {
				hmL0.insert(L2[i].label.value(), i);
			}
		}

		ElementType tmpe1;
		LabelType t1, iT;
		iT.random(0, 1ull << k_upper1);
		join2lists_on_iT_hashmap_v2
		        <k_lower1, k_upper1>
		        (hmL1, L1, L2, hmL0, iT, false);

		LabelType::sub(t1, target, iT);
		twolevel_streamjoin_on_iT_hashmap_v2
				<k_lower1, k_upper1, k_lower2, k_upper2>
				(out, hmL1, L1, L2, hmL0, target, t1);
	}

	/// Schematic view on the Algorithm.
	/// The algorithm ignores any facts about th weight, nor does it guess the output size.
	/// The intermediate list `iL` is created every time you call this function. So if called repeatedly, maybe write it yourself, or set the list to static.
	/// The difference to the function `streamjoin4lists` is that the lists `L3` and `L4` are simulated on the fly. This
	/// means that list `L1` and `L2` are used for this.
	///                                                  Out
	///                                       +-----------+----------+
	///                                       | k_lower   |  k_higher|                                         Level 2
	///                                       +----------------------+
	/// Merge on k_upper                                  |
	///                         +-------------------------+-------------------------+  Stream Merge
	///                         |                Merge on target                    |
	///             +-----------------------+                          + - - - - - - - - - - - +
	///             | k_lower   |  k_higher | Intermediate List        |            |          |               Level 1
	///             +-----------------------+ Will be save.            +- - - - - - - - - - - -+ NOT Saved
	/// Merge on k_lower        |                                                   |
	///            +------------+------------+                        +-------------+-----------+
	///            |      Merge on random R  |                        |    merge on R+target    |
	///+-----------------------+ +-----------------------+ + - - - - - - - - - - - + + - - - - - - - - - - - +
	///| k_lower   |  k_higher | | k_lower   |  k_higher | | NOT SAVED/PASSED      | | NOT_SAVED/PASSED      | Level 0
	///+-----------------------+ +-----------------------+ +- - - - - - - - - - - -+ +- - - - - - - - - - - -+
	///           L_1                       L_2                      L_1                       L_4
	///
	///
	/// \param out output list. All targets which are a solution to the 4 kxor problem.
	/// \param L1 	(sorted)
	/// \param L2 	(sorted+R+target) intermediate target is added.
	/// \param target
	/// \param lta
	static void streamjoin4lists_twolists(List &out, List &L1, List &L2,
	                                      const LabelType &target, const std::vector<uint64_t> &lta,
	                                      bool prepare = true) noexcept {
		ASSERT(lta.size() == 3);
		// limits: k_lower1, k_upper1 for the lowest level tree. And k_lower2, k_upper2 for highest level. There are
		// only two levels..., so obviously k_upper1=k_lower2
		const uint64_t k_lower1 = lta[0], k_upper1 = lta[1];
		const uint64_t k_lower2 = lta[1], k_upper2 = lta[2];
		streamjoin4lists_twolists(out, L1, L2, target, k_lower1, k_upper1, k_lower2, k_upper2, prepare);
	}

	static void streamjoin4lists_twolists(List &out, List &L1, List &L2,
	                                      const LabelType &target,
	                                      const uint64_t k_lower1, const uint64_t k_upper1, const uint64_t k_lower2, const uint64_t k_upper2,
	                                      bool prepare = true) noexcept {
		ASSERT(k_lower1 < k_upper1 &&
		       0 < k_upper1 && k_lower2 < k_upper2 &&
		       0 < k_upper2 && k_lower1 <= k_lower2 &&
		       k_upper1 <= k_upper2 &&
		       L1.load() > 0 && L2.load() > 0);

		// Intermediate Element, List, Target
		List iL{out.size() * 2};
		LabelType R, zero;
		R.random();
		zero.zero();

		// prepare list L2.
		if ((!target.is_zero()) && (prepare)) {
			for (size_t i = 0; i < L2.load(); ++i) {
				LabelType::add(L2[i].label, L2[i].label, R, k_lower1, k_upper1);
			}
		}

		join2lists(iL, L1, L2, zero, k_lower1, k_upper1, false);

		// early exit
		if (iL.load() == 0) {
			return;
		}

		// prepare list L2 as L4
		if ((!target.is_zero()) && (prepare)) {
			for (size_t i = 0; i < L2.load(); ++i) {
				LabelType::sub(L2[i].label, L2[i].label, target, k_lower1, k_upper2);
			}
		}

		// Now run the merge procedure for the right part of the tree.
		twolevel_streamjoin(out, iL, L1, L2, k_lower1, k_upper1, k_lower2, k_upper2);
	}


	/// \param out		output lists
	/// \param L		input lists, L must contain 8 Lists
	/// \param target	target to match on
	/// \param lta		translation const_array with the limits to match on, for each lvl
	static void streamjoin8lists(List &out, std::vector<List> &L, const LabelType &target,
	                             const std::vector<uint64_t> &lta) noexcept {
		ASSERT(lta.size() == 4 && L.size() == 8);

		// limits:
		const uint64_t k_lower1 = lta[0], k_upper1 = lta[1];
		const uint64_t k_lower2 = lta[1], k_upper2 = lta[2];
		const uint64_t k_lower3 = lta[2], k_upper3 = lta[3];
		std::vector<uint64_t> lower_lta{{k_lower1, k_upper1}};
		std::vector<uint64_t> middle_lta{{k_lower2, k_upper2}};

		// Intermediate List
		// 2 for the first Level, 2 for the second level
		for (uint32_t k = 0; k < 4; ++k) {
			L.push_back(List{out.size() * 2});
		}

		// Intermediate Target
		LabelType R, R1, R3, R5, R13, R57, zero;
		R1.random();
		R3.random();
		R5.random();
		R13.random();
		R57.random();
		zero.zero();
		LabelType::add(R, R1, R3, k_lower1, k_upper1);
		LabelType::add(R, R, R5, k_lower1, k_upper1);
		LabelType::add(R, R, R13, k_lower2, k_upper2);
		LabelType::add(R, R, R57, k_lower2, k_upper2);

		std::vector<LabelType> vr{{zero, zero}};
		// prepare Lists
		if (!target.is_zero()) {
			for (size_t k = 0; k < L[1].load(); ++k) {
				LabelType::add(L[1][k].label, L[1][k].label, R1, k_lower1, k_upper1);
			}

			for (size_t k = 0; k < L[3].load(); ++k) {
				LabelType::add(L[3][k].label, L[3][k].label, R3, k_lower1, k_upper1);
				LabelType::add(L[3][k].label, L[3][k].label, R13, k_lower2, k_upper2);
			}

			for (size_t k = 0; k < L[5].load(); ++k) {
				LabelType::add(L[5][k].label, L[5][k].label, R5, k_lower1, k_upper1);
			}

			for (size_t k = 0; k < L[7].load(); ++k) {
				LabelType::sub(L[7][k].label, L[7][k].label, R, k_lower1, k_upper1);
				// add the target on the full length
				LabelType::add(L[7][k].label, L[7][k].label, target, k_lower1, k_upper3);
			}
		}

		for (uint32_t k = 0; k < 3; k += 2) {
			List &L1 = L[k * 2];
			List &L2 = L[k * 2 + 1];
			List &iL = L[8 + k / 2];      // Intermediate list for the normal join
			List &tout = L[8 + k / 2 + 2];// Intermediate list for the stream join

			// base list join
			join2lists(iL, L1, L2, vr[k / 2], lower_lta, false);

			// then streamjoin
			twolevel_streamjoin(tout, iL, L1, L2, k_lower1, k_upper1, k_lower2, k_upper2);
		}

		// Last Join
		// in L[10] is the result of the stream join between L[0-3]
		// in L[11] is the result of the stream join between L[4-7]
		std::vector<uint64_t> last_lta{{k_lower3, k_upper3}};
		join2lists(out, L[10], L[11], target, last_lta, false);
	}

	/// \param out
	/// \param L1 dont care
	/// \param L2 must be sorted
	/// \param target
	/// \param k_lower1
	/// \param k_upper1
	/// \param k_lower2
	/// \param k_upper2
	/// \param k_lower3
	/// \param k_upper3
	/// \param prepare
	static void join8lists_twolists_on_iT_v2(List &out,
											 const List &L1, List &L2,
											 const LabelType &target,
											 const uint64_t k_lower1, const uint64_t k_upper1,
											 const uint64_t k_lower2, const uint64_t k_upper2,
											 const uint64_t k_lower3, const uint64_t k_upper3,
											 const bool prepare = true) noexcept {
		ASSERT(k_lower1 < k_upper1);
		ASSERT(k_lower2 < k_upper2);
		ASSERT(k_upper1 < k_upper2);
		ASSERT(k_upper2 < k_upper3);
		ASSERT(k_upper2 < k_upper3);

		(void)k_lower2;
		(void)k_lower3;
		constexpr uint32_t filter = -1u;
		List iL1{L1.size() * 2}, iL2{L1.size() * 2};

		// reset everything
		if (prepare) {
			out.set_load(0);
			L2.sort_level(k_lower1, k_upper1);
		}

		ElementType tmpe1;
		LabelType t1, t2, t3, iT1, iT2, iT3, tmpl, tmpl2;
		iT1.random(0, 1ull << k_upper3);
		iT2.random(0, 1ull << k_upper3);
		iT3.random(0, 1ull << k_upper3);

		// L1 and L2
		join2lists_on_iT_v2(iL1, L1, L2, iT1, k_lower1, k_upper1);
		iL1.sort_level(0, k_upper2);
		if (iL1.load() == 0) {
			return ;
		}

		// L1, L2, L3 and L4
		LabelType::sub(t1, iT2, iT1);
		twolevel_streamjoin_on_iT_v2(iL2, iL1, L1, L2, iT2, t1,
									 k_lower1, k_upper1, k_lower2, k_upper2);
		iL2.sort_level(0, k_upper3);
		if (iL2.load() == 0) {
			return ;
		}


		// L5 and L6
		iL1.set_load(0);
		join2lists_on_iT_v2(iL1, L1, L2, iT3, k_lower1, k_upper1);
		iL1.sort_level(0, k_upper2);
		if (iL1.load() == 0) {
			return ;
		}

		LabelType::sub(tmpl2, target, iT2);
		LabelType::sub(tmpl, target, iT3);
		LabelType::sub(tmpl, tmpl, iT2);
		for (size_t i1 = 0; i1 < L1.load(); ++i1) {
			LabelType::sub(t1, tmpl, L1[i1].label);
			size_t j1 = L2.search_level(t1, 0, k_upper1);
			for (; (j1 < L2.load()) &&
				   (t1.is_equal(L2[j1].label, 0, k_upper1));
				   ++j1) {

				LabelType::sub(t2, tmpl2, L1[i1].label);
				LabelType::sub(t2, t2, L2[j1].label);
				size_t j2 = iL1.search_level(t2, 0, k_upper2);
				for (; (j2 < iL1.load()) &&
					   (t2.is_equal(iL1[j2].label, 0, k_upper2));
					   ++j2) {

					LabelType::sub(t3, target, L1[i1].label);
					LabelType::sub(t3, t3, L2[j1].label);
					LabelType::sub(t3, t3, iL1[j2].label);

					size_t j3 = iL2.search_level(t3, 0, k_upper3);
					for (; (j3 < iL2.load()) &&
						   (t3.is_equal(iL2[j3].label, 0, k_upper3));
						   ++j3) {

						tmpe1 = iL2[j3];
						ElementType::add(tmpe1, tmpe1, iL1[j2]);
						ElementType::add(tmpe1, tmpe1, L1[i1]);
						out.add_and_append(tmpe1, L2[j1], 0, k_upper3, filter);
					}
				}
			}
		}
	}

	template<const uint32_t k_lower1, const uint32_t k_upper1,
			 const uint32_t k_lower2, const uint32_t k_upper2,
			 const uint32_t k_lower3, const uint32_t k_upper3>
	static void join8lists_twolists_on_iT_v2(List &out,
											 const List &L1, List &L2,
											 const LabelType &target,
											 const bool prepare = true) noexcept {
		static_assert(k_lower1 < k_upper1);
		static_assert(k_lower2 < k_upper2);
		static_assert(k_upper1 < k_upper2);
		static_assert(k_upper2 < k_upper3);
		static_assert(k_upper2 < k_upper3);

		(void)k_lower2;
		(void)k_lower3;
		constexpr uint32_t filter = -1u;
		List iL1{L1.size() * 2}, iL2{L1.size() * 2};

		// reset everything
		if (prepare) {
			out.set_load(0);
			L2.template sort_level<k_lower1, k_upper1>();
		}

		ElementType tmpe1;
		LabelType t1, t2, t3, iT1, iT2, iT3, tmpl, tmpl2;
		iT1.random(0, 1ull << k_upper3);
		iT2.random(0, 1ull << k_upper3);
		iT3.random(0, 1ull << k_upper3);

		// L1 and L2
		join2lists_on_iT_v2<k_lower1, k_upper1>(iL1, L1, L2, iT1);
		iL1.template sort_level<0, k_upper2>();
		if (iL1.load() == 0) {
			return ;
		}

		// L1, L2, L3 and L4
		LabelType::sub(t1, iT2, iT1);
		twolevel_streamjoin_on_iT_v2
		        <k_lower1, k_upper1, k_lower2, k_upper2>
		        (iL2, iL1, L1, L2, iT2, t1);
		iL2.template sort_level<0, k_upper3>();
		if (iL2.load() == 0) {
			return ;
		}


		// L5 and L6
		iL1.set_load(0);
		join2lists_on_iT_v2<k_lower1, k_upper1>(iL1, L1, L2, iT3);
		iL1.template sort_level<0, k_upper2>();
		if (iL1.load() == 0) {
			return ;
		}

		LabelType::sub(tmpl2, target, iT2);
		LabelType::sub(tmpl, target, iT3);
		LabelType::sub(tmpl, tmpl, iT2);
		for (size_t i1 = 0; i1 < L1.load(); ++i1) {
			LabelType::sub(t1, tmpl, L1[i1].label);
			size_t j1 = L2.template search_level<0, k_upper1>(t1);
			for (; (j1 < L2.load()) &&
				   (t1.template is_equal<0, k_upper1>(L2[j1].label));
				   ++j1) {

				LabelType::sub(t2, tmpl2, L1[i1].label);
				LabelType::sub(t2, t2, L2[j1].label);
				size_t j2 = iL1.template search_level<0, k_upper2>(t2);
				for (; (j2 < iL1.load()) &&
					   (t2.template is_equal<0, k_upper2>(iL1[j2].label));
					   ++j2) {

					LabelType::sub(t3, target, L1[i1].label);
					LabelType::sub(t3, t3, L2[j1].label);
					LabelType::sub(t3, t3, iL1[j2].label);

					size_t j3 = iL2.template search_level<0, k_upper3>(t3);
					for (; (j3 < iL2.load()) &&
						   (t3.template is_equal<0, k_upper3>(iL2[j3].label));
						   ++j3) {

						tmpe1 = iL2[j3];
						ElementType::add(tmpe1, tmpe1, iL1[j2]);
						ElementType::add(tmpe1, tmpe1, L1[i1]);
						out.template add_and_append
						        <0, k_upper3, filter, false>
						        (tmpe1, L2[j1]);
					}
				}
			}
		}
	}


	/// helper function wich computes
	///		`search_in_level`
	/// 	`increment_previous_level`
	/// \tparam l
	/// \tparam h
	template<const uint32_t l, const uint32_t h>
	class helper {
	public:
		/// \param level  the current level we are in. NOT the level we are looking into
		/// \param lists
		/// \param e
		/// \param boundaries
		/// \param indices
		/// \return  the next level to search in
		constexpr static inline uint32_t run(const uint32_t &level,
		                                     const List *lists,
		                                     const ElementType &e,
		                                     std::vector<std::pair<size_t, size_t>> &boundaries,
		                                     std::vector<size_t> &indices) {
			boundaries[level] = lists[level + 2].template search_boundaries<l, h>(e);
			indices[level] = boundaries[level].first;

			// NOTE: the cast here
			int32_t lvl = (int32_t)level;
			while (lvl >= 0) {
				if (indices[lvl] == boundaries[lvl].second) {
					if (lvl == 0) {
						return 0;
					} else {
						indices[--lvl]++;
					}
				} else {
					return lvl + 1;
				}
			}

			return 0;
		}
	};


	// This example shows how to enumerate all intermediate target
	// if a full join (no stream join) is done.
	//
	// Output Example: (d = 2) => 4 base lists
	//	iT1 iT2-iT1 iT3 T-iT3-iT2
	//  iT2 T-iT2
	//
	// std::vector<Label> iTs(1u<<(depth-1u));
	// for (uint32_t i = 0; i < ((1u<<(depth-1u))-1u); ++i) {
	// 	iTs[i].random(0, 1ull << n);
	// }
	// iTs[(1u<<(depth-1u))-1u] = target;
	// std::vector<std::vector<Label>> sum_iTs(depth);
	// for (uint32_t i = 0; i < depth; ++i) {
	// 	const uint32_t size = 1u << (depth - i - 1u);
	// 	sum_iTs[i].resize(size);
	// 	// -1, because the last on is the target
	// 	for (uint32_t j = 0; j < size; ++j) {
	// 		sum_iTs[i][j] = iTs[i+j];
	// 		if (j & 1u) {
	// 			const uint32_t ctz = __builtin_ctz(j+1);
	// 			for (uint32_t k = 1; k <= std::min(2u, ctz); ++k) {
	// 				Label::sub(sum_iTs[i][j], sum_iTs[i][j], iTs[i+j-k]);
	// 			}
	// 		}
	// 	}
	// }

	template<const uint32_t level,
	         const uint32_t ...ks>
	void join_stream_internal(List &out,
							  const std::vector<std::vector<LabelType>> &iTs,
	                          const bool prepare=true) {
		static_assert(sizeof...(ks) >= (level + 2u),
		              "make sure that you give enough limits");

		constexpr uint32_t filter = -1u;
		constexpr bool sub = false;

		/// NOMENCLATUR:
		///		lists[0] = left base list
		/// 	lists[1] = right base lists
		constexpr uint32_t k_lower = std::get<0>(std::forward_as_tuple(ks...));
		constexpr uint32_t k_upper = std::get<1>(std::forward_as_tuple(ks...));;
		if (prepare) {
			out.set_load(0);
			lists[1].template sort_level<k_lower, k_upper>();
		}

		if constexpr (level == 0) {
			// easy case, we want to joint on only two lists,
			// so basically no streaming
			join2lists_on_iT_v2<k_lower, k_upper>(out, lists[0], lists[1], iTs[0][0], prepare);

			// TODO
			lists[2].template sort_level<0, 16>();
			return;
		}

		constexpr uint32_t ll = std::get<0>(std::forward_as_tuple(ks...));
		constexpr uint32_t lh = std::get<sizeof...(ks) - 1u>(std::forward_as_tuple(ks...));;

		// keep track of the partial sums we are computing to search within the next level
		std::vector<ElementType> search_sums(level+1);
		// keep track of the actual sums of elements
		std::vector<ElementType> sums(level+1);

		//
		std::vector<std::pair<size_t, size_t>> boundaries(level);

		//
		std::vector<size_t> indices(level);

		/// storage for the helper functions: these allow the translation
		/// between the (non-constexpr) current level we are matching on
		/// and the (constexpr) boundaries we are matching on.
		std::vector<std::function< decltype(helper<0, 1>::run)>> vec;

		/// NOTE: the starting point 1: we want to merge on lvl 0 with elements from lvl 1
		/// NOTE: the -1 is because in `constexpr_for` the upper bound is inclusive
		constexpr_for<1u, sizeof...(ks) - 1u, 1u>([&vec](const auto I) {
			constexpr uint32_t l = std::get<I+0>(std::forward_as_tuple(ks...));
			constexpr uint32_t h = std::get<I+1>(std::forward_as_tuple(ks...));;
			std::function<decltype(helper<l, h>::run)> t2 = helper<l, h>::run;
			vec.push_back(t2);
		});

		/// start the actual algorithm
		for (size_t i = 0; i < lists[0].load(); ++i) {
			LabelType::template sub<k_lower, k_upper>(search_sums[0].label, iTs[0][level], lists[0][i].label);
			size_t j = lists[1].template search_level<k_lower, k_upper>(search_sums[0]);

			for (; (j < lists[1].load()) &&
				   (search_sums[0].label.template is_equal<k_lower, k_upper>(lists[1][j].label)); ++j) {
				ElementType::add(sums[0], lists[0][i], lists[1][j]);

				// now the actual stream join starts
				bool stop = false;
				uint32_t l = 0;
				while (!(stop)) {
					while (l < level) {
						search_sums[l+1].zero();
						search_sums[l+1].label = iTs[l+1][0];
						ElementType::sub(search_sums[l+1], search_sums[l+1], sums[l]);

						// NOTE: vec[l] should be vec[l+1], but we do not save the
						// lowerst matching routine, this shift can be ignored.
						l = vec[l].operator()(l, lists.data(), search_sums[l+1], boundaries, indices);

						//if on lowest level the index reaches the boundary: stop
						if (indices[0] == boundaries[0].second) {
							stop = true;
							break;
						}
					}

					// save all matching elements
					for (; indices[level - 1] < boundaries[level - 1].second; ++indices[level - 1]) {
						out.template add_and_append
						        <ll, lh, filter, sub>
						        (sums[level-1], lists[level + 1][indices[level - 1]]);
					}

					// continue stream-join with the next element of previous level
					if (l == level) {
						l--;
					}

					l = increment_previous_level(l, boundaries, indices);
					if (indices[0] == boundaries[0].second) {
						stop = true;
					}
				}
			}
		}

		// TODO sort out list
	}



	// TODO hashmap version, and non constexpr version
	// SRC: https://eprint.iacr.org/2010/189.pdf
	// implementation of the modular 4-way merge
	template<
			 const uint32_t k_lower1=0,
	         const uint32_t k_upper1=ValueLENGTH/4u,
			 const uint32_t k_upper2=ValueLENGTH>
	static void constexpr_dissection4(List &out,
									  const LabelType &target,
									  const MatrixType &MT) noexcept {
		// reset the output list
		out.set_load(0);

		constexpr static uint32_t filter = -1u;
		constexpr static double factor = 1.5;
		constexpr static size_t size = (1ull << k_upper1) - 1ull;

		constexpr size_t baselist_size = sum_bc(k_upper1, k_upper1/2);
		//constexpr size_t baselist_size = sum_bc(n/2, n/4);
		static List L1{baselist_size}, L2{baselist_size}, L3{baselist_size}, L4{baselist_size}, iL{(size_t) ((double) size * factor)};
		L1.set_load(0); L2.set_load(0); L3.set_load(0); L4.set_load(0);

		// enumerate the base lists
		// using Enumerator = BinaryLexicographicEnumerator<List, n/2, n/4>;
		// Enumerator e{MT};
		// e.run(&L1, &L2, n/2);
		// e.run(&L3, &L4, n/2);
		using Enumerator = BinaryLexicographicEnumerator<List, k_upper1, k_upper1/2>;
		Enumerator e{MT};
		e.run(&L1, &L2, k_upper1);
		e.run(&L3, &L4, k_upper1, k_upper2/2);

		L2.sort_level(k_lower1, k_upper1);
		L4.sort_level(k_lower1, k_upper1);

		ElementType tmpe1;
		LabelType sigma_M, sigma_t, tprime;
		for (size_t sigma_M_ = 0; sigma_M_ < size; ++sigma_M_) {
			iL.set_load(0);
			sigma_M.set(sigma_M_, 0);
			for (size_t i = 0; i < L1.load(); ++i) {
				LabelType::sub(sigma_t, sigma_M, L1[i].label, 0, k_upper1);
				size_t j = L2.search_level(sigma_t, 0, k_upper1);
				for (; (j < L2.load()) &&
				       (sigma_t.is_equal(L2[j].label, 0, k_upper1)); ++j) {
					iL.add_and_append(L1[i], L2[j], 0, k_upper2, filter);
				}
			}

			if (iL.load() == 0) {
				continue;
			}

			iL.sort_level(0, k_upper2);
			for (size_t k = 0; k < L3.load(); ++k) {
				LabelType::sub(sigma_t, target, sigma_M);
				LabelType::sub(sigma_t, sigma_t, L3[k].label);
				size_t l = L4.search_level(sigma_t, 0, k_upper1);
				for (; (l < L4.load()) &&
				       (sigma_t.is_equal(L4[l].label, 0, k_upper1)); ++l) {
					LabelType::sub(tprime, target, L3[k].label);
					LabelType::sub(tprime, tprime, L4[l].label);
					size_t o = iL.search_level(tprime, 0, k_upper2);
					for (; (o < iL.load()) &&
					       (tprime.is_equal(iL[o].label, 0, k_upper2)); ++o) {
						tmpe1 = iL[o];
						ElementType::add(tmpe1, tmpe1, L3[k]);
						out.add_and_append(tmpe1, L4[l], 0, k_upper2, filter);
					}
				}
			}
		}
	}

	/// builds the cross product between in1 x in2 (only on the values).
	///	The coordinates [k_lower, k_middle] are taken from the input list in1, whereas the coordinates
	///	[k_middle, k_upper] are taken from the second input list in2.
	/// \param out			Output list
	/// \param in1			const Input list
	/// \param in2			const Input list
	/// \param k_lower
	/// \param k_middle
	/// \param k_upper
	static size_t cross_product(List &out,
	                            const List &in1,
	                            const List &in2,
	                            const uint64_t k_lower,
	                            const uint64_t k_middle,
	                            const uint64_t k_upper) noexcept {
		ASSERT(k_lower < k_middle && k_middle < k_upper && 0 < k_middle);

		const uint64_t size = in1.size() * in2.size();
		out.resize(size);
		out.set_load(size);

		uint64_t counter = 0;
		for (uint64_t i = 0; i < in1.size(); ++i) {
			for (uint64_t j = 0; j < in2.size(); ++j) {
				ValueContainerType::set(out[counter].value, in1[i].value, k_lower, k_middle);
				ValueContainerType::set(out[counter].value, in2[j].value, k_middle, k_upper);

				counter += 1;
			}
		}

		return counter;
	}

	// const and non-const list access functions.
	List &operator[](uint64_t i) noexcept {
		ASSERT(i < (depth + additional_baselists) && "Wrong index");
		return this->lists[i];
	}

	const List &operator[](const uint64_t i) const noexcept {
		ASSERT(i < (depth + additional_baselists) && "Wrong index");
		return this->lists[i];
	}

	/// some getters
	[[nodiscard]] constexpr inline uint64_t get_size() const noexcept { return lists.size(); }
	[[nodiscard]] constexpr inline uint64_t get_basesize() const noexcept { return base_size; }
	[[nodiscard]] constexpr inline const auto &get_level_translation_array() const noexcept { return level_translation_array; }

private:
	// drop the default constructor.
	Tree_T();

	/// search for e1=e2+e3 in level 'level'+1 list and update parameters
	/// \param e1 			output element
	/// \param e2 			input element
	/// \param e3 			input element
	/// \param level 		current lvl within the tree. Sets the "k_lower", "k_higher" to search between.
	/// \param boundaries
	/// \param indices
	constexpr inline void search_in_level_l(ElementType &e1,
	                              const ElementType &e2,
	                              const ElementType &e3,
	                              const uint8_t level,
	                              std::vector<std::pair<size_t, size_t>> &boundaries,
	                              std::vector<size_t> &indices) noexcept {
		uint64_t k_lower, k_higher;
		translate_level(&k_lower, &k_higher, level + 1, level_translation_array);
		search_in_level_l(e1, e2, e3, level, k_lower, k_higher, boundaries, indices);
	}

	/// \param e1
	/// \param e2
	/// \param e3
	/// \param level
	/// \param k_lower
	/// \param k_higher
	/// \param boundaries
	/// \param indices
	/// \return
	constexpr inline void search_in_level_l(ElementType &e1,
								  const ElementType &e2,
								  const ElementType &e3,
								  const uint8_t level,
	                              const uint32_t k_lower,
	                              const uint32_t k_higher,
								  std::vector<std::pair<size_t, size_t>> &boundaries,
								  std::vector<size_t> &indices) noexcept {
		ElementType::add(e1, e2, e3);

		ASSERT(level < boundaries.size());
		ASSERT(level < indices.size());
		boundaries[level] = lists[level + 2].search_boundaries(e1, k_lower, k_higher);
		indices[level] = boundaries[level].first;
	}

	/// Called from `join_stream_join`. DO NOT USE IT.
	/// \param l			lvl
	/// \param boundaries
	/// \param indices
	/// \return new lvl
	int increment_previous_level(int l,
	                             std::vector<std::pair<size_t, size_t>> &boundaries,
	                             std::vector<size_t> &indices) {
		while (l >= 0) {
			if (indices[l] == boundaries[l].second) {
				if (l == 0)
					return 0;
				else
					indices[--l]++;
			} else
				return l + 1;
		}

		return 0;
	}

	/// accessed via stream_join function
	/// non-recursive version of stream-join (that stores only depth+2 lists)
	/// \param level level of the tree to join to.
	/// \param target a list where every solution is saved into.
	void join_stream_internal(const uint64_t level, List &target) {
		std::vector<std::pair<size_t, size_t>> boundaries(level);
		std::vector<ElementType> a(level);
		std::vector<size_t> indices(level);

		// reset output list
		target.set_load(0);

		// prepare filter variables. Filter means that filter every element out that has a norm bigger than filer0
		const uint32_t filter = translate_filter(level, filter_nr, level_filter_array);
		const uint32_t filter0 = translate_filter(0, filter_nr, level_filter_array);

		// prepare coordinate limits. This function will only match on the first to lists on the l_lower0, k_higher0 coordinates.
		uint64_t k_lower0, k_higher0;
		translate_level(&k_lower0, &k_higher0, 0, level_translation_array);

		uint64_t i = 0, j = 0;
		while (i < lists[0].load() && j < lists[1].load()) {
			if (lists[1][j].is_greater(lists[0][i], k_lower0, k_higher0)) {
				i++;
			} else if (lists[0][i].is_greater(lists[1][j], k_lower0, k_higher0)) {
				j++;
			} else {
				size_t i_max=i+1ull, j_max=j+1ull;
				// if elements are equal find max index in each list, such that they remain equal
				for (; i_max < lists[0].load() && lists[0][i].is_equal(lists[0][i_max], k_lower0, k_higher0); i_max++) {}
				for (; j_max < lists[1].load() && lists[1][j].is_equal(lists[1][j_max], k_lower0, k_higher0); j_max++) {}

				// for each matching tuple
				size_t jprev = j;

				for (; i < i_max; ++i) {
					for (j = jprev; j < j_max; ++j) {
						//if base-list join is performed store directly in level 1 list
						if (level == 0) {
							target.add_and_append(lists[0][i], lists[1][j], filter0);
							//if level>=1: perform stream join
						} else {
							auto stop = false;
							uint8_t l = 0;
							while (!(stop)) {
								while (l < level) {
									// perform search on current level (streamjoining against level l list)
									if (l == 0) {
										search_in_level_l(a[0], lists[0][i], lists[1][j], l, boundaries, indices);
									} else {
										search_in_level_l(a[l], a[l - 1], lists[l + 1][indices[l - 1]], l, boundaries,
										                  indices);
									}

									//if nothing found continue with the next element of the previous level
									l = increment_previous_level(l, boundaries, indices);

									//if on lowest level the index reaches the boundary: stop
									if (indices[0] == boundaries[0].second) {
										stop = true;
										break;
									}
								}

								// save all matching elements
								for (; indices[level - 1] < boundaries[level - 1].second; ++indices[level - 1]) {
									target.add_and_append(a[level - 1], lists[level + 1][indices[level - 1]], filter);
								}

								// continue stream-join with the next element of previous level
								if (l == level) {
									l--;
								}

								l = increment_previous_level(l, boundaries, indices);
								if (indices[0] == boundaries[0].second) {
									stop = true;
								}
							}
						}
					}
				}
			}
		}
	}

	/// print every list in this tree
	/// \param k_lower
	/// \param k_upper
	void print(const uint64_t k_lower, const uint64_t k_higher) {
		for (const auto &l: lists) {
			l.print(k_lower, k_higher);
		}
	}

	/// NTRU / LWE / decoding matrix
	const MatrixType matrix;

	/// const_array to translate lvl x to upper and lower bound
	const std::vector<uint64_t> level_translation_array;

	/// const_array to translate lvl x to filter
	const std::vector<std::vector<uint8_t>> level_filter_array;

	/// minimum nr of 2 in an element till we discard it
	const uint32_t filter_nr = 200;

public:
	// TODO access functions
	std::vector<List> lists;
private:
	unsigned int depth;
	uint64_t base_size;
};

/// \param out
/// \param obj tree objext
template<class List>
std::ostream &operator<<(std::ostream &out, const Tree_T<List> &obj) {
	/// print each list
	for (uint64_t i = 0; i < obj.get_size(); ++i) {
		out << "List: " << i << std::endl;
		out << obj[i];
		out << std::endl << std::endl;
	}

	return out;
}
#endif//SMALLSECRETLWE_TREE_H
