#ifndef SMALLSECRETLWE_TREE_H
#define SMALLSECRETLWE_TREE_H

#include <utility>
#include <vector>

#ifdef __unix__
#include <sys/mman.h>
#endif

// internal includes
#include "kAry_type.h"
#include "container/vector.h"
#include "element.h"
#include "matrix/matrix.h"
#include "list/list.h"
#include "thread/thread.h"
#include "matrix/fq_matrix.h"

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
	//typename HM::Hash;
	//typename HM::Extractor;

	//requires requres(const typename HM::List &list, typename HM::Hash hash, typename HM::Extractor extractor,
	// 		   const size_t s, const uint32_t u32, const uint64_t u64) {
	//	hash(list, u32);
	//	hash(list, u64, u32, hash);
	// 	insert(const typename HM::T, typename::IndexType *pos, u32);
	// 	find(const typename HM::T, typename::IndexType load) -> std::convertable_to<typename HM::IndexType>;
	//};
};

template<class List>
concept TreeAble = requires(List l){
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
		List(u64);  // constructor
		l.sort_level(u64, u64);
		l.sort_level(u32, u32, u32);

		l.search_level(e, u64, u64);
		l.search(e);
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

template<class List>
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
	constexpr static uint32_t ValueLENGTH = ValueType::LENGTH;
	constexpr static uint32_t LabelLENGTH = LabelType::LENGTH;
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
    /// \return
    [[nodiscard]] constexpr unsigned int intermediat_level_limit(const unsigned int i) const noexcept {
        // the first -1 half the amount of lists.
        return (1ULL << (depth - i - 1)) - 1;
    }

    /// count how many carries would occur of one add 2,4,... and so till no
    /// \param in
    /// \return
    [[nodiscard]] constexpr static unsigned int count_carry_propagates(const unsigned int in) noexcept {
        ASSERT(in >= 2 && "insert bigger than 2");

        auto prev = in - 2u;
        auto mask = 1ULL << 1u;
        unsigned int amount_negates = 1;
        while (mask & prev) {
            amount_negates++;
            mask <<= 1u;
        }

        return amount_negates;
    }

public:
    ///	On normal usage only d+2 (two baselists) __MUST__ be allocated and saved in memory. So technically that implements
    /// a non-recursive k-List algorithm.
    /// \param d depth of the tree.
    /// \param A Matrix
    /// \param baselist_size size of the baselists to generate (log)
    /// \param level_translation_array array containing the limits of the coorditates to merge on each lvl.
    ///         Example:
    ///				[0, 10, 20, ... 100]
    ///			means: that on the first lvl the lists are matched on the coordinates 0-9 (9 inclusive)
    ///         and so on
    explicit Tree_T(const unsigned int d,
	                const MatrixType &A,
					const unsigned int baselist_size, // 10
					const std::vector<uint64_t> &level_translation_array,
					const std::vector<std::vector<uint8_t>> &level_filter_array) noexcept :
            matrix(A),
            level_translation_array(level_translation_array),
			level_filter_array(level_filter_array)
    {
        ASSERT(d > 0 && "at least level 1");
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
	/// \param baselist_size	size of the baselists
	/// \param level_translation_array array containing the limits of the coorditates to merge on each lvl.
	///         Example:
	///				[0, 10, 20, ... 100]
	///			means: that on the first lvl the lists are matched on the coordinates 0-9 (9 inclusive)
	///         and so on
	explicit Tree_T(const unsigned int d,
	                const MatrixType &A,
	                const List &b1,
	                const List &b2,
	                const unsigned int baselist_size,
	                std::vector<uint64_t> &level_translation_array) noexcept :
			matrix(A),
			level_translation_array(level_translation_array)
	{
		ASSERT(d > 0 && "at least level 1");
		lists.push_back(b1);
		lists.push_back(b2);

		for (int i = 2; i < d + additional_baselists; ++i) {
			List a{0};
			lists.push_back(a);
		}

		depth = d;
		base_size = 1u << baselist_size;
	}

    // Andre: this just saves in default target list (lists[level+2])
    /// \param level
    void join_stream(uint64_t level) noexcept {
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
    void join_stream(uint64_t level, List &target) noexcept {
        join_stream_internal(level, target);
    }


    /// \param target
    /// \param gen_lists
    /// \param two_result_lists
    void build_tree(LabelType &target, bool gen_lists = true, bool two_result_lists = false) noexcept {
	    uint64_t k_lower, k_higher;

	    if (gen_lists) {
            lists[0].set_load(0);
            lists[1].set_load(0);
            lists[0].generate_base_random(this->base_size, matrix);
            lists[1].generate_base_random(this->base_size, matrix);
        }

        // generate the intermediate targets for all levels
        std::vector<std::vector<LabelType>> intermediate_targets(depth);
        for (unsigned int i = 0; i < depth; ++i) {
            intermediate_targets[i].resize(intermediat_level_limit(i) + 1); // +1 to have enough space.

            // set random intermediate targets
            for (unsigned int j = 0; j < intermediat_level_limit(i); ++j) {
                intermediate_targets[i][j].random();
            }

            intermediate_targets[i][intermediat_level_limit(i)] = target;
        }

        // adjust last intermediate target of each level, such that targets of each level add up the the true target
        for (int i = 0; i < depth - 1; ++i) {
            for (int j = 0; j < intermediat_level_limit(i); ++j) {
	            translate_level(&k_lower, &k_higher, -1, level_translation_array);

	            LabelType::sub(intermediate_targets[i][intermediat_level_limit(i)],
                           intermediate_targets[i][intermediat_level_limit(i)], intermediate_targets[i][j], k_lower, k_higher);
            }
        }


        // start joining the lists
        for (uint32_t i = 0; i < (1ULL << (this->depth - 1)); ++i) {
            // calc number of trailing zeros of i, which indicates to which level we have to join
            int join_to_level = Count_Trailing_Zeros(i + 1);

            // prepare the base-lists, as the join itself just checks for *equality* on all levels and performs only *addition* of labels
            prepare_lists(i, intermediate_targets);

            // sort the baselists
            lists[0].sort_level(0, level_translation_array);
            lists[1].sort_level(0, level_translation_array);

            // perform join
            // if two_result_lists is set, the tree saves the last two lists and does not joint to the final level
            if (!two_result_lists || join_to_level != depth - 1)
                join_stream(join_to_level);
            else
                join_stream(join_to_level - 1, lists[depth + 1]);

            // Empty List.
            ASSERT(lists[join_to_level + 2].get_load() != 0 && "list empty");

            // if not finished: sort the resulting list
            if (likely(i != (1ULL << (this->depth - 1)) - 1)) {
                lists[join_to_level + 2].sort_level(join_to_level + 1, level_translation_array);
            }
                //else restore the original baselists for the next run
            else {
                restore_baselists(i + 1, intermediate_targets);
                restore_label(intermediate_targets, two_result_lists);
            }
        }
    }

    ///
    /// \param i
    /// \param intermediate_targets
    void restore_label(std::vector<std::vector<LabelType>> &intermediate_targets, bool two_result_lists) noexcept {
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
	        for (int a = 0; a < 2; ++a) {
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

        //NOTE: Before applying BDD-Solver the label has actually to be negated on all coordinates, is here the right place to do so?
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
	        for (int k = 0; k < amount_negates; ++k) {
		        translate_level(&k_lower, &k_higher, k + 1, level_translation_array);
		        lists[0][ind].get_label().neg(k_lower, k_higher);
	        }
        }

        // calculate number of old targets to eliminate from baselists
        int old_targets = Count_Trailing_Zeros(second - 1);

        // determine which parts of the labels have to be negate
        amount_negates = count_carry_propagates(second);

        for (uint64_t ind = 0; ind < lists[1].get_load(); ++ind) {
            //get rid of intermediate targets from previous levels
            for (unsigned int j = 0; j < old_targets; ++j) {
	            translate_level(&k_lower, &k_higher, j, level_translation_array);
	            LabelType::sub(lists[1][ind].get_label(), lists[1][ind].get_label(),
	                       intermediate_targets[j][((second - 1u) >> (j + 1u)) - 1], k_lower, k_higher);
            }

            for (int k = 0; k < amount_negates; ++k) {
	            translate_level(&k_lower, &k_higher, k, level_translation_array);
	            lists[1][ind].get_label().neg(k_lower, k_higher);
            }
        }
    }

    ///
    /// \param i
    /// \param intermediate_targets
    void prepare_lists(int i, std::vector<std::vector<LabelType>> &intermediate_targets) noexcept {
        /// first, second are the positions of the lists to prepare
        int first = 2 * i;
        int second = 2 * i + 1;

	    uint64_t k_lower, k_higher;

        // determine which parts of the labels of first baselist have to be negated
        if (first != 0) {
            // IMPORTANT: the first-2 is within the function. Is this intuitively?
            unsigned int amount_negates = count_carry_propagates(first);

            // negate corresponding parts of the labels
            for (uint64_t ind = 0; ind < lists[0].get_load(); ++ind) {
	            for (int k = 0; k < amount_negates; ++k) {
		            translate_level(&k_lower, &k_higher, k+1, level_translation_array);
		            lists[0][ind].get_label().neg(k_lower, k_higher);
	            }
            }
        }

        if (second == 1) {
            // if we work on the first two baselists (which means the ones on the far left side of the tree.) We need to
            // negate the label of each element within first list.
            for (uint64_t ind = 0; ind < lists[0].get_load(); ++ind) {
	            translate_level(&k_lower, &k_higher, 0, level_translation_array);

	            lists[1][ind].get_label().neg(k_lower, k_higher);
                LabelType::add(lists[1][ind].get_label(), lists[1][ind].get_label(), intermediate_targets[0][0], k_lower, k_higher);
            }
        } else {
            // calculate number of old targets to eliminate from baselists and number of new intermediate targets to add
            int old_targets = Count_Trailing_Zeros(second - 1);
            int new_targets = Count_Trailing_Zeros(second + 1);

            // determine which parts of the labels have to be negated
            // IMPORTANT: the second-2 is within the function. Is this intuitively?
            unsigned int amount_negates = count_carry_propagates(second);

            for (uint64_t ind = 0; ind < lists[1].get_load(); ++ind) {
                //get rid of intermediate targets from previous levels
                for (unsigned int j = 0; j < old_targets; ++j) {
	                translate_level(&k_lower, &k_higher, j, level_translation_array);
	                LabelType::sub(lists[1][ind].get_label(), lists[1][ind].get_label(),
	                           intermediate_targets[j][((second - 1u) >> (j + 1u)) - 1], k_lower, k_higher);
                }

                for (int k = 0; k < amount_negates; ++k) {
	                translate_level(&k_lower, &k_higher, k+1, level_translation_array);
	                lists[1][ind].get_label().neg(k_lower, k_higher);
                }

                //add new intermediate targets
                for (unsigned int j = 0; j < new_targets; ++j) {
	                translate_level(&k_lower, &k_higher, j, level_translation_array);
	                LabelType::add(lists[1][ind].get_label(), lists[1][ind].get_label(),
	                           intermediate_targets[j][((second + 1u) >> (j + 1u)) - 1], k_lower, k_higher);
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
	static void create_odlyzko_list(BDD_list & in, Odlyzko_list & out, const uint32_t n, const uint32_t l) noexcept {
        Odlyzko_element el{};
        //NOTE: 10 ok?
        std::vector<uint64_t> border_indices(10);

        //NOTE: resize odlzyko list to expected size
        for (uint64_t i = 0; i < in.get_load(); ++i) {
            int num_border_cases = 0;
            //remember index of current bdd-element via value field of odl-list
            el.get_value().data()[0] = i;
            for (int j = 0; j < n + l; ++j) {
                auto coordinate = in[i].get_label().data()[j].get_value();

                //if border case, remember position
                if ((coordinate == 0) || (coordinate == q - 1) ||
		                (coordinate == q/2 - 1) || (coordinate == q/2)) {
                    border_indices[num_border_cases] = j;
                    num_border_cases++;
                }
                //otherwise hash to the sign
                else if (coordinate < q/2)
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

    // IMPORTANT: The two lists L1, L2 __MUST__ be prepared befpr calling this function.
	//                                                 Out
	//                                      +----------------------+
	//                                      |           |          |
	//                                      +----------------------+
	// Merge on k_upper                                 |
	//                        +-------------------------+-------------------------+  Stream Merge
	//                        |                Merge on target                    |
	//            +-----------------------+                          + - - - - - --- - - - - +
	//            |           |           | Intermediate List        |            |          | NOT Saved
	//            +-----------+-----------+                          +- - - - - - | - - - - -+
	//                       i_L                                                  |  Merge on k_lower
	//                                                              +-------------+-----------+
	//                                                              |    merge on equal       |
	//                                                   +-----------------------+ +-----------------------+
	//                                                   |          |            | |          |            |
	//                                                   +----------+------------+ +----------+------------+
	//                                                  k_lower1   L_1      k_upper2         L_2      k_upper2
	//                                                            k_upper1=k_lower2
    /// \param out			Output list
    /// \param iL			Input list, will be sorted on [k_lower2, k_upper2]
    /// \param L1			Input list, will be sorted on [k_lower1, k_upper1]
    /// \param L2			Input list, will be sorted on [k_lower1, k_upper1]
    /// \param k_lower1
    /// \param k_upper1
    /// \param k_lower2
    /// \param k_upper2
	static void twolevel_streamjoin(List &out, List iL, List &L1, List &L2,
								 const uint64_t k_lower1, const uint64_t k_upper1,const uint64_t k_lower2, const uint64_t k_upper2) noexcept {
    	ASSERT(k_lower1 < k_upper1 && 0 < k_upper1 && k_lower2 < k_upper2 && 0 < k_upper2 && k_lower1 <= k_lower2 && k_upper1 <= k_upper2);
		// internal variables.
	    const uint32_t filter = -1;
	    std::pair<uint64_t, uint64_t> boundaries;
	    ElementType e;
		uint64_t i = 0, j = 0;

		iL.sort_level(k_lower2, k_upper2);
		L1.sort_level(k_lower1, k_upper1);
		L2.sort_level(k_lower1, k_upper1);

		while (i < L1.load() && j < L2.load()) {
			if (L2[j].is_greater(L1[i], k_lower1, k_upper1)) {
				i++;
			} else if (L1[i].is_greater(L2[j], k_lower1, k_upper1)) {
				j++;
			} else {
				uint64_t i_max, j_max;
				for (i_max = i + 1; i_max < L1.load() && L1[i].is_equal(L1[i_max], k_lower1, k_upper1); i_max++) {}
				for (j_max = j + 1; j_max < L2.load() && L2[j].is_equal(L2[j_max], k_lower1, k_upper1); j_max++) {}

				uint64_t jprev = j;

				// we have found equal elements. But this time we dont have to save the result. Rather we stream join everything up to the final solution.
				for (; i < i_max; ++i) {
					for (j = jprev; j < j_max; ++j) {
						ElementType::add(e, L1[i], L2[j], 0, LabelLENGTH, -1);
						boundaries = iL.search_boundaries(e, k_lower2, k_upper2);

						// finished?
						if (boundaries.first == boundaries.second) {
							break;
						}

						for (size_t l = boundaries.first; l < boundaries.second; ++l) {
							out.add_and_append(e, iL[l], filter);
						}
					}
				}
			}
		}
    }


	// Schematic View on how the algorithm Works.
	// The algorithm does not care what weight is on each element, nor does it care about the output size.
	// IMPORTANT: The Output size IS NOT GUESSED ahead. Because we dont know hom many representations of the solution there are.
	//                        Out
	//             +-----------------------+
	//             |                       |
	//             +-----------^-----------+
	//                         |
	//            +------------+------------+
	//            |        Target           |   The merge is on [k_lower, k_upper] coordinates.
	//+-----------+-----------+ +-----------+-----------+
	//|                       | |                       |
	//+-----------------------+ +-----------------------+
	//           L_1                       L_2
    /// given two lists L1, L2. It outputs every elements which are the same on the specified coordinates in 'out'
    /// \param out 	Output List
    /// \param L1 	Input List (sorted)
    /// \param L2 	Input List (will be changed. Target is added to every element in it.)
    /// \param ta __MUST__ be an uint64_t array containing two elements. The first will be used as the lower cooridnate bound to match the elements on, where as the latter one will be the upper bound.
    static void join2lists(List &out, List &L1, List &L2, const LabelType &target,
							const std::vector<uint64_t> &lta, const bool prepare=true) noexcept {
    	ASSERT(lta.size() >=1 );
	    join2lists(out, L1, L2, target, lta[0], lta[1], prepare);
    }

	///
	/// \param out
	/// \param L1
	/// \param L2
	/// \param target
	/// \param k_lower
	/// \param k_upper
	/// \param prepare
	static void join2lists(List &out, List &L1, List &L2,
	                       const LabelType &target,
	                       const uint32_t k_lower, const uint32_t k_upper,
	                       bool prepare=true) noexcept {
    	ASSERT(k_lower < k_upper && 0 < k_upper);
    	uint64_t i= 0,j = 0;
	    const uint32_t filter = -1;

	    if ((!target.is_zero()) && (prepare)) {
		    for (size_t s = 0; s < L2.load(); ++s) {
			    LabelType::sub(L2[s].label, L2[s].label, target, k_lower, k_upper);
		    }
	    }

	    L1.sort_level(k_lower, k_upper);
	    L2.sort_level(k_lower, k_upper);

	    while (i < L1.load() && j < L2.load()) {
		    if (L2[j].is_greater(L1[i], k_lower, k_upper)) {
				i++;
			} else if (L1[i].is_greater(L2[j], k_lower, k_upper)) {
				j++;
			} else {
			    uint64_t i_max, j_max;
			    // if elements are equal find max index in each list, such that they remain equal
			    for (i_max = i + 1; i_max < L1.load() && L1[i].is_equal(L1[i_max], k_lower, k_upper); i_max++) {}
			    for (j_max = j + 1; j_max < L2.load() && L2[j].is_equal(L2[j_max], k_lower, k_upper); j_max++) {}

			    uint64_t jprev = j;

			    for (; i < i_max; ++i) {
				    for (j = jprev; j < j_max; ++j) {
					    out.add_and_append(L1[i], L2[j], filter);
				    }
			    }
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
    static void streamjoin4lists(List &out, List &L1, List &L2, List &L3, List &L4,
	                        	 const LabelType &target,
	                             const std::vector<uint64_t> &lta,
	                        	 const bool prepare=true) noexcept {
		ASSERT(lta.size() == 3);
		// limits: k_lower1, k_upper1 for the lowest level tree. And k_lower2, k_upper2 for highest level. There are
		// only two levels..., so obviously k_upper1=k_lower2
		const uint64_t k_lower1 = lta[0], k_upper1 = lta[1];
		const uint64_t k_lower2 = lta[1], k_upper2 = lta[2];
		streamjoin4lists(out, L1, L2, L3, L4, target, k_lower1, k_upper1, k_lower2, k_upper2, prepare);
    }

	/// TODO
	/// \param out
	/// \param L1
	/// \param L2
	/// \param L3
	/// \param L4
	/// \param target
	/// \param k_lower1
	/// \param k_upper1
	/// \param k_lower2
	/// \param k_upper2
	/// \param prepare
	static void streamjoin4lists(List &out, List &L1, List &L2, List &L3, List &L4,
	                             const LabelType &target,
	                             const uint64_t k_lower1, const uint64_t k_upper1,
	                             const uint64_t k_lower2, const uint64_t k_upper2,
	                             const bool prepare=true) noexcept {
		ASSERT(k_lower1 < k_upper1 && 0 < k_upper1 && k_lower2 < k_upper2 && 0 < k_upper2 && k_lower1 <= k_lower2 && k_upper1 < k_upper2
		        && L1.load() > 0 && L2.load() > 0 && L3.load() > 0 && L4.load() > 0);
		// Intermediate Element, List, Target
		List iL{L1.size()*4};
		LabelType R, zero; R.random(); zero.zero();

		// prepare baselists
	    if ((!target.is_zero()) && (prepare)) {
		    for (size_t i = 0; i < L2.load(); ++i) {
		    	LabelType::add(L2[i].label, L2[i].label, R, k_lower1, k_upper1);
		    }

		    for (size_t i = 0; i < L4.load(); ++i) {
			    LabelType::sub(L4[i].label, L4[i].label, R, k_lower1, k_upper1);
			    // add is on the full length
			    LabelType::sub(L4[i].label, L4[i].label, target, k_lower1, k_upper2);
		    }
	    }

		join2lists(iL, L1, L2, zero, k_lower1, k_upper1, false);

		// early exit
		if (iL.load() == 0) {
			return;
		}

		// Now run the merge procedure for the right part of the tree.
		twolevel_streamjoin(out, iL, L3, L4, k_lower1, k_upper1, k_lower2, k_upper2);
	}

	// Schematic view on the Algorithm.
	// The algorithm ignores any facts about th weight, nor does it guess the output size.
	// The intermediate list `iL` is created every time you call this function. So if called repeatedly, maybe write it yourself, or set the list to static.
	// The difference to the function `streamjoin4lists` is that the lists `L3` and `L4` are simulated on the fly. This
	// means that list `L1` and `L2` are used for this.
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
	//+-----------------------+ +-----------------------+ + - - - - - - - - - - - + + - - - - - - - - - - - +
	//| k_lower   |  k_higher | | k_lower   |  k_higher | | NOT SAVED/PASSED      | | NOT_SAVED/PASSED      | Level 0
	//+-----------------------+ +-----------------------+ +- - - - - - - - - - - -+ +- - - - - - - - - - - -+
	//           L_1                       L_2                      L_1                       L_4
	//
	///
	/// \param out output list. All targets which are a solution to the 4 kxor problem.
	/// \param L1 	(sorted)
	/// \param L2 	(sorted+R+target) intermediate target is added.
	/// \param target
	/// \param lta
	static void streamjoin4lists_twolists(List &out, List &L1, List &L2,
	                             const LabelType &target, const std::vector<uint64_t> &lta,
	                             bool prepare=true) noexcept {
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
	                             bool prepare=true) noexcept {
		ASSERT(k_lower1 < k_upper1 && 0 < k_upper1 && k_lower2 < k_upper2 && 0 < k_upper2 && k_lower1 <= k_lower2 && k_upper1 <= k_upper2 && L1.load() > 0 && L2.load() > 0);
		// Intermediate Element, List, Target
		List iL{out.size()*2};
		LabelType R, zero;
		R.random(); zero.zero();

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
	/// \param lta		translation array with the limits to match on, for each lvl
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
			L.push_back(List{out.size()*2});
		}

		// Intermediate Target
		LabelType R, R1, R3, R5, R13, R57, zero;
		R1.random(); R3.random(); R5.random(); zero.zero();
		R13.random(); R57.random();
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
				LabelType::add(L[3][k].label, L[3][k].label, R3,  k_lower1, k_upper1);
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
			List &L1    = L[k*2];
			List &L2    = L[k*2 + 1];
			List &iL    = L[8 + k/2];      // Intermediate list for the normal join
			List &tout  = L[8 + k/2 + 2];  // Intermediate list for the stream join

			// base list join
			join2lists(iL, L1, L2, vr[k/2], lower_lta, false);

			// then streamjoin
			twolevel_streamjoin(tout, iL, L1, L2, k_lower1, k_upper1, k_lower2, k_upper2);
		}

		// Last Join
		// in L[10] is the result of the stream join between L[0-3]
		// in L[11] is the result of the stream join between L[4-7]
		std::vector<uint64_t> last_lta{{k_lower3, k_upper3}};
		join2lists(out, L[10], L[11], target, last_lta, false);
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

    	const uint64_t size = in1.size()* in2.size();
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
    const uint64_t get_size() const { return lists.size(); }
    const uint64_t get_basesize() const { return base_size; }
	const auto& get_level_translation_array() const noexcept { return level_translation_array; }
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
    inline void search_in_level_l(ElementType &e1, const ElementType &e2, const ElementType &e3, const uint8_t level,
                                  std::vector<std::pair<uint64_t, uint64_t>> &boundaries,
                                  std::vector<uint64_t> &indices) {
	    ElementType::add(e1, e2, e3);

	    uint64_t k_lower, k_higher;
	    translate_level(&k_lower, &k_higher, level + 1, level_translation_array);

        boundaries[level] = lists[level + 2].search_boundaries(e1, k_lower, k_higher);
        indices[level] = boundaries[level].first;
    }

    /// Called from `join_stream_join`. DO NOT USE IT.
    /// \param l			lvl
    /// \param boundaries
    /// \param indices
    /// \return new lvl
    int increment_previous_level(int l, std::vector<std::pair<uint64_t, uint64_t>> &boundaries,
                                        std::vector<uint64_t> &indices) {
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
    void join_stream_internal(uint64_t level, List &target) {
        std::vector<std::pair<uint64_t, uint64_t>> boundaries(level);
        std::vector<ElementType> a(level);
        std::vector<uint64_t> indices(level);

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
            if (lists[1][j].is_greater(lists[0][i], k_lower0, k_higher0))
                i++;

            else if (lists[0][i].is_greater(lists[1][j], k_lower0, k_higher0))
                j++;

            else {
                uint64_t i_max, j_max;
                // if elements are equal find max index in each list, such that they remain equal
                for (i_max = i + 1; i_max < lists[0].load() && lists[0][i].is_equal(lists[0][i_max], k_lower0, k_higher0); i_max++) {}
                for (j_max = j + 1; j_max < lists[1].load() && lists[1][j].is_equal(lists[1][j_max], k_lower0, k_higher0); j_max++) {}

                // for each matching tuple
                uint64_t jprev = j;

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
                                    if (l == 0)
                                        search_in_level_l(a[0], lists[0][i], lists[1][j], l, boundaries, indices);
                                    else
                                        search_in_level_l(a[l], a[l - 1], lists[l + 1][indices[l - 1]], l, boundaries,
                                                          indices);

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
                                if (l == level)
                                	l--;

                                l = increment_previous_level(l, boundaries, indices);
                                if (indices[0] == boundaries[0].second)
                                    stop = true;
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

    /// array to translate lvl x to upper and lower bound
	const std::vector<uint64_t> level_translation_array;

	/// array to translate lvl x to filter
	const std::vector<std::vector<uint8_t>> level_filter_array;

	/// minimum nr of 2 in an element till we discard it
	const uint32_t filter_nr = 200;

	std::vector<List> lists;
    unsigned int depth;
    uint64_t base_size;
};

/// \param out
/// \param obj
/// \return
template<class List>
std::ostream &operator<<(std::ostream &out, const Tree_T<List> &obj) {
    for (uint64_t i = 0; i < obj.get_size(); ++i) {
        out << obj[i];
    }

    return out;
}



struct ExtendenTreeConfig {
public:
	const uint32_t n = 100;

	const uint32_t d = 2;

	const uint32_t threads = 1;

	// Das ding ist momentan der platzhalter fuer n-k. Das das ist mehr oder weniger das obere ende der coordonaten auf denen gematch werden kann
	const uint32_t MatchingOffset = 10;
	// coordinates to match on.
	const std::array<uint32_t, 4> l = {0, 2, 5, 9};
};


#include <tuple>
// The structure that encapsulates index lists
template <size_t... Is>
struct index_list {};

// Collects internal details for generating index ranges [MIN, MAX)
namespace detail
{
	// Declare primary template for index range builder
	template <size_t MIN, size_t N, size_t... Is>
	struct range_builder;

	// Base step
	template <size_t MIN, size_t... Is>
	struct range_builder<MIN, MIN, Is...> {
		typedef index_list<Is...> type;
	};

	// Induction step
	template <size_t MIN, size_t N, size_t... Is>
	struct range_builder : public range_builder<MIN, N - 1, N - 1, Is...>{};
}

// Meta-function that returns a [MIN, MAX) index range
template<size_t MIN, size_t MAX>
using index_range = typename detail::range_builder<MIN, MAX>::type;

template <typename T, std::size_t ...Is>
struct selector {
	using type = std::tuple<typename std::tuple_element<Is, T>::type...>;
};


template <size_t I, typename... Ts>
struct get_type_at_helper;

template <size_t I, typename Head, typename... Tail>
struct get_type_at_helper<I, Head, Tail...>
{ typedef typename get_type_at_helper<I - 1, Tail...>::type type; };

template <typename Head, typename... Tail>
struct get_type_at_helper<0, Head, Tail...>
{ typedef Head type; };

template <int I, typename... Ts>
using get_type_at_t = typename get_type_at_helper<I, Ts...>::type;

template <typename...> struct type_list {};


template<class List>
#if __cplusplus > 201709L
	requires TreeAble<List>
#endif
class ExtendedTree {
public:
	// Internal datatypes of the lists
	typedef typename List::ElementType ElementType;
	typedef typename List::ValueType ValueType;
	typedef typename List::LabelType LabelType;

	typedef typename List::ValueContainerType ValueContainerType;
	typedef typename List::LabelContainerType LabelContainerType;

	typedef typename List::ValueDataType ValueDataType;
	typedef typename List::ValueDataType LabelDataType;

	typedef typename List::MatrixType MatrixType;

	// TODO generalize
	using ArgumentLimbType = uint64_t;
	using Extractor = WindowExtractor<LabelType, ArgumentLimbType>;

	// internal data types lengths
	constexpr static uint32_t ValueLENGTH = ValueType::LENGTH;
	constexpr static uint32_t LabelLENGTH = LabelType::LENGTH;

	// Hashes
	template<const uint32_t l, const uint32_t h, typename T1, typename T2>
	static T1 Hash(T2 a) {
		constexpr T2 mask = (~((T2(1u) << l) - 1u)) &
							( (T2(1u)  << h) - 1u);
		return (a&mask)>>l;
	}

	// first lvl of stream join, write the output elements into a hashmap.
	template<const ExtendenTreeConfig &config, const uint32_t cl, class HMOut, class HMIn>
	static void streamJoin_internal(HMOut &out, HMIn &in, const List &L2) {
		// internal data types of the hashmap
		typedef typename HMIn::H            HMInHashType;
		typedef typename HMIn::T            HMInType;
		typedef typename HMIn::IndexType    HMInIndexType;
		typedef typename HMOut::H           HMOutHashType;
		typedef typename HMOut::T           HMOutType;
		typedef typename HMOut::IndexType   HMOutIndexType;

		// thread id
		const uint32_t tid = config.threads == 1 ? 0 : Thread::get_tid();

		// list offsets for threads
		const size_t spos = L2.start_pos(tid);
		const size_t epos = L2.end_pos(tid);

		auto extractor = [] (const LabelType &e) {
			return Extractor::template extract<config.MatchingOffset - config.l[cl+1], config.MatchingOffset - config.l[cl]>(e);
		};

		// helper variables
		HMOutIndexType npos[2];
		HMInIndexType load, pos;

		for (npos[1] = spos; npos[1] < epos; npos[1]++) {
			auto extracted_label = extractor(L2.data_label(npos[1]));
			HMInType data = in.hash(extracted_label);
			pos = in.find1(data, load);

			while (pos < load) {
				HMInIndexType data1 = in.template traverse<0, 1>(data, pos, npos, load);
				out.insert1(data1, npos, tid);
			}
		}
	}

	// variadic type of the function above
	template<const ExtendenTreeConfig &config, const uint32_t cl, class HMOut, class HMIn, class... HMs>
	static void streamJoin_internal(HMOut *out, HMIn *in, const List &L2) {
		// TODO finish implementing constexpr size_t n = sizeof...(HMs);
	}

	///
	/// \tparam config
	/// \tparam cl		current lvl
	/// \tparam HM
	/// \tparam HMs
	/// \param L1
	/// \param L2
	/// \param target
	template<const ExtendenTreeConfig &config, uint32_t cl, typename ...HMs>
	static void streamjoin(List &L1, List &L2, const LabelType &target) {
		static_assert(cl < config.d);
		static_assert(sizeof...(HMs) == config.d+1);

		// thread id
		const uint32_t tid = config.threads == 1 ? 0 : Thread::get_tid();

		// generate intermediate targets
		LabelType targets[config.d];
		for (uint32_t i = 0; i < config.d-1; ++i) {
			targets[i].random();
		}
		targets[config.d-1] = target;

		// TODO add first intermediate target.

		// create the first hash map and hash the first list.
		using HM = get_type_at_t<0, HMs...>;


		auto *hm = new HM();
		hm->reset(tid);
		OMP_BARRIER

		// this hashes the first list into the first hashmap
		hm->hash1(L1, L1.size(tid), tid, [](const LabelType&e) -> ArgumentLimbType {
			return Extractor::template extract<config.MatchingOffset - config.l[1], config.MatchingOffset>(e);
		});


		using HM2 = get_type_at_t<1, HMs...>;
		auto *hm2 = new HM2();

		// test
		// using HMs2 = typename selector<std::tuple<HMs...>, index_range<0, 1>>::type;


		streamJoin_internal<config, cl+1, HM2, HM, HMs...>(hm2, hm, L2);
	}
};

#endif//SMALLSECRETLWE_TREE_H
