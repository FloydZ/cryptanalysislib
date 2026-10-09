#ifndef SMALLSECRETLWE_METASORT_H
#define SMALLSECRETLWE_METASORT_H

#include "common.h"
#include "counting_sort.h"
#include "heapsort.h"
#include "merge_sort.h"
#include "robinhoodsort.h"
#include "timsort.h"
#include "ska_sort.h"
#include "vergesort.h"
#include "vv_radixsort.h"


#ifdef USE_AVX2
#include "djb_sort.h"
#endif

#include "sort/sorting_network/common.h"

#include "parallel.h"
#include "ips4o.h"

namespace cryptanalysislib {
	/// sequential comparison sort of [first, last), same contract as `std::sort`
	/// NOTE: currently forwards to `std::sort`. This is the single entry point,
	/// 	so the backend can be exchanged later.
	/// NOTE: the `requires` clause makes these overloads more constrained than
	/// 	`std::sort`, so an unqualified `sort(...)` which sees both (e.g. via
	/// 	`using namespace std;` and ADL) is not ambiguous.
	template<class RandIt>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt>
#endif
	constexpr inline void sort(RandIt first,
	                           RandIt last) noexcept {
		std::sort(first, last);
	}

	/// same as above, but with a custom comparator
	template<class RandIt,
	         class Compare>
#if __cplusplus > 201709L
		requires std::random_access_iterator<RandIt>
#endif
	constexpr inline void sort(RandIt first,
	                           RandIt last,
	                           Compare comp) noexcept {
		std::sort(first, last, comp);
	}
} // end namespace cryptanalysislib
#endif//SMALLSECRETLWE_METASORT_H
