#ifndef CRYPTANALYZELIB_CONTAINER_VECTOR
#define CRYPTANALYZELIB_CONTAINER_VECTOR

#include <cstdint>
#include "helper.h"
#include "container/common.h"
#include "alloc/alloc.h"
#include "math/math.h"


/// simple data container holding up to `size` Ts. Every allocation of the
/// vector is a page, freed pages are reused. A page holds the next power of
/// two >= `size` elements, so the doubling growth of `std::vector` fits.
/// NOTE: the capacity is limited to a page: a larger allocation fails
/// 	(returns `nullptr`).
/// NOTE: before, the page size was `size` bytes (not elements), and the page
/// 	allocator had no `allocate(n)`, i.e. this did not compile.
/// \tparam T base type
/// \tparam size number of elements
template<typename T, const size_t size>
using page_vector = std::vector<T, STDAllocatorWrapper<T, FreeListPageMallocator<1u << 12u, (size_t(1) << ceil_log2(size)) * sizeof(T)> >>;
#endif
