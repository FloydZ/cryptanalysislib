#ifndef CRYPTANALYZELIB_CONTAINER_VECTOR
#define CRYPTANALYZELIB_CONTAINER_VECTOR

#include <cstdint>
#include "helper.h"
#include "container/common.h"
#include "alloc/alloc.h"


/// simple data container holding `length` Ts
/// \tparam T base type
/// \tparam size number of elements
template<typename T, const size_t size>
using page_vector = std::vector<T, STDAllocatorWrapper<T, FreeListPageMallocator<1u << 12u, size> >>;
#endif
