#ifndef SMALLSECRETLWE_METASORT_H
#define SMALLSECRETLWE_METASORT_H

#include "common.h"
#include "crumsort.h"
#include "quadsort.h"
#include "ska_sort.h"
#include "crumsort.h"
#include "robinhoodsort.h"
#include "quadsort.h"
#include "vergesort.h"
#include "vv_radixsort.h"

#ifdef USE_AVX2
#include "djb_sort.h"
#include "sort/sorting_network/avx2.h"
#endif

#include "sort/sorting_network/common.h"
#endif //SMALLSECRETLWE_METASORT_H
