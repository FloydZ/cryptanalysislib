#ifndef SMALLSECRETLWE_METASORT_H
#define SMALLSECRETLWE_METASORT_H

#include "crumsort.h"
#include "quadsort.h"
#include "ska_sort.h"
#include "crumsort.h"
#include "robinhoodsort.h"
#include "quadsort.h"
#include "vergesort.h"

#ifdef USE_AVX2
#include "sort/sorting_network/avx2.h"
#endif

#endif //SMALLSECRETLWE_METASORT_H
