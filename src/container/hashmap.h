#ifndef CRYPTANALYSISLIB_HASHMAP_H
#define CRYPTANALYSISLIB_HASHMAP_H

#include "hash/simple.h"

#include "container/hashmap/common.h"
#include "container/hashmap/simple.h"
#include "container/hashmap/simple2.h"

#include "container/hashmap/hopscotch.h"
#include "container/hashmap/ska_flat.h"

#ifdef USE_AVX2
#include "container/hashmap/avx2.h"
#endif
#endif //CRYPTANALYSISLIB
