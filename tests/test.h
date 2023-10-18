#ifndef SMALLSECRETLWE_TEST_H
#define SMALLSECRETLWE_TEST_H

#define TESTSIZE 100
#define SORT_INCREASING_ORDER

#include "element.h"
#include "container/fq_vector.h"
#include "matrix/fq_matrix.h"
#include "list/list.h"
#include "tree.h"


constexpr uint32_t n    = 30;
constexpr uint32_t k    = 30;
constexpr uint32_t q    = 3;
constexpr uint32_t d    = 2;
using T 				= uint8_t;
using kAryType          = kAry_Type_T<uint32_t, uint64_t, q>;
using kAryContainer     = kAryContainer_T<T, n, q>;
using kAryContainer2    = kAryContainer_T<T, k, q>;
using Label             = kAryContainer2;
using Value             = kAryContainer;


static std::vector<uint64_t> __level_translation_array{{0, 5, 10, 15, n}};
static std::vector<std::vector<uint8_t>> __level_filter_array{{ {{4,0,0}}, {{1,0,0}}, {{1,0,0}}, {{0,0,0}} }};
#endif //SMALLSECRETLWE_TEST_H
